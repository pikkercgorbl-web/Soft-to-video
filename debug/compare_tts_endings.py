#!/usr/bin/env python3
"""Controlled raw TTS comparisons. Default: prepare a plan; --run synthesizes locally.

No Gemini, stress correction, postprocessing or production cache is involved.
Forced EOS and speech-end annotations must come from logs/listening, not duration.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import download_pexels as dp
from project_state import atomic_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def make_cases(text: str, seeds: list[int], high: float, low: float,
               punctuation: bool = False) -> list[dict]:
    if not text.strip():
        raise ValueError('Text is empty')
    if '\u0301' not in text:
        raise ValueError('Input must contain U+0301 stress marks for the paired comparison')
    if high == low:
        raise ValueError('Baseline and comparison exaggeration must differ')
    plain = text.replace('\u0301', '')
    cases = []
    for seed in seeds:
        variants = [('stress_baseline', text, high), ('plain_baseline', plain, high),
                    ('stress_lower', text, low), ('plain_lower', plain, low),
                    ('stress_repeat', text, high)]
        if punctuation:
            variants.append(('stress_ellipsis', text.rstrip().rstrip('.!?…') + '…', high))
        # Fixed shuffle reduces simple position/order bias; repeat checks server consistency.
        random.Random(seed).shuffle(variants)
        for label, variant_text, exaggeration in variants:
            payload = dp.build_chatterbox_payload(variant_text, seed=seed, split_text=False)
            payload['exaggeration'] = exaggeration
            cases.append({'id': f'{seed}_{label}', 'seed': seed, 'variant': label,
                          'payload': payload})
    return cases


def audio_info(path: Path) -> dict:
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                             '-show_entries', 'stream=sample_rate,channels:format=duration',
                             '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    stream = data['streams'][0]
    return {'duration_seconds': float(data['format']['duration']),
            'sample_rate': int(stream['sample_rate']), 'channels': int(stream['channels'])}


def write_reviews(output: Path, records: list[dict]) -> None:
    # This file is for human annotations; never auto-label noise or forced EOS.
    fields = ['id', 'variant', 'seed', 'status', 'file', 'started_utc', 'ended_utc',
              'elapsed_seconds', 'duration_seconds', 'speech_end_seconds',
              'tail_noise_0_to_3', 'naturalness_1_to_5', 'missing_words',
              'forced_eos_from_server_log', 'notes']
    with (output / 'review.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def execute(cases: list[dict], output: Path) -> list[dict]:
    records = []
    for case in cases:
        record = {key: case[key] for key in ('id', 'seed', 'variant')}
        record.update(status='running', started_utc=utc_now(), payload=case['payload'])
        path = output / (case['id'] + '_raw.wav')
        atomic_json(output / (case['id'] + '.json'), record)
        start = time.perf_counter()
        print(f"Generating {case['id']}", flush=True)
        try:
            response = dp.requests.post(dp.CHATTERBOX_TTS_URL, json=case['payload'],
                                        timeout=dp.TTS_SERVER_TIMEOUT)
            if not response.ok:
                raise RuntimeError(f'TTS HTTP {response.status_code}')
            if len(response.content) < 12 or response.content[:4] not in (b'RIFF', b'RF64') or response.content[8:12] != b'WAVE':
                raise ValueError('TTS returned non-WAV content')
            path.write_bytes(response.content)  # Preserve exact response bytes, including tail.
            record['file'] = path.name
            record['sha256'] = hashlib.sha256(response.content).hexdigest()
            record.update(audio_info(path))
            record['status'] = 'completed'
        except Exception as error:
            # Avoid response bodies/URLs which could contain credentials.
            record.update(status='failed', error_type=type(error).__name__)
        finally:
            record['ended_utc'] = utc_now()
            record['elapsed_seconds'] = round(time.perf_counter() - start, 3)
            records.append(record)
            atomic_json(output / (case['id'] + '.json'), record)
            write_reviews(output, records)
        if record['status'] == 'failed':
            # A timed-out server may still be generating. Do not enqueue more work.
            print('Stopped after failure. Check server log before starting a new run.')
            break
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-file', type=Path,
                        default=ROOT / 'debug/chatterbox_report/chunk2_seg1.txt')
    parser.add_argument('--seeds', type=int, nargs='+', default=[100001, 100002, 100003])
    parser.add_argument('--baseline-exaggeration', type=float, default=1.0)
    parser.add_argument('--comparison-exaggeration', type=float, default=0.5)
    parser.add_argument('--punctuation', action='store_true', help='Add an ellipsis variant')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'debug/chatterbox_report/ab_runs')
    parser.add_argument('--run', action='store_true', help='Send requests; otherwise only save the plan')
    args = parser.parse_args()
    dp.load_settings()
    try:
        text = args.text_file.read_text(encoding='utf-8-sig').strip()
        if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
            raise ValueError('Seeds must be unique nonnegative integers')
        cases = make_cases(text, args.seeds, args.baseline_exaggeration,
                           args.comparison_exaggeration, args.punctuation)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.run:
        try:
            subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        except (OSError, subprocess.SubprocessError):
            parser.error('ffprobe is required in PATH before generation')
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    output = args.output_root / name
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'plan.json', {'created_utc': utc_now(), 'cases': cases,
                                     'postprocessing': False, 'splitting': False})
    print(f'{len(cases)} requests planned. Results: {output}')
    if not args.run:
        print('Plan only: no TTS requests sent. Add --run to generate.')
        return 0
    records = execute(cases, output)
    complete = len(records) == len(cases) and all(r['status'] == 'completed' for r in records)
    atomic_json(output / 'summary.json', {'status': 'completed' if complete else 'partial',
                                        'planned': len(cases), 'attempted': len(records)})
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
