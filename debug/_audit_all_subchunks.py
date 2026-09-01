#!/usr/bin/env python3
"""Все подчанки прогона: текст против фактической длительности в WAV.

Отвечает на вопрос, режет ли repetition guard живую речь. Для каждого подчанка
сопоставляются длина текста (нарезка настоящим кодом сервера) и измеренная
длительность сегмента в готовом WAV. Слишком высокая плотность символов
на секунду означает, что часть текста не была озвучена.

Запускать встроенным интерпретатором сервера — нужен его utils:
    C:\\AI_Servers\\Chatterbox\\Chatterbox-TTS-Server\\python_embedded\\python.exe
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SERVER_DIR = Path(r"C:\AI_Servers\Chatterbox\Chatterbox-TTS-Server")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

logging.disable(logging.CRITICAL)

import numpy as np
import soundfile as sf

import utils  # сервера

SERVER_LOG = SERVER_DIR / "logs" / "tts_server.log"
DOWNLOADS = ROOT / "downloads"
TEXTS_DIR = ROOT / "debug" / "logs"
RUN_DATE = "2026-08-20"
CHUNK_SIZE = 100
S3_TOKEN_RATE = 25

# Порядок отправки запросов виден в консольном логе пайплайна по строкам
# "↓ Chatterbox TTS (ya.wav) → NNN_audio.wav".
REQUEST_TO_PIPELINE_CHUNK = [3, 1, 2, 5, 4]


def parse_log() -> list[dict]:
    requests: list[dict] = []
    current: dict | None = None

    for line in SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(RUN_DATE):
            continue
        if "Received /tts request" in line:
            current = {"time": line[11:19], "chunks": [], "total": None}
            requests.append(current)
            continue
        if current is None:
            continue

        match = re.search(r"Generated (\d+) chunk", line)
        if match:
            current["total"] = int(match.group(1))
        if re.search(r"Synthesizing chunk (\d+)/", line):
            current["chunks"].append({"steps": None, "guard": None, "token": None})
        match = re.search(r"repetition of token (\d+)", line)
        if match and current["chunks"]:
            current["chunks"][-1]["token"] = int(match.group(1))
        match = re.search(r"token_repetition=(\w+)", line)
        if match and current["chunks"]:
            current["chunks"][-1]["guard"] = (
                "repetition" if match.group(1) == "True" else "long_tail"
            )
        match = re.search(r"Stopping generation at step (\d+)", line)
        if match and current["chunks"]:
            current["chunks"][-1]["steps"] = int(match.group(1))

    return [r for r in requests if r["chunks"]]


def find_segments(path: Path) -> list[tuple[float, float]]:
    audio, rate = sf.read(str(path), always_2d=True)
    mono = np.abs(audio.mean(axis=1))
    silent = mono < 1e-4
    min_len = int(0.14 * rate)

    gaps: list[tuple[int, int]] = []
    start = None
    for index, is_silent in enumerate(silent):
        if is_silent:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= min_len:
                gaps.append((start, index))
            start = None
    if start is not None and len(silent) - start >= min_len:
        gaps.append((start, len(silent)))

    segments: list[tuple[float, float]] = []
    cursor = 0.0
    for gap_start, gap_end in gaps:
        if gap_start / rate - cursor > 0.15:
            segments.append((cursor, gap_start / rate))
        cursor = gap_end / rate
    total = len(mono) / rate
    if total - cursor > 0.15:
        segments.append((cursor, total))
    return segments


def main() -> int:
    requests = parse_log()
    print(f"Запросов в прогоне: {len(requests)}\n")

    rows: list[dict] = []

    for number, request in enumerate(requests):
        pipeline_chunk = REQUEST_TO_PIPELINE_CHUNK[number]
        text_path = TEXTS_DIR / f"chunk_{pipeline_chunk:02d}_tts_text.txt"
        wav_path = DOWNLOADS / f"{pipeline_chunk:03d}_audio.wav"
        if not text_path.is_file() or not wav_path.is_file():
            print(f"  пропуск запроса #{number + 1}: нет {text_path.name} или {wav_path.name}")
            continue

        text = text_path.read_text(encoding="utf-8")
        subchunks = utils.chunk_text_by_sentences(text, CHUNK_SIZE)
        segments = find_segments(wav_path)

        fingerprint_ok = len(subchunks) == request["total"]
        print("=" * 88)
        print(
            f"Запрос #{number + 1} {request['time']} → пайплайн-чанк {pipeline_chunk} "
            f"({wav_path.name})"
        )
        print(
            f"  подчанков: в логе {request['total']}, в реконструкции {len(subchunks)} "
            f"{'✓' if fingerprint_ok else '✗ реконструкция не совпала'}; "
            f"сегментов в WAV: {len(segments)}"
        )
        if not fingerprint_ok:
            continue

        print(f"  {'№':<3} {'симв':>5} {'шагов':>6} {'ожид,с':>7} {'факт,с':>7} "
              f"{'симв/с':>7}  {'гвард':<11} текст")
        for index, subchunk in enumerate(subchunks):
            meta = request["chunks"][index] if index < len(request["chunks"]) else {}
            steps = meta.get("steps")
            expected = steps / S3_TOKEN_RATE if steps else None
            actual = segments[index][1] - segments[index][0] if index < len(segments) else None
            density = len(subchunk) / actual if actual else None

            rows.append(
                {
                    "chunk": pipeline_chunk,
                    "index": index + 1,
                    "chars": len(subchunk),
                    "guard": meta.get("guard"),
                    "actual": actual,
                    "density": density,
                }
            )

            preview = subchunk if len(subchunk) <= 44 else subchunk[:41] + "..."
            print(
                f"  {index + 1:<3} {len(subchunk):>5} {steps or 0:>6} "
                f"{expected or 0:>7.2f} {actual or 0:>7.2f} {density or 0:>7.1f}  "
                f"{meta.get('guard') or '—':<11} {preview}"
            )
        print()

    print("=" * 88)
    print("ПЛОТНОСТЬ РЕЧИ: символов текста на секунду звука")
    print("=" * 88)
    valid = [r for r in rows if r["density"]]
    densities = sorted(r["density"] for r in valid)
    median = densities[len(densities) // 2]
    print(f"  подчанков измерено: {len(valid)}")
    print(f"  медиана: {median:.1f} симв/с")
    print(f"  диапазон: {densities[0]:.1f} … {densities[-1]:.1f} симв/с")

    for guard in ("repetition", "long_tail"):
        subset = [r["density"] for r in valid if r["guard"] == guard]
        if subset:
            print(
                f"  {guard:<11}: n={len(subset):<3} "
                f"медиана {sorted(subset)[len(subset) // 2]:.1f}, "
                f"максимум {max(subset):.1f} симв/с"
            )

    print()
    print("  Аномалии (плотность выше медианы на 40%+ → текст не успел озвучиться):")
    anomalies = [r for r in valid if r["density"] > median * 1.4]
    if not anomalies:
        print("    не найдено — ни один подчанк не оборван")
    for row in anomalies:
        print(
            f"    чанк {row['chunk']} подчанк {row['index']}: {row['chars']} симв "
            f"за {row['actual']:.2f} с = {row['density']:.1f} симв/с, гвард={row['guard']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
