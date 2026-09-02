#!/usr/bin/env python3
"""Показывает, какие паузы найдёт чистильщик в готовом WAV, и что в них звучит.

Нужен для подбора порогов: по таймкодам из отчёта можно прослушать место в
плеере и понять, попал ли детектор в тот самый "вздох" между предложениями.

Использование:
    python debug/_inspect_audio_pauses.py                 # все downloads/*.wav
    python debug/_inspect_audio_pauses.py downloads/001_audio.wav
    python debug/_inspect_audio_pauses.py 001_audio.wav --enter -28
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

from audio_postprocess import (
    _to_mono,
    expand_pause_regions,
    find_pause_regions,
    find_vad_nonspeech_regions,
)


def describe_region(mono: np.ndarray, sr: int, start: int, end: int) -> str:
    """Уровень в паузе относительно пика: чем выше, тем слышнее артефакт."""
    segment = mono[start:end]
    if len(segment) == 0:
        return "пусто"
    peak = float(np.max(np.abs(mono))) or 1.0
    rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2) + 1e-12))
    seg_peak = float(np.max(np.abs(segment)))
    return (
        f"rms {20 * np.log10(rms / peak + 1e-12):6.1f} dB, "
        f"peak {20 * np.log10(seg_peak / peak + 1e-12):6.1f} dB"
    )


def inspect(path: Path, args: argparse.Namespace) -> None:
    wav, sr = sf.read(str(path), always_2d=False)
    mono, _ = _to_mono(wav)
    duration = len(mono) / sr

    regions = find_pause_regions(
        mono,
        sr,
        enter_db=args.enter,
        exit_db=args.exit_db,
        min_pause_ms=args.min_ms,
        guard_ms=args.guard_ms,
    )
    raw_count = len(regions)
    if args.bridge_ms > 0:
        regions = expand_pause_regions(
            regions,
            mono,
            sr,
            bridge_max_ms=args.bridge_ms,
            bridge_peak_db=args.bridge_peak_db,
        )
    vad_regions = find_vad_nonspeech_regions(
        mono,
        sr,
        aggressiveness=args.vad_mode,
        speech_pad_ms=args.vad_pad_ms,
        min_nonspeech_ms=args.vad_min_ms,
    )

    print(f"\n{path.name}  —  {duration:.2f} s, {sr} Hz")
    print(f"  порог входа {args.enter} dB / выхода {args.exit_db} dB, "
          f"мин. пауза {args.min_ms:.0f} ms, запас {args.guard_ms:.0f} ms")
    print(f"  мост: до {args.bridge_ms:.0f} ms, пик < {args.bridge_peak_db:.0f} dB")
    if raw_count != len(regions):
        print(f"  склеено мостов: {raw_count} → {len(regions)} регионов")

    total_ms = 0.0
    print("  RMS-паузы:")
    for i, (start, end) in enumerate(regions, start=1):
        length_ms = (end - start) * 1000.0 / sr
        total_ms += length_ms
        print(
            f"  {i:2d}. {start / sr:6.2f} s → {end / sr:6.2f} s  "
            f"({length_ms:6.1f} ms)  {describe_region(mono, sr, start, end)}"
        )
    print(f"  итого: {len(regions)} пауз, {total_ms:.0f} ms будет заглушено")

    print(f"  WebRTC VAD не-речь (mode {args.vad_mode}):")
    for i, (start, end) in enumerate(vad_regions, start=1):
        print(
            f"  {i:2d}. {start / sr:6.2f} s → {end / sr:6.2f} s  "
            f"({(end - start) * 1000.0 / sr:6.1f} ms)"
        )
    print(f"  итого VAD: {len(vad_regions)} участков")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="WAV-файлы (по умолчанию downloads/*.wav)")
    parser.add_argument("--enter", type=float, default=-36.0, help="порог входа в речь, dB")
    parser.add_argument("--exit", dest="exit_db", type=float, default=-48.0, help="порог выхода, dB")
    parser.add_argument("--min-ms", type=float, default=180.0, help="минимальная пауза, ms")
    parser.add_argument("--guard-ms", type=float, default=80.0, help="запас у границ речи, ms")
    parser.add_argument("--bridge-ms", type=float, default=0.0, help="макс. длина моста между паузами, ms")
    parser.add_argument("--bridge-peak-db", type=float, default=-20.0, help="пик моста для склейки, dB")
    parser.add_argument("--vad-mode", type=int, default=3, choices=range(4))
    parser.add_argument("--vad-pad-ms", type=float, default=180.0)
    parser.add_argument("--vad-min-ms", type=float, default=180.0)
    args = parser.parse_args()

    if args.files:
        paths = []
        for raw in args.files:
            candidate = Path(raw)
            if not candidate.is_absolute() and not candidate.exists():
                candidate = ROOT / "downloads" / raw
            paths.append(candidate)
    else:
        paths = sorted((ROOT / "downloads").glob("*.wav"))

    if not paths:
        print("WAV-файлы не найдены.")
        return 1

    for path in paths:
        if not path.is_file():
            print(f"\n{path}: файл не найден")
            continue
        inspect(path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
