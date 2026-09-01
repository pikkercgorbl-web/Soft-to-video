#!/usr/bin/env python3
"""Разбор события repetition guard: какой подчанк оборвался и оборвался ли реально.

Сервер не логирует текст запроса, поэтому запросы привязываются к файлам downloads/
по размеру ответа, а длительности подчанков берутся из самого WAV: между подчанками
сервер вставляет 200 мс тишины (server.py, SENTENCE_PAUSE_MS), по этим паузам
границы и находятся. Ожидаемая длительность считается из числа шагов генерации:
S3_TOKEN_RATE = 25 токенов в секунду.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

SERVER_LOG = Path(r"C:\AI_Servers\Chatterbox\Chatterbox-TTS-Server\logs\tts_server.log")
DOWNLOADS = ROOT / "downloads"
RUN_DATE = "2026-08-20"
S3_TOKEN_RATE = 25
PAUSE_MS = 200


def parse_server_log() -> list[dict]:
    """Вытаскивает запросы прогона: число подчанков, шаги, причины EOS, размер ответа."""
    requests: list[dict] = []
    current: dict | None = None

    for line in SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(RUN_DATE):
            continue

        if "Received /tts request" in line:
            current = {"chunks": [], "total": None, "bytes": None, "time": line[11:19]}
            requests.append(current)
            continue
        if current is None:
            continue

        match = re.search(r"Text chunking complete\. Generated (\d+) chunk", line)
        if match:
            current["total"] = int(match.group(1))

        match = re.search(r"A single segment \(length (\d+)\) exceeds", line)
        if match:
            current.setdefault("oversized", []).append(int(match.group(1)))

        match = re.search(r"Synthesizing chunk (\d+)/(\d+)", line)
        if match:
            current["chunks"].append({"index": int(match.group(1)), "guard": None, "steps": None})

        match = re.search(r"Detected 2x repetition of token (\d+)", line)
        if match and current["chunks"]:
            current["chunks"][-1]["repeated_token"] = int(match.group(1))

        match = re.search(r"forcing EOS token, long_tail=tensor\((\w+)\).*token_repetition=(\w+)", line)
        if match and current["chunks"]:
            current["chunks"][-1]["guard"] = (
                "token_repetition" if match.group(2) == "True" else "long_tail"
            )

        match = re.search(r"Stopping generation at step (\d+)", line)
        if match and current["chunks"]:
            current["chunks"][-1]["steps"] = int(match.group(1))

        match = re.search(r"Successfully generated audio: \S+, (\d+) bytes", line)
        if match:
            current["bytes"] = int(match.group(1))

    return [r for r in requests if r["chunks"]]


def match_to_downloads(requests: list[dict]) -> None:
    """Привязывает запрос к файлу downloads/ по размеру (постобработка режет только хвост)."""
    files = sorted(DOWNLOADS.glob("*_audio.wav"))
    sizes = {path: path.stat().st_size for path in files}

    for request in requests:
        if request["bytes"] is None:
            continue
        best, best_delta = None, None
        for path, size in sizes.items():
            delta = request["bytes"] - size
            if delta < -64:
                continue  # файл больше ответа — не тот
            if best_delta is None or delta < best_delta:
                best, best_delta = path, delta
        request["file"] = best
        request["trimmed_bytes"] = best_delta


def find_segments(path: Path, pause_ms: int = PAUSE_MS) -> list[tuple[float, float]]:
    """Границы подчанков по вставленным сервером паузам.

    Между подчанками лежит буфер настоящих нулей (server.py: np.zeros на 200 мс,
    по 20 мс с каждого края съедает кроссфейд). Внутрифразовые паузы речи такой
    глубины не дают, поэтому порог берётся почти нулевой — иначе детектор ловит
    дыхания и паузы внутри предложения.
    """
    audio, rate = sf.read(str(path), always_2d=True)
    mono = np.abs(audio.mean(axis=1))

    silent = mono < 1e-4
    min_len = int((pause_ms - 60) / 1000 * rate)

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
        gap_start_sec = gap_start / rate
        if gap_start_sec - cursor > 0.15:
            segments.append((cursor, gap_start_sec))
        cursor = gap_end / rate
    total = len(mono) / rate
    if total - cursor > 0.15:
        segments.append((cursor, total))
    return segments


def describe_tail(path: Path, end_sec: float, window: float = 0.25) -> str:
    """Обрыв на полуслове звучит как резкий срез: энергия у самого конца остаётся высокой."""
    audio, rate = sf.read(str(path), always_2d=True)
    mono = audio.mean(axis=1)

    end = int(end_sec * rate)
    start = max(0, end - int(window * rate))
    tail = mono[start:end]
    if len(tail) < rate // 50:
        return "слишком короткий хвост"

    half = len(tail) // 2
    first_rms = float(np.sqrt(np.mean(tail[:half] ** 2)))
    last_rms = float(np.sqrt(np.mean(tail[half:] ** 2)))
    peak = float(np.abs(mono).max()) or 1.0
    ratio = last_rms / first_rms if first_rms else 0.0

    verdict = "затухание (естественный конец фразы)" if ratio < 0.5 else "энергия держится (похоже на срез)"
    return (
        f"RMS первая половина {first_rms / peak:.4f}, вторая {last_rms / peak:.4f}, "
        f"отношение {ratio:.2f} → {verdict}"
    )


def main() -> int:
    requests = parse_server_log()
    match_to_downloads(requests)

    print("=" * 78)
    print("ЗАПРОСЫ ПРОГОНА И ПРИВЯЗКА К ФАЙЛАМ")
    print("=" * 78)
    for number, request in enumerate(requests, start=1):
        name = request.get("file").name if request.get("file") else "—"
        oversized = request.get("oversized") or []
        print(
            f"  #{number} {request['time']}  подчанков: {request['total']}  "
            f"ответ: {request['bytes']} Б  → {name}"
            + (f"  (обрезано постобработкой {request['trimmed_bytes']} Б)" if request.get("trimmed_bytes") else "")
        )
        if oversized:
            print(f"        сегменты длиннее chunk_size=100: {oversized}")
        guards = [
            f"{c['index']}:{c['steps']}шаг/{c['guard']}"
            + (f"/tok{c['repeated_token']}" if c.get("repeated_token") else "")
            for c in request["chunks"]
        ]
        print(f"        {'  '.join(guards)}")

    target = next(
        (
            r
            for r in requests
            if any(c["index"] == 1 and c["steps"] == 50 and c["guard"] == "token_repetition"
                   for c in r["chunks"])
        ),
        None,
    )
    if target is None or not target.get("file"):
        print("\nЗапрос с обрывом на 50-м шаге не найден.")
        return 1

    path = target["file"]
    print()
    print("=" * 78)
    print(f"РАЗБОР СОБЫТИЯ: {path.name} (запрос {target['time']}, подчанк 1/{target['total']})")
    print("=" * 78)

    segments = find_segments(path)
    print(f"  Найдено сегментов в WAV: {len(segments)} (ожидалось {target['total']})")
    print()
    print("  №  шагов  ожидание, с  факт, с   расхождение")
    for index, chunk in enumerate(target["chunks"]):
        expected = chunk["steps"] / S3_TOKEN_RATE if chunk["steps"] else None
        if index < len(segments):
            start, end = segments[index]
            actual = end - start
            delta = f"{actual - expected:+.2f}" if expected else "—"
            print(
                f"  {chunk['index']}  {chunk['steps'] or '—':>5}  "
                f"{expected:>11.2f}  {actual:>7.2f}  {delta:>11}"
            )
        else:
            print(f"  {chunk['index']}  {chunk['steps'] or '—':>5}  {expected:>11.2f}  {'—':>7}")

    print()
    if segments:
        start, end = segments[0]
        print(f"  Подчанк 1: {start:.2f}–{end:.2f} с, длительность {end - start:.2f} с")
        print(f"  Конец подчанка 1: {describe_tail(path, end)}")

    print()
    print("=" * 78)
    print("СРАБАТЫВАНИЯ ГВАРДА ЗА ВЕСЬ ПРОГОН")
    print("=" * 78)
    all_chunks = [c for r in requests for c in r["chunks"]]
    by_token = [c for c in all_chunks if c["guard"] == "token_repetition"]
    by_tail = [c for c in all_chunks if c["guard"] == "long_tail"]
    print(f"  Всего подчанков: {len(all_chunks)}")
    print(f"  token_repetition: {len(by_token)} ({len(by_token) / len(all_chunks) * 100:.0f}%)")
    print(f"  long_tail:        {len(by_tail)} ({len(by_tail) / len(all_chunks) * 100:.0f}%)")
    tokens = sorted({c["repeated_token"] for c in by_token if c.get("repeated_token")})
    print(f"  Повторявшиеся токены: {tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
