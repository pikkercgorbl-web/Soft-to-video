#!/usr/bin/env python3
"""Замер TTS на полном тексте «Москва, октябрь 1974» (как в боевом пайплайне)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from google import genai

import download_pexels as dp
from download_pexels import (
    SCRIPT_FILE,
    chunk_script,
    generate_audio_with_chatterbox,
    load_settings,
    prepare_text_with_gemini,
    read_script,
)
from stress_verify import process_gemini_stress

COMBINING_ACUTE = "\u0301"


def count_words(text: str) -> int:
    return len(re.findall(r"[\w" + COMBINING_ACUTE + r"]+", text))


async def main() -> None:
    load_dotenv()
    load_settings()
    script = read_script(SCRIPT_FILE)
    chunks = chunk_script(script)

    print("=== chunk_script (current) ===")
    print(f"Chunks: {len(chunks)}")
    chunk = chunks[0]
    print(f"Chunk 1 chars: {len(chunk)}")
    print(f"Chunk 1 words: {count_words(chunk)}")
    print(f"Chunk 1 lines: {chunk.count(chr(10)) + 1}")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    print("\nGemini TTS prep...")
    t0 = time.perf_counter()
    gemini_text = await prepare_text_with_gemini(script, client)
    print(f"  Gemini prep: {time.perf_counter() - t0:.1f}s")
    print(f"  Gemini text chars: {len(gemini_text)}")
    print(f"  Gemini text words: {count_words(gemini_text)}")

    stress = process_gemini_stress(
        gemini_text,
        chunk_index=0,
        log_path=ROOT / "debug/logs/_bench_noop.log",
        verify_enabled=True,
        autocorrect_enabled=True,
    )
    tts_text = stress.text
    print(f"\nAfter stress:")
    print(f"  filled_count (StressRNN): {stress.filled_count}")
    print(f"  chars: {len(tts_text)}")
    print(f"  words: {count_words(tts_text)}")

    out = ROOT / "debug/logs/_bench_full_tts.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    old_timeout = dp.TTS_SERVER_TIMEOUT
    dp.TTS_SERVER_TIMEOUT = 600
    print(f"\nChatterbox TTS (timeout=600s)...")
    t1 = time.perf_counter()
    try:
        generate_audio_with_chatterbox(tts_text, out)
        elapsed = time.perf_counter() - t1
        size = out.stat().st_size
        print(f"  TTS OK: {elapsed:.1f}s, file {size} bytes")
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            duration = float(json.loads(result.stdout)["format"]["duration"])
            print(f"  Audio duration: {duration:.2f}s")
            print(f"  gen_time / audio_dur: {elapsed / duration:.2f}x realtime")
    except Exception as error:
        elapsed = time.perf_counter() - t1
        print(f"  TTS FAILED after {elapsed:.1f}s: {error}")
    finally:
        dp.TTS_SERVER_TIMEOUT = old_timeout


if __name__ == "__main__":
    asyncio.run(main())
