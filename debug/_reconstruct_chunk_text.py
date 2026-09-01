#!/usr/bin/env python3
"""Восстанавливает финальный TTS-текст пайплайн-чанка и сохраняет его для нарезки.

Сервер не пишет текст запроса в лог, поэтому текст воспроизводится тем же путём,
что и в прогоне: Gemini-подготовка → доразметка ударений StressRNN. Достоверность
реконструкции затем проверяется отпечатком нарезки (число подчанков и длины
сегментов, превысивших chunk_size), который в логе сервера есть.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import download_pexels as dp

CHUNK_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 4
OUT_PATH = ROOT / "debug" / "logs" / f"chunk_{CHUNK_INDEX:02d}_tts_text.txt"


async def main() -> int:
    dp.load_settings()
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    chunks = dp.chunk_script(dp.read_script(dp.SCRIPT_FILE))
    if not 1 <= CHUNK_INDEX <= len(chunks):
        print(f"Чанк {CHUNK_INDEX} вне диапазона 1..{len(chunks)}")
        return 1

    source = chunks[CHUNK_INDEX - 1]
    print(f"Пайплайн-чанк {CHUNK_INDEX} из {len(chunks)}, исходник {len(source)} символов")
    print("-" * 78)
    print(source)
    print("-" * 78)

    text = await dp.prepare_text_with_gemini(source, client)
    print(f"\nПосле Gemini-подготовки: {len(text)} символов")

    from stress_verify import process_gemini_stress_async

    result = await process_gemini_stress_async(
        text,
        chunk_index=CHUNK_INDEX,
        # Боевой stress_mismatches.log не пачкаем: это диагностический прогон.
        log_path=str(ROOT / "debug" / "logs" / "_reconstruct_stress.log"),
        engine_name=dp.STRESS_VERIFY_ENGINE,
        verify_enabled=dp.STRESS_VERIFY_ENABLED,
        autocorrect_enabled=dp.STRESS_AUTOCORRECT_ENABLED,
    )
    final = result.text
    print(f"После доразметки ударений: {len(final)} символов, "
          f"{len(final.split())} слов, доразмечено {result.filled_count}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(final, encoding="utf-8")
    print(f"\nСохранено: {OUT_PATH}")
    print("=" * 78)
    print(final)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
