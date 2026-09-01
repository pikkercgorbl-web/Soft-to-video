#!/usr/bin/env python3
"""Прогоняет текст через настоящую нарезку сервера Chatterbox.

Запускать встроенным интерпретатором сервера, иначе не найдутся torch/pydub/config:
    C:\\AI_Servers\\Chatterbox\\Chatterbox-TTS-Server\\python_embedded\\python.exe
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SERVER_DIR = Path(r"C:\AI_Servers\Chatterbox\Chatterbox-TTS-Server")
sys.path.insert(0, str(SERVER_DIR))

import logging

logging.disable(logging.CRITICAL)

import utils  # noqa: E402  сервера, не проекта

CHUNK_SIZE = 100


def main() -> int:
    if len(sys.argv) < 2:
        print("Использование: _server_chunking.py <файл с текстом>")
        return 1

    text = Path(sys.argv[1]).read_text(encoding="utf-8")
    chunks = utils.chunk_text_by_sentences(text, CHUNK_SIZE)

    print(f"Текст: {len(text)} символов, chunk_size={CHUNK_SIZE}")
    print(f"Подчанков: {len(chunks)}")
    oversized = [len(c) for c in chunks if len(c) > CHUNK_SIZE]
    print(f"Сегменты длиннее chunk_size: {oversized}")
    print("=" * 78)

    for index, chunk in enumerate(chunks, start=1):
        flag = "  ← превышает chunk_size" if len(chunk) > CHUNK_SIZE else ""
        print(f"\nПодчанк {index}/{len(chunks)}: {len(chunk)} символов, "
              f"{len(chunk.split())} слов{flag}")
        print(f"  {chunk}")
        if index == 1:
            print(f"  codepoints: {' '.join(f'U+{ord(ch):04X}' for ch in chunk)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
