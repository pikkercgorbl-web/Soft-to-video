#!/usr/bin/env python3
"""Какие текстовые метаданные реально отдают Pexels и Pixabay для видео."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aiohttp
from dotenv import load_dotenv


async def main() -> None:
    load_dotenv()

    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.pexels.com/v1/videos/search",
            headers={"Authorization": os.getenv("PEXELS_API_KEY") or ""},
            params={"query": "dark hallway", "orientation": "landscape", "per_page": 5, "page": 1},
        ) as response:
            pexels = await response.json()

    video = pexels["videos"][0]
    print("=== PEXELS VIDEO: поля объекта ===")
    for key, value in video.items():
        if isinstance(value, (list, dict)):
            print(f"  {key}: <{type(value).__name__}> len={len(value)}")
        else:
            print(f"  {key}: {value!r}")

    print()
    print("=== Текстовые поля ===")
    for probe in ("title", "tags", "alt", "description", "tag", "keywords"):
        present = "ДА" if probe in video else "нет"
        print(f"  {probe}: {present}")

    print()
    print("=== url: slug содержит человекочитаемое описание ===")
    for item in pexels["videos"]:
        print(f"  {item['url']}")

    print()
    print(f"  user: {video['user']}")

    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://pixabay.com/api/videos/",
            params={"key": os.getenv("PIXABAY_API_KEY") or "", "q": "dark hallway", "per_page": 5},
        ) as response:
            pixabay = await response.json()

    hit = pixabay["hits"][0]
    print()
    print("=== PIXABAY VIDEO: поля объекта ===")
    for key, value in hit.items():
        if isinstance(value, (list, dict)):
            print(f"  {key}: <{type(value).__name__}>")
        else:
            print(f"  {key}: {value!r}")


if __name__ == "__main__":
    asyncio.run(main())
