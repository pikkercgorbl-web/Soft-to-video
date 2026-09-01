#!/usr/bin/env python3
"""Проверка компонентов смыслового подбора на реальной выдаче Pexels."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aiohttp
from dotenv import load_dotenv

from media_semantic import (
    MediaCandidate,
    MediaSelectionLog,
    SelectionOutcome,
    UsedMediaStore,
    VisualBrief,
    build_ranking_prompt,
    extract_caption,
    lexical_score,
    parse_ranking_choice,
    parse_visual_brief,
    rank_lexically,
)


def test_brief_parsing() -> None:
    print("=== parse_visual_brief ===")
    raw = (
        '```json\n{"visual": "Пустой тёмный коридор с мигающей лампой в конце",\n'
        ' "visual_en": "empty dark corridor with flickering lamp at the end",\n'
        ' "query": "dark corridor flickering"}\n```'
    )
    brief = parse_visual_brief(raw)
    assert brief is not None, "markdown-обёртка должна сниматься"
    print(f"  visual:    {brief.visual}")
    print(f"  visual_en: {brief.visual_en}")
    print(f"  query:     {brief.query}")

    assert parse_visual_brief("не json") is None
    assert parse_visual_brief("") is None
    partial = parse_visual_brief('{"visual_en": "old wooden door in a forest"}')
    assert partial is not None and partial.query, "query должен выводиться из visual_en"
    print(f"  query без явного поля: {partial.query!r}")

    print("=== parse_ranking_choice ===")
    for raw_answer, pool_size, expected in [
        ("3", 10, 2),
        ("Ответ: 7", 10, 6),
        ("11", 10, None),
        ("0", 10, None),
        ("не знаю", 10, None),
    ]:
        got = parse_ranking_choice(raw_answer, pool_size)
        assert got == expected, f"{raw_answer!r} → {got}, ожидалось {expected}"
        print(f"  {raw_answer!r:<12} → {got}")


def test_store_and_log() -> None:
    print("=== UsedMediaStore ===")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "used_media.json"
        store = UsedMediaStore(path)
        assert len(store) == 0

        store.remember("pexels:video:111", query="dark corridor", score=0.42)
        store.remember("pexels:video:222", query="old door")
        assert path.is_file(), "файл должен появиться сразу после первой записи"

        reloaded = UsedMediaStore(path)
        print(f"  после перезагрузки: {len(reloaded)} записей, keys={sorted(reloaded.keys())}")
        assert len(reloaded) == 2
        assert reloaded.contains("pexels:video:111")
        assert not reloaded.contains("pexels:video:999")

        print("=== MediaSelectionLog ===")
        log_path = Path(tmp) / "media_selection.log"
        log = MediaSelectionLog(log_path, run_id="test-run")
        brief = VisualBrief("Тёмный коридор", "dark corridor", "dark corridor")
        candidate = MediaCandidate(
            source="pexels",
            media_id=111,
            download_url="http://x",
            duration=10,
            width=1920,
            height=1080,
            caption="a dark hallway with a light",
            score=0.71,
        )
        log.write(
            chunk_index=1,
            brief=brief,
            outcome=SelectionOutcome(
                chosen=candidate, ranker="llm", pool=[candidate], reason="выбор модели"
            ),
            chunk_text="Он шёл по коридору очень долго.",
        )
        content = log_path.read_text(encoding="utf-8").strip()
        print(f"  строка лога: {content[:160]}...")
        assert '"ranker": "llm"' in content
        assert '"media_id": 111' in content


async def test_captions_and_ranking() -> None:
    load_dotenv()
    print("=== Подписи и ранжирование на живой выдаче Pexels ===")

    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.pexels.com/v1/videos/search",
            headers={"Authorization": os.getenv("PEXELS_API_KEY") or ""},
            params={"query": "dark corridor", "orientation": "landscape", "per_page": 10, "page": 1},
        ) as response:
            data = await response.json()

    candidates = [
        MediaCandidate(
            source="pexels",
            media_id=item["id"],
            download_url="",
            duration=item.get("duration"),
            width=item.get("width"),
            height=item.get("height"),
            caption=extract_caption(item, "pexels"),
        )
        for item in data["videos"]
    ]

    empty = [c for c in candidates if not c.caption]
    print(f"  кандидатов: {len(candidates)}, без подписи: {len(empty)}")
    assert not empty, "slug из url должен давать подпись каждому кандидату"

    brief = VisualBrief(
        visual="Пустой тёмный коридор с мигающей лампой в конце",
        visual_en="empty dark corridor with flickering lamp at the end",
        query="dark corridor",
    )
    ordered = rank_lexically(brief, candidates)
    print(f"  лексический топ-5 под {brief.visual_en!r}:")
    for candidate in ordered[:5]:
        print(f"    {candidate.score:<8} id={candidate.media_id:<10} {candidate.caption[:70]}")

    assert ordered[0].score >= ordered[-1].score
    assert lexical_score("dark corridor", "") == 0.0
    assert lexical_score("", "dark corridor") == 0.0

    print("=== Промпт ранжирования (первые строки) ===")
    prompt = build_ranking_prompt(brief, ordered[:3])
    for line in prompt.splitlines()[:6]:
        print(f"  {line}")


def main() -> None:
    test_brief_parsing()
    print()
    test_store_and_log()
    print()
    asyncio.run(test_captions_and_ranking())
    print("\nВсе проверки прошли.")


if __name__ == "__main__":
    main()
