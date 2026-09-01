"""Переигрывает шаг LLM-ранжирования media_semantic по уже записанным логам.

Идея: не гонять весь пайплайн (Pexels/TTS/аудио), а взять пулы кандидатов,
которые уже были собраны и залогированы в media_selection.log, и заново
попросить Gemini выбрать номер — но на этот раз сохранить и напечатать
её СЫРОЙ ответ, а не только распарсенный индекс.

Важная оговорка: в логе кандидаты хранятся уже ПЕРЕСТАВЛЕННЫМИ (выбранный —
на позиции 0, см. rank_candidates() в download_pexels.py), а не в том порядке,
в котором их видела модель при первом запросе. Поэтому индекс, который вернёт
Gemini сейчас, не обязан совпадать с исходным выбором "1" — это нормально,
задача скрипта не в побитовом повторе, а в том, чтобы увидеть:
  а) отвечает ли модель голым числом или рассуждением (тест парсера);
  б) насколько стабилен выбор при повторном запросе на тех же кандидатах.

Использование:
    export GEMINI_API_KEY=...
    python debug/replay_ranking.py media_selection.log
    python debug/replay_ranking.py media_selection.log --chunk 2
    python debug/replay_ranking.py media_selection.log --repeat 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai  # type: ignore
from google.genai import types  # type: ignore

from media_semantic import (
    MediaCandidate,
    VisualBrief,
    build_ranking_prompt,
    parse_ranking_choice,
)

GEMINI_MODEL = "gemini-2.5-flash"


def load_entries(log_path: Path, chunk: int | None) -> list[dict]:
    entries = []
    with log_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("ranker") != "llm":
                continue
            if not record.get("candidates"):
                continue
            if chunk is not None and record.get("chunk") != chunk:
                continue
            entries.append(record)
    return entries


def replay_entry(client: "genai.Client", record: dict, repeat: int, shuffle: bool) -> None:
    brief = VisualBrief(
        visual=record.get("visual") or "",
        visual_en=record.get("visual_en") or "",
        query=record.get("query") or "",
    )
    base_candidates = [
        MediaCandidate(
            source=item["source"],
            media_id=item["media_id"],
            download_url="",
            duration=None,
            width=None,
            height=None,
            caption=item.get("caption") or "",
            score=item.get("score"),
        )
        for item in record["candidates"]
    ]

    original_choice_id = (record.get("chosen") or {}).get("media_id")

    print("=" * 88)
    print(f"chunk {record.get('chunk')}  run_id={record.get('run_id')}")
    print(f"visual: {brief.visual}")
    print(f"original chosen media_id={original_choice_id} "
          f"caption={(record.get('chosen') or {}).get('caption')!r}")
    print(f"original reason: {record.get('reason')}")
    if record.get("raw_llm_response") is not None:
        print(f"original raw_llm_response: {record['raw_llm_response']!r}")
    print("-" * 88)

    if not shuffle:
        print("Кандидаты, как они переданы модели сейчас (порядок из лога, не исходный):")
        for index, candidate in enumerate(base_candidates, start=1):
            print(f"  {index}. [{candidate.media_id}] {candidate.caption}")
        print("-" * 88)

    for attempt in range(1, repeat + 1):
        candidates = list(base_candidates)
        if shuffle:
            random.shuffle(candidates)
            print(f"[replay #{attempt}] порядок перемешан:")
            for index, candidate in enumerate(candidates, start=1):
                print(f"    {index}. [{candidate.media_id}] {candidate.caption}")

        prompt = build_ranking_prompt(brief, candidates)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0),
        )
        raw_text = response.text or ""
        parsed = parse_ranking_choice(raw_text, len(candidates))
        picked = candidates[parsed] if parsed is not None else None
        print(f"[replay #{attempt}] raw_response={raw_text!r}")
        if picked is not None:
            print(
                f"[replay #{attempt}] parsed_index={parsed} "
                f"-> media_id={picked.media_id} caption={picked.caption!r}"
            )
        else:
            print(f"[replay #{attempt}] parsed_index=None (парсер не нашёл номер)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="Путь к media_selection.log")
    parser.add_argument("--chunk", type=int, default=None, help="Фильтр по номеру чанка")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Сколько раз переспросить каждую сцену"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Перемешивать порядок кандидатов перед каждой попыткой "
        "(проверка на склонность модели к первому варианту в списке)",
    )
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠ GEMINI_API_KEY не задан в окружении.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    entries = load_entries(args.log_path, args.chunk)
    if not entries:
        print("Нет подходящих записей (ranker=llm с непустым пулом кандидатов).")
        return

    print(f"Найдено записей для переигровки: {len(entries)}\n")
    for record in entries:
        replay_entry(client, record, args.repeat, args.shuffle)


if __name__ == "__main__":
    main()
