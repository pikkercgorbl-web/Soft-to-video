#!/usr/bin/env python3
"""Боевая проверка нового подбора видеоряда без TTS.

TTS-сервер в этой проверке не участвует (задача его не касается), поэтому вместо
озвучки подставляется синтетический WAV нужной длительности. Всё остальное —
Gemini, Pexels/Pixabay, скачивание, FFmpeg-обрезка, реестр и логи — боевое.

Второй проход с тем же сценарием проверяет, что клипы из used_media.json
больше не предлагаются.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aiohttp
from google import genai

import download_pexels as dp
from media_semantic import MediaSelectionLog, UsedMediaStore

WORK_DIR = ROOT / "debug" / "logs" / "media_check"
STORE_PATH = WORK_DIR / "used_media.json"
LOG_PATH = WORK_DIR / "media_selection.log"
CHUNK_LIMIT = 3
FAKE_AUDIO_SECONDS = 6.0


async def make_fake_audio(path: Path, seconds: float) -> None:
    """Синтетический WAV вместо TTS — нужен только как эталон длительности для обрезки."""
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
        "-ac", "1", "-ar", "24000", str(path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await process.wait()


async def build_tasks(
    chunks: list[str], client: genai.Client
) -> list[dp.MediaTask]:
    tasks: list[dp.MediaTask] = []

    for index, chunk in enumerate(chunks, start=1):
        brief = await dp.generate_visual_brief_for_chunk(chunk, client)
        if brief is None:
            print(f"  [{index}] визуальное описание не получено, пропуск")
            continue

        print(f"  [{index}] visual : {brief.visual}")
        print(f"       en     : {brief.visual_en}")
        print(f"       query  : {brief.query}")

        audio_path = WORK_DIR / f"chunk_{index:02d}.wav"
        await make_fake_audio(audio_path, FAKE_AUDIO_SECONDS)

        tasks.append(
            dp.MediaTask(
                index=index,
                prompt=brief.query,
                chunk_text=chunk,
                audio_path=audio_path,
                audio_duration=FAKE_AUDIO_SECONDS,
                video_path=WORK_DIR / f"chunk_{index:02d}.mp4",
                raw_video_path=WORK_DIR / f"chunk_{index:02d}_raw.mp4",
                visual_brief=brief,
            )
        )

    return tasks


async def run_pass(
    label: str,
    tasks: list[dp.MediaTask],
    client: genai.Client,
    pexels_key: str,
    pixabay_key: str,
    run_id: str,
) -> tuple[int, set[str]]:
    """Один прогон подбора. Реестр создаётся заново — историю он читает с диска."""
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")

    store = UsedMediaStore(STORE_PATH)
    print(f"  used_media.json на входе: {len(store)} клипов")
    registry = dp.MediaRegistry(store=store)
    selection_log = MediaSelectionLog(LOG_PATH, run_id)

    timeout = aiohttp.ClientTimeout(total=dp.REQUEST_TIMEOUT)
    semaphore = asyncio.Semaphore(dp.MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *[
                dp.process_video_task(
                    task,
                    len(tasks),
                    session,
                    pexels_key,
                    pixabay_key,
                    registry,
                    semaphore,
                    client,
                    selection_log,
                )
                for task in tasks
            ]
        )

    success = sum(1 for ok in results if ok)
    print(f"\n  Успешно собрано пар: {success} из {len(tasks)}")
    return success, UsedMediaStore(STORE_PATH).keys()


def report_log() -> None:
    print(f"\n{'=' * 70}\nmedia_selection.log\n{'=' * 70}")
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        chosen = record["chosen"]
        print(f"\n  run={record['run_id']} chunk={record['chunk']}")
        print(f"    visual    : {record['visual']}")
        print(f"    query     : {record['query']}")
        print(f"    ranker    : {record['ranker']}  (fallback={record['fallback_used']})")
        print(f"    pool_size : {record['pool_size']}")
        print(f"    reason    : {record['reason']}")
        if chosen:
            print(
                f"    chosen    : {chosen['source']}:{chosen['media_id']} "
                f"score={chosen['score']} — {chosen['caption'][:60]}"
            )
        else:
            print("    chosen    : —")


async def main() -> int:
    dp.load_settings()
    dp.MEDIA_SEMANTIC_ENABLED = True
    dp.MEDIA_RANKER = "llm"
    dp.MEDIA_CANDIDATE_POOL = 10

    gemini_key = os.getenv("GEMINI_API_KEY")
    pexels_key = os.getenv("PEXELS_API_KEY")
    pixabay_key = os.getenv("PIXABAY_API_KEY", "")
    if not gemini_key or not pexels_key:
        print("Нет ключей GEMINI_API_KEY / PEXELS_API_KEY")
        return 1

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    chunks = dp.chunk_script(dp.read_script(dp.SCRIPT_FILE))[:CHUNK_LIMIT]
    print(f"Чанков в проверке: {len(chunks)}")
    print(f"Ранжирование: {dp.MEDIA_RANKER}, пул: {dp.MEDIA_CANDIDATE_POOL}\n")

    client = genai.Client(api_key=gemini_key)

    print("=" * 70)
    print("ВИЗУАЛЬНЫЕ ОПИСАНИЯ СЦЕН (Gemini)")
    print("=" * 70)
    tasks = await build_tasks(chunks, client)
    if not tasks:
        print("Ни одного задания не собрано.")
        return 1

    _, ids_after_first = await run_pass(
        "ПРОХОД 1 — чистая история", tasks, client, pexels_key, pixabay_key, "pass-1"
    )

    # Второй проход теми же запросами: клипы из прохода 1 уже на диске,
    # поэтому подбор обязан выдать другие ID.
    for task in tasks:
        task.video_path = WORK_DIR / f"{task.video_path.stem}_p2.mp4"
        task.raw_video_path = WORK_DIR / f"{task.raw_video_path.stem}_p2.mp4"

    _, ids_after_second = await run_pass(
        "ПРОХОД 2 — те же запросы, история с диска",
        tasks,
        client,
        pexels_key,
        pixabay_key,
        "pass-2",
    )

    report_log()

    new_ids = ids_after_second - ids_after_first
    print(f"\n{'=' * 70}\nПЕРСИСТЕНТНОСТЬ\n{'=' * 70}")
    print(f"  ID после прохода 1 : {len(ids_after_first)}")
    print(f"  ID после прохода 2 : {len(ids_after_second)}")
    print(f"  Пересечение выборов проходов: {len(ids_after_first & (ids_after_second - new_ids))}")
    print(f"  Новых ID во проходе 2: {len(new_ids)} → {sorted(new_ids)}")

    if not new_ids:
        print("  ✗ Проход 2 не добавил ни одного клипа")
        return 1
    if len(ids_after_second) != len(ids_after_first) + len(new_ids):
        print("  ✗ История повреждена")
        return 1

    print("  ✓ Проход 2 выбрал другие клипы, история сохранилась между прогонами")

    produced = sorted(p.name for p in WORK_DIR.glob("*.mp4"))
    print(f"\n  Файлы видео: {produced}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
