#!/usr/bin/env python3
"""Автоматическое скачивание медиа со стоков: сценарий → Gemini → TTS-prep (Gemini) → Chatterbox → Pexels / Pixabay → FFmpeg."""

import asyncio
import argparse
import json
import logging
import os
import re
import shutil
import sys
import warnings
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.raiseExceptions = False
logging.getLogger("google.genai").setLevel(logging.WARNING)
logging.getLogger("google.genai.models").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

import aiohttp
import requests
from google import genai
from dotenv import load_dotenv
from filelock import FileLock, Timeout
from project_state import Project, atomic_json, file_hash, fingerprint

from media_semantic import (
    EmbeddingRanker,
    MediaCandidate,
    MediaSelectionLog,
    SelectionOutcome,
    UsedMediaStore,
    VisualBrief,
    build_ranking_prompt,
    build_visual_brief_prompt,
    extract_caption,
    parse_ranking_choice,
    parse_visual_brief,
    rank_lexically,
)
from tts_text_sanitize import sanitize_tts_text

# ==========================================
# 🎛️ НАСТРОЙКИ ГОЛОСА CHATTERBOX
# ==========================================
TTS_VOICE_FILE = "1_.wav"      # Имя файла голоса (твоего клона или системного пресета)
TTS_VOICE_MODE = "clone"       # Режим: "clone" (твои голоса) или "predefined" (системные)
TTS_LANGUAGE = "ru"            # Язык
TTS_SPEED_FACTOR = 1.0         # Скорость
TTS_TEMPERATURE = 0.7          # Креативность (ниже = стабильнее, меньше «съедания» слов)
TTS_EXAGGERATION = 1.0         # Эмоциональность (выше 1.2 — модель начинает частить и глотать слоги)
TTS_CFG_WEIGHT = 0.5           # Строгость следования тексту
TTS_CHUNK_SIZE = 100           # Размер sub-chunk на сервере Chatterbox (split_text)
# ==========================================

# ==========================================
# 🔇 ЧИСТКА ПАУЗ В ОЗВУЧКЕ
# ==========================================
AUDIO_PAUSE_CLEAN = True       # глушить придыхания/шорохи на стыках sub-chunk'ов
AUDIO_PAUSE_ENTER_DB = -36.0   # ниже = консервативнее, лучше сохраняет тихую речь
AUDIO_PAUSE_EXIT_DB = -48.0    # порог выхода из речи (гистерезис)
AUDIO_PAUSE_MIN_MS = 180.0     # короткие провалы внутри слов не трогаем
AUDIO_PAUSE_GUARD_MS = 80.0    # большой запас у границ речи
AUDIO_PAUSE_BRIDGE_MAX_MS = 0.0     # опасное склеивание выключено: оно съедало тихие слова
AUDIO_PAUSE_BRIDGE_PEAK_DB = -20.0  # пик моста (отн. пика файла), ниже — глушим
AUDIO_PAUSE_TAIL_EXTEND_MS = 0.0    # опасное продление выключено
AUDIO_PAUSE_TAIL_PEAK_DB = -18.0    # порог «тихого хвоста» для продления
AUDIO_STITCH_CLEAN = True       # мягко гасить хвост перед точной тишиной склейки
AUDIO_STITCH_MIN_ZERO_MS = 100.0
AUDIO_STITCH_FADE_BEFORE_MS = 40.0
AUDIO_VAD_CLEAN = True          # WebRTC VAD: ветер/шорох/вой против человеческой речи
AUDIO_VAD_AGGRESSIVENESS = 3    # 0..3; 3 строже всего отбрасывает не-речь
AUDIO_VAD_SPEECH_PAD_MS = 180.0 # защищать речь на 180 ms с обеих сторон
AUDIO_VAD_MIN_NONSPEECH_MS = 120.0
AUDIO_SPECTRAL_CLEAN = False    # глобально опасен для шипящих; применяется только у стыков
AUDIO_SPECTRAL_MIN_REGION_MS = 220.0  # короче — может быть обычная шипящая согласная
AUDIO_SPECTRAL_MAX_GAP_MS = 80.0
AUDIO_KEEP_RAW = True          # сохранять оригинал до обработки в downloads/raw
# Переопределение через .env: AUDIO_PAUSE_CLEAN, AUDIO_PAUSE_ENTER_DB, ...
# ==========================================

# ==========================================
# 🎬 СМЫСЛОВОЙ ПОДБОР ВИДЕОРЯДА
# ==========================================
MEDIA_SEMANTIC_ENABLED = True          # визуальное описание сцены + ранжирование кандидатов
MEDIA_DOWNLOAD_ENABLED = True          # скачивание и синхронизация видеоряда (Pexels/Pixabay)
MEDIA_CANDIDATE_POOL = 10              # сколько кандидатов собирать на ранжирование
MEDIA_RANKER = "llm"                   # llm | embedding | lexical
MEDIA_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
USED_MEDIA_STORE = "used_media.json"   # персистентный реестр использованных клипов
MEDIA_SELECTION_LOG = "media_selection.log"
# Переопределение через .env: MEDIA_DOWNLOAD_ENABLED, MEDIA_SEMANTIC_ENABLED, ...
# ==========================================

# ==========================================
# 🔍 ПРОВЕРКА УДАРЕНИЙ (StressRNN, локально)
# ==========================================
STRESS_VERIFY_ENABLED = True          # логировать расхождения (и омографы)
STRESS_AUTOCORRECT_ENABLED = True     # StressRNN исправляет ошибки Gemini (кроме омографов)
STRESS_FILL_UNMARKED = False          # доразметка StressRNN для слов без U+0301 от Gemini
STRESS_MISMATCH_LOG = "stress_mismatches.log"
STRESS_EXCEPTION_DICT = "exception_dictionary.txt"  # U+0301, дополняет Зализняка
STRESS_VERIFY_ENGINE = "stressrnn"    # stressrnn | russtress
# Переопределение через .env: STRESS_VERIFY_ENABLED, STRESS_AUTOCORRECT_ENABLED,
# STRESS_FILL_UNMARKED
# ==========================================

# ---------------------------------------------------------------------------
# Глобальные настройки (переопределяются через .env)
# ---------------------------------------------------------------------------

MEDIA_TYPE = "video"       # video | photo
ORIENTATION = "portrait"     # portrait | landscape | square
GLOBAL_VIBE = ""             # например: creepy atmosphere, bright, cyberpunk
MAX_DURATION_SECONDS = 20
MAX_SEARCH_PAGES = 5
MAX_CONCURRENT_REQUESTS = 4
CHATTERBOX_TTS_URL = "http://localhost:8004/tts"
TTS_SERVER_TIMEOUT = 300
SCRIPT_CHUNK_MAX_CHARS = 400
FAILED_CHUNKS_LOG = "failed_chunks.log"
VIDEO_TRIM_PADDING = 0.3
AUDIO_WRITE_POLL_INTERVAL = 0.5
AUDIO_WRITE_MAX_WAIT = 60.0

PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/v1/videos/search"
PEXELS_PHOTO_SEARCH_URL = "https://api.pexels.com/v1/search"
PIXABAY_VIDEO_SEARCH_URL = "https://pixabay.com/api/videos/"
PIXABAY_PHOTO_SEARCH_URL = "https://pixabay.com/api/"

PEXELS_PER_PAGE = 80
PIXABAY_PER_PAGE = 200

SCRIPT_FILE = "script.txt"
PROMPTS_FILE = "prompts.txt"
DOWNLOADS_DIR = "downloads"
REQUEST_TIMEOUT = 60
GEMINI_MODEL = "gemini-2.5-flash"
PROJECT: Project | None = None
TTS_SEMAPHORE: asyncio.Semaphore | None = None
RUN_MEDIA: dict[int, str] = {}


def stage_settings(kind: str) -> dict:
    # Explicit groups keep visual changes independent of the audio cache.
    if kind == "audio":
        names = [name for name in globals() if name.startswith(("TTS_", "AUDIO_", "STRESS_"))
                 and name not in {"TTS_SEMAPHORE", "STRESS_MISMATCH_LOG"}]
        names += ["GEMINI_MODEL", "CHATTERBOX_TTS_URL"]
    else:
        names = ["GEMINI_MODEL", "GLOBAL_VIBE", "MEDIA_TYPE", "MEDIA_SEMANTIC_ENABLED",
                 "MEDIA_DOWNLOAD_ENABLED", "MEDIA_RANKER", "MEDIA_EMBEDDING_MODEL",
                 "MEDIA_CANDIDATE_POOL", "ORIENTATION", "VIDEO_TRIM_PADDING"]
    result = {name: globals()[name] for name in names
              if isinstance(globals()[name], (str, int, float, bool))}
    if "CHATTERBOX_TTS_URL" in result:
        result["CHATTERBOX_TTS_URL"] = fingerprint(result["CHATTERBOX_TTS_URL"])
    result["pipeline_revision"] = fingerprint({
        name: file_hash(Path(__file__).parent / name)
        for name in ("download_pexels.py", "media_semantic.py", "audio_postprocess.py",
                     "tts_text_sanitize.py", "stress_verify.py", "project_state.py")
    })
    if kind == "audio":
        dictionary = Path(STRESS_EXCEPTION_DICT)
        result["dictionary_hash"] = file_hash(dictionary) if dictionary.is_file() else None
        result["server_revision"] = os.getenv("TTS_CACHE_REVISION", "1")
        result["declared_model"] = os.getenv("TTS_MODEL_ID", "unknown")
    return result



@dataclass
class ChunkResult:
    chunk_index: int
    chunk_text: str
    search_prompt: str | None = None
    audio_path: Path | None = None
    audio_duration: float | None = None
    visual_brief: VisualBrief | None = None


@dataclass
class MediaTask:
    index: int
    prompt: str
    chunk_text: str
    audio_path: Path
    audio_duration: float | None
    video_path: Path
    raw_video_path: Path
    visual_brief: VisualBrief | None = None


class MediaRegistry:
    """Реестр использованных медиа: клипы не повторяются ни внутри прогона, ни между прогонами.

    Захват (`try_claim`) held только в памяти — иначе неудачное скачивание навсегда
    забанило бы годный клип. На диск запись уходит в `commit()`, то есть после того,
    как пара audio+video реально собрана.
    """

    def __init__(self, store: UsedMediaStore | None = None) -> None:
        self.store = store
        self.used_ids: set[str] = set(store.keys()) if store else set()
        self._lock = asyncio.Lock()

    def _passes_filters(
        self,
        source: str,
        media_id: int | None,
        duration: int | None = None,
        width: int | None = None,
        height: int | None = None,
        min_duration: float | None = None,
    ) -> bool:
        if media_id is None:
            return False

        if MEDIA_TYPE == "video":
            if duration is None:
                return False
            if min_duration is not None:
                if duration < min_duration:
                    return False
            elif duration > MAX_DURATION_SECONDS:
                return False

        if width is not None and height is not None:
            if not matches_orientation(width, height, ORIENTATION):
                return False

        if make_media_key(source, media_id) in self.used_ids:
            return False
        return True

    async def is_available(
        self,
        source: str,
        media_id: int | None,
        duration: int | None = None,
        width: int | None = None,
        height: int | None = None,
        min_duration: float | None = None,
    ) -> bool:
        """Проходит ли элемент фильтры — без захвата. Нужно для сбора пула кандидатов."""
        async with self._lock:
            return self._passes_filters(
                source, media_id, duration, width, height, min_duration
            )

    async def try_claim(
        self,
        source: str,
        media_id: int | None,
        duration: int | None = None,
        width: int | None = None,
        height: int | None = None,
        min_duration: float | None = None,
    ) -> bool:
        async with self._lock:
            if not self._passes_filters(
                source, media_id, duration, width, height, min_duration
            ):
                return False
            self.used_ids.add(make_media_key(source, media_id))
            return True

    async def release(self, source: str, media_id: int) -> None:
        async with self._lock:
            self.used_ids.discard(make_media_key(source, media_id))

    async def commit(self, source: str, media_id: int, **meta) -> None:
        """Фиксирует клип на диске после успешной сборки пары audio+video."""
        if self.store is None:
            return
        key = make_media_key(source, media_id)
        await asyncio.to_thread(self.store.remember, key, **meta)


# ---------------------------------------------------------------------------
# Загрузка настроек
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def load_settings() -> None:
    """Загружает глобальные настройки из .env."""
    global MEDIA_TYPE, ORIENTATION, GLOBAL_VIBE, CHATTERBOX_TTS_URL
    global TTS_LANGUAGE, TTS_VOICE_FILE, TTS_VOICE_MODE
    global TTS_SERVER_TIMEOUT, SCRIPT_CHUNK_MAX_CHARS, FAILED_CHUNKS_LOG
    global TTS_SPEED_FACTOR, TTS_TEMPERATURE, TTS_EXAGGERATION, TTS_CFG_WEIGHT, TTS_CHUNK_SIZE
    global AUDIO_PAUSE_CLEAN, AUDIO_PAUSE_ENTER_DB, AUDIO_PAUSE_EXIT_DB
    global AUDIO_PAUSE_MIN_MS, AUDIO_PAUSE_GUARD_MS
    global AUDIO_PAUSE_BRIDGE_MAX_MS, AUDIO_PAUSE_BRIDGE_PEAK_DB
    global AUDIO_PAUSE_TAIL_EXTEND_MS, AUDIO_PAUSE_TAIL_PEAK_DB, AUDIO_KEEP_RAW
    global AUDIO_STITCH_CLEAN, AUDIO_STITCH_MIN_ZERO_MS, AUDIO_STITCH_FADE_BEFORE_MS
    global AUDIO_VAD_CLEAN, AUDIO_VAD_AGGRESSIVENESS
    global AUDIO_VAD_SPEECH_PAD_MS, AUDIO_VAD_MIN_NONSPEECH_MS
    global AUDIO_SPECTRAL_CLEAN, AUDIO_SPECTRAL_MIN_REGION_MS
    global AUDIO_SPECTRAL_MAX_GAP_MS
    global STRESS_VERIFY_ENABLED, STRESS_AUTOCORRECT_ENABLED, STRESS_FILL_UNMARKED
    global STRESS_MISMATCH_LOG, STRESS_EXCEPTION_DICT, STRESS_VERIFY_ENGINE
    global MEDIA_SEMANTIC_ENABLED, MEDIA_DOWNLOAD_ENABLED, MEDIA_CANDIDATE_POOL, MEDIA_RANKER
    global MEDIA_EMBEDDING_MODEL, USED_MEDIA_STORE, MEDIA_SELECTION_LOG

    load_dotenv()

    TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", TTS_LANGUAGE).strip()
    TTS_VOICE_FILE = os.getenv("TTS_VOICE_FILE", TTS_VOICE_FILE).strip()
    TTS_VOICE_MODE = os.getenv("TTS_VOICE_MODE", TTS_VOICE_MODE).strip()
    MEDIA_TYPE = os.getenv("MEDIA_TYPE", MEDIA_TYPE).strip().lower()
    ORIENTATION = os.getenv("ORIENTATION", ORIENTATION).strip().lower()
    GLOBAL_VIBE = os.getenv("GLOBAL_VIBE", GLOBAL_VIBE).strip()
    CHATTERBOX_TTS_URL = os.getenv("CHATTERBOX_TTS_URL", CHATTERBOX_TTS_URL).strip()
    TTS_SERVER_TIMEOUT = int(os.getenv("TTS_SERVER_TIMEOUT", str(TTS_SERVER_TIMEOUT)))
    SCRIPT_CHUNK_MAX_CHARS = int(
        os.getenv("SCRIPT_CHUNK_MAX_CHARS", str(SCRIPT_CHUNK_MAX_CHARS))
    )
    FAILED_CHUNKS_LOG = os.getenv("FAILED_CHUNKS_LOG", FAILED_CHUNKS_LOG).strip()
    TTS_SPEED_FACTOR = float(os.getenv("TTS_SPEED_FACTOR", str(TTS_SPEED_FACTOR)))
    TTS_TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", str(TTS_TEMPERATURE)))
    TTS_EXAGGERATION = float(os.getenv("TTS_EXAGGERATION", str(TTS_EXAGGERATION)))
    TTS_CFG_WEIGHT = float(os.getenv("TTS_CFG_WEIGHT", str(TTS_CFG_WEIGHT)))
    TTS_CHUNK_SIZE = int(os.getenv("TTS_CHUNK_SIZE", str(TTS_CHUNK_SIZE)))
    if TTS_CHUNK_SIZE < 50:
        raise ValueError("TTS_CHUNK_SIZE должен быть >= 50")

    AUDIO_PAUSE_CLEAN = _env_bool("AUDIO_PAUSE_CLEAN", AUDIO_PAUSE_CLEAN)
    AUDIO_PAUSE_ENTER_DB = float(os.getenv("AUDIO_PAUSE_ENTER_DB", str(AUDIO_PAUSE_ENTER_DB)))
    AUDIO_PAUSE_EXIT_DB = float(os.getenv("AUDIO_PAUSE_EXIT_DB", str(AUDIO_PAUSE_EXIT_DB)))
    AUDIO_PAUSE_MIN_MS = float(os.getenv("AUDIO_PAUSE_MIN_MS", str(AUDIO_PAUSE_MIN_MS)))
    AUDIO_PAUSE_GUARD_MS = float(os.getenv("AUDIO_PAUSE_GUARD_MS", str(AUDIO_PAUSE_GUARD_MS)))
    AUDIO_PAUSE_BRIDGE_MAX_MS = float(
        os.getenv("AUDIO_PAUSE_BRIDGE_MAX_MS", str(AUDIO_PAUSE_BRIDGE_MAX_MS))
    )
    AUDIO_PAUSE_BRIDGE_PEAK_DB = float(
        os.getenv("AUDIO_PAUSE_BRIDGE_PEAK_DB", str(AUDIO_PAUSE_BRIDGE_PEAK_DB))
    )
    AUDIO_PAUSE_TAIL_EXTEND_MS = float(
        os.getenv("AUDIO_PAUSE_TAIL_EXTEND_MS", str(AUDIO_PAUSE_TAIL_EXTEND_MS))
    )
    AUDIO_PAUSE_TAIL_PEAK_DB = float(
        os.getenv("AUDIO_PAUSE_TAIL_PEAK_DB", str(AUDIO_PAUSE_TAIL_PEAK_DB))
    )
    AUDIO_STITCH_CLEAN = _env_bool("AUDIO_STITCH_CLEAN", AUDIO_STITCH_CLEAN)
    AUDIO_STITCH_MIN_ZERO_MS = float(
        os.getenv("AUDIO_STITCH_MIN_ZERO_MS", str(AUDIO_STITCH_MIN_ZERO_MS))
    )
    AUDIO_STITCH_FADE_BEFORE_MS = float(
        os.getenv("AUDIO_STITCH_FADE_BEFORE_MS", str(AUDIO_STITCH_FADE_BEFORE_MS))
    )
    AUDIO_VAD_CLEAN = _env_bool("AUDIO_VAD_CLEAN", AUDIO_VAD_CLEAN)
    AUDIO_VAD_AGGRESSIVENESS = int(
        os.getenv("AUDIO_VAD_AGGRESSIVENESS", str(AUDIO_VAD_AGGRESSIVENESS))
    )
    AUDIO_VAD_SPEECH_PAD_MS = float(
        os.getenv("AUDIO_VAD_SPEECH_PAD_MS", str(AUDIO_VAD_SPEECH_PAD_MS))
    )
    AUDIO_VAD_MIN_NONSPEECH_MS = float(
        os.getenv("AUDIO_VAD_MIN_NONSPEECH_MS", str(AUDIO_VAD_MIN_NONSPEECH_MS))
    )
    AUDIO_SPECTRAL_CLEAN = _env_bool("AUDIO_SPECTRAL_CLEAN", AUDIO_SPECTRAL_CLEAN)
    AUDIO_SPECTRAL_MIN_REGION_MS = float(
        os.getenv("AUDIO_SPECTRAL_MIN_REGION_MS", str(AUDIO_SPECTRAL_MIN_REGION_MS))
    )
    AUDIO_SPECTRAL_MAX_GAP_MS = float(
        os.getenv("AUDIO_SPECTRAL_MAX_GAP_MS", str(AUDIO_SPECTRAL_MAX_GAP_MS))
    )
    if AUDIO_VAD_AGGRESSIVENESS not in (0, 1, 2, 3):
        raise ValueError("AUDIO_VAD_AGGRESSIVENESS должен быть от 0 до 3")
    AUDIO_KEEP_RAW = _env_bool("AUDIO_KEEP_RAW", AUDIO_KEEP_RAW)
    if AUDIO_PAUSE_ENTER_DB < AUDIO_PAUSE_EXIT_DB:
        raise ValueError("AUDIO_PAUSE_ENTER_DB должен быть выше AUDIO_PAUSE_EXIT_DB")
    STRESS_VERIFY_ENABLED = _env_bool("STRESS_VERIFY_ENABLED", STRESS_VERIFY_ENABLED)
    STRESS_AUTOCORRECT_ENABLED = _env_bool(
        "STRESS_AUTOCORRECT_ENABLED", STRESS_AUTOCORRECT_ENABLED
    )
    STRESS_FILL_UNMARKED = _env_bool("STRESS_FILL_UNMARKED", STRESS_FILL_UNMARKED)
    STRESS_MISMATCH_LOG = os.getenv("STRESS_MISMATCH_LOG", STRESS_MISMATCH_LOG).strip()
    STRESS_EXCEPTION_DICT = os.getenv("STRESS_EXCEPTION_DICT", STRESS_EXCEPTION_DICT).strip()
    STRESS_VERIFY_ENGINE = os.getenv("STRESS_VERIFY_ENGINE", STRESS_VERIFY_ENGINE).strip().lower()

    try:
        from stress_verify import configure_exception_dictionary

        configure_exception_dictionary(STRESS_EXCEPTION_DICT)
    except ImportError:
        pass

    MEDIA_DOWNLOAD_ENABLED = _env_bool("MEDIA_DOWNLOAD_ENABLED", MEDIA_DOWNLOAD_ENABLED)
    MEDIA_SEMANTIC_ENABLED = _env_bool("MEDIA_SEMANTIC_ENABLED", MEDIA_SEMANTIC_ENABLED)
    MEDIA_CANDIDATE_POOL = int(
        os.getenv("MEDIA_CANDIDATE_POOL", str(MEDIA_CANDIDATE_POOL))
    )
    MEDIA_RANKER = os.getenv("MEDIA_RANKER", MEDIA_RANKER).strip().lower()
    MEDIA_EMBEDDING_MODEL = os.getenv("MEDIA_EMBEDDING_MODEL", MEDIA_EMBEDDING_MODEL).strip()
    USED_MEDIA_STORE = os.getenv("USED_MEDIA_STORE", USED_MEDIA_STORE).strip()
    MEDIA_SELECTION_LOG = os.getenv("MEDIA_SELECTION_LOG", MEDIA_SELECTION_LOG).strip()

    if MEDIA_TYPE not in ("video", "photo"):
        raise ValueError("MEDIA_TYPE должен быть 'video' или 'photo'")
    if ORIENTATION not in ("portrait", "landscape", "square"):
        raise ValueError("ORIENTATION должен быть 'portrait', 'landscape' или 'square'")
    if MEDIA_RANKER not in ("llm", "embedding", "lexical"):
        raise ValueError("MEDIA_RANKER должен быть 'llm', 'embedding' или 'lexical'")
    if MEDIA_CANDIDATE_POOL < 1:
        raise ValueError("MEDIA_CANDIDATE_POOL должен быть >= 1")


def get_file_extension() -> str:
    return ".mp4" if MEDIA_TYPE == "video" else ".jpg"


def get_pexels_search_url() -> str:
    return PEXELS_PHOTO_SEARCH_URL if MEDIA_TYPE == "photo" else PEXELS_VIDEO_SEARCH_URL


def get_pixabay_search_url() -> str:
    return PIXABAY_PHOTO_SEARCH_URL if MEDIA_TYPE == "photo" else PIXABAY_VIDEO_SEARCH_URL


def get_pexels_results_key() -> str:
    return "photos" if MEDIA_TYPE == "photo" else "videos"


def get_pixabay_orientation_param() -> str | None:
    """Pixabay использует vertical/horizontal вместо portrait/landscape."""
    mapping = {"portrait": "vertical", "landscape": "horizontal"}
    return mapping.get(ORIENTATION)


def build_audio_path(index: int, output_dir: Path) -> Path:
    return output_dir / f"{index:03d}_audio.wav"


def build_video_path(index: int, output_dir: Path) -> Path:
    return output_dir / f"{index:03d}_video.mp4"


def build_raw_video_path(index: int, output_dir: Path) -> Path:
    return output_dir / f"{index:03d}_raw.mp4"


def build_photo_path(index: int, prompt: str, output_dir: Path) -> Path:
    name = re.sub(r'[<>:"/\\|?*]', "", prompt.replace(" ", "_"))
    return output_dir / f"{index:03d}_{name}{get_file_extension()}"


def is_valid_audio_file(audio_path: Path) -> bool:
    """Проверяет, что аудиофайл существует и не пустой."""
    return audio_path.is_file() and audio_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------


def log_section(title: str) -> None:
    line = "─" * 60
    print(f"\n{line}\n  {title}\n{line}")


def log_info(label: str, value: str) -> None:
    print(f"  • {label}: {value}")


def log_search_prompt(search_prompt: str) -> None:
    print(f"  ✦ Gemini prompt: {search_prompt!r}")


# ---------------------------------------------------------------------------
# Чтение и дробление сценария
# ---------------------------------------------------------------------------


def read_script(filepath: str) -> str:
    """Читает файл сценария."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    return path.read_text(encoding="utf-8").strip()


def chunk_script(text: str, max_chars: int | None = None) -> list[str]:
    """Дробит сценарий на TTS-чанки: сначала по абзацам (\\n\\n), затем по лимиту длины.

    Длинные абзацы дополнительно режутся по границам предложений (.!?…),
    при необходимости — по строкам и словам (без разрыва U+0301 внутри токена).
    """
    limit = max_chars if max_chars is not None else SCRIPT_CHUNK_MAX_CHARS
    if limit <= 0:
        raise ValueError("SCRIPT_CHUNK_MAX_CHARS должен быть > 0")

    stripped = text.strip()
    if not stripped:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", stripped)
        if paragraph.strip()
    ]
    if not paragraphs:
        paragraphs = [stripped]

    chunks: list[str] = []
    for paragraph in paragraphs:
        chunks.extend(_split_paragraph_to_chunks(paragraph, limit))
    return chunks


def _split_paragraph_to_chunks(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    units = _split_paragraph_into_units(paragraph)
    if len(units) == 1 and len(units[0]) > max_chars:
        return _split_oversized_text(units[0], max_chars)

    packed: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for unit in units:
        extra = len(unit) + (1 if buffer else 0)
        if buffer and buffer_len + extra > max_chars:
            packed.append("\n".join(buffer))
            buffer = [unit]
            buffer_len = len(unit)
        else:
            if buffer:
                buffer_len += 1 + len(unit)
            else:
                buffer_len = len(unit)
            buffer.append(unit)

    if buffer:
        packed.append("\n".join(buffer))

    final: list[str] = []
    for piece in packed:
        if len(piece) <= max_chars:
            final.append(piece)
        else:
            final.extend(_split_oversized_text(piece, max_chars))
    return final


def _split_paragraph_into_units(paragraph: str) -> list[str]:
    """Предложения; если одно длинное «предложение» — строки абзаца."""
    sentences = _split_into_sentences(paragraph)
    if len(sentences) > 1:
        return sentences

    lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
    if len(lines) > 1:
        return lines
    return sentences or [paragraph]


def _split_into_sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []
    length = len(paragraph)
    index = 0

    while index < length:
        char = paragraph[index]
        current.append(char)
        if char in ".!?…":
            trailing = index + 1
            while trailing < length and paragraph[trailing] in "\"»')}]":
                current.append(paragraph[trailing])
                trailing += 1
            if trailing >= length or paragraph[trailing].isspace():
                sentence = "".join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []
                index = trailing
                while index < length and paragraph[index].isspace():
                    index += 1
                continue
        index += 1

    remainder = "".join(current).strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def _split_oversized_text(text: str, max_chars: int) -> list[str]:
    """Последний рубеж: нарезка по словам (U+0301 остаётся внутри токена)."""
    words = text.split()
    if not words:
        return [text]

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for word in words:
        extra = len(word) + (1 if buffer else 0)
        if buffer and buffer_len + extra > max_chars:
            chunks.append(" ".join(buffer))
            buffer = [word]
            buffer_len = len(word)
        elif not buffer and len(word) > max_chars:
            chunks.append(word)
            buffer_len = 0
        else:
            if buffer:
                buffer_len += 1 + len(word)
            else:
                buffer_len = len(word)
            buffer.append(word)

    if buffer:
        chunks.append(" ".join(buffer))
    return chunks


def log_failed_chunk(
    *,
    chunk_index: int,
    total_chunks: int,
    stage: str,
    error: str,
    script_text: str,
    tts_text: str | None = None,
    log_path: str | Path | None = None,
) -> None:
    """Пишет FAILED-чанк в failed_chunks.log для ручного перезапуска."""
    path = Path(log_path or FAILED_CHUNKS_LOG)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    script_one_line = " ".join(script_text.split())
    parts = [
        f"{timestamp} | FAILED",
        f"chunk={chunk_index}/{total_chunks}",
        f"stage={stage}",
        f"error={error}",
        f"script={script_one_line}",
    ]
    if tts_text is not None:
        parts.append(f"tts_text={' '.join(tts_text.split())}")
    line = " | ".join(parts)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")
    print(f"  ✗ FAILED chunk {chunk_index}/{total_chunks} → {path.name}")


def save_prompts(prompts: list[str], filepath: str = PROMPTS_FILE) -> None:
    """Сохраняет сгенерированные промпты в файл."""
    Path(filepath).write_text("\n".join(prompts) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------


def build_gemini_system_prompt() -> str:
    """Формирует системный промпт режиссёра монтажа для поиска стокового видео."""
    vibe_section = ""
    if GLOBAL_VIBE:
        vibe_section = f"""
Общая атмосфера ролика: {GLOBAL_VIBE}. Подбери физический объект или фон, который передаёт это настроение, не используя абстрактные слова.
"""

    return f"""Ты — режиссер видеомонтажа. Твоя задача — перевести кусок текста в ОДИН поисковый запрос (1-3 слова на английском) для бесплатного стока видео.
{vibe_section}
ПРАВИЛО 1: НИКАКИХ абстракций, эмоций, метафор или сложных действий (запрещено: sadness, double life, lonely man thinking).
ПРАВИЛО 2: ТОЛЬКО физические, базовые объекты, фоны или текстуры, которые легко найти на стоках (разрешено: dark empty room, rainy window, smoking cigarette, shadow on wall, typing keyboard).
Верни строго JSON массив с одной строкой."""


def parse_gemini_response(raw_text: str) -> str | None:
    """Парсит JSON-массив и возвращает только первый поисковый запрос."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Ответ Gemini не является JSON-массивом")

    prompts = [str(item).strip() for item in data if str(item).strip()]
    if not prompts:
        return None
    return prompts[0]


def generate_search_prompt_for_chunk(chunk: str, client: genai.Client) -> str | None:
    """Отправляет чанк в Gemini и возвращает один визуальный промпт для поиска."""
    system_prompt = build_gemini_system_prompt()
    prompt = f"{system_prompt}\n\nScript fragment:\n{chunk}"
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    raw_text = response.text or ""
    return parse_gemini_response(raw_text)


def _generate_visual_brief_sync(chunk: str, client: genai.Client) -> VisualBrief | None:
    """Визуальное описание сцены: что должно быть в кадре, чтобы смысл читался без слов."""
    prompt = f"{build_visual_brief_prompt()}\n\nТекст сцены:\n{chunk}"
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return parse_visual_brief(response.text or "")


async def generate_visual_brief_for_chunk(
    chunk: str, client: genai.Client
) -> VisualBrief | None:
    return await asyncio.to_thread(_generate_visual_brief_sync, chunk, client)


def _rank_candidates_with_llm_sync(
    brief: VisualBrief,
    candidates: list[MediaCandidate],
    client: genai.Client,
) -> tuple[int | None, str]:
    """Просит модель выбрать кандидата по смыслу.

    Возвращает (0-based индекс или None, сырой текст ответа) — сырой текст
    нужен в логе, чтобы отличить реальный (пусть и неудачный) выбор модели
    от сбоя парсинга её ответа.
    """
    prompt = build_ranking_prompt(brief, candidates)
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    raw_text = response.text or ""
    return parse_ranking_choice(raw_text, len(candidates)), raw_text


async def rank_candidates(
    brief: VisualBrief,
    candidates: list[MediaCandidate],
    client: genai.Client,
) -> SelectionOutcome:
    """Упорядочивает кандидатов по смысловому соответствию описанию сцены.

    Порядок важен: если верхний кандидат уже занят параллельным чанком,
    вызывающая сторона идёт по списку дальше.
    """
    if not candidates:
        return SelectionOutcome(chosen=None, ranker=MEDIA_RANKER, pool=[], reason="пустой пул")

    if MEDIA_RANKER == "embedding":
        try:
            ranker = await asyncio.to_thread(EmbeddingRanker.get, MEDIA_EMBEDDING_MODEL)
            ordered = await asyncio.to_thread(ranker.rank, brief, candidates)
            return SelectionOutcome(
                chosen=ordered[0],
                ranker="embedding",
                pool=ordered,
                reason=f"косинусная близость {ordered[0].score}",
            )
        except ImportError:
            print("  ⚠ sentence-transformers не установлен → лексическое ранжирование")
        except Exception as error:
            print(f"  ⚠ Ошибка эмбеддингов ({error}) → лексическое ранжирование")
        ordered = rank_lexically(brief, candidates)
        return SelectionOutcome(
            chosen=ordered[0],
            ranker="lexical",
            pool=ordered,
            reason=f"лексическое совпадение {ordered[0].score}",
            fallback_used=True,
        )

    # Лексический скор считаем всегда: он задаёт порядок «второго выбора»
    # и попадает в лог как объективная метрика рядом с решением модели.
    ordered = rank_lexically(brief, candidates)

    if MEDIA_RANKER == "lexical":
        return SelectionOutcome(
            chosen=ordered[0],
            ranker="lexical",
            pool=ordered,
            reason=f"лексическое совпадение {ordered[0].score}",
        )

    try:
        choice, raw_response = await asyncio.to_thread(
            _rank_candidates_with_llm_sync, brief, ordered, client
        )
    except Exception as error:
        print(f"  ⚠ Ошибка LLM-ранжирования ({error}) → лексическое")
        return SelectionOutcome(
            chosen=ordered[0],
            ranker="lexical",
            pool=ordered,
            reason=f"лексическое совпадение {ordered[0].score}",
            fallback_used=True,
            raw_llm_response=None,
        )

    if choice is None:
        return SelectionOutcome(
            chosen=ordered[0],
            ranker="lexical",
            pool=ordered,
            reason="модель не вернула номер → лексическое",
            fallback_used=True,
            raw_llm_response=raw_response,
        )

    picked = ordered.pop(choice)
    ordered.insert(0, picked)
    return SelectionOutcome(
        chosen=picked,
        ranker="llm",
        pool=ordered,
        reason=f"выбор модели, лексический скор {picked.score}",
        raw_llm_response=raw_response,
    )


def build_gemini_tts_system_prompt() -> str:
    """Системный промпт для фонетической подготовки и режиссуры пауз перед Chatterbox TTS.

    Формат ударения и пунктуации согласован с реальным токенизатором Chatterbox
    (MTLTokenizer): текст приводится к нижнему регистру, а ударная гласная кодируется
    отдельным токеном U+0301 (id 764 в grapheme_mtl_merged_expanded_v1.json).
    Заглавные буквы и '+' теряются, «—» и «...» сервер (punc_norm) заменяет на '-'/','.
    """
    stress = "\u0301"  # COMBINING ACUTE ACCENT (U+0301)
    return f"""Ты — лингвистический препроцессор и режиссер озвучания для системы синтеза речи (TTS). Твоя задача — подготовить текст для идеального и выразительного прочтения диктором.
Твои строгие правила:
1. Расставь ударения символом U+0301 СРАЗУ ПОСЛЕ ударной гласной во ВСЕХ словах от двух слогов и во всех формах, где TTS может ошибиться (существительные, глаголы, прилагательные, наречия, числительные, имена). В односложных словах ударения не ставь. Для омографов (замо{stress}к/за{stress}мок, до{stress}ма/дома{stress}) выбирай ударение по смыслу фразы.
2. Ударение обозначай ТОЛЬКО символом юникода U+0301, поставленным СРАЗУ ПОСЛЕ обычной кириллической гласной. Пример: "замо{stress}к", "дома{stress}", "блочны{stress}е", "просыпа{stress}ться", "вашингто{stress}н". КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО: заглавная буква, плюс (+), апостроф, а также латинские буквы с акутом (á, é, í, ó, ú) — модель их не читает и начинает шуметь. Буква под ударением всегда остаётся кириллической.
3. Орфография е/ё: сохраняй написание оригинала СИМВОЛ В СИМВОЛ. «все» (все люди, почти все) и «всё» (всё хорошо, всё как вчера) — разные слова. Если в исходнике «все», в ответе должно быть «все»; если «всё» — «всё». Не «улучшай» орфографию.
4. Переведи все числа, даты, время и аббревиатуры в текст прописью в правильном падеже (например: "в 1998 году" -> "в ты{stress}сяча девятьсо{stress}т девяно{stress}сто восьмо{stress}м году{stress}", "06:14" -> "шесть часо{stress}в четы{stress}рнадцать мину{stress}т"). ЦИФР В ОТВЕТЕ БЫТЬ НЕ ДОЛЖНО ВООБЩЕ: токенизатор их не знает и выдаёт вместо речи шум.
5. Буквенные аббревиатуры записывай русскими названиями букв через пробел, каждое слово с ударением: ЦРУ -> "цэ{stress} эр у{stress}", КГБ -> "ка{stress} гэ бэ{stress}", СССР -> "эс эс эс э{stress}р", ФБР -> "эф бэ{stress} эр". Латиницу не используй.
6. Убери эмодзи и лишние спецсимволы. Оставь буквы, символы ударения U+0301 и базовую пунктуацию: запятая, точка, вопросительный и восклицательный знаки. Кавычки «» убери.
7. Режиссура пауз: короткую паузу обозначай ЗАПЯТОЙ, длинную драматическую — концом предложения (ТОЧКОЙ). НЕ используй тире (—) и многоточия (...): сервер заменяет их на дефис и запятую. Не ставь лишних знаков в начале и конце текста. Если текст монотонный, добавь максимум одну запятую в середине длинного предложения.
8. В ответе выдай ТОЛЬКО обработанный текст. Никаких приветствий, подтверждений или лишних комментариев."""


def clean_gemini_plain_text(raw_text: str) -> str:
    """Убирает markdown-обёртки и лишние пробелы из ответа Gemini."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _prepare_text_with_gemini_sync(text: str, client: genai.Client) -> str:
    """Синхронно отправляет текст в Gemini для TTS-препроцессинга."""
    system_prompt = build_gemini_tts_system_prompt()
    prompt = f"{system_prompt}\n\nText:\n{text}"
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    processed = clean_gemini_plain_text(response.text or "")
    if not processed:
        raise ValueError("Gemini вернула пустой текст для TTS.")
    return processed


async def prepare_text_with_gemini(text: str, client: genai.Client) -> str:
    """Асинхронно готовит текст для Chatterbox через Gemini."""
    return await asyncio.to_thread(_prepare_text_with_gemini_sync, text, client)


# ---------------------------------------------------------------------------
# Chatterbox TTS API и FFmpeg
# ---------------------------------------------------------------------------


def build_chatterbox_payload(text: str, *, seed: int = 0, split_text: bool = True) -> dict:
    """Shared request contract for production and controlled raw-audio comparisons."""
    payload = {
        "text": text,
        "voice_mode": TTS_VOICE_MODE,
        "language": TTS_LANGUAGE,
        "speed_factor": TTS_SPEED_FACTOR,
        "temperature": TTS_TEMPERATURE,
        "exaggeration": TTS_EXAGGERATION,
        "cfg_weight": TTS_CFG_WEIGHT,
        "chunk_size": TTS_CHUNK_SIZE,
        "output_format": "wav",
        "seed": seed,
        "split_text": split_text,
    }

    if TTS_VOICE_MODE == "clone":
        payload["reference_audio_filename"] = TTS_VOICE_FILE
    else:
        payload["predefined_voice_id"] = TTS_VOICE_FILE

    return payload


def generate_audio_with_chatterbox(text: str, output_path: Path) -> None:
    """Отправляет текст на Chatterbox TTS-сервер и сохраняет бинарный ответ."""
    response = requests.post(
        CHATTERBOX_TTS_URL,
        json=build_chatterbox_payload(text),
        timeout=TTS_SERVER_TIMEOUT,
    )

    if not response.ok:
        detail = response.text.strip()
        raise RuntimeError(
            f"Chatterbox TTS вернул {response.status_code}: {detail or response.reason}"
        )

    with open(output_path, "wb") as file:
        file.write(response.content)


async def generate_audio_with_chatterbox_async(text: str, output_path: Path) -> None:
    """Асинхронная обёртка над HTTP-запросом к Chatterbox TTS."""
    if TTS_SEMAPHORE is None:
        await asyncio.to_thread(generate_audio_with_chatterbox, text, output_path)
    else:
        async with TTS_SEMAPHORE:
            await asyncio.to_thread(generate_audio_with_chatterbox, text, output_path)


def _cleanup_audio_tail(audio_path: Path) -> None:
    """Чистит паузы, обрезает хвостовые артефакты и делает fade-out (best-effort)."""
    try:
        from audio_postprocess import postprocess_file
    except ImportError:
        return  # numpy/soundfile не установлены — пропускаем пост-обработку
    try:
        if AUDIO_KEEP_RAW:
            raw_dir = audio_path.parent / "raw"
            raw_dir.mkdir(exist_ok=True)
            shutil.copy2(audio_path, raw_dir / audio_path.name)

        stats: dict = {}
        postprocess_file(
            audio_path,
            pause_clean=AUDIO_PAUSE_CLEAN,
            pause_enter_db=AUDIO_PAUSE_ENTER_DB,
            pause_exit_db=AUDIO_PAUSE_EXIT_DB,
            pause_min_ms=AUDIO_PAUSE_MIN_MS,
            pause_guard_ms=AUDIO_PAUSE_GUARD_MS,
            pause_bridge_max_ms=AUDIO_PAUSE_BRIDGE_MAX_MS,
            pause_bridge_peak_db=AUDIO_PAUSE_BRIDGE_PEAK_DB,
            pause_tail_extend_ms=AUDIO_PAUSE_TAIL_EXTEND_MS,
            pause_tail_peak_db=AUDIO_PAUSE_TAIL_PEAK_DB,
            stitch_clean=AUDIO_STITCH_CLEAN,
            stitch_min_zero_ms=AUDIO_STITCH_MIN_ZERO_MS,
            stitch_fade_before_ms=AUDIO_STITCH_FADE_BEFORE_MS,
            vad_clean=AUDIO_VAD_CLEAN,
            vad_aggressiveness=AUDIO_VAD_AGGRESSIVENESS,
            vad_speech_pad_ms=AUDIO_VAD_SPEECH_PAD_MS,
            vad_min_nonspeech_ms=AUDIO_VAD_MIN_NONSPEECH_MS,
            spectral_clean=AUDIO_SPECTRAL_CLEAN,
            spectral_min_region_ms=AUDIO_SPECTRAL_MIN_REGION_MS,
            spectral_max_gap_ms=AUDIO_SPECTRAL_MAX_GAP_MS,
            stats=stats,
        )
        cleaned = stats.get("pauses_cleaned", 0)
        if cleaned:
            print(
                f"  ✓ Очищено пауз в {audio_path.name}: {cleaned} "
                f"({stats.get('pauses_ms', 0.0):.0f} ms)"
            )
        stitches = stats.get("stitches_cleaned", 0)
        if stitches:
            print(f"  ✓ Смягчено стыков в {audio_path.name}: {stitches}")
        vad_regions = stats.get("vad_regions_cleaned", 0)
        if vad_regions:
            print(
                f"  ✓ VAD удалил не-речь в {audio_path.name}: {vad_regions} "
                f"участков ({stats.get('vad_regions_ms', 0.0):.0f} ms)"
            )
        spectral_regions = stats.get("spectral_regions_cleaned", 0)
        if spectral_regions:
            print(
                f"  ✓ Спектрально удалён шум в {audio_path.name}: {spectral_regions} "
                f"участков ({stats.get('spectral_regions_ms', 0.0):.0f} ms)"
            )
    except Exception as error:
        print(f"  ⚠ Пост-обработка аудио пропущена: {error}")


async def cleanup_audio_tail_async(audio_path: Path) -> None:
    await asyncio.to_thread(_cleanup_audio_tail, audio_path)


async def wait_for_audio_file_ready(
    audio_path: Path,
    poll_interval: float = AUDIO_WRITE_POLL_INTERVAL,
    max_wait: float = AUDIO_WRITE_MAX_WAIT,
) -> None:
    """Ждёт полной записи аудиофайла: размер > 0 и перестал расти."""
    waited = 0.0
    previous_size = -1

    while waited < max_wait:
        if audio_path.is_file():
            current_size = audio_path.stat().st_size
            if current_size > 0 and current_size == previous_size:
                return
            previous_size = current_size

        await asyncio.sleep(poll_interval)
        waited += poll_interval

    if not is_valid_audio_file(audio_path):
        raise RuntimeError(
            f"Аудиофайл {audio_path.name} не готов после ожидания {max_wait:.0f} с."
        )


def get_trim_duration(audio_duration: float) -> float:
    """Длительность обрезки видео с запасом под последнее слово."""
    return audio_duration + VIDEO_TRIM_PADDING


async def get_audio_duration(audio_path: Path) -> float:
    """Измеряет длительность аудио через ffprobe."""
    process = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"ffprobe не смог прочитать {audio_path.name}: {stderr.decode().strip()}"
        )
    return float(stdout.decode().strip())


async def resolve_audio_duration(audio_path: Path) -> float | None:
    """Проверяет аудиофайл и возвращает актуальную длительность или None."""
    if not is_valid_audio_file(audio_path):
        return None
    try:
        duration = await get_audio_duration(audio_path)
    except (RuntimeError, ValueError, OSError):
        return None
    if duration <= 0:
        return None
    return duration


async def trim_video_to_duration(
    input_path: Path,
    output_path: Path,
    audio_path: Path,
) -> bool:
    """Обрезает видео до длительности аудио. Длительность читается заново из файла."""
    audio_duration = await resolve_audio_duration(audio_path)
    if audio_duration is None:
        print(
            f"  ⚠ Аудио {audio_path.name} отсутствует или пустое — обрезка пропущена."
        )
        return False

    trim_duration = get_trim_duration(audio_duration)

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-t",
        f"{trim_duration:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(output_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        print(f"  ⚠ Ошибка FFmpeg: {stderr.decode().strip()}")
        if output_path.exists():
            output_path.unlink()
        return False
    return True


async def check_ffmpeg_tools() -> None:
    """Проверяет наличие ffmpeg и ffprobe в системе."""
    for tool in ("ffmpeg", "ffprobe"):
        process = await asyncio.create_subprocess_exec(
            tool,
            "-version",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"{tool} не найден. Установите FFmpeg и добавьте его в PATH."
            )


# ---------------------------------------------------------------------------
# Обработка чанков: Gemini + TTS
# ---------------------------------------------------------------------------


async def process_chunk(
    index: int,
    total: int,
    script_fragment: str,
    client: genai.Client,
    output_dir: Path,
    gemini_semaphore: asyncio.Semaphore,
) -> ChunkResult:
    """Генерирует поисковые промпты (Gemini), TTS-препроцессинг (Gemini) и озвучку (Chatterbox)."""
    log_section(f"CHUNK {index}/{total}")
    preview = (
        script_fragment
        if len(script_fragment) <= 120
        else script_fragment[:117] + "..."
    )
    log_info("Script fragment", preview)

    search_prompt, visual_brief = await prepare_visual(index, script_fragment, client, gemini_semaphore)
    if MEDIA_TYPE != "video":
        return ChunkResult(index, script_fragment, search_prompt=search_prompt, visual_brief=visual_brief)

    stage = PROJECT.stage("audio", index, {"text": script_fragment, "settings": stage_settings("audio")}) if PROJECT else None
    if stage:
        cached = stage.cached()
        if cached:
            audio_path = stage.path / cached["audio_file"]
            duration = await resolve_audio_duration(audio_path)
            if duration is not None:
                log_info("Resume audio", audio_path.name)
                return ChunkResult(index, script_fragment, search_prompt, audio_path, duration, visual_brief)
        stage.start()
        output_dir = stage.path
    result = await generate_chunk_audio(index, total, script_fragment, client, output_dir,
                                        gemini_semaphore, search_prompt, visual_brief)
    if stage:
        if result.audio_path and result.audio_duration:
            stage.finish({"audio_file": result.audio_path.name}, [result.audio_path])
        else:
            stage.fail("Audio generation failed; see project logs")
    return result


async def _prepare_visual_uncached(index, script_fragment, client, gemini_semaphore):
    search_prompt: str | None = None
    visual_brief: VisualBrief | None = None
    if MEDIA_DOWNLOAD_ENABLED or MEDIA_TYPE == "photo":
        async with gemini_semaphore:
            try:
                if MEDIA_SEMANTIC_ENABLED and MEDIA_DOWNLOAD_ENABLED:
                    visual_brief = await generate_visual_brief_for_chunk(
                        script_fragment, client
                    )
                    if visual_brief is None:
                        print(
                            "  ⚠ Визуальное описание не разобрано "
                            "→ откат на подбор по ключевым словам"
                        )
                    else:
                        search_prompt = visual_brief.query
                        log_info("Visual", visual_brief.visual)

                if not search_prompt:
                    search_prompt = await asyncio.to_thread(
                        generate_search_prompt_for_chunk, script_fragment, client
                    )
                if not search_prompt:
                    print("  ⚠ Gemini не вернула промпт, пропуск чанка.")
                    return None, None
                log_search_prompt(search_prompt)
            except (json.JSONDecodeError, ValueError) as error:
                print(f"  ⚠ Ошибка парсинга ответа Gemini: {error}")
                return None, None
            except Exception as error:
                print(f"  ⚠ Ошибка Gemini API: {error}")
                return None, None

    return search_prompt, visual_brief


async def prepare_visual(index, script_fragment, client, gemini_semaphore):
    if not (MEDIA_DOWNLOAD_ENABLED or MEDIA_TYPE == "photo"):
        return None, None
    stage = PROJECT.stage("visual", index, {"text": script_fragment, "settings": stage_settings("visual")}) if PROJECT else None
    if stage:
        cached = stage.cached()
        if cached:
            return cached["prompt"], VisualBrief(**cached["brief"]) if cached["brief"] else None
        stage.start()
    prompt, brief = await _prepare_visual_uncached(index, script_fragment, client, gemini_semaphore)
    if stage:
        if prompt:
            stage.finish({"prompt": prompt, "brief": asdict(brief) if brief else None})
        else:
            stage.fail("Visual request failed; audio can still proceed")
    return prompt, brief


async def generate_chunk_audio(index, total, script_fragment, client, output_dir,
                               gemini_semaphore, search_prompt=None, visual_brief=None):
    final_audio_path = build_audio_path(index, output_dir)
    processed_text: str | None = None

    try:
        async with gemini_semaphore:
            processed_text = await prepare_text_with_gemini(script_fragment, client)

        sanitized = sanitize_tts_text(processed_text, script_fragment)
        processed_text = sanitized.text
        if sanitized.latin_fixed:
            print(f"  ✓ Латиница с акутом исправлена: {sanitized.latin_fixed} симв.")
        if sanitized.mono_stripped:
            print(f"  ✓ Снято ударение с односложных: {sanitized.mono_stripped} слов")
        if sanitized.yo_restored:
            print(f"  ✓ Восстановлено е/ё по исходнику: {sanitized.yo_restored} слов")
        if sanitized.digits_left:
            print(
                f"  ⚠ В тексте остались цифры: {sanitized.digits_left} — "
                "Chatterbox читает их нестабильно"
            )

        gemini_tts_text = processed_text
        if STRESS_VERIFY_ENABLED or STRESS_AUTOCORRECT_ENABLED or STRESS_FILL_UNMARKED:
            try:
                from stress_verify import process_gemini_stress_async

                stress_result = await process_gemini_stress_async(
                    processed_text,
                    chunk_index=index,
                    log_path=STRESS_MISMATCH_LOG,
                    engine_name=STRESS_VERIFY_ENGINE,
                    verify_enabled=STRESS_VERIFY_ENABLED,
                    autocorrect_enabled=STRESS_AUTOCORRECT_ENABLED,
                    fill_unmarked=STRESS_FILL_UNMARKED,
                )
                processed_text = stress_result.text
                gemini_preview = (
                    gemini_tts_text
                    if len(gemini_tts_text) <= 120
                    else gemini_tts_text[:117] + "..."
                )
                log_info("TTS input (Gemini raw)", gemini_preview)
                if processed_text != gemini_tts_text:
                    corrected_preview = (
                        processed_text
                        if len(processed_text) <= 120
                        else processed_text[:117] + "..."
                    )
                    log_info("TTS input (after stress)", corrected_preview)
                if stress_result.filled_count:
                    print(
                        f"  ✓ Доразметка StressRNN: {stress_result.filled_count} слов "
                        f"→ {STRESS_MISMATCH_LOG}"
                    )
                if stress_result.missed_homograph_count:
                    print(
                        f"  ⚠ Gemini пропустила омограф: {stress_result.missed_homograph_count} слов "
                        f"→ {STRESS_MISMATCH_LOG}"
                    )
                if stress_result.missed_unmarked_count:
                    print(
                        f"  ⚠ Gemini не разметила: {stress_result.missed_unmarked_count} слов "
                        f"(без автозаполнения StressRNN) → {STRESS_MISMATCH_LOG}"
                    )
                if stress_result.corrected_count:
                    print(
                        f"  ✓ Автокоррекция ударений: {stress_result.corrected_count} слов "
                        f"→ {STRESS_MISMATCH_LOG}"
                    )
                if stress_result.mismatch_count:
                    print(
                        f"  ⚠ Расхождение ударений: {stress_result.mismatch_count} слов "
                        f"→ {STRESS_MISMATCH_LOG}"
                    )
            except ImportError:
                print("  ⚠ stress_verify недоступен (установите stressrnn)")
            except Exception as error:
                print(f"  ⚠ Проверка ударений пропущена: {error}")

        tts_preview = (
            processed_text
            if len(processed_text) <= 80
            else processed_text[:77] + "..."
        )
        log_info("TTS voice file", TTS_VOICE_FILE)
        if not (STRESS_VERIFY_ENABLED or STRESS_AUTOCORRECT_ENABLED or STRESS_FILL_UNMARKED):
            log_info("TTS input (Gemini)", tts_preview)
        else:
            log_info("TTS input (final)", tts_preview)
        print(f"  ↓ Chatterbox TTS ({TTS_VOICE_FILE}) → {final_audio_path.name}")
        await generate_audio_with_chatterbox_async(processed_text, final_audio_path)
        await wait_for_audio_file_ready(final_audio_path)

        if not is_valid_audio_file(final_audio_path):
            raise RuntimeError(
                f"Файл {final_audio_path.name} не создан или пустой после Chatterbox TTS."
            )

        await cleanup_audio_tail_async(final_audio_path)
    except Exception as error:
        print(f"  ⚠ Ошибка Gemini TTS/Chatterbox: {error}")
        log_failed_chunk(
            chunk_index=index,
            total_chunks=total,
            stage="tts",
            error=str(error),
            script_text=script_fragment,
            tts_text=processed_text,
        )
        if final_audio_path.exists():
            final_audio_path.unlink()
        return ChunkResult(index, script_fragment, search_prompt=search_prompt)

    audio_duration: float | None = None
    try:
        audio_duration = await resolve_audio_duration(final_audio_path)
        if audio_duration is None:
            raise RuntimeError(
                f"Не удалось определить длительность {final_audio_path.name}."
            )
        log_info("Audio duration", f"{audio_duration:.2f} s")
        return ChunkResult(
            chunk_index=index,
            chunk_text=script_fragment,
            search_prompt=search_prompt,
            audio_path=final_audio_path,
            audio_duration=audio_duration,
            visual_brief=visual_brief,
        )
    except Exception as error:
        print(f"  ⚠ Ошибка ffprobe: {error}")
        if final_audio_path.exists():
            final_audio_path.unlink()
        return ChunkResult(index, script_fragment, search_prompt=search_prompt)


async def process_all_chunks(
    chunks: list[str],
    client: genai.Client,
    output_dir: Path,
    gemini_semaphore: asyncio.Semaphore,
) -> list[ChunkResult]:
    """Параллельно обрабатывает чанки: Gemini и Chatterbox TTS."""
    tasks = [
        asyncio.create_task(
            process_chunk(
                index=index,
                total=len(chunks),
                script_fragment=script_fragment,
                client=client,
                output_dir=output_dir,
                gemini_semaphore=gemini_semaphore,
            )
        )
        for index, script_fragment in enumerate(chunks, start=1)
    ]
    return await asyncio.gather(*tasks)


def build_media_tasks(chunk_results: list[ChunkResult], output_dir: Path) -> list[MediaTask]:
    """Собирает задачи: 1 чанк = 1 audio + 1 video."""
    tasks: list[MediaTask] = []

    for chunk_result in chunk_results:
        if not chunk_result.search_prompt:
            continue

        index = chunk_result.chunk_index

        if (
            chunk_result.audio_path is None
            or not is_valid_audio_file(chunk_result.audio_path)
            or chunk_result.audio_duration is None
        ):
            print(
                f"  ⚠ Чанк {index}: нет валидного аудио, "
                "пропуск видео для этого фрагмента."
            )
            continue

        audio_path = chunk_result.audio_path

        if not is_valid_audio_file(audio_path):
            print(
                f"  ⚠ Чанк {index}: аудио {audio_path.name} "
                "отсутствует или пустое, пропуск."
            )
            continue

        tasks.append(
            MediaTask(
                index=index,
                prompt=chunk_result.search_prompt,
                chunk_text=chunk_result.chunk_text,
                audio_path=audio_path,
                audio_duration=chunk_result.audio_duration,
                video_path=build_video_path(index, output_dir),
                raw_video_path=build_raw_video_path(index, output_dir),
                visual_brief=chunk_result.visual_brief,
            )
        )

    return tasks


# ---------------------------------------------------------------------------
# Фильтры и URL медиа
# ---------------------------------------------------------------------------


def make_media_key(source: str, media_id: int) -> str:
    return f"{source}:{MEDIA_TYPE}:{media_id}"


def matches_orientation(width: int, height: int, orientation: str) -> bool:
    if orientation == "portrait":
        return height > width
    if orientation == "landscape":
        return width > height
    if orientation == "square":
        if width == 0 or height == 0:
            return False
        ratio = width / height
        return 0.85 <= ratio <= 1.15
    return True


def get_pexels_download_info(item: dict) -> tuple[str | None, int | None, int | None, int | None]:
    """Возвращает URL, duration, width, height для элемента Pexels."""
    if MEDIA_TYPE == "photo":
        src = item.get("src", {})
        url = (
            src.get(ORIENTATION)
            or src.get("large2x")
            or src.get("large")
            or src.get("original")
        )
        return url, None, item.get("width"), item.get("height")

    video_files = item.get("video_files", [])
    if not video_files:
        return None, item.get("duration"), None, None
    best = max(video_files, key=lambda file_info: file_info.get("width", 0))
    return (
        best.get("link"),
        item.get("duration"),
        best.get("width"),
        best.get("height"),
    )


def get_pixabay_download_info(item: dict) -> tuple[str | None, int | None, int | None, int | None]:
    """Возвращает URL, duration, width, height для элемента Pixabay."""
    if MEDIA_TYPE == "photo":
        width = item.get("imageWidth")
        height = item.get("imageHeight")
        url = item.get("largeImageURL") or item.get("webformatURL")
        return url, None, width, height

    videos = item.get("videos", {})
    for quality in ("large", "medium", "small", "tiny"):
        file_info = videos.get(quality, {})
        if file_info.get("url"):
            return (
                file_info["url"],
                item.get("duration"),
                file_info.get("width"),
                file_info.get("height"),
            )
    return None, item.get("duration"), None, None


# ---------------------------------------------------------------------------
# Pexels API (async)
# ---------------------------------------------------------------------------


async def fetch_pexels_page(
    session: aiohttp.ClientSession,
    query: str,
    api_key: str,
    page: int,
) -> tuple[list[dict], bool]:
    headers = {"Authorization": api_key}
    params: dict[str, str | int] = {
        "query": query,
        "orientation": ORIENTATION,
        "per_page": PEXELS_PER_PAGE,
        "page": page,
    }

    try:
        async with session.get(
            get_pexels_search_url(),
            headers=headers,
            params=params,
        ) as response:
            response.raise_for_status()
            data = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as error:
        print(f"  ⚠ Ошибка Pexels API: {error}")
        return [], False

    results_key = get_pexels_results_key()
    items = data.get(results_key, [])
    has_next_page = data.get("next_page") is not None
    return items, has_next_page


async def collect_pexels_candidates(
    session: aiohttp.ClientSession,
    query: str,
    api_key: str,
    registry: MediaRegistry,
    min_duration: float | None = None,
    pool_size: int = 1,
) -> list[MediaCandidate]:
    """Собирает до pool_size свободных кандидатов Pexels вместе с их подписями."""
    pool: list[MediaCandidate] = []

    for page in range(1, MAX_SEARCH_PAGES + 1):
        items, has_next_page = await fetch_pexels_page(session, query, api_key, page)
        if not items:
            break

        for item in items:
            url, duration, width, height = get_pexels_download_info(item)
            if url is None:
                continue
            if not await registry.is_available(
                source="pexels",
                media_id=item.get("id"),
                duration=duration,
                width=width,
                height=height,
                min_duration=min_duration,
            ):
                continue
            pool.append(
                MediaCandidate(
                    source="pexels",
                    media_id=item["id"],
                    download_url=url,
                    duration=duration,
                    width=width,
                    height=height,
                    caption=extract_caption(item, "pexels"),
                )
            )
            if len(pool) >= pool_size:
                return pool

        if not has_next_page:
            break

    return pool


# ---------------------------------------------------------------------------
# Pixabay API (async fallback)
# ---------------------------------------------------------------------------


async def fetch_pixabay_page(
    session: aiohttp.ClientSession,
    query: str,
    api_key: str,
    page: int,
) -> tuple[list[dict], bool]:
    params: dict[str, str | int] = {
        "key": api_key,
        "q": query,
        "per_page": PIXABAY_PER_PAGE,
        "page": page,
    }

    pixabay_orientation = get_pixabay_orientation_param()
    if pixabay_orientation and MEDIA_TYPE == "photo":
        params["orientation"] = pixabay_orientation

    try:
        async with session.get(get_pixabay_search_url(), params=params) as response:
            response.raise_for_status()
            data = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as error:
        print(f"  ⚠ Ошибка Pixabay API: {error}")
        return [], False

    hits = data.get("hits", [])
    total_hits = data.get("totalHits", 0)
    has_next_page = page * PIXABAY_PER_PAGE < total_hits
    return hits, has_next_page


async def collect_pixabay_candidates(
    session: aiohttp.ClientSession,
    query: str,
    api_key: str,
    registry: MediaRegistry,
    min_duration: float | None = None,
    pool_size: int = 1,
) -> list[MediaCandidate]:
    """Собирает до pool_size свободных кандидатов Pixabay вместе с их tags."""
    pool: list[MediaCandidate] = []

    for page in range(1, MAX_SEARCH_PAGES + 1):
        items, has_next_page = await fetch_pixabay_page(session, query, api_key, page)
        if not items:
            break

        for item in items:
            url, duration, width, height = get_pixabay_download_info(item)
            if url is None:
                continue
            if not await registry.is_available(
                source="pixabay",
                media_id=item.get("id"),
                duration=duration,
                width=width,
                height=height,
                min_duration=min_duration,
            ):
                continue
            pool.append(
                MediaCandidate(
                    source="pixabay",
                    media_id=item["id"],
                    download_url=url,
                    duration=duration,
                    width=width,
                    height=height,
                    caption=extract_caption(item, "pixabay"),
                )
            )
            if len(pool) >= pool_size:
                return pool

        if not has_next_page:
            break

    return pool


# ---------------------------------------------------------------------------
# Поиск с fallback и скачивание (async)
# ---------------------------------------------------------------------------


async def collect_candidates_with_fallback(
    session: aiohttp.ClientSession,
    query: str,
    pexels_api_key: str,
    pixabay_api_key: str,
    registry: MediaRegistry,
    min_duration: float | None = None,
    pool_size: int = 1,
) -> list[MediaCandidate]:
    """Набирает пул кандидатов на Pexels, при недоборе дополняет Pixabay."""
    media_label = "фото" if MEDIA_TYPE == "photo" else "видео"
    print(f"  → Поиск на Pexels ({media_label}): {query!r}")

    pool = await collect_pexels_candidates(
        session, query, pexels_api_key, registry, min_duration, pool_size
    )
    if len(pool) >= pool_size:
        return pool

    if not pixabay_api_key:
        if not pool:
            print("  ⚠ PIXABAY_API_KEY не задан, fallback невозможен.")
        return pool

    print(
        f"  ↳ Pexels: {len(pool)} из {pool_size} кандидатов "
        f"→ дополняю с Pixabay: {query!r}"
    )
    pool.extend(
        await collect_pixabay_candidates(
            session,
            query,
            pixabay_api_key,
            registry,
            min_duration,
            pool_size - len(pool),
        )
    )
    return pool


async def claim_candidate(
    pool: list[MediaCandidate],
    registry: MediaRegistry,
    min_duration: float | None = None,
) -> MediaCandidate | None:
    """Захватывает первого доступного кандидата в порядке ранжирования.

    Порядок обхода нужен потому, что параллельный чанк мог занять верхний
    кандидат между сбором пула и захватом.
    """
    for candidate in pool:
        if await registry.try_claim(
            source=candidate.source,
            media_id=candidate.media_id,
            duration=candidate.duration,
            width=candidate.width,
            height=candidate.height,
            min_duration=min_duration,
        ):
            return candidate
    return None


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    filepath: Path,
) -> bool:
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            with filepath.open("wb") as file:
                async for chunk in response.content.iter_chunked(8192):
                    file.write(chunk)
        return True
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
        print(f"  ⚠ Ошибка скачивания: {error}")
        if filepath.exists():
            filepath.unlink()
        return False


def format_not_found_message(min_duration: float | None = None) -> str:
    parts = [f"ориентация: {ORIENTATION}"]
    if MEDIA_TYPE == "video":
        if min_duration is not None:
            parts.append(f"мин. длительность: {min_duration:.1f} с")
        else:
            parts.append(f"макс. {MAX_DURATION_SECONDS} с")
    parts.append("без дубликатов")
    return ", ".join(parts)


async def process_video_task(task, total, session, pexels_api_key, pixabay_api_key,
                             registry, semaphore, gemini_client=None, selection_log=None):
    stage = PROJECT.stage("video", task.index, {
        "prompt": task.prompt, "brief": asdict(task.visual_brief) if task.visual_brief else None,
        "audio_hash": file_hash(task.audio_path), "settings": stage_settings("visual"),
    }) if PROJECT else None
    if stage:
        cached = stage.cached()
        if cached:
            RUN_MEDIA[task.index] = str((stage.path / cached["video_file"]).resolve())
            log_info("Resume video", str(stage.path))
            return True
        stage.start()
        task = replace(task, video_path=build_video_path(task.index, stage.path),
                       raw_video_path=build_raw_video_path(task.index, stage.path))
    try:
        success = await _process_video_task_uncached(task, total, session, pexels_api_key,
                    pixabay_api_key, registry, semaphore, gemini_client, selection_log)
        if stage:
            if success:
                stage.finish({"video_file": task.video_path.name}, [task.video_path])
                RUN_MEDIA[task.index] = str(task.video_path.resolve())
            else:
                stage.fail("Video selection/download failed")
        return success
    except Exception:
        if stage:
            stage.fail("Video task raised an exception; rerun to retry")
        raise


async def _process_video_task_uncached(
    task: MediaTask,
    total: int,
    session: aiohttp.ClientSession,
    pexels_api_key: str,
    pixabay_api_key: str,
    registry: MediaRegistry,
    semaphore: asyncio.Semaphore,
    gemini_client: genai.Client | None = None,
    selection_log: MediaSelectionLog | None = None,
) -> bool:
    """Скачивает видео, обрезает под длительность аудио и сохраняет пару файлов."""
    async with semaphore:
        log_section(f"CHUNK {task.index}/{total}")
        log_info("Search query", task.prompt)
        if task.visual_brief is not None:
            log_info("Visual", task.visual_brief.visual)

        audio_duration = await resolve_audio_duration(task.audio_path)
        if audio_duration is None:
            print(
                f"  ✗ Пропуск: аудио {task.audio_path.name} отсутствует, "
                "пустое или без длительности."
            )
            return False

        log_info("Audio", f"{task.audio_path.name} ({audio_duration:.2f} s)")
        trim_duration = get_trim_duration(audio_duration)
        log_info("Trim target", f"{trim_duration:.2f} s (+{VIDEO_TRIM_PADDING} s padding)")

        use_ranking = (
            MEDIA_SEMANTIC_ENABLED
            and task.visual_brief is not None
            and gemini_client is not None
        )
        pool_size = MEDIA_CANDIDATE_POOL if use_ranking else 1

        pool = await collect_candidates_with_fallback(
            session,
            task.prompt,
            pexels_api_key,
            pixabay_api_key,
            registry,
            min_duration=trim_duration,
            pool_size=pool_size,
        )
        if not pool:
            print(f"  ✗ Видео не найдено ({format_not_found_message(trim_duration)}).")
            if selection_log is not None:
                selection_log.write(
                    chunk_index=task.index,
                    brief=task.visual_brief,
                    outcome=SelectionOutcome(
                        chosen=None,
                        ranker="none",
                        reason="кандидатов не найдено",
                    ),
                    chunk_text=task.chunk_text,
                )
            return False

        if use_ranking:
            log_info("Candidates", str(len(pool)))
            outcome = await rank_candidates(task.visual_brief, pool, gemini_client)
        else:
            outcome = SelectionOutcome(
                chosen=pool[0], ranker="none", pool=pool, reason="ранжирование отключено"
            )

        found = await claim_candidate(outcome.pool, registry, trim_duration)
        if found is None:
            print("  ✗ Все кандидаты заняты параллельными сценами.")
            outcome.chosen = None
            outcome.reason = "все кандидаты заняты"
            if selection_log is not None:
                selection_log.write(
                    chunk_index=task.index,
                    brief=task.visual_brief,
                    outcome=outcome,
                    chunk_text=task.chunk_text,
                )
            return False

        outcome.chosen = found
        source_label = found.source.capitalize()
        log_info("Source", source_label)
        log_info("Media ID", str(found.media_id))
        log_info("Caption", found.caption or "—")
        log_info("Match", f"{outcome.ranker}, {outcome.reason}")
        if found.duration is not None:
            log_info("Stock duration", f"{found.duration} s")

        print(f"  ↓ Скачивание сырого видео → {task.raw_video_path.name}")
        if not await download_file(session, found.download_url, task.raw_video_path):
            await registry.release(found.source, found.media_id)
            return False

        print(f"  ↓ FFmpeg: обрезка по {task.audio_path.name} → {task.video_path.name}")
        if not await trim_video_to_duration(
            task.raw_video_path,
            task.video_path,
            task.audio_path,
        ):
            await registry.release(found.source, found.media_id)
            task.raw_video_path.unlink(missing_ok=True)
            return False

        task.raw_video_path.unlink(missing_ok=True)

        await registry.commit(
            found.source,
            found.media_id,
            query=task.prompt,
            visual=task.visual_brief.visual if task.visual_brief else None,
            caption=found.caption or None,
            score=found.score,
            ranker=outcome.ranker,
        )
        if selection_log is not None:
            selection_log.write(
                chunk_index=task.index,
                brief=task.visual_brief,
                outcome=outcome,
                chunk_text=task.chunk_text,
            )

        print(
            f"  ✓ Готово ({source_label}): {task.audio_path.name} + {task.video_path.name}"
        )
        return True


async def process_photo_prompt(index, total, prompt, session, pexels_api_key,
                               pixabay_api_key, output_path, registry, semaphore):
    stage = PROJECT.stage("photo", index, {"prompt": prompt, "settings": stage_settings("visual")}) if PROJECT else None
    if stage:
        cached = stage.cached()
        if cached:
            RUN_MEDIA[index] = str((stage.path / cached["photo_file"]).resolve())
            log_info("Resume photo", str(stage.path))
            return True
        stage.start()
        output_path = build_photo_path(index, prompt, stage.path)
    success = await _process_photo_prompt_uncached(index, total, prompt, session,
                    pexels_api_key, pixabay_api_key, output_path, registry, semaphore)
    if stage:
        if success:
            stage.finish({"photo_file": output_path.name}, [output_path])
            RUN_MEDIA[index] = str(output_path.resolve())
        else:
            stage.fail("Photo selection/download failed")
    return success


async def _process_photo_prompt_uncached(
    index: int,
    total: int,
    prompt: str,
    session: aiohttp.ClientSession,
    pexels_api_key: str,
    pixabay_api_key: str,
    output_path: Path,
    registry: MediaRegistry,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Скачивает фото без TTS и FFmpeg."""
    async with semaphore:
        log_section(f"PROMPT {index}/{total}")
        log_info("Query", prompt)

        pool = await collect_candidates_with_fallback(
            session,
            prompt,
            pexels_api_key,
            pixabay_api_key,
            registry,
            pool_size=1,
        )
        found = await claim_candidate(pool, registry)
        if found is None:
            print(f"  ✗ Медиа не найдено ({format_not_found_message()}).")
            return False

        source_label = found.source.capitalize()
        log_info("Source", source_label)
        log_info("Media ID", str(found.media_id))
        print(f"  ↓ Скачивание → {output_path.name}")

        if not await download_file(session, found.download_url, output_path):
            await registry.release(found.source, found.media_id)
            return False

        await registry.commit(found.source, found.media_id, query=prompt)
        print(f"  ✓ Готово ({source_label}): {output_path.name}")
        return True


async def process_all_media(
    chunk_results: list[ChunkResult],
    pexels_api_key: str,
    pixabay_api_key: str,
    output_dir: Path,
    registry: MediaRegistry,
    gemini_client: genai.Client | None = None,
    selection_log: MediaSelectionLog | None = None,
) -> int:
    """Параллельно обрабатывает все задачи скачивания."""
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        if MEDIA_TYPE == "video":
            tasks_list = build_media_tasks(chunk_results, output_dir)
            if not tasks_list:
                return 0

            tasks = [
                process_video_task(
                    task,
                    len(tasks_list),
                    session,
                    pexels_api_key,
                    pixabay_api_key,
                    registry,
                    semaphore,
                    gemini_client,
                    selection_log,
                )
                for task in tasks_list
            ]
        else:
            photo_chunks = [
                chunk_result
                for chunk_result in chunk_results
                if chunk_result.search_prompt
            ]

            tasks = [
                process_photo_prompt(
                    chunk_result.chunk_index,
                    len(photo_chunks),
                    chunk_result.search_prompt,
                    session,
                    pexels_api_key,
                    pixabay_api_key,
                    build_photo_path(
                        chunk_result.chunk_index,
                        chunk_result.search_prompt,
                        output_dir,
                    ),
                    registry,
                    semaphore,
                )
                for chunk_result in photo_chunks
            ]

        results = await asyncio.gather(*tasks)

    return sum(1 for success in results if success)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main() -> int:
    global TTS_SEMAPHORE, FAILED_CHUNKS_LOG, STRESS_MISMATCH_LOG, MEDIA_SELECTION_LOG
    TTS_SEMAPHORE = asyncio.Semaphore(1)
    RUN_MEDIA.clear()
    try:
        load_settings()
    except ValueError as error:
        print(f"Ошибка настроек: {error}")
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    pexels_api_key = os.getenv("PEXELS_API_KEY")
    pixabay_api_key = os.getenv("PIXABAY_API_KEY", "")

    missing_keys = []
    if not gemini_api_key:
        missing_keys.append("GEMINI_API_KEY")
    if MEDIA_DOWNLOAD_ENABLED and not pexels_api_key:
        missing_keys.append("PEXELS_API_KEY")
    if missing_keys:
        print(f"Ошибка: не найдены ключи в .env: {', '.join(missing_keys)}")
        return 1

    try:
        script_text = read_script(SCRIPT_FILE)
    except FileNotFoundError as error:
        print(f"Ошибка: {error}")
        return 1

    if not script_text:
        print(f"Файл {SCRIPT_FILE} пуст.")
        return 0

    chunks = chunk_script(script_text)
    if not chunks:
        print("Не удалось разбить сценарий на части.")
        return 1

    output_dir = PROJECT.path if PROJECT else Path(DOWNLOADS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    if PROJECT:
        (output_dir / "scripts").mkdir(exist_ok=True)
        (output_dir / "scripts" / (fingerprint(script_text) + ".txt")).write_text(script_text, encoding="utf-8")
        FAILED_CHUNKS_LOG = str(output_dir / "failed_chunks.log")
        STRESS_MISMATCH_LOG = str(output_dir / "stress_mismatches.log")
        MEDIA_SELECTION_LOG = str(output_dir / "media_selection.log")

    log_section("НАСТРОЙКИ")
    log_info("Project", str(output_dir.resolve()))
    log_info("Media type", MEDIA_TYPE)
    log_info(
        "Media download",
        "on" if MEDIA_DOWNLOAD_ENABLED else "off (только аудио)",
    )
    log_info("Orientation", ORIENTATION)
    log_info("Global vibe", GLOBAL_VIBE or "(не задан)")
    if MEDIA_TYPE == "video":
        log_info("TTS engine", f"Chatterbox API ({CHATTERBOX_TTS_URL})")
        log_info("TTS read timeout", f"{TTS_SERVER_TIMEOUT} s")
        log_info("Script chunk max", f"{SCRIPT_CHUNK_MAX_CHARS} chars")
        log_info("Failed chunks log", FAILED_CHUNKS_LOG)
        log_info("TTS voice file", TTS_VOICE_FILE)
        log_info("TTS voice mode", TTS_VOICE_MODE)
        log_info("TTS language", TTS_LANGUAGE)
        log_info("TTS speed factor", str(TTS_SPEED_FACTOR))
        log_info("TTS temperature", str(TTS_TEMPERATURE))
        log_info("TTS exaggeration", str(TTS_EXAGGERATION))
        log_info("TTS cfg_weight", str(TTS_CFG_WEIGHT))
        log_info("TTS chunk size", str(TTS_CHUNK_SIZE))
        log_info("Stress verify", "on" if STRESS_VERIFY_ENABLED else "off")
        log_info(
            "Stress autocorrect",
            "on" if STRESS_AUTOCORRECT_ENABLED else "off",
        )
        log_info(
            "Stress fill unmarked",
            "on" if STRESS_FILL_UNMARKED else "off",
        )
        if STRESS_VERIFY_ENABLED or STRESS_AUTOCORRECT_ENABLED or STRESS_FILL_UNMARKED:
            log_info("Stress engine", STRESS_VERIFY_ENGINE)
            log_info("Stress log", STRESS_MISMATCH_LOG)
            log_info("Stress exception dict", STRESS_EXCEPTION_DICT)
        log_info(
            "Audio postprocess",
            "per chunk (VAD tail trim + fade-out до склейки; склейки нет — 1 чанк = 1 WAV)",
        )
        log_info("Pause cleanup", "on" if AUDIO_PAUSE_CLEAN else "off")
        log_info(
            "Stitch cleanup",
            (
                f"on ({AUDIO_STITCH_FADE_BEFORE_MS:.0f} ms before boundary)"
                if AUDIO_STITCH_CLEAN
                else "off"
            ),
        )
        log_info(
            "WebRTC VAD cleanup",
            (
                f"on (mode {AUDIO_VAD_AGGRESSIVENESS}, "
                f"speech guard {AUDIO_VAD_SPEECH_PAD_MS:.0f} ms)"
                if AUDIO_VAD_CLEAN
                else "off"
            ),
        )
        log_info(
            "Spectral noise cleanup",
            (
                f"on (min {AUDIO_SPECTRAL_MIN_REGION_MS:.0f} ms)"
                if AUDIO_SPECTRAL_CLEAN
                else "off"
            ),
        )
        log_info("Keep raw audio", "on (raw beside each audio result)" if AUDIO_KEEP_RAW else "off")
        if AUDIO_PAUSE_CLEAN:
            log_info(
                "Pause thresholds",
                f"enter {AUDIO_PAUSE_ENTER_DB} dB / exit {AUDIO_PAUSE_EXIT_DB} dB, "
                f"min {AUDIO_PAUSE_MIN_MS:.0f} ms, guard {AUDIO_PAUSE_GUARD_MS:.0f} ms, "
                f"bridge {AUDIO_PAUSE_BRIDGE_MAX_MS:.0f} ms @ {AUDIO_PAUSE_BRIDGE_PEAK_DB:.0f} dB",
            )

    log_section("ГЕНЕРАЦИЯ ПРОМПТОВ И ОЗВУЧКИ")
    log_info("Model", GEMINI_MODEL)
    log_info("Chunks", str(len(chunks)))
    for index, chunk in enumerate(chunks, start=1):
        preview = chunk if len(chunk) <= 60 else chunk[:57] + "..."
        log_info(f"Chunk {index} script", f"{len(chunk)} chars — {preview}")
    log_info("Concurrency", str(MAX_CONCURRENT_REQUESTS))

    if MEDIA_TYPE == "video":
        try:
            await check_ffmpeg_tools()
        except RuntimeError as error:
            print(f"Ошибка: {error}")
            return 1
        except Exception as error:
            print(f"Ошибка инициализации: {error}")
            return 1

    client = genai.Client(api_key=gemini_api_key)
    gemini_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    chunk_results = await process_all_chunks(
        chunks,
        client,
        output_dir,
        gemini_semaphore,
    )

    if MEDIA_DOWNLOAD_ENABLED:
        ready_chunks = [
            chunk_result
            for chunk_result in chunk_results
            if chunk_result.search_prompt
        ]

        save_prompts([chunk.search_prompt for chunk in ready_chunks], str(output_dir / "prompts.txt"))
        print(f"\n  ✓ Сохранено {len(ready_chunks)} промптов в {output_dir / 'prompts.txt'}")

        used_media_store = UsedMediaStore(USED_MEDIA_STORE)
        registry = MediaRegistry(store=used_media_store)
        selection_log = MediaSelectionLog(MEDIA_SELECTION_LOG, run_id)

        download_label = (
            "СКАЧИВАНИЕ ФОТО" if MEDIA_TYPE == "photo" else "СКАЧИВАНИЕ И СИНХРОНИЗАЦИЯ ВИДЕО"
        )
        log_section(download_label)
        log_info("Orientation", ORIENTATION)
        if MEDIA_TYPE == "video":
            log_info("Sync mode", "FFmpeg trim to audio duration")
        log_info("Output dir", str(output_dir.resolve()))
        log_info("Chunks ready", str(len(ready_chunks)))
        log_info("Concurrency", str(MAX_CONCURRENT_REQUESTS))
        log_info("Semantic match", "on" if MEDIA_SEMANTIC_ENABLED else "off")
        if MEDIA_SEMANTIC_ENABLED:
            log_info("Ranker", MEDIA_RANKER)
            log_info("Candidate pool", str(MEDIA_CANDIDATE_POOL))
        log_info(
            "Used media store",
            f"{USED_MEDIA_STORE} ({len(used_media_store)} клипов из прошлых прогонов)",
        )
        log_info("Selection log", MEDIA_SELECTION_LOG)

        success_count = await process_all_media(
            chunk_results,
            pexels_api_key,
            pixabay_api_key,
            output_dir,
            registry,
            client,
            selection_log,
        )
    else:
        ready_chunks = [
            chunk_result
            for chunk_result in chunk_results
            if chunk_result.audio_path is not None
            and is_valid_audio_file(chunk_result.audio_path)
        ]
        success_count = len(ready_chunks)

    log_section("ИТОГ")
    if not MEDIA_DOWNLOAD_ENABLED and MEDIA_TYPE == "video":
        print(f"  Готово аудио: {success_count} из {len(chunks)}")
    elif MEDIA_TYPE == "video":
        print(f"  Готово пар (audio + video): {success_count} из {len(chunks)}")
    else:
        print(f"  Скачано: {success_count} из {len(chunks)}")
    complete = success_count == len(chunks)
    atomic_json(output_dir / "run_summary.json", {
        "status": "completed" if complete else "partial", "expected": len(chunks),
        "completed": success_count, "script_hash": fingerprint(script_text),
        "chunks": [{"index": c.chunk_index, "text": c.chunk_text,
                    "audio": str(c.audio_path.resolve()) if c.audio_path else None,
                    "search_prompt": c.search_prompt,
                    "media": RUN_MEDIA.get(c.chunk_index)} for c in chunk_results],
    })
    return 0 if complete else 1


def main() -> int:
    global PROJECT, SCRIPT_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", default=SCRIPT_FILE)
    parser.add_argument("--project", help="Stable project name (letters, digits, _ and -)")
    parser.add_argument("--projects-dir", default="projects")
    args = parser.parse_args()
    SCRIPT_FILE = args.script
    try:
        script = read_script(SCRIPT_FILE)
        name = args.project or "video-" + fingerprint(script)[:12]
        PROJECT = Project(Path(args.projects_dir), name)
        with FileLock(str(PROJECT.path / ".run.lock"), timeout=0):
            return asyncio.run(async_main())
    except Timeout:
        print("Этот проект уже обрабатывается другим процессом.")
        return 1
    except (OSError, ValueError) as error:
        print(f"Ошибка проекта: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
