"""Смысловой подбор видеоряда: визуальное описание сцены, ранжирование кандидатов, реестр использованных клипов.

Стоки не отдают готовых описаний в удобном виде:
    * Pexels video — поля `title`/`description` отсутствуют, `tags` приходит пустым списком.
      Единственный человекочитаемый текст — slug в `url`
      (`.../video/a-dark-hallway-with-a-light-on-the-floor-19217895/`) и ключевые слова
      в имени файла превью (`aisle-alone-dark-darkness-19217895.jpeg`).
    * Pixabay video — отдаёт настоящие `tags` (строка через запятую) и `name`.

Поэтому подпись кандидата собирается из доступных полей (см. extract_caption), а уже она
сравнивается с визуальным описанием сцены — через LLM или через эмбеддинги.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

# Слова, которые не несут визуального смысла при лексическом сопоставлении.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "of", "on", "in", "at", "to", "with", "and", "or", "for",
        "from", "by", "into", "over", "under", "near", "is", "are", "was", "were",
        "be", "being", "been", "it", "its", "this", "that", "these", "those",
        "и", "в", "на", "с", "по", "за", "из", "у", "к", "о", "об", "для", "как",
        "что", "это", "то", "же", "не", "ни", "а", "но", "или",
    }
)

# Технический мусор из имён файлов превью и slug'ов стоков: смысла не несёт,
# но забивает подпись, которая уходит в промпт ранжирования.
_CAPTION_NOISE: frozenset[str] = frozenset(
    {
        "pexels", "pixabay", "photo", "photos", "image", "images",
        "video", "videos", "footage", "clip", "stock", "free",
        "jpeg", "jpg", "png", "compress", "tinysrgb", "auto", "crop",
    }
)

_TOKEN_RE = re.compile(r"[a-zа-яё]+", re.IGNORECASE)
_TRAILING_ID_RE = re.compile(r"[-_]\d+$")


# ---------------------------------------------------------------------------
# Модели данных
# ---------------------------------------------------------------------------


@dataclass
class VisualBrief:
    """Визуальное описание сцены: что должно быть в кадре."""

    visual: str      # 7-12 слов на русском — то, что уходит в лог и в LLM-ранжирование
    visual_en: str   # тот же образ по-английски — для сравнения с англоязычными подписями
    query: str       # 2-4 слова для поискового API стока

    @property
    def match_text(self) -> str:
        """Текст для сопоставления с подписями кандидатов."""
        return self.visual_en or self.visual


@dataclass
class MediaCandidate:
    """Кандидат из выдачи стока вместе с его текстовой подписью."""

    source: str
    media_id: int
    download_url: str
    duration: int | None
    width: int | None
    height: int | None
    caption: str
    score: float | None = None


@dataclass
class SelectionOutcome:
    """Результат выбора клипа под сцену — для лога."""

    chosen: MediaCandidate | None
    ranker: str
    pool: list[MediaCandidate] = field(default_factory=list)
    reason: str = ""
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# Извлечение подписей из ответов стоков
# ---------------------------------------------------------------------------


def _slug_to_text(slug: str) -> str:
    cleaned = _TRAILING_ID_RE.sub("", slug.strip("/"))
    return cleaned.replace("-", " ").replace("_", " ").strip()


def extract_pexels_caption(item: dict) -> str:
    """Собирает подпись Pexels-видео из slug в url, имени превью и (если есть) tags."""
    parts: list[str] = []

    url = item.get("url") or ""
    if url:
        segments = [seg for seg in urlparse(url).path.split("/") if seg]
        if segments:
            parts.append(_slug_to_text(segments[-1]))

    image = item.get("image") or ""
    if image:
        stem = Path(urlparse(image).path).stem
        parts.append(_slug_to_text(stem))

    tags = item.get("tags")
    if isinstance(tags, list):
        parts.extend(str(tag) for tag in tags if str(tag).strip())
    elif isinstance(tags, str) and tags.strip():
        parts.append(tags)

    return _dedupe_caption(parts)


def extract_pixabay_caption(item: dict) -> str:
    """Собирает подпись Pixabay-видео из tags и name."""
    parts: list[str] = []

    tags = item.get("tags")
    if isinstance(tags, str) and tags.strip():
        parts.append(tags.replace(",", " "))
    elif isinstance(tags, list):
        parts.extend(str(tag) for tag in tags if str(tag).strip())

    name = item.get("name")
    if isinstance(name, str) and name.strip():
        parts.append(name.replace(",", " "))

    return _dedupe_caption(parts)


def extract_caption(item: dict, source: str) -> str:
    if source == "pixabay":
        return extract_pixabay_caption(item)
    return extract_pexels_caption(item)


def _dedupe_caption(parts: list[str]) -> str:
    """Склеивает части, убирая повторы и технический мусор, с сохранением порядка слов."""
    seen: set[str] = set()
    words: list[str] = []
    for part in parts:
        for token in _TOKEN_RE.findall(part):
            lower = token.lower()
            if lower in seen or lower in _CAPTION_NOISE:
                continue
            seen.add(lower)
            words.append(lower)
    return " ".join(words)


# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------


def build_visual_brief_prompt() -> str:
    """Промпт визуального описания сцены.

    Первый абзац — постановка задачи; JSON-контракт нужен потому, что поисковому API
    стока нельзя отдать предложение из 7-12 слов (выдача становится пустой),
    поэтому короткий запрос выводится из этого же описания, а не из сырого текста сцены.
    """
    return (
        "Опиши одним предложением (7-12 слов), что именно должно быть в кадре для этой "
        "сцены, чтобы зритель понял смысл без слов. Не пересказывай текст — опиши "
        "конкретный визуальный образ: место, объект, действие. Если сцена абстрактная — "
        "дай метафоричный, но конкретный визуал. Ответ — только описание.\n\n"
        "Верни строго JSON-объект без markdown-обёрток:\n"
        '{"visual": "<описание по правилу выше, на русском>", '
        '"visual_en": "<то же описание на английском>", '
        '"query": "<2-4 слова на английском для поиска по стоку видео>"}'
    )


def build_ranking_prompt(brief: VisualBrief, candidates: list[MediaCandidate]) -> str:
    """Промпт выбора кандидата по смыслу."""
    listing = "\n".join(
        f"{index}. {candidate.caption or '(без описания)'}"
        for index, candidate in enumerate(candidates, start=1)
    )
    return (
        f"Визуальное описание сцены: {brief.visual}\n"
        f"Кандидаты:\n{listing}\n\n"
        "Выбери номер клипа, который лучше всего передаёт этот визуал по смыслу, "
        "игнорируя эстетику. Ответ — только номер."
    )


def _strip_markdown_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_visual_brief(raw_text: str) -> VisualBrief | None:
    """Разбирает JSON-ответ модели в VisualBrief."""
    text = _strip_markdown_fence(raw_text)
    if not text:
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    visual = str(data.get("visual") or "").strip()
    visual_en = str(data.get("visual_en") or "").strip()
    query = str(data.get("query") or "").strip()

    if not query:
        # Без запроса искать нечего; пробуем вытащить его из англоязычного описания.
        query = " ".join(meaningful_tokens(visual_en)[:3])
    if not visual and not visual_en:
        return None
    if not query:
        return None

    return VisualBrief(visual=visual or visual_en, visual_en=visual_en, query=query)


def parse_ranking_choice(raw_text: str, pool_size: int) -> int | None:
    """Возвращает 0-based индекс выбранного кандидата."""
    match = re.search(r"\d+", raw_text or "")
    if not match:
        return None
    number = int(match.group(0))
    if 1 <= number <= pool_size:
        return number - 1
    return None


# ---------------------------------------------------------------------------
# Лексическое сопоставление (без внешних зависимостей)
# ---------------------------------------------------------------------------


def meaningful_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if token.lower() not in _STOPWORDS and len(token) > 2
    ]


def lexical_score(brief_text: str, caption: str) -> float:
    """Доля значимых слов описания, встретившихся в подписи кандидата (0..1)."""
    brief_tokens = set(meaningful_tokens(brief_text))
    if not brief_tokens:
        return 0.0
    caption_tokens = set(meaningful_tokens(caption))
    if not caption_tokens:
        return 0.0

    overlap = brief_tokens & caption_tokens
    coverage = len(overlap) / len(brief_tokens)
    # Небольшая поправка на «шумные» подписи с десятками ключевых слов.
    precision = len(overlap) / len(caption_tokens)
    return round(0.8 * coverage + 0.2 * precision, 4)


def rank_lexically(brief: VisualBrief, candidates: list[MediaCandidate]) -> list[MediaCandidate]:
    for candidate in candidates:
        candidate.score = lexical_score(brief.match_text, candidate.caption)
    return sorted(candidates, key=lambda item: item.score or 0.0, reverse=True)


# ---------------------------------------------------------------------------
# Ранжирование эмбеддингами (опционально, sentence-transformers)
# ---------------------------------------------------------------------------


class EmbeddingRanker:
    """Косинусная близость описания сцены и подписей кандидатов.

    Модель загружается лениво и только один раз. sentence-transformers —
    необязательная зависимость: при её отсутствии ранжирование откатывается
    на лексическое, а пайплайн не падает.
    """

    _instance: "EmbeddingRanker | None" = None
    _lock = threading.Lock()

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    @classmethod
    def get(cls, model_name: str) -> "EmbeddingRanker":
        with cls._lock:
            if cls._instance is None or cls._instance.model_name != model_name:
                cls._instance = cls(model_name)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def rank(
        self, brief: VisualBrief, candidates: list[MediaCandidate]
    ) -> list[MediaCandidate]:
        captions = [candidate.caption or "" for candidate in candidates]
        vectors = self._model.encode(
            [brief.match_text] + captions,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        brief_vector = vectors[0]
        for candidate, vector in zip(candidates, vectors[1:]):
            candidate.score = round(float(brief_vector @ vector), 4)
        return sorted(candidates, key=lambda item: item.score or 0.0, reverse=True)


# ---------------------------------------------------------------------------
# Персистентный реестр использованных клипов
# ---------------------------------------------------------------------------


class UsedMediaStore:
    """Использованные media_id на диске, чтобы клипы не повторялись между прогонами.

    Формат: JSON с ключами вида "pexels:video:12345". Запись сбрасывается на диск
    сразу после каждого подтверждённого клипа (атомарно, через os.replace),
    поэтому падение в середине прогона не теряет историю.
    """

    VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        entries = data.get("entries") if isinstance(data, dict) else None
        if isinstance(entries, dict):
            self._entries = entries

    def keys(self) -> set[str]:
        with self._lock:
            return set(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def remember(self, key: str, **meta) -> None:
        """Добавляет клип и сразу сохраняет файл."""
        with self._lock:
            if key in self._entries:
                self._entries[key].setdefault("reused", 0)
                self._entries[key]["reused"] += 1
            else:
                self._entries[key] = {
                    "first_used": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    **{name: value for name, value in meta.items() if value is not None},
                }
            self._flush_locked()

    def _flush_locked(self) -> None:
        payload = {"version": self.VERSION, "entries": self._entries}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.path.parent),
            prefix=self.path.name + ".",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            os.replace(handle.name, self.path)
        except OSError:
            Path(handle.name).unlink(missing_ok=True)
            raise


# ---------------------------------------------------------------------------
# Лог подбора видеоряда
# ---------------------------------------------------------------------------


class MediaSelectionLog:
    """JSONL-лог подбора: описание сцены, кандидаты, выбор и его score."""

    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self._lock = threading.Lock()

    def write(
        self,
        *,
        chunk_index: int,
        brief: VisualBrief | None,
        outcome: SelectionOutcome,
        chunk_text: str = "",
    ) -> None:
        chosen = outcome.chosen
        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_id": self.run_id,
            "chunk": chunk_index,
            "scene_text": " ".join(chunk_text.split())[:200],
            "visual": brief.visual if brief else None,
            "visual_en": brief.visual_en if brief else None,
            "query": brief.query if brief else None,
            "ranker": outcome.ranker,
            "fallback_used": outcome.fallback_used,
            "pool_size": len(outcome.pool),
            "reason": outcome.reason,
            "chosen": (
                {
                    "source": chosen.source,
                    "media_id": chosen.media_id,
                    "score": chosen.score,
                    "duration": chosen.duration,
                    "caption": chosen.caption,
                }
                if chosen
                else None
            ),
            "candidates": [
                {
                    "source": candidate.source,
                    "media_id": candidate.media_id,
                    "score": candidate.score,
                    "caption": candidate.caption,
                }
                for candidate in outcome.pool
            ],
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
