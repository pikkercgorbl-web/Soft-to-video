"""Санитайзер текста от Gemini перед подачей в Chatterbox TTS.

Gemini регулярно ломает три вещи, каждая из которых даёт слышимый брак:

1. Ставит ударение precomposed-символом («Москвá» = U+00E1) вместо «а» + U+0301.
   Латиница не входит в русский словарь токенизатора → шум/галлюцинация.
2. Помечает ударением односложные слова («ле́т»). Само по себе безвредно, но
   U+0301 разрывает слово для контекстных правил в stress_verify.
3. Подменяет е/ё по своему усмотрению («почти все» → «почти всё»), меняя смысл.

Санитайзер выполняется до проверки ударений, поэтому stress_verify видит уже
починенный текст.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

COMBINING_ACUTE = "\u0301"

CYRILLIC_VOWELS = frozenset("аеёиоуыэюя")

# Latin precomposed с акутом → кириллическая гласная + U+0301.
_LATIN_ACCENTED = {
    "\u00e1": "а",  # á
    "\u00e9": "е",  # é
    "\u00ed": "и",  # í
    "\u00f3": "о",  # ó
    "\u00fa": "у",  # ú
    "\u00fd": "у",  # ý
    "\u00c1": "А",
    "\u00c9": "Е",
    "\u00cd": "И",
    "\u00d3": "О",
    "\u00da": "У",
    "\u00dd": "У",
}

# Голая латиница перед U+0301 — тот же дефект, только в разложенной форме.
_LATIN_BARE = {
    "a": "а", "e": "е", "i": "и", "o": "о", "u": "у", "y": "у",
    "A": "А", "E": "Е", "I": "И", "O": "О", "U": "У", "Y": "У",
}

_WORD_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z\u0301-]+")
_DIGIT_RE = re.compile(r"\d")


@dataclass
class SanitizeResult:
    text: str
    latin_fixed: int = 0
    mono_stripped: int = 0
    yo_restored: int = 0
    digits_left: int = 0


def _fix_latin_stress(text: str) -> tuple[str, int]:
    """Чинит «Москвá» и «Москвa\u0301» → «Москва\u0301»."""
    result: list[str] = []
    fixed = 0
    for i, ch in enumerate(text):
        if ch in _LATIN_ACCENTED:
            result.append(_LATIN_ACCENTED[ch])
            result.append(COMBINING_ACUTE)
            fixed += 1
            continue
        if ch in _LATIN_BARE and text[i + 1 : i + 2] == COMBINING_ACUTE:
            result.append(_LATIN_BARE[ch])
            fixed += 1
            continue
        result.append(ch)
    return "".join(result), fixed


def _letters(word: str) -> list[str]:
    return [ch for ch in word if ch.isalpha()]


def _key(word: str) -> str:
    """Форма для сопоставления: буквы, lower, ё сведено к е."""
    return "".join(_letters(word)).lower().replace("ё", "е")


def _vowel_count(word: str) -> int:
    return sum(1 for ch in _letters(word) if ch.lower() in CYRILLIC_VOWELS)


def _strip_mono_stress(text: str) -> tuple[str, int]:
    """Снимает U+0301 с односложных слов: там ударение бессмысленно."""
    stripped = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal stripped
        word = match.group()
        if COMBINING_ACUTE not in word or _vowel_count(word) > 1:
            return word
        stripped += 1
        return word.replace(COMBINING_ACUTE, "")

    return _WORD_RE.sub(replace, text), stripped


def _apply_source_yo(processed_word: str, source_word: str) -> str:
    """Переносит е/ё из исходного слова в обработанное, сохраняя регистр и U+0301."""
    source_letters = _letters(source_word)
    result: list[str] = []
    pos = 0
    for ch in processed_word:
        if not ch.isalpha():
            result.append(ch)
            continue
        if pos >= len(source_letters):
            result.append(ch)
            continue
        source_ch = source_letters[pos]
        pos += 1
        if ch.lower() in ("е", "ё") and source_ch.lower() in ("е", "ё"):
            if ch.lower() != source_ch.lower():
                ch = source_ch.lower() if ch.islower() else source_ch.upper()
        result.append(ch)
    return "".join(result)


def _restore_yo(text: str, source_text: str) -> tuple[str, int]:
    """Возвращает е/ё исходника словам, которые Gemini не переписывала.

    Выравнивание идёт по последовательности слов, поэтому «все» и «всё» в одном
    фрагменте не путаются: каждое слово получает свою форму по позиции.
    """
    source_words = _WORD_RE.findall(source_text)
    processed_spans = [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(text)]
    if not source_words or not processed_spans:
        return text, 0

    source_keys = [_key(word) for word in source_words]
    processed_keys = [_key(word) for _, _, word in processed_spans]

    matcher = SequenceMatcher(None, source_keys, processed_keys, autojunk=False)
    replacements: list[tuple[int, int, str]] = []
    restored = 0

    for tag, i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(j2 - j1):
            source_word = source_words[i1 + offset]
            start, end, processed_word = processed_spans[j1 + offset]
            if "ё" not in source_word.lower() and "ё" not in processed_word.lower():
                continue
            rebuilt = _apply_source_yo(processed_word, source_word)
            if rebuilt != processed_word:
                replacements.append((start, end, rebuilt))
                restored += 1

    for start, end, rebuilt in reversed(replacements):
        text = text[:start] + rebuilt + text[end:]
    return text, restored


def sanitize_tts_text(processed_text: str, source_text: str) -> SanitizeResult:
    """Чинит вывод Gemini перед TTS: латиница с акутом, лишние ударения, е/ё."""
    text, latin_fixed = _fix_latin_stress(processed_text)
    text, mono_stripped = _strip_mono_stress(text)
    text, yo_restored = _restore_yo(text, source_text)
    return SanitizeResult(
        text=text,
        latin_fixed=latin_fixed,
        mono_stripped=mono_stripped,
        yo_restored=yo_restored,
        digits_left=len(_DIGIT_RE.findall(text)),
    )
