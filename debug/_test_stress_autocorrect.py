#!/usr/bin/env python3
"""Тест автокоррекции ударений на примере из сценария."""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from stress_verify import (
    process_gemini_stress,
    parse_u0301_stress,
    set_u0301_stress_on_token,
    StressRnnEngine,
    parse_plus_stress,
)

AC = "\u0301"
_engine = StressRnnEngine.get()


def _wrong_stress(word: str, wrong_idx: int) -> str:
    return set_u0301_stress_on_token(word, wrong_idx)


def _dict_stress_idx(word: str) -> int:
    _, idx = parse_plus_stress(_engine.stress_word(word))
    assert idx is not None
    return idx


# Имитация ошибочной разметки Gemini (U+0301 после неверной гласной)
GEMINI_WRONG = (
    f"{_wrong_stress('мужчина', 0)} лет {_wrong_stress('сорока', 2)} выходит из подъезда на "
    f"улице Профсоюзная... Инженер-конструктор второй "
    f"{_wrong_stress('категории', 3)}"
)

LOG = Path("_test_stress_autocorrect.log")
LOG.unlink(missing_ok=True)

result = process_gemini_stress(
    GEMINI_WRONG,
    chunk_index=1,
    log_path=LOG,
    verify_enabled=True,
    autocorrect_enabled=True,
)

print("=== Исходный текст (ошибки Gemini) ===")
print(GEMINI_WRONG)
print()
print("=== После автокоррекции ===")
print(result.text)
print()
print(f"Исправлено: {result.corrected_count}, расхождений (без подмены): {result.mismatch_count}")
print()
print("=== Лог ===")
print(LOG.read_text(encoding="utf-8"))

# Проверка трёх целевых слов
assert result.corrected_count == 3, f"ожидалось 3 auto_corrected, получено {result.corrected_count}"
print("=== Индексы ударных гласных после коррекции ===")
for word in ("мужчина", "сорока", "категории"):
    expected = _dict_stress_idx(word)
    found = False
    for token in result.text.replace("...", " ").replace(",", " ").split():
        if AC not in token:
            continue
        plain, idx = parse_u0301_stress(token)
        if plain == word:
            ok = idx == expected
            print(f"  {word}: vowel_idx={idx} (ожид. {expected}) {'OK' if ok else 'FAIL'}  ({token})")
            found = True
    if not found:
        print(f"  {word}: НЕ НАЙДЕНО")
