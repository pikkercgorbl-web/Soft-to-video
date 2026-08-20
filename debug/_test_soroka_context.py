#!/usr/bin/env python3
"""Тест контекстного правила для «сорока» (age_context vs bird_context)."""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from stress_verify import process_gemini_stress, set_u0301_stress_on_token

LOG = Path("_test_soroka_context.log")


def run_case(name: str, text_wrong: str) -> None:
    LOG.unlink(missing_ok=True)
    result = process_gemini_stress(
        text_wrong,
        chunk_index=1,
        log_path=LOG,
        verify_enabled=True,
        autocorrect_enabled=True,
    )
    print(f"=== {name} ===")
    print("Вход (ошибка Gemini):", text_wrong)
    print("Выход:", result.text)
    print(f"corrected={result.corrected_count} mismatch={result.mismatch_count}")
    if LOG.exists():
        print("Лог:", LOG.read_text(encoding="utf-8").strip())
    print()


# Ошибка: ударение на последней «а» (как у птицы), в контексте возраста
age_wrong = (
    f"Мужчина лет {set_u0301_stress_on_token('сорока', 2)} выходит из подъезда"
)
# Ошибка: ударение как у числительного, но контекст — птица
bird_wrong = (
    f"{set_u0301_stress_on_token('Сорока', 2)} прилетела на подоконник"
)

run_case("мужчина лет сорока → автокоррекция (age_context)", age_wrong)
run_case("сорока прилетела → Gemini (bird_context)", bird_wrong)
