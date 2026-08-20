#!/usr/bin/env python3
"""Тест полной доразметки + бенчмарк производительности."""

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from stress_verify import (
    COMBINING_ACUTE,
    MARKED_WORD_RE,
    StressRnnEngine,
    count_vowels_in_plain,
    parse_u0301_stress,
    process_gemini_stress,
    set_u0301_stress_on_token,
    token_plain_form,
)

AC = COMBINING_ACUTE
TEST_SENTENCE = (
    "Мужчина лет сорока выходит из подъезда на улице Профсоюзная — или, "
    "как его знают здесь, Роберт Александрович Малинин. "
    "Инженер-конструктор второй категории."
)
LOG = Path("_test_full_stress.log")


def words_with_stress(text: str) -> list[tuple[str, str, int | None]]:
    out = []
    for m in MARKED_WORD_RE.finditer(text):
        t = m.group(0)
        if count_vowels_in_plain(token_plain_form(t)) < 2:
            continue
        plain, idx = parse_u0301_stress(t)
        out.append((plain, t, idx))
    return out


def benchmark_marked_only(text: str, engine_name: str = "stressrnn") -> float:
    """Старая логика: только слова с U+0301 от Gemini."""
    from stress_verify import get_engine, parse_dict_stress

    engine = get_engine(engine_name)
    t0 = time.perf_counter()
    for match in MARKED_WORD_RE.finditer(text):
        token = match.group(0)
        if AC not in token:
            continue
        plain, idx = parse_u0301_stress(token)
        if not plain or idx is None:
            continue
        dm = engine.stress_word(plain)
        parse_dict_stress(dm, engine_name)
    return time.perf_counter() - t0


def benchmark_full(text: str, **kwargs) -> float:
    StressRnnEngine.reset()
    t0 = time.perf_counter()
    process_gemini_stress(
        text,
        chunk_index=99,
        log_path=Path("_bench_null.log"),
        verify_enabled=True,
        autocorrect_enabled=True,
        **kwargs,
    )
    return time.perf_counter() - t0


def main() -> None:
    # Имитация Gemini: почти без ударений + одна ошибка на «категории»
    gemini_text = TEST_SENTENCE.replace(
        "категории",
        set_u0301_stress_on_token("категории", 3),  # неверное ударение
    )

    LOG.unlink(missing_ok=True)
    Path("_bench_null.log").unlink(missing_ok=True)

    StressRnnEngine.reset()
    result = process_gemini_stress(
        gemini_text,
        chunk_index=1,
        log_path=LOG,
        verify_enabled=True,
        autocorrect_enabled=True,
    )

    print("=" * 70)
    print("ТЕСТ: финальный текст для Chatterbox")
    print("=" * 70)
    print("Gemini (вход):", gemini_text)
    print()
    print("Chatterbox (выход):", result.text)
    print()
    print(
        f"filled={result.filled_count} corrected={result.corrected_count} "
        f"mismatch={result.mismatch_count} missed_homograph={result.missed_homograph_count}"
    )

    print("\n--- Слова с 2+ гласными после обработки ---")
    for plain, token, idx in words_with_stress(result.text):
        status = "OK" if idx is not None else "NO STRESS"
        print(f"  {plain:<22} idx={idx}  {status}  {token}")

    unstressed = [
        plain
        for plain, token, idx in words_with_stress(result.text)
        if idx is None
    ]
    print(f"\nБез ударения: {unstressed or '(нет)'}")

    print("\n--- Лог ---")
    print(LOG.read_text(encoding="utf-8"))

    # Perf
    print("=" * 70)
    print("БЕНЧМАРК (3 прогона, среднее)")
    print("=" * 70)
    StressRnnEngine.reset()
    old_times = [benchmark_marked_only(gemini_text) for _ in range(3)]
    new_times = [benchmark_full(gemini_text) for _ in range(3)]
    old_avg = sum(old_times) / len(old_times)
    new_avg = sum(new_times) / len(new_times)
    n_words = sum(
        1
        for m in MARKED_WORD_RE.finditer(gemini_text)
        if count_vowels_in_plain(token_plain_form(m.group(0))) >= 2
    )
    n_marked = sum(1 for m in MARKED_WORD_RE.finditer(gemini_text) if AC in m.group(0))
    print(f"Слов 2+ гласных в чанке: {n_words}  (с U+0301 от Gemini: {n_marked})")
    print(f"Старая логика (только marked): {old_avg*1000:.1f} ms")
    print(f"Новая логика (все слова):      {new_avg*1000:.1f} ms")
    print(f"Замедление:                    ×{new_avg/old_avg:.1f}  (+{(new_avg-old_avg)*1000:.0f} ms)")


if __name__ == "__main__":
    main()
