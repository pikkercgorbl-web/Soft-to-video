#!/usr/bin/env python3
"""Интерактивное подтверждение расхождений из stress_mismatches.log → exception_dictionary.txt."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from stress_verify import (
    STRESS_HOMOGRAPHS,
    append_exception_dictionary,
    configure_exception_dictionary,
    is_homograph,
    load_exception_dictionary_entries,
    parse_u0301_stress,
    plus_marked_to_u0301,
    set_u0301_stress_on_token,
    StressRnnEngine,
)

LOG_LINE_RE = re.compile(
    r"event=(?P<event>\w+) \| .*? plain=(?P<plain>[^ |]+) \| "
    r"gemini=(?P<gemini>[^ |]+) \| dict=(?P<dict>[^ |]+) \| "
    r"gemini_vowel_idx=(?P<gidx>-?\d+|None) \| dict_vowel_idx=(?P<didx>-?\d+|None)"
)


@dataclass
class LogMismatch:
    plain: str
    gemini_marked: str
    dict_marked: str
    gemini_idx: int | None
    dict_idx: int | None
    event: str


def parse_log(path: Path) -> list[LogMismatch]:
    if not path.is_file():
        return []
    items: list[LogMismatch] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LOG_LINE_RE.search(line)
        if not match:
            continue
        if match.group("event") not in ("mismatch", "auto_corrected"):
            continue
        plain = match.group("plain")
        if plain in seen:
            continue
        seen.add(plain)

        def _idx(raw: str) -> int | None:
            if raw == "None":
                return None
            return int(raw)

        items.append(
            LogMismatch(
                plain=plain,
                gemini_marked=match.group("gemini"),
                dict_marked=match.group("dict"),
                gemini_idx=_idx(match.group("gidx")),
                dict_idx=_idx(match.group("didx")),
                event=match.group("event"),
            )
        )
    return items


def dict_marked_to_u0301(dict_marked: str, plain: str, dict_idx: int | None) -> str:
    if "\u0301" in dict_marked:
        return dict_marked
    if "+" in dict_marked or "'" in dict_marked:
        return plus_marked_to_u0301(dict_marked.replace("'", "+"))
    if dict_idx is not None:
        return set_u0301_stress_on_token(plain, dict_idx)
    return plain


def already_in_dict(path: Path, plain: str) -> bool:
    for entry in load_exception_dictionary_entries(path):
        entry_plain, _ = parse_u0301_stress(entry)
        if entry_plain == plain:
            return True
    return False


def add_confirmed_exception(dict_path: Path, plain: str, vowel_idx: int) -> bool:
    """Добавляет подтверждённое исключение (U+0301) тем же путём, что интерактивный режим."""
    if is_homograph(plain) or plain in STRESS_HOMOGRAPHS:
        print(f"— {plain}: омограф, в словарь исключений не добавляем")
        return False
    if already_in_dict(dict_path, plain):
        print(f"— {plain}: уже в {dict_path.name}")
        return False
    u0301_word = set_u0301_stress_on_token(plain, vowel_idx)
    if append_exception_dictionary(dict_path, u0301_word):
        print(f"✓ Добавлено в {dict_path.name}: {u0301_word}")
        StressRnnEngine.reset()
        configure_exception_dictionary(dict_path)
        return True
    print(f"⚠ Не добавлено: {plain}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Подтверждение ударений из лога расхождений")
    parser.add_argument(
        "--log",
        default="stress_mismatches.log",
        help="Файл лога (по умолчанию stress_mismatches.log)",
    )
    parser.add_argument(
        "--dict",
        default="exception_dictionary.txt",
        help="Словарь исключений U+0301 (по умолчанию exception_dictionary.txt)",
    )
    parser.add_argument(
        "--add",
        nargs=2,
        metavar=("WORD", "VOWEL_IDX"),
        action="append",
        help="Подтверждённое исключение: слово + индекс ударной гласной (0-based)",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    dict_path = Path(args.dict)
    configure_exception_dictionary(dict_path)

    if args.add:
        added = 0
        for plain, idx_raw in args.add:
            try:
                vowel_idx = int(idx_raw)
            except ValueError:
                print(f"⚠ Неверный vowel_idx для {plain!r}: {idx_raw!r}")
                continue
            if add_confirmed_exception(dict_path, plain.strip().lower(), vowel_idx):
                added += 1
        print(f"\nГотово. Добавлено записей: {added}")
        if added:
            print("Перезапустите пайплайн — StressRNN подхватит словарь при load_settings().")
        return 0

    mismatches = parse_log(log_path)
    if not mismatches:
        print(f"В {log_path} нет строк event=mismatch/auto_corrected для разбора.")
        return 0

    print(f"Найдено {len(mismatches)} уникальных слов в {log_path}")
    print("Команды: [1] Gemini  [2] StressRNN  [s] пропустить  [q] выход\n")

    added = 0
    for item in mismatches:
        if is_homograph(item.plain) or item.plain in STRESS_HOMOGRAPHS:
            print(f"— {item.plain}: омограф, пропуск (решение за Gemini)\n")
            continue
        if already_in_dict(dict_path, item.plain):
            print(f"— {item.plain}: уже в {dict_path.name}\n")
            continue

        print(f"Слово: {item.plain}  (из лога: {item.event})")
        print(f"  [1] Gemini:    {item.gemini_marked}  (vowel_idx={item.gemini_idx})")
        print(f"  [2] StressRNN: {item.dict_marked}  (vowel_idx={item.dict_idx})")

        while True:
            choice = input("Выбор [1/2/s/q]: ").strip().lower()
            if choice in ("1", "2", "s", "q"):
                break
            print("  Введите 1, 2, s или q")

        if choice == "q":
            break
        if choice == "s":
            print()
            continue

        if choice == "1":
            if "\u0301" in item.gemini_marked:
                u0301_word = item.gemini_marked
            elif item.gemini_idx is not None:
                u0301_word = set_u0301_stress_on_token(item.plain, item.gemini_idx)
            else:
                print("  ⚠ Не удалось восстановить U+0301 из Gemini, пропуск\n")
                continue
        else:
            u0301_word = dict_marked_to_u0301(item.dict_marked, item.plain, item.dict_idx)

        if append_exception_dictionary(dict_path, u0301_word):
            added += 1
            print(f"  ✓ Добавлено в {dict_path}: {u0301_word}\n")
            StressRnnEngine.reset()
            configure_exception_dictionary(dict_path)
        else:
            print(f"  ⚠ Не добавлено (дубликат или ошибка)\n")

    print(f"Готово. Добавлено записей: {added}")
    if added:
        print("Перезапустите пайплайн — StressRNN подхватит словарь при load_settings().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
