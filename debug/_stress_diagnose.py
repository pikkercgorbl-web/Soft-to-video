#!/usr/bin/env python3
"""Диагностика StressRNN vs пайплайн stress_verify (задача 1)."""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from stressrnn import StressRNN
from stress_verify import (
    COMBINING_ACUTE,
    MARKED_WORD_RE,
    StressRnnEngine,
    configure_exception_dictionary,
    parse_plus_stress,
    parse_u0301_stress,
    process_gemini_stress,
    set_u0301_stress_on_token,
    strip_stress_marks,
)

WORDS = ["мужчина", "сорока", "категории"]
AC = COMBINING_ACUTE


def main() -> None:
    configure_exception_dictionary("exception_dictionary.txt")

    print("=" * 60)
    print("1. ПРЯМОЙ StressRNN")
    print("=" * 60)
    direct = StressRNN()
    for word in WORDS:
        raw = direct.put_stress(word, stress_symbol="+")
        _, idx = parse_plus_stress(raw)
        print(f"  {word!r} -> {raw!r}  vowel_idx={idx}")

    print("\n" + "=" * 60)
    print("2. ВХОД В stress_word() ИЗ ПАЙПЛАЙНА")
    print("=" * 60)
    engine = StressRnnEngine.get()
    text = (
        f"{set_u0301_stress_on_token('мужчина', 1)} лет "
        f"{set_u0301_stress_on_token('сорока', 1)} ... второй "
        f"{set_u0301_stress_on_token('категории', 2)}"
    )
    for match in MARKED_WORD_RE.finditer(text):
        token = match.group(0)
        if AC not in token:
            continue
        plain, gidx = parse_u0301_stress(token)
        stripped = strip_stress_marks(token)
        dm = engine.stress_word(plain)
        _, didx = parse_plus_stress(dm)
        print(f"  token={token!r}")
        print(f"    plain={plain!r}  U+0301_остатки={AC in plain}  strip==plain={stripped == plain}")
        print(f"    gemini_idx={gidx}  dict={dm!r}  dict_idx={didx}  match={gidx == didx}")

    print("\n" + "=" * 60)
    print("3. ЗАлизняк (bundled exception_dictionary.txt)")
    print("=" * 60)
    import stressrnn.constants as c

    zpath = Path(c.F_NAME_EXCEPTION_DICT)
    ztext = zpath.read_text(encoding="utf-8")
    for word in WORDS:
        exact = [
            ln.strip()
            for ln in ztext.splitlines()
            if ln.strip().replace("+", "").replace("'", "") == word
        ]
        print(f"  {word}: {exact if exact else '(нет точной формы — ответ RNN)'}")

    print("\n" + "=" * 60)
    print("4. АВТОКОРРЕКЦИЯ (ошибочный Gemini)")
    print("=" * 60)
    wrong = (
        f"{set_u0301_stress_on_token('мужчина', 0)} лет "
        f"{set_u0301_stress_on_token('сорока', 2)} ... второй "
        f"{set_u0301_stress_on_token('категории', 3)}"
    )
    log = Path("_diag_autocorrect.log")
    log.unlink(missing_ok=True)
    result = process_gemini_stress(
        wrong, autocorrect_enabled=True, verify_enabled=True, log_path=log
    )
    print(f"  corrected={result.corrected_count}")
    print(f"  text={result.text}")
    if log.exists():
        print(f"  log:\n{log.read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()
