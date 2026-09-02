#!/usr/bin/env python3
"""Сводка по stress_mismatches.log за последние N дней."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from stress_verify import STRESS_HOMOGRAPHS, is_homograph
except ImportError:
    STRESS_HOMOGRAPHS = frozenset()

    def is_homograph(word: str) -> bool:
        return word in STRESS_HOMOGRAPHS

LOG_LINE_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T[\d:]+Z) \| event=(?P<event>\w+) \| .*?"
    r"plain=(?P<plain>[^ |]+) \| gemini=(?P<gemini>[^ |]+) \| dict=(?P<dict>[^ |]+) \| "
    r"gemini_vowel_idx=(?P<gidx>-?\d+|None) \| dict_vowel_idx=(?P<didx>-?\d+|None)"
)


@dataclass
class WordStats:
    total: int = 0
    auto_corrected: int = 0
    mismatch: int = 0
    filled: int = 0
    missed_homograph: int = 0
    missed_unmarked: int = 0
    gemini_idx: int | None = None
    dict_idx: int | None = None
    last_event: str = ""
    last_ts: datetime | None = None


@dataclass
class Summary:
    by_word: dict[str, WordStats] = field(default_factory=lambda: defaultdict(WordStats))
    total_lines: int = 0
    skipped_old: int = 0


def parse_ts(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def load_summary(log_path: Path, days: int) -> Summary:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    summary = Summary()

    if not log_path.is_file():
        return summary

    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = LOG_LINE_RE.search(line)
        if not match:
            continue
        event = match.group("event")
        if event not in ("auto_corrected", "mismatch", "gemini_missed_filled", "gemini_missed_homograph", "gemini_missed_unmarked"):
            continue

        ts = parse_ts(match.group("ts"))
        if ts < cutoff:
            summary.skipped_old += 1
            continue

        plain = match.group("plain")
        stats = summary.by_word[plain]
        stats.total += 1
        stats.last_event = event
        stats.last_ts = ts
        if match.group("gidx") != "None":
            stats.gemini_idx = int(match.group("gidx"))
        if match.group("didx") != "None":
            stats.dict_idx = int(match.group("didx"))
        if event == "auto_corrected":
            stats.auto_corrected += 1
        elif event == "gemini_missed_filled":
            stats.filled += 1
        elif event == "gemini_missed_homograph":
            stats.missed_homograph += 1
        elif event == "gemini_missed_unmarked":
            stats.missed_unmarked += 1
        else:
            stats.mismatch += 1
        summary.total_lines += 1

    return summary


def homograph_candidates(summary: Summary, min_count: int = 3) -> list[tuple[str, WordStats]]:
    """Слова с повторяющимся auto_corrected — возможные омографы."""
    candidates: list[tuple[str, WordStats]] = []
    for word, stats in summary.by_word.items():
        if stats.auto_corrected < min_count:
            continue
        if is_homograph(word):
            continue
        candidates.append((word, stats))
    candidates.sort(key=lambda item: item[1].auto_corrected, reverse=True)
    return candidates


def print_report(log_path: Path, days: int, min_candidate: int) -> None:
    summary = load_summary(log_path, days)

    print(f"Файл: {log_path}")
    print(f"Период: последние {days} дн. (UTC)")
    print(f"Строк в периоде: {summary.total_lines}  (старше периода пропущено: {summary.skipped_old})")
    print()

    if not summary.by_word:
        print("Записей за период нет.")
        return

    ranked = sorted(summary.by_word.items(), key=lambda x: x[1].total, reverse=True)[:20]

    print("ТОП-20 слов по числу расхождений (Gemini vs словарь)")
    print("-" * 72)
    print(f"{'#':>3}  {'слово':<20} {'всего':>6} {'filled':>7} {'auto_corr':>10} {'mismatch':>9} {'missed_hg':>10}")
    print("-" * 82)
    for i, (word, stats) in enumerate(ranked, start=1):
        print(
            f"{i:>3}  {word:<20} {stats.total:>6} {stats.filled:>7} "
            f"{stats.auto_corrected:>10} {stats.mismatch:>9} {stats.missed_homograph:>10}"
        )

    total_ac = sum(s.auto_corrected for s in summary.by_word.values())
    total_fill = sum(s.filled for s in summary.by_word.values())
    total_mm = sum(s.mismatch for s in summary.by_word.values())
    total_mh = sum(s.missed_homograph for s in summary.by_word.values())
    total_mu = sum(s.missed_unmarked for s in summary.by_word.values())
    print("-" * 82)
    print(
        f"Итого filled: {total_fill}  auto_corrected: {total_ac}  "
        f"mismatch: {total_mm}  missed_homograph: {total_mh}  missed_unmarked: {total_mu}"
    )
    print()

    candidates = homograph_candidates(summary, min_candidate)
    print(f"Кандидаты в STRESS_HOMOGRAPHS (auto_corrected ≥ {min_candidate} раз, не в списке):")
    if not candidates:
        print("  (нет)")
    else:
        for word, stats in candidates:
            print(
                f"  • {word}: auto_corrected={stats.auto_corrected}, "
                f"gemini_idx={stats.gemini_idx} → dict_idx={stats.dict_idx}"
            )
        print()
        print("  Подтверждайте вручную через confirm_stress_exceptions.py — не добавляйте автоматически.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Сводка stress_mismatches.log")
    parser.add_argument(
        "--log",
        default="stress_mismatches.log",
        help="Путь к логу (по умолчанию stress_mismatches.log)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="За сколько последних дней (UTC). По умолчанию 7.",
    )
    parser.add_argument(
        "--min-candidate",
        type=int,
        default=3,
        help="Порог повторов auto_corrected для кандидата в омографы (по умолчанию 3).",
    )
    args = parser.parse_args()
    print_report(Path(args.log), args.days, args.min_candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
