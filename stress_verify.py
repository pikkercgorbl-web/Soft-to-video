"""Локальная проверка и опциональная автокоррекция ударений Gemini (StressRNN / russtress).

Сравнивает позицию U+0301 в словах Gemini с офлайн-словарём. Омографы остаются
за Gemini; для однозначных слов при включённом STRESS_AUTOCORRECT — подмена ударения.
"""

from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

COMBINING_ACUTE = "\u0301"
CYRILLIC_VOWELS = frozenset("аеёиоууыэюя")
MARKED_WORD_RE = re.compile(r"[а-яёА-ЯЁ\u0301\-]+", re.IGNORECASE | re.UNICODE)

# Путь к пользовательскому словарю исключений (U+0301). Переопределяется из конфига.
_EXCEPTION_DICT_PATH: Path | None = None

# Омографы: ударение зависит от смысла — решение остаётся за Gemini.
# Формы в lower case; дефисные слова проверяются по каждой части.
STRESS_HOMOGRAPHS: frozenset[str] = frozenset(
    {
        # замо́к (замок) / за́мок (замок)
        "замок",
        "замка",
        "замку",
        "замком",
        "замке",
        "замки",
        "замков",
        # му́ка / мука́
        "мука",
        "муки",
        "муку",
        "мукой",
        "муке",
        # а́тлас / атла́с
        "атлас",
        "атласа",
        "атласу",
        "атласом",
        "атласе",
        "атласы",
        # и́рис / ири́с
        "ирис",
        "ириса",
        "ирису",
        "ирисом",
        "ирисе",
        "ирисы",
        # ко́мпас / компа́с
        "компас",
        "компаса",
        "компасу",
        "компасом",
        "компасе",
        "компасы",
        # о́сужденный / осуждённый (контекстная омонимия)
        "осужденный",
        "осужденная",
        "осужденное",
        "осужденные",
        "осужден",
        "осуждена",
        "осуждено",
        "осуждены",
        # стоя́щий / сто́ящий
        "стоящий",
        "стоящая",
        "стоящее",
        "стоящие",
        # у́гольный / угольный
        "угольный",
        "угольная",
        "угольное",
        "угольные",
        # до́суг / досу́г
        "досуг",
        "досуга",
        # хло́пок / хлопо́к
        "хлопок",
        "хлопка",
        "хлопку",
        "хлопком",
        "хлопке",
        # банки́ / ба́нки
        "банки",
        "банка",
        "банку",
        "банкой",
        "банке",
        # пироги́ / пиро́ги
        "пироги",
        "пирог",
        "пирога",
        "пирогу",
        "пирогом",
        "пироге",
        # приказу́ / приказу́ (приказу́ vs приказу́ — контекст)
        "приказ",
        "приказа",
        "приказу",
        "приказом",
        "приказе",
        # договор / договор (до́говор / догово́р)
        "договор",
        "договора",
        "договору",
        "договором",
        "договоре",
        "договоры",
        # замер / замер (за́мер / заме́р)
        "замер",
        "замера",
        "замеру",
        "замером",
        "замере",
        # средства / средства (сре́дства / средства́)
        "средства",
        "средство",
        "средств",
        # сорока — контекстное правило (age_context / bird_context), не в списке омографов
        # плачу́ / пла́чу
        "плачу",
        # плати́ / пла́ти
        "плати",
        "платит",
        "платил",
        "платила",
        # слы́шал / слышал
        "слышал",
        "слышала",
        "слышали",
        # включи́т / включит
        "включит",
        "включишь",
        # начнёт / начнет
        "начнет",
        "начнёт",
        # подари́т / подарит
        "подарит",
        "подаришь",
        # уколол / уколол
        "уколол",
        "уколола",
        # откры́тый / открытый
        "открытый",
        "открытая",
        "открытое",
        "открытые",
        # огорчи́т / огорчит
        "огорчит",
        "огорчишь",
        # пору́ / по́ру
        "пору",
        # вору́ / во́ру
        "вору",
        # белит / белит
        "белит",
        "белят",
        # углуби́т / углубит
        "углубит",
        "углубишь",
        # облегчи́т / облегчит
        "облегчит",
        "облегчишь",
        # нарочи́т / нарочит
        "нарочит",
        # проживёшь / проживешь
        "проживешь",
        "проживёшь",
        # большие / большие (больши́е vs бóльшие)
        "большие",
        "большее",
        # доро́г / дорог (мало vs много)
        "дорог",
        "дорога",
        "дороги",
        "дорогу",
        "дорогой",
        "дороге",
        "дорогам",
        "дорогами",
        # бе́лки / белки́
        "белки",
        "белка",
        "белку",
        "белкой",
        "белке",
        "белок",
        # о́рган / орга́н
        "орган",
        "органа",
        "органу",
        "органом",
        "органе",
        "органы",
        "органов",
        # ду́хи / духи́
        "духи",
        "духов",
        "духам",
        "духами",
        # за́пах / запа́х
        "запах",
        "запаха",
        "запаху",
        "запахом",
        "запахе",
        "запахи",
    }
)

# «Сорока» — контекстный омограф; см. should_preserve_gemini_stress().
SOROKA_FORMS: frozenset[str] = frozenset(
    {"сорока", "сорок", "сороки", "сороку", "сорокой", "сороке"}
)
AGE_CONTEXT_MARKERS: frozenset[str] = frozenset(
    {"лет", "год", "года", "году", "годом", "годе", "годам", "годами", "годов"}
)


class StressEngine(Protocol):
    def stress_word(self, plain_word: str) -> str:
        """Возвращает слово с маркером ударения (формат зависит от движка)."""
        ...


class StressRnnEngine:
    """StressRNN (dbklim): RNN + словарь Зализняка + пользовательские исключения."""

    _instance: "StressRnnEngine | None" = None
    _lock = threading.Lock()
    _loaded_exception_path: Path | None = None

    def __init__(self, exception_dict_path: Path | None = None) -> None:
        from stressrnn import StressRNN

        stressrnn_dict: str | None = None
        if exception_dict_path and exception_dict_path.is_file():
            cache = build_stressrnn_exception_cache(exception_dict_path)
            if cache is not None:
                stressrnn_dict = str(cache)
        self._model = StressRNN(f_name_add_exception_dict=stressrnn_dict)
        self._loaded_exception_path = exception_dict_path

    @classmethod
    def get(cls, exception_dict_path: Path | None = None) -> "StressRnnEngine":
        with cls._lock:
            path = exception_dict_path or _EXCEPTION_DICT_PATH
            if cls._instance is None or cls._loaded_exception_path != path:
                cls._instance = cls(path)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None
            cls._loaded_exception_path = None

    def stress_word(self, plain_word: str) -> str:
        return self._model.put_stress(plain_word, stress_symbol="+").strip()


class RusstressEngine:
    """russtress (MashaPo): маркер apostrophe (') после ударной гласной."""

    _instance: "RusstressEngine | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        from russtress import Accent

        self._model = Accent()

    @classmethod
    def get(cls) -> "RusstressEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def stress_word(self, plain_word: str) -> str:
        return self._model.put_stress(plain_word).strip()


@dataclass
class StressProcessResult:
    text: str
    mismatch_count: int = 0
    corrected_count: int = 0
    homograph_mismatch_count: int = 0
    filled_count: int = 0
    missed_homograph_count: int = 0


def is_vowel(ch: str) -> bool:
    return ch.lower() in CYRILLIC_VOWELS


def configure_exception_dictionary(path: str | Path | None) -> None:
    """Задаёт путь к exception_dictionary.txt и сбрасывает кэш StressRNN."""
    global _EXCEPTION_DICT_PATH
    _EXCEPTION_DICT_PATH = Path(path) if path else None
    StressRnnEngine.reset()


def u0301_marked_to_plus(word: str) -> str:
    """Конвертирует U+0301 → '+' (формат StressRNN exception dict)."""
    return word.replace(COMBINING_ACUTE, "+")


def plus_marked_to_u0301(word: str) -> str:
    """Конвертирует '+' → U+0301 (пользовательский формат словаря)."""
    if "+" not in word:
        return word
    plain, idx = parse_plus_stress(word)
    if idx is None:
        return word
    return set_u0301_stress_on_token(plain, idx)


def load_exception_dictionary_entries(path: Path) -> list[str]:
    """Читает строки словаря (U+0301 или '+'). Пропускает пустые и комментарии."""
    if not path.is_file():
        return []
    entries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if COMBINING_ACUTE in stripped:
            entries.append(stripped)
        elif "+" in stripped:
            entries.append(plus_marked_to_u0301(stripped))
        else:
            entries.append(stripped)
    return entries


def build_stressrnn_exception_cache(u0301_path: Path) -> Path | None:
    """Генерирует .stressrnn.txt с '+' для конструктора StressRNN."""
    entries = load_exception_dictionary_entries(u0301_path)
    if not entries:
        return None
    cache_path = u0301_path.with_suffix(u0301_path.suffix + ".stressrnn.txt")
    lines = [u0301_marked_to_plus(entry) for entry in entries]
    cache_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cache_path


def append_exception_dictionary(path: Path, u0301_word: str) -> bool:
    """Добавляет слово в словарь, если его plain-форма ещё не там. Возвращает True если добавлено."""
    plain, _ = parse_u0301_stress(u0301_word)
    if not plain:
        return False
    existing = load_exception_dictionary_entries(path)
    for entry in existing:
        entry_plain, _ = parse_u0301_stress(entry)
        if entry_plain == plain:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_leading_newline = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8") as file:
        if needs_leading_newline:
            file.write("\n")
        file.write(u0301_word + "\n")
    return True


def is_homograph(plain_word: str) -> bool:
    """True, если слово (или любая часть через дефис) — омограф."""
    if plain_word in STRESS_HOMOGRAPHS:
        return True
    if "-" in plain_word:
        return any(part in STRESS_HOMOGRAPHS for part in plain_word.split("-") if part)
    return False


def is_soroka_age_context(full_text: str, word_start: int) -> bool:
    """True, если «сорок*» в контексте возраста (лет/год в 1–2 словах до)."""
    before = full_text[:word_start]
    tokens = re.findall(r"[а-яё]+", before.lower())
    if not tokens:
        return False
    window = tokens[-2:]
    return any(token in AGE_CONTEXT_MARKERS for token in window)


def should_preserve_gemini_stress(
    plain_word: str, full_text: str, word_start: int
) -> tuple[bool, str | None]:
    """Нужно ли оставить ударение Gemini. Второй элемент — тег контекста для лога."""
    if is_homograph(plain_word):
        return True, "homograph"
    if plain_word in SOROKA_FORMS:
        if is_soroka_age_context(full_text, word_start):
            return False, "age_context"
        return True, "bird_context"
    return False, None


def strip_stress_marks(word: str) -> str:
    """Снимает U+0301 и оставляет только кириллические буквы (lower)."""
    letters: list[str] = []
    for ch in word:
        if ch == COMBINING_ACUTE:
            continue
        if ch.isalpha():
            letters.append(ch.lower())
    return "".join(letters)


def parse_u0301_stress(marked_word: str) -> tuple[str, int | None]:
    """U+0301 стоит сразу после ударной гласной. Возвращает (plain, vowel_index)."""
    plain_chars: list[str] = []
    stress_idx: int | None = None
    vowel_count = 0

    for ch in marked_word:
        if ch == COMBINING_ACUTE:
            if plain_chars and is_vowel(plain_chars[-1]):
                stress_idx = vowel_count - 1
            continue
        if not ch.isalpha():
            continue
        lower = ch.lower()
        plain_chars.append(lower)
        if is_vowel(lower):
            vowel_count += 1

    return "".join(plain_chars), stress_idx


def parse_plus_stress(marked_word: str) -> tuple[str, int | None]:
    """StressRNN: '+' стоит сразу после ударной гласной."""
    plain_chars: list[str] = []
    stress_idx: int | None = None
    vowel_count = 0

    for ch in marked_word:
        if ch == "+":
            if plain_chars and is_vowel(plain_chars[-1]):
                stress_idx = vowel_count - 1
            continue
        if not ch.isalpha():
            continue
        lower = ch.lower()
        plain_chars.append(lower)
        if is_vowel(lower):
            vowel_count += 1

    return "".join(plain_chars), stress_idx


def parse_apostrophe_stress(marked_word: str) -> tuple[str, int | None]:
    """russtress: apostrophe (') сразу после ударной гласной."""
    plain_chars: list[str] = []
    stress_idx: int | None = None
    vowel_count = 0

    for ch in marked_word:
        if ch == "'":
            if plain_chars and is_vowel(plain_chars[-1]):
                stress_idx = vowel_count - 1
            continue
        if not ch.isalpha():
            continue
        lower = ch.lower()
        plain_chars.append(lower)
        if is_vowel(lower):
            vowel_count += 1

    return "".join(plain_chars), stress_idx


def set_u0301_stress_on_token(token: str, vowel_index: int) -> str:
    """Снимает U+0301 и ставит ударение на vowel_index (0-based среди гласных)."""
    letters = [ch for ch in token if ch != COMBINING_ACUTE]
    result: list[str] = []
    vowel_count = 0

    for ch in letters:
        result.append(ch)
        if is_vowel(ch):
            if vowel_count == vowel_index:
                result.append(COMBINING_ACUTE)
            vowel_count += 1

    return "".join(result)


def extract_gemini_stressed_words(text: str) -> list[tuple[str, str, int | None]]:
    """Слова с U+0301 из текста Gemini: (marked, plain, gemini_vowel_index)."""
    results: list[tuple[str, str, int | None]] = []
    seen: set[tuple[str, str, int | None]] = set()

    for match in MARKED_WORD_RE.finditer(text):
        token = match.group(0)
        if COMBINING_ACUTE not in token:
            continue
        plain, stress_idx = parse_u0301_stress(token)
        if not plain or stress_idx is None:
            continue
        key = (token, plain, stress_idx)
        if key not in seen:
            seen.add(key)
            results.append(key)

    return results


def count_vowels_in_plain(plain: str) -> int:
    return sum(1 for ch in plain if is_vowel(ch))


def token_plain_form(token: str) -> str:
    return strip_stress_marks(token)


def apply_dict_stress_to_token(token: str, dict_idx: int) -> str:
    """Ставит U+0301 по индексу гласной, сохраняя регистр исходного токена."""
    return set_u0301_stress_on_token(token, dict_idx)


def _lookup_dict_stress(
    engine: StressEngine,
    plain: str,
    engine_name: str,
) -> tuple[str, int | None]:
    dict_marked = engine.stress_word(plain)
    _, dict_idx = parse_dict_stress(dict_marked, engine_name)
    return dict_marked, dict_idx


def _process_marked_word(
    *,
    token: str,
    plain: str,
    gemini_idx: int,
    gemini_text: str,
    word_start: int,
    engine: StressEngine,
    engine_name: str,
    chunk_index: int | None,
    verify_enabled: bool,
    autocorrect_enabled: bool,
    log_lines: list[str],
    counters: dict[str, int],
) -> str:
    """Слово с U+0301 от Gemini: сравнение, автокоррекция, mismatch."""
    try:
        dict_marked, dict_idx = _lookup_dict_stress(engine, plain, engine_name)
    except Exception as error:
        if verify_enabled:
            log_lines.append(
                _format_log_line(
                    "mismatch",
                    chunk_index,
                    plain,
                    token,
                    f"(ошибка движка: {error})",
                    gemini_idx,
                    None,
                )
            )
            counters["mismatch"] += 1
        return token

    if dict_idx is None or gemini_idx == dict_idx:
        return token

    preserve_gemini, context_tag = should_preserve_gemini_stress(
        plain, gemini_text, word_start
    )
    output_token = token
    event: str | None = None

    if preserve_gemini:
        counters["homograph_mismatch"] += 1
        if verify_enabled:
            event = "mismatch"
            counters["mismatch"] += 1
    elif autocorrect_enabled:
        output_token = apply_dict_stress_to_token(token, dict_idx)
        event = "auto_corrected"
        counters["corrected"] += 1
    elif verify_enabled:
        event = "mismatch"
        counters["mismatch"] += 1

    if event:
        log_lines.append(
            _format_log_line(
                event,
                chunk_index,
                plain,
                token,
                dict_marked,
                gemini_idx,
                dict_idx,
                corrected=output_token if event == "auto_corrected" else None,
                context_tag=context_tag,
            )
        )

    return output_token


def _process_unmarked_word(
    *,
    token: str,
    plain: str,
    engine: StressEngine,
    engine_name: str,
    chunk_index: int | None,
    log_lines: list[str],
    counters: dict[str, int],
) -> str:
    """Слово без U+0301: доразметка StressRNN или пропуск омографа."""
    if is_homograph(plain):
        try:
            dict_marked, dict_idx = _lookup_dict_stress(engine, plain, engine_name)
        except Exception as error:
            dict_marked = f"(ошибка движка: {error})"
            dict_idx = None
        log_lines.append(
            _format_log_line(
                "gemini_missed_homograph",
                chunk_index,
                plain,
                token,
                dict_marked,
                None,
                dict_idx,
            )
        )
        counters["missed_homograph"] += 1
        return token

    try:
        dict_marked, dict_idx = _lookup_dict_stress(engine, plain, engine_name)
    except Exception as error:
        return token

    if dict_idx is None:
        return token

    output_token = apply_dict_stress_to_token(token, dict_idx)
    log_lines.append(
        _format_log_line(
            "gemini_missed_filled",
            chunk_index,
            plain,
            token,
            dict_marked,
            None,
            dict_idx,
            corrected=output_token,
        )
    )
    counters["filled"] += 1
    return output_token


def get_engine(engine_name: str) -> StressEngine:
    name = engine_name.strip().lower()
    if name == "russtress":
        return RusstressEngine.get()
    if name in ("stressrnn", "stress_rnn", "default"):
        return StressRnnEngine.get(_EXCEPTION_DICT_PATH)
    raise ValueError(f"Неизвестный движок проверки ударений: {engine_name!r}")


def parse_dict_stress(marked_word: str, engine_name: str) -> tuple[str, int | None]:
    name = engine_name.strip().lower()
    if name == "russtress":
        return parse_apostrophe_stress(marked_word)
    return parse_plus_stress(marked_word)


def process_gemini_stress(
    gemini_text: str,
    *,
    chunk_index: int | None = None,
    log_path: str | Path = "stress_mismatches.log",
    engine_name: str = "stressrnn",
    verify_enabled: bool = True,
    autocorrect_enabled: bool = False,
) -> StressProcessResult:
    """Проверяет/доразмечает ударения для всех слов с 2+ гласными."""
    if not verify_enabled and not autocorrect_enabled:
        return StressProcessResult(text=gemini_text)

    engine = get_engine(engine_name)
    result_parts: list[str] = []
    log_lines: list[str] = []
    last_end = 0
    counters = {
        "mismatch": 0,
        "corrected": 0,
        "homograph_mismatch": 0,
        "filled": 0,
        "missed_homograph": 0,
    }

    for match in MARKED_WORD_RE.finditer(gemini_text):
        result_parts.append(gemini_text[last_end : match.start()])
        token = match.group(0)
        word_start = match.start()
        last_end = match.end()

        plain = token_plain_form(token)
        if count_vowels_in_plain(plain) < 2:
            result_parts.append(token)
            continue

        if COMBINING_ACUTE in token:
            gemini_idx = parse_u0301_stress(token)[1]
            if gemini_idx is None:
                result_parts.append(token)
                continue
            output_token = _process_marked_word(
                token=token,
                plain=plain,
                gemini_idx=gemini_idx,
                gemini_text=gemini_text,
                word_start=word_start,
                engine=engine,
                engine_name=engine_name,
                chunk_index=chunk_index,
                verify_enabled=verify_enabled,
                autocorrect_enabled=autocorrect_enabled,
                log_lines=log_lines,
                counters=counters,
            )
        else:
            output_token = _process_unmarked_word(
                token=token,
                plain=plain,
                engine=engine,
                engine_name=engine_name,
                chunk_index=chunk_index,
                log_lines=log_lines,
                counters=counters,
            )

        result_parts.append(output_token)

    result_parts.append(gemini_text[last_end:])
    new_text = "".join(result_parts)

    if log_lines:
        _append_log(log_path, log_lines)

    return StressProcessResult(
        text=new_text,
        mismatch_count=counters["mismatch"],
        corrected_count=counters["corrected"],
        homograph_mismatch_count=counters["homograph_mismatch"],
        filled_count=counters["filled"],
        missed_homograph_count=counters["missed_homograph"],
    )


def verify_gemini_stress(
    gemini_text: str,
    *,
    chunk_index: int | None = None,
    log_path: str | Path = "stress_mismatches.log",
    engine_name: str = "stressrnn",
) -> int:
    """Обратная совместимость: только логирование без автокоррекции."""
    result = process_gemini_stress(
        gemini_text,
        chunk_index=chunk_index,
        log_path=log_path,
        engine_name=engine_name,
        verify_enabled=True,
        autocorrect_enabled=False,
    )
    return result.mismatch_count


def _format_log_line(
    event: str,
    chunk_index: int | None,
    plain: str,
    gemini_marked: str,
    dict_marked: str,
    gemini_idx: int | None,
    dict_idx: int | None,
    *,
    corrected: str | None = None,
    context_tag: str | None = None,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    chunk_part = f"chunk={chunk_index}" if chunk_index is not None else "chunk=?"
    line = (
        f"{ts} | event={event} | {chunk_part} | plain={plain} | "
        f"gemini={gemini_marked} | dict={dict_marked} | "
        f"gemini_vowel_idx={gemini_idx if gemini_idx is not None else 'None'} | "
        f"dict_vowel_idx={dict_idx if dict_idx is not None else 'None'}"
    )
    if context_tag:
        line += f" | context={context_tag}"
    if corrected is not None:
        line += f" | corrected={corrected}"
    return line


def _append_log(log_path: str | Path, lines: list[str]) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")


async def process_gemini_stress_async(
    gemini_text: str,
    *,
    chunk_index: int | None = None,
    log_path: str | Path = "stress_mismatches.log",
    engine_name: str = "stressrnn",
    verify_enabled: bool = True,
    autocorrect_enabled: bool = False,
) -> StressProcessResult:
    """CPU-bound обёртка для asyncio."""
    return await asyncio.to_thread(
        process_gemini_stress,
        gemini_text,
        chunk_index=chunk_index,
        log_path=log_path,
        engine_name=engine_name,
        verify_enabled=verify_enabled,
        autocorrect_enabled=autocorrect_enabled,
    )


async def verify_gemini_stress_async(
    gemini_text: str,
    *,
    chunk_index: int | None = None,
    log_path: str | Path = "stress_mismatches.log",
    engine_name: str = "stressrnn",
) -> int:
    """Обратная совместимость."""
    result = await process_gemini_stress_async(
        gemini_text,
        chunk_index=chunk_index,
        log_path=log_path,
        engine_name=engine_name,
        verify_enabled=True,
        autocorrect_enabled=False,
    )
    return result.mismatch_count
