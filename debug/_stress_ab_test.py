"""Мини-тест A/B против ЖИВОГО сервера Chatterbox (http://localhost:8004).

Сравнивает прочтение омографа "замок" и стабильность хвоста аудио для трёх вариантов:
    (a) без ударения           -> "дверной замок"
    (b) ударение через U+0301   -> "дверной замо́к"   (правильный формат для vocab)
    (c) ударение заглавной      -> "дверной замОк"   (ломается: сервер делает .lower())

Для каждого варианта:
    * шлёт POST /tts,
    * сохраняет WAV в downloads/_test_*.wav,
    * считает RMS последних 150 мс (индикатор хвостового шума/щелчка).

Запуск:
    python _stress_ab_test.py
Нужен работающий сервер на 8004 и установленные numpy, soundfile, requests.
"""

from pathlib import Path

import numpy as np
import requests
import soundfile as sf

URL = "http://localhost:8004/tts"
VOICE_FILE = "ya.WAV"       # поменяй на свой голос
VOICE_MODE = "clone"        # "clone" или "predefined"
OUT_DIR = Path("downloads")
OUT_DIR.mkdir(exist_ok=True)

STRESS = "\u0301"
CASES = {
    "a_plain": "Он повесил на дверь тяжёлый замок.",
    "b_u0301": f"Он повесил на дверь тяжёлый замо{STRESS}к.",
    "c_upper": "Он повесил на дверь тяжёлый замОк.",
}


def synth(text: str, out_path: Path) -> None:
    payload = {
        "text": text,
        "voice_mode": VOICE_MODE,
        "language": "ru",
        "speed_factor": 1.0,
        "temperature": 0.8,
        "exaggeration": 1.3,
        "cfg_weight": 0.5,
        "chunk_size": 100,
        "output_format": "wav",
        "seed": 12345,          # фиксированный seed -> сравнение честное
        "split_text": True,
    }
    if VOICE_MODE == "clone":
        payload["reference_audio_filename"] = VOICE_FILE
    else:
        payload["predefined_voice_id"] = VOICE_FILE

    r = requests.post(URL, json=payload, timeout=180)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def tail_rms_db(path: Path, tail_ms: float = 150.0) -> float:
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    n = int(sr * tail_ms / 1000)
    tail = wav[-n:] if len(wav) >= n else wav
    rms = float(np.sqrt(np.mean(tail.astype(np.float64) ** 2) + 1e-12))
    peak = float(np.max(np.abs(wav)) or 1.0)
    return 20.0 * np.log10(rms / peak + 1e-12)


def main() -> None:
    print(f"{'case':10} {'input':45} {'tail dB (ниже=чище)':>20}")
    for name, text in CASES.items():
        out = OUT_DIR / f"_test_{name}.wav"
        synth(text, out)
        db = tail_rms_db(out)
        marks = " ".join("U+%04X" % ord(c) for c in text if ord(c) > 0x2000)
        print(f"{name:10} {text[:45]:45} {db:20.1f}   {marks}")
        print(f"           -> {out}")
    print("\nПрослушай файлы: правильное ударение должно звучать в (b_u0301).")
    print("(c_upper) должен читаться как (a_plain) — заглавная теряется из-за .lower().")


if __name__ == "__main__":
    main()
