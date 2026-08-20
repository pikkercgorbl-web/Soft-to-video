"""Пост-обработка WAV из Chatterbox TTS: обрезка хвостовых артефактов + fade-out.

Убирает остаточный шум/щелчок в конце клипа (типичный артефакт авторегрессии T3),
независимо от того, как расставлены ударения во входном тексте.

Порядок обработки:
    1. Энергетический VAD: находим последний "речевой" сэмпл по скользящему RMS
       и отрезаем всё, что тише порога (хвост тишины/шума/щелчка).
    2. Fade-out, выровненный по ближайшему переходу через ноль (zero-crossing),
       чтобы срез не давал собственного щелчка.
    3. Опциональный мягкий noise-gate по всему сигналу.

Зависимости: numpy, soundfile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def _to_mono(wav: np.ndarray) -> tuple[np.ndarray, bool]:
    if wav.ndim == 1:
        return wav, False
    return wav.mean(axis=1), True


def _frame_rms(mono: np.ndarray, frame: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (rms_по_кадрам, индексы_центров_кадров)."""
    if len(mono) < frame:
        frame = max(1, len(mono))
    n_frames = 1 + max(0, (len(mono) - frame) // hop)
    rms = np.empty(n_frames, dtype=np.float64)
    centers = np.empty(n_frames, dtype=np.int64)
    for i in range(n_frames):
        start = i * hop
        seg = mono[start : start + frame]
        rms[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
        centers[i] = start + frame // 2
    return rms, centers


def find_last_voice_sample(
    mono: np.ndarray,
    sr: int,
    frame_ms: float = 20.0,
    hop_ms: float = 5.0,
    threshold_db: float = -45.0,
) -> int:
    """Индекс последнего сэмпла, где сигнал ещё "громче" порога относительно пика."""
    if len(mono) == 0:
        return 0
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    rms, centers = _frame_rms(mono, frame, hop)

    peak = float(rms.max())
    if peak <= 0:
        return len(mono)
    thresh = peak * (10.0 ** (threshold_db / 20.0))

    voiced = np.where(rms >= thresh)[0]
    if len(voiced) == 0:
        return len(mono)
    last_frame = int(voiced[-1])
    # конец последнего речевого кадра
    return min(len(mono), centers[last_frame] + frame // 2)


def _nearest_zero_crossing(mono: np.ndarray, idx: int, search: int) -> int:
    """Ближайший к idx переход через ноль в окне [idx-search, idx]."""
    lo = max(1, idx - search)
    hi = min(len(mono), idx + 1)
    seg = mono[lo:hi]
    if len(seg) < 2:
        return idx
    signs = np.signbit(seg)
    crossings = np.where(signs[:-1] != signs[1:])[0]
    if len(crossings) == 0:
        return idx
    # берём переход, ближайший к правому краю окна (к idx)
    return lo + int(crossings[-1])


def postprocess_wav(
    wav: np.ndarray,
    sr: int,
    tail_threshold_db: float = -45.0,
    tail_pad_ms: float = 40.0,
    fade_ms: float = 25.0,
    gate_db: float | None = None,
) -> np.ndarray:
    """Обрезает хвост, делает fade-out по zero-crossing, опционально гейтит.

    Args:
        wav: массив сэмплов float (-1..1), моно (N,) или стерео (N, C).
        sr: частота дискретизации.
        tail_threshold_db: порог тишины относительно пика для обрезки хвоста.
        tail_pad_ms: сколько миллисекунд оставить после последнего голоса.
        fade_ms: длительность fade-out.
        gate_db: если задан, всё тише этого порога (отн. пика) обнуляется.
    """
    original_ndim = wav.ndim
    mono, _ = _to_mono(wav)

    last_voice = find_last_voice_sample(mono, sr, threshold_db=tail_threshold_db)
    pad = int(sr * tail_pad_ms / 1000)
    cut = min(len(mono), last_voice + pad)

    fade_len = int(sr * fade_ms / 1000)
    # выравниваем точку среза по zero-crossing, чтобы не было щелчка
    cut = _nearest_zero_crossing(mono, cut, search=fade_len)
    cut = max(1, cut)

    trimmed = wav[:cut].copy()

    # fade-out
    fade_len = min(fade_len, cut)
    if fade_len > 1:
        ramp = np.linspace(1.0, 0.0, fade_len, dtype=wav.dtype)
        if original_ndim == 1:
            trimmed[-fade_len:] *= ramp
        else:
            trimmed[-fade_len:] *= ramp[:, None]

    if gate_db is not None:
        peak = float(np.max(np.abs(trimmed))) or 1.0
        floor = peak * (10.0 ** (gate_db / 20.0))
        mask = np.abs(trimmed) < floor
        trimmed[mask] = 0.0

    return trimmed


def postprocess_file(
    in_path: str | Path,
    out_path: str | Path | None = None,
    **kwargs,
) -> Path:
    """Читает WAV, применяет postprocess_wav, пишет результат (по умолчанию — на место)."""
    in_path = Path(in_path)
    out_path = Path(out_path) if out_path else in_path
    wav, sr = sf.read(str(in_path), always_2d=False)
    processed = postprocess_wav(wav, sr, **kwargs)
    sf.write(str(out_path), processed, sr)
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python audio_postprocess.py input.wav [output.wav]")
        raise SystemExit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else None
    result = postprocess_file(src, dst)
    print(f"Готово: {result}")
