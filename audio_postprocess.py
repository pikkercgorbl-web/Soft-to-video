"""Пост-обработка WAV из Chatterbox TTS: чистка пауз, обрезка хвоста, fade-out.

Chatterbox генерирует длинный текст по sub-chunk'ам (split_text) и склеивает их
встык. Каждый sub-chunk заканчивается артефактом авторегрессии T3 — придыханием,
шорохом или "порывом ветра". На склейках эти артефакты попадают в середину
дорожки, где их слышно как посторонний вздох между предложениями.

Порядок обработки:
    1. Чистка пауз: ищем межфразовые интервалы без речи и глушим их до тишины
       плавными склонами, оставляя защитный запас у границ речи.
    2. Энергетический VAD: находим последний "речевой" сэмпл по скользящему RMS
       и отрезаем всё, что тише порога (хвост тишины/шума/щелчка).
    3. Fade-out, выровненный по ближайшему переходу через ноль (zero-crossing),
       чтобы срез не давал собственного щелчка.
    4. Опциональный мягкий noise-gate по всему сигналу.

Зависимости: numpy, soundfile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def _resample_mono_linear(mono: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Лёгкий ресемплинг для VAD; качество звука не меняет — это только анализ."""
    if source_sr == target_sr or len(mono) == 0:
        return mono.astype(np.float32, copy=False)
    target_len = max(1, int(round(len(mono) * target_sr / source_sr)))
    old_x = np.linspace(0.0, 1.0, len(mono), endpoint=False)
    new_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
    return np.interp(new_x, old_x, mono).astype(np.float32)


def find_vad_nonspeech_regions(
    mono: np.ndarray,
    sr: int,
    aggressiveness: int = 3,
    frame_ms: int = 30,
    speech_pad_ms: float = 180.0,
    min_nonspeech_ms: float = 180.0,
) -> list[tuple[int, int]]:
    """Ищет внутренние участки без речи через WebRTC VAD.

    В отличие от RMS-порога, VAD анализирует признаки человеческой речи и
    поэтому способен отделить громкий ветер, шорох и вой от слов.
    """
    try:
        import webrtcvad
    except ImportError:
        return []

    target_sr = 16000
    analysis = _resample_mono_linear(mono, sr, target_sr)
    if len(analysis) == 0:
        return []

    peak = float(np.max(np.abs(analysis))) or 1.0
    pcm = np.clip(analysis / peak * 30000.0, -32768, 32767).astype("<i2")
    frame_samples = target_sr * frame_ms // 1000
    frame_count = len(pcm) // frame_samples
    if frame_count < 3:
        return []

    vad = webrtcvad.Vad(max(0, min(3, aggressiveness)))
    speech = np.zeros(frame_count, dtype=bool)
    for index in range(frame_count):
        frame = pcm[index * frame_samples : (index + 1) * frame_samples]
        speech[index] = vad.is_speech(frame.tobytes(), target_sr)

    # Защитная дилатация речи: даже единичный речевой кадр сохраняет окружение.
    pad_frames = max(1, int(np.ceil(speech_pad_ms / frame_ms)))
    protected = speech.copy()
    for offset in range(1, pad_frames + 1):
        protected[offset:] |= speech[:-offset]
        protected[:-offset] |= speech[offset:]

    min_frames = max(1, int(np.ceil(min_nonspeech_ms / frame_ms)))
    regions: list[tuple[int, int]] = []
    index = 0
    while index < frame_count:
        if protected[index]:
            index += 1
            continue
        start_frame = index
        while index < frame_count and not protected[index]:
            index += 1
        end_frame = index
        if end_frame - start_frame < min_frames:
            continue
        # Только внутренние интервалы: начало и конец файла обрабатывает tail VAD.
        if start_frame == 0 or end_frame == frame_count:
            continue
        start = int(start_frame * frame_ms / 1000 * sr)
        end = min(len(mono), int(end_frame * frame_ms / 1000 * sr))
        regions.append((start, end))
    return regions


def clean_vad_nonspeech(
    wav: np.ndarray,
    sr: int,
    aggressiveness: int = 3,
    speech_pad_ms: float = 180.0,
    min_nonspeech_ms: float = 180.0,
    edge_fade_ms: float = 20.0,
) -> tuple[np.ndarray, int, float]:
    """Глушит только внутренние интервалы, классифицированные VAD как не-речь."""
    mono, _ = _to_mono(wav)
    regions = find_vad_nonspeech_regions(
        mono,
        sr,
        aggressiveness=aggressiveness,
        speech_pad_ms=speech_pad_ms,
        min_nonspeech_ms=min_nonspeech_ms,
    )
    if not regions:
        return wav, 0, 0.0

    result = wav.copy()
    fade = max(1, int(sr * edge_fade_ms / 1000))
    cleaned_samples = 0
    for start, end in regions:
        length = end - start
        edge = min(fade, length // 2)
        gain = np.zeros(length, dtype=np.float64)
        if edge > 1:
            down = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, edge)))
            gain[:edge] = down
            gain[-edge:] = down[::-1]
        shaped = gain.astype(result.dtype)
        if result.ndim == 1:
            result[start:end] *= shaped
        else:
            result[start:end] *= shaped[:, None]
        cleaned_samples += length
    return result, len(regions), cleaned_samples * 1000.0 / sr


def find_spectral_noise_regions(
    mono: np.ndarray,
    sr: int,
    frame_ms: float = 40.0,
    hop_ms: float = 20.0,
    min_region_ms: float = 220.0,
    max_gap_ms: float = 80.0,
    zcr_threshold: float = 0.32,
    centroid_hz: float = 3500.0,
    high_ratio_threshold: float = 0.55,
    max_rms_db: float = -28.0,
) -> list[tuple[int, int]]:
    """Ищет длительный ВЧ-шорох/ветер, который WebRTC ошибочно считает речью.

    Одиночные шипящие согласные тоже высокочастотны, но обычно длятся менее
    220 ms. Поэтому удаляется только устойчивый шум, совпавший сразу по ZCR,
    спектральному центроиду и доле энергии выше 4 kHz.
    """
    if len(mono) == 0:
        return []
    frame = max(16, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(mono) < frame:
        return []

    peak = float(np.max(np.abs(mono))) or 1.0
    window = np.hanning(frame)
    frequencies = np.fft.rfftfreq(frame, 1.0 / sr)
    high_mask = frequencies >= 4000.0
    flags: list[bool] = []

    for start in range(0, len(mono) - frame + 1, hop):
        segment = mono[start : start + frame]
        rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2) + 1e-15))
        rms_db = 20.0 * np.log10(rms / peak + 1e-12)
        zcr = float(np.mean(np.signbit(segment[:-1]) != np.signbit(segment[1:])))
        spectrum = np.abs(np.fft.rfft(segment * window))
        power = spectrum * spectrum
        total = float(np.sum(power)) + 1e-15
        centroid = float(np.sum(frequencies * power) / total)
        high_ratio = float(np.sum(power[high_mask]) / total)
        flags.append(
            rms_db <= max_rms_db
            and zcr >= zcr_threshold
            and centroid >= centroid_hz
            and high_ratio >= high_ratio_threshold
        )

    min_frames = max(1, int(np.ceil(min_region_ms / hop_ms)))
    max_gap_frames = max(0, int(np.floor(max_gap_ms / hop_ms)))
    regions: list[tuple[int, int]] = []
    index = 0
    while index < len(flags):
        if not flags[index]:
            index += 1
            continue
        run_start = index
        last_true = index
        gap = 0
        index += 1
        while index < len(flags):
            if flags[index]:
                last_true = index
                gap = 0
            else:
                gap += 1
                if gap > max_gap_frames:
                    break
            index += 1

        if last_true - run_start + 1 < min_frames:
            continue
        start = run_start * hop
        end = min(len(mono), last_true * hop + frame)
        regions.append((start, end))
    return regions


def clean_spectral_noise(
    wav: np.ndarray,
    sr: int,
    min_region_ms: float = 220.0,
    max_gap_ms: float = 80.0,
    edge_fade_ms: float = 20.0,
) -> tuple[np.ndarray, int, float]:
    """Глушит устойчивые ВЧ-артефакты, сохраняя краткие согласные речи."""
    mono, _ = _to_mono(wav)
    regions = find_spectral_noise_regions(
        mono,
        sr,
        min_region_ms=min_region_ms,
        max_gap_ms=max_gap_ms,
    )
    if not regions:
        return wav, 0, 0.0

    result = wav.copy()
    fade = max(1, int(sr * edge_fade_ms / 1000))
    cleaned_samples = 0
    for start, end in regions:
        length = end - start
        edge = min(fade, length // 2)
        gain = np.zeros(length, dtype=np.float64)
        if edge > 1:
            down = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, edge)))
            gain[:edge] = down
            gain[-edge:] = down[::-1]
        shaped = gain.astype(result.dtype)
        if result.ndim == 1:
            result[start:end] *= shaped
        else:
            result[start:end] *= shaped[:, None]
        cleaned_samples += length
    return result, len(regions), cleaned_samples * 1000.0 / sr


def _to_mono(wav: np.ndarray) -> tuple[np.ndarray, bool]:
    if wav.ndim == 1:
        return wav, False
    return wav.mean(axis=1), True


def clean_stitch_boundaries(
    wav: np.ndarray,
    sr: int,
    min_zero_ms: float = 100.0,
    fade_before_ms: float = 40.0,
    candidate_window_ms: float = 1300.0,
    speech_threshold_db: float = -38.0,
    speech_guard_ms: float = 40.0,
    zero_epsilon: float = 1e-7,
) -> tuple[np.ndarray, int]:
    """Плавно гасит хвост непосредственно перед точной тишиной склейки сервера.

    Smart stitching Chatterbox вставляет между sub-chunk'ами около 200 ms
    цифрового нуля. Это надёжный маркер стыка: в отличие от VAD, здесь не нужно
    решать по громкости, является ли тихий звук речью. Затрагивается только
    короткий хвост перед стыком, поэтому целое слово удалить невозможно.
    """
    mono, _ = _to_mono(wav)
    silent = np.abs(mono) <= zero_epsilon
    min_samples = max(1, int(sr * min_zero_ms / 1000))
    fade_samples = max(1, int(sr * fade_before_ms / 1000))
    candidate_window = int(sr * candidate_window_ms / 1000)

    result = wav.copy()
    boundaries = 0
    index = 0
    previous_zero_end = 0
    peak = float(np.max(np.abs(mono))) or 1.0
    strong_threshold = peak * (10.0 ** (speech_threshold_db / 20.0))
    analysis_frame = max(1, int(sr * 30.0 / 1000))
    analysis_hop = max(1, int(sr * 10.0 / 1000))
    speech_guard = int(sr * speech_guard_ms / 1000)

    while index < len(silent):
        if not silent[index]:
            index += 1
            continue
        start = index
        while index < len(silent) and silent[index]:
            index += 1
        end = index
        if end - start < min_samples or start == 0:
            continue

        window_start = max(previous_zero_end, start - candidate_window)
        last_strong_end: int | None = None
        final_frame_start = max(window_start, start - analysis_frame)
        for frame_start in range(window_start, final_frame_start + 1, analysis_hop):
            segment = mono[frame_start : min(start, frame_start + analysis_frame)]
            if len(segment) == 0:
                continue
            rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2) + 1e-15))
            if rms >= strong_threshold:
                last_strong_end = frame_start + len(segment)

        # Всё после последнего уверенного речевого кадра — хвост генерации.
        # Guard сохраняет тихое окончание последнего слова.
        cleanup_start = (
            min(start, last_strong_end + speech_guard)
            if last_strong_end is not None
            else start
        )
        fade_start = max(previous_zero_end, cleanup_start - fade_samples)
        length = start - fade_start
        if length > 1:
            fade_length = min(fade_samples, length)
            gain = np.zeros(length, dtype=np.float64)
            gain[:fade_length] = 0.5 * (
                1.0 + np.cos(np.linspace(0.0, np.pi, fade_length))
            )
            shaped = gain.astype(result.dtype)
            if result.ndim == 1:
                result[fade_start:start] *= shaped
            else:
                result[fade_start:start] *= shaped[:, None]
        boundaries += 1
        previous_zero_end = end

    return result, boundaries


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


def _speech_frame_mask(
    rms: np.ndarray,
    enter_db: float,
    exit_db: float,
) -> np.ndarray:
    """Маска речевых кадров с гистерезисом.

    Два порога вместо одного нужны, чтобы тихие окончания слов не обрывались:
    вход в речь требует уровня enter_db, а выход — падения ниже exit_db.
    """
    peak = float(rms.max()) if len(rms) else 0.0
    if peak <= 0:
        return np.zeros(len(rms), dtype=bool)

    enter = peak * (10.0 ** (enter_db / 20.0))
    exit_ = peak * (10.0 ** (exit_db / 20.0))

    mask = np.zeros(len(rms), dtype=bool)
    active = False
    for i, value in enumerate(rms):
        if active:
            if value < exit_:
                active = False
        elif value >= enter:
            active = True
        mask[i] = active
    return mask


def find_pause_regions(
    mono: np.ndarray,
    sr: int,
    frame_ms: float = 20.0,
    hop_ms: float = 5.0,
    enter_db: float = -32.0,
    exit_db: float = -42.0,
    min_pause_ms: float = 120.0,
    guard_ms: float = 25.0,
) -> list[tuple[int, int]]:
    """Границы пауз (в сэмплах), пригодных для заглушения.

    Возвращаются только интервалы длиннее min_pause_ms, ужатые на guard_ms с
    каждой стороны: запас защищает придыхательные согласные и тихие окончания.
    """
    if len(mono) == 0:
        return []

    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    rms, centers = _frame_rms(mono, frame, hop)
    speech = _speech_frame_mask(rms, enter_db, exit_db)
    if not speech.any():
        return []

    min_pause = int(sr * min_pause_ms / 1000)
    guard = int(sr * guard_ms / 1000)
    half_frame = frame // 2

    regions: list[tuple[int, int]] = []
    index = 0
    total_frames = len(speech)

    while index < total_frames:
        if speech[index]:
            index += 1
            continue
        run_start = index
        while index < total_frames and not speech[index]:
            index += 1
        run_end = index - 1

        start = max(0, int(centers[run_start]) - half_frame)
        end = min(len(mono), int(centers[run_end]) + half_frame)
        if end - start < min_pause:
            continue

        # у самых краёв файла защитный запас не нужен: там нет речи для защиты
        start = start + guard if run_start > 0 else start
        end = end - guard if run_end < total_frames - 1 else end
        if end > start:
            regions.append((start, end))

    return regions


def _segment_peak_db(mono: np.ndarray, start: int, end: int, peak: float) -> float:
    if start >= end or peak <= 0:
        return -240.0
    segment = mono[start:end]
    return float(20.0 * np.log10(float(np.max(np.abs(segment))) / peak + 1e-12))


def expand_pause_regions(
    regions: list[tuple[int, int]],
    mono: np.ndarray,
    sr: int,
    bridge_max_ms: float = 550.0,
    bridge_peak_db: float = -20.0,
) -> list[tuple[int, int]]:
    """Склеивает паузы через короткий «мост» с тихим шумом/вздохом между ними.

    Типичный случай: VAD видит две паузы, а между ними 300–500 ms «псевдоречи»
    с пиком −25…−30 dB — это и есть слышимый вздох на стыке sub-chunk'ов.
    """
    if not regions:
        return regions

    peak = float(np.max(np.abs(mono))) or 1.0
    bridge_thresh = peak * (10.0 ** (bridge_peak_db / 20.0))
    max_bridge = int(sr * bridge_max_ms / 1000)

    merged: list[tuple[int, int]] = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        gap_start = prev_end
        gap_end = start
        gap_len = gap_end - gap_start
        if 0 < gap_len <= max_bridge:
            gap_peak = float(np.max(np.abs(mono[gap_start:gap_end]))) if gap_len else 0.0
            if gap_peak <= bridge_thresh:
                merged[-1] = (prev_start, end)
                continue
        merged.append((start, end))
    return merged


def extend_pause_tails(
    regions: list[tuple[int, int]],
    mono: np.ndarray,
    sr: int,
    tail_peak_db: float = -18.0,
    max_extend_ms: float = 400.0,
    hop_ms: float = 5.0,
) -> list[tuple[int, int]]:
    """Продлевает конец паузы, пока сигнал остаётся тихим (хвост вздоха после VAD)."""
    if not regions:
        return regions

    peak = float(np.max(np.abs(mono))) or 1.0
    thresh = peak * (10.0 ** (tail_peak_db / 20.0))
    hop = max(1, int(sr * hop_ms / 1000))
    max_extend = int(sr * max_extend_ms / 1000)

    extended: list[tuple[int, int]] = []
    for start, end in regions:
        new_end = end
        limit = min(len(mono), end + max_extend)
        while new_end < limit:
            window = mono[new_end : min(new_end + hop, limit)]
            if len(window) == 0 or float(np.max(np.abs(window))) > thresh:
                break
            new_end += len(window)
        extended.append((start, new_end))
    return extended


def clean_pause_artifacts(
    wav: np.ndarray,
    sr: int,
    enter_db: float = -36.0,
    exit_db: float = -48.0,
    min_pause_ms: float = 180.0,
    guard_ms: float = 80.0,
    bridge_max_ms: float = 0.0,
    bridge_peak_db: float = -20.0,
    tail_extend_ms: float = 0.0,
    tail_peak_db: float = -18.0,
    edge_fade_ms: float = 12.0,
    residual_db: float | None = None,
) -> tuple[np.ndarray, int, float]:
    """Глушит паузы между фразами. Возвращает (сигнал, число пауз, суммарные мс)."""
    mono, _ = _to_mono(wav)
    regions = find_pause_regions(
        mono,
        sr,
        enter_db=enter_db,
        exit_db=exit_db,
        min_pause_ms=min_pause_ms,
        guard_ms=guard_ms,
    )
    if bridge_max_ms > 0:
        regions = expand_pause_regions(
            regions,
            mono,
            sr,
            bridge_max_ms=bridge_max_ms,
            bridge_peak_db=bridge_peak_db,
        )
    if tail_extend_ms > 0:
        regions = extend_pause_tails(
            regions,
            mono,
            sr,
            tail_peak_db=tail_peak_db,
            max_extend_ms=tail_extend_ms,
        )
    if not regions:
        return wav, 0, 0.0

    residual = 0.0 if residual_db is None else 10.0 ** (residual_db / 20.0)
    edge = max(1, int(sr * edge_fade_ms / 1000))
    result = wav.copy()
    cleaned_samples = 0

    for start, end in regions:
        length = end - start
        edge_len = min(edge, length // 2)
        gain = np.full(length, residual, dtype=np.float64)
        if edge_len > 0:
            # Оба склона нужны: жёсткий переход из нуля обратно в звук даёт щелчок.
            ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, edge_len)))
            gain[:edge_len] = residual + (1.0 - residual) * ramp
            gain[-edge_len:] = residual + (1.0 - residual) * ramp[::-1]

        shaped = gain.astype(result.dtype)
        if result.ndim == 1:
            result[start:end] *= shaped
        else:
            result[start:end] *= shaped[:, None]
        cleaned_samples += length

    return result, len(regions), cleaned_samples * 1000.0 / sr


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
    pause_clean: bool = True,
    pause_enter_db: float = -36.0,
    pause_exit_db: float = -48.0,
    pause_min_ms: float = 180.0,
    pause_guard_ms: float = 80.0,
    pause_bridge_max_ms: float = 0.0,
    pause_bridge_peak_db: float = -20.0,
    pause_tail_extend_ms: float = 0.0,
    pause_tail_peak_db: float = -18.0,
    stitch_clean: bool = True,
    stitch_min_zero_ms: float = 100.0,
    stitch_fade_before_ms: float = 40.0,
    vad_clean: bool = True,
    vad_aggressiveness: int = 3,
    vad_speech_pad_ms: float = 180.0,
    vad_min_nonspeech_ms: float = 180.0,
    spectral_clean: bool = False,
    spectral_min_region_ms: float = 220.0,
    spectral_max_gap_ms: float = 80.0,
    stats: dict | None = None,
) -> np.ndarray:
    """Чистит паузы, обрезает хвост, делает fade-out по zero-crossing.

    Args:
        wav: массив сэмплов float (-1..1), моно (N,) или стерео (N, C).
        sr: частота дискретизации.
        tail_threshold_db: порог тишины относительно пика для обрезки хвоста.
        tail_pad_ms: сколько миллисекунд оставить после последнего голоса.
        fade_ms: длительность fade-out.
        gate_db: если задан, всё тише этого порога (отн. пика) обнуляется.
        pause_clean: глушить ли артефакты в межфразовых паузах.
        pause_enter_db: порог входа в речь для детектора пауз.
        pause_exit_db: порог выхода из речи (гистерезис).
        pause_min_ms: минимальная длина паузы, которую есть смысл чистить.
        pause_guard_ms: защитный запас у границ речи.
        pause_bridge_max_ms: склеивать паузы через «мост» не длиннее этого.
        pause_bridge_peak_db: пик моста (отн. пика файла), ниже которого его глушим.
        stats: если передан словарь, в него пишется отчёт по чистке пауз.
    """
    original_ndim = wav.ndim

    # Сначала работаем с настоящими цифровыми паузами сервера. Остальные
    # фильтры сами создают нулевые участки, которые нельзя принимать за стыки.
    if stitch_clean:
        wav, stitch_count = clean_stitch_boundaries(
            wav,
            sr,
            min_zero_ms=stitch_min_zero_ms,
            fade_before_ms=stitch_fade_before_ms,
        )
        if stats is not None:
            stats["stitches_cleaned"] = stitch_count

    if spectral_clean:
        wav, spectral_count, spectral_ms = clean_spectral_noise(
            wav,
            sr,
            min_region_ms=spectral_min_region_ms,
            max_gap_ms=spectral_max_gap_ms,
        )
        if stats is not None:
            stats["spectral_regions_cleaned"] = spectral_count
            stats["spectral_regions_ms"] = spectral_ms

    if vad_clean:
        wav, vad_count, vad_ms = clean_vad_nonspeech(
            wav,
            sr,
            aggressiveness=vad_aggressiveness,
            speech_pad_ms=vad_speech_pad_ms,
            min_nonspeech_ms=vad_min_nonspeech_ms,
        )
        if stats is not None:
            stats["vad_regions_cleaned"] = vad_count
            stats["vad_regions_ms"] = vad_ms

    if pause_clean:
        wav, pause_count, pause_ms = clean_pause_artifacts(
            wav,
            sr,
            enter_db=pause_enter_db,
            exit_db=pause_exit_db,
            min_pause_ms=pause_min_ms,
            guard_ms=pause_guard_ms,
            bridge_max_ms=pause_bridge_max_ms,
            bridge_peak_db=pause_bridge_peak_db,
            tail_extend_ms=pause_tail_extend_ms,
            tail_peak_db=pause_tail_peak_db,
        )
        if stats is not None:
            stats["pauses_cleaned"] = pause_count
            stats["pauses_ms"] = pause_ms

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
