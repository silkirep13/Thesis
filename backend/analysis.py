"""
DSP/MIR analysis pipeline — split architecture:

  1. HPSS           — isolate harmonic content (helps both stages)
  2. Chroma (CQT)   — mode & tonic detection   [polyphonic-safe]
  3. pYIN           — microtonal deviations     [monophonic only, flagged when unreliable]

CREPE slot: replace _extract_pitch() once crepe/torchcrepe supports Python 3.14+.
"""

import io
import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# Scale templates — intervals in cents from tonic
# Sources: Chrysanthos of Madytos (1832), d'Erlanger Arabic maqam corpus,
#          RISM Byzantine chant studies, compIAM
#
# Byzantine intervals use the 72-moira system (1 moira ≈ 16.67 ¢):
#   large step  = 12 moira = 200 ¢
#   medium step = 10 moira = 167 ¢
#   small step  =  8 moira = 133 ¢
#   semitone    =  6 moira = 100 ¢
#
# NOTE: chroma-based mode detection works at ~50 ¢ resolution.
# Quarter-tone distinctions (e.g. Bayati's 150 ¢ 2nd vs. 200 ¢) are captured
# by the separate pYIN microtonal deviation analysis, not here.
# ---------------------------------------------------------------------------
SCALE_TEMPLATES: dict[str, list[int]] = {

    # -----------------------------------------------------------------------
    # Byzantine / Greek Orthodox — Octoechos (8 modes)
    # -----------------------------------------------------------------------
    "Mode 1 – Protos":       [0, 200, 367, 500, 700, 900, 1067],
    "Mode 2 – Deuteros":     [0, 100, 367, 500, 700, 800, 1067],
    "Mode 3 – Tritos":       [0, 167, 367, 500, 700, 867, 1067],
    "Mode 4 – Tetartos":     [0, 133, 367, 567, 700, 900, 1067],
    "Mode 5 – Plagios A'":   [0, 200, 367, 500, 700, 900, 1067],
    "Mode 6 – Plagios B'":   [0, 100, 367, 500, 700, 800,  967],
    "Mode 7 – Barys":        [0, 167, 367, 500, 700, 767,  967],
    "Mode 8 – Plagios D'":   [0, 200, 367, 500, 700, 900, 1067],

    # -----------------------------------------------------------------------
    # Byzantine chromatic & enharmonic genera
    # -----------------------------------------------------------------------
    "Chromatic – Hard":      [0,  67, 400, 500, 700, 767, 1100],
    "Chromatic – Soft":      [0, 100, 367, 500, 700, 800, 1067],
    "Enharmonic":            [0,  50, 400, 500, 700, 750, 1100],

    # -----------------------------------------------------------------------
    # Greek & Cypriot folk
    # -----------------------------------------------------------------------
    "Greek Minor (Minore)":  [0, 200, 300, 500, 700, 800, 1000],
    "Greek Hijaz":           [0, 100, 400, 500, 700, 800, 1000],
    "Cypriot Pentachord":    [0, 200, 367, 500, 700, 900, 1067],

    # -----------------------------------------------------------------------
    # Arabic maqamat
    # -----------------------------------------------------------------------
    "Maqam Rast":            [0, 200, 350, 500, 700, 900, 1050],
    "Maqam Bayati":          [0, 150, 300, 500, 700, 800, 1000],
    "Maqam Hijaz":           [0, 100, 400, 500, 700, 800, 1000],
    "Maqam Nahawand":        [0, 200, 300, 500, 700, 800, 1100],
    "Maqam Saba":            [0, 150, 300, 400, 600, 800, 1000],
    "Maqam Kurd":            [0, 100, 300, 500, 700, 800, 1000],
    "Maqam Ajam":            [0, 200, 400, 600, 700, 900, 1100],
    "Maqam Jiharkah":        [0, 200, 400, 550, 700, 900, 1100],
}

TONIC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Byzantine kyrios → plagios pairs (same scale, different melodic position)
KYRIOS_TO_PLAGAL = {
    "Mode 1 – Protos":   "Mode 5 – Plagios A'",
    "Mode 2 – Deuteros": "Mode 6 – Plagios B'",
    "Mode 4 – Tetartos": "Mode 8 – Plagios D'",
}

# Templates available per tradition — restricts the joint chroma search to
# the relevant scale family, eliminating cross-tradition confusion.
TRADITION_TEMPLATES: dict[str, list[str]] = {
    "byzantine": [
        # Mode 8 (Plagios D') is intentionally excluded from direct detection:
        # its chroma template is identical to Mode 1 (Protos) at 12-TET resolution,
        # so the joint search scores them equally and picks arbitrarily.
        # Mode 8 is instead reached via the Mode 4 → kyrios/plagal refinement path.
        "Mode 1 – Protos", "Mode 2 – Deuteros", "Mode 3 – Tritos",
        "Mode 4 – Tetartos", "Mode 5 – Plagios A'", "Mode 6 – Plagios B'",
        "Mode 7 – Barys",
        "Chromatic – Hard", "Chromatic – Soft", "Enharmonic",
    ],
    "greek": [
        "Greek Minor (Minore)", "Greek Hijaz",
    ],
    "cypriot": [
        "Cypriot Pentachord",
    ],
    "arabic": [
        "Maqam Rast", "Maqam Bayati", "Maqam Hijaz", "Maqam Nahawand",
        "Maqam Saba", "Maqam Kurd", "Maqam Ajam", "Maqam Jiharkah",
    ],
}

MAX_DURATION        = 90     # seconds
VOICED_THRESHOLD    = 0.65   # pYIN voiced probability minimum
MICROTONAL_MIN_VOICED = 0.25 # below this → microtonal results unreliable
CENTS_PER_BIN       = 10
N_BINS              = 1200 // CENTS_PER_BIN

# Minimum RMS energy for a Demucs stem to be considered "present"
STEM_ENERGY_THRESHOLD       = 0.005   # vocals / other
STEM_ENERGY_THRESHOLD_BASS  = 0.015   # bass needs a higher bar — low-register oud/laouto
                                       # bleeds into the bass stem and causes false positives
STEM_ENERGY_THRESHOLD_DRUMS = 0.020   # drums need a higher bar — reverb from
                                       # chant / acoustic recordings bleeds into
                                       # the drums stem and would cause false positives

# Human-readable stem labels per tradition (for the "other" melody stem)
TRADITION_MELODY_LABEL: dict[str, str] = {
    "byzantine": "Choir / Isocratima",
    "greek":     "Bouzouki / Laouto / Guitar",
    "cypriot":   "Laouto / Violin / Guitar",
    "arabic":    "Oud / Kanon / Ney",
}

# ---------------------------------------------------------------------------
# Demucs source separation — lazy-loaded model
# ---------------------------------------------------------------------------

_DEMUCS_MODEL = None   # loaded once on first use


def _get_demucs_model():
    """Load htdemucs model once and cache it for the process lifetime."""
    global _DEMUCS_MODEL
    if _DEMUCS_MODEL is None:
        from demucs.pretrained import get_model
        _DEMUCS_MODEL = get_model("htdemucs")
        _DEMUCS_MODEL.eval()
    return _DEMUCS_MODEL


def _separate_stems(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """
    Run Demucs htdemucs on the audio and return a dict of
    stem_name → mono numpy array resampled to `sr`.

    Stems: 'drums', 'bass', 'other', 'vocals'

    Returns an empty dict if Demucs is not installed or fails, so the
    rest of the pipeline can fall back gracefully to the mixed audio.
    """
    try:
        import torch
        from demucs.audio import convert_audio
        from demucs.apply import apply_model

        model = _get_demucs_model()

        # Build (batch=1, channels=1, samples) float32 tensor
        wav = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

        # Demucs expects stereo at its native 44 100 Hz — convert_audio handles both
        wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)

        with torch.no_grad():
            sources = apply_model(
                model, wav,
                shifts=1,       # one random shift for quality/speed balance
                overlap=0.25,
                progress=False,
            )
        # sources: (batch=1, n_stems, channels, samples) at model.samplerate

        stems: dict[str, np.ndarray] = {}
        for i, name in enumerate(model.sources):
            stem_mono = sources[0, i].mean(dim=0).cpu().numpy()   # → mono
            if model.samplerate != sr:
                stem_mono = librosa.resample(
                    stem_mono, orig_sr=model.samplerate, target_sr=sr
                )
            # Enforce MAX_DURATION in case of floating-point length drift
            stem_mono = stem_mono[: int(sr * MAX_DURATION)]
            stems[name] = stem_mono.astype(np.float32)

        return stems

    except Exception:
        # Demucs not installed or GPU OOM — return empty so caller falls back
        return {}


def _detect_instruments(
    stems: dict[str, np.ndarray],
    tradition: str | None,
) -> list[str]:
    """
    Decide which instrument categories are audible based on per-stem RMS.
    Returns a list of human-readable labels shown in the UI.
    """
    if not stems:
        return []

    rms = {name: float(np.sqrt(np.mean(stem ** 2))) for name, stem in stems.items()}
    # drums and bass use their own higher thresholds to avoid false positives
    # from low-register melodic instruments (oud, laouto) and choral reverb
    present = {name for name, energy in rms.items()
               if name not in ("drums", "bass") and energy >= STEM_ENERGY_THRESHOLD}

    labels: list[str] = []

    if "vocals" in present:
        labels.append("Voice")

    if "other" in present:
        labels.append(TRADITION_MELODY_LABEL.get(tradition or "", "Melody instrument"))

    # Bass and percussion: Byzantine chant has neither. Demucs stem bleed from
    # reverb and low-register vocal harmonics causes false positives — skip both
    # for Byzantine. Other traditions may genuinely have bass/drums.
    TRADITIONS_WITH_BASS = {"greek", "cypriot", "arabic", None}
    if tradition in TRADITIONS_WITH_BASS:
        if rms.get("bass", 0.0) >= STEM_ENERGY_THRESHOLD_BASS:
            labels.append("Bass")

    TRADITIONS_WITH_PERCUSSION = {"greek", "cypriot", "arabic", None}
    if tradition in TRADITIONS_WITH_PERCUSSION:
        drums_rms = rms.get("drums", 0.0)
        if drums_rms >= STEM_ENERGY_THRESHOLD_DRUMS:
            labels.append("Percussion")

    return labels


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _load_audio(file_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Load audio from raw bytes.
    Tries soundfile (fast, strict) first, falls back to pydub (ffmpeg-backed,
    handles WebM, WebA, OGG, AAC, and other formats soundfile rejects).
    """
    TARGET_SR = 22050
    try:
        audio, sr = librosa.load(io.BytesIO(file_bytes), sr=TARGET_SR, mono=True, duration=MAX_DURATION)
        return audio, sr
    except Exception:
        pass

    # Fallback: pydub (requires ffmpeg on PATH)
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(file_bytes))
        seg = seg.set_channels(1).set_frame_rate(TARGET_SR)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= float(2 ** (seg.sample_width * 8 - 1))  # normalise to [-1, 1]
        # Enforce MAX_DURATION
        max_samples = int(TARGET_SR * MAX_DURATION)
        return samples[:max_samples], TARGET_SR
    except Exception as e:
        raise ValueError(
            f"Could not decode audio. Make sure ffmpeg is installed for non-MP3/WAV formats. ({e})"
        )


def analyze_audio(file_bytes: bytes, filename: str, tradition: str | None = None) -> dict:
    audio, sr = _load_audio(file_bytes)
    duration = float(librosa.get_duration(y=audio, sr=sr))

    # Step 1 — Demucs source separation
    # Isolates vocal stem (used for CREPE) and detects which instruments are present.
    # Falls back gracefully to mixed audio if Demucs is unavailable.
    stems = _separate_stems(audio, sr)
    detected_instruments = _detect_instruments(stems, tradition)

    # Step 2 — HPSS on the full mix: keeps harmonic layer for chroma analysis.
    # We still use the full mix (not just the vocal stem) for mode detection
    # because chroma works best with all melodic voices present.
    harmonic, _ = librosa.effects.hpss(audio)

    # Step 3 — Chroma-based mode & tonic detection (polyphonic-safe)
    # Restrict templates to the declared tradition so cross-family modes
    # (e.g. Maqam Nahawand vs Byzantine Mode 6) never compete.
    if tradition and tradition in TRADITION_TEMPLATES:
        active_templates = {k: _CHROMA_TEMPLATES[k] for k in TRADITION_TEMPLATES[tradition]}
    else:
        active_templates = _CHROMA_TEMPLATES
    tonic_name, mode, mode_confidence = _detect_mode_chroma(harmonic, sr, active_templates)

    # Step 4 — CREPE pitch tracking for microtonal analysis
    # Choose the best stem for pitch tracking:
    #   - vocal recording  → vocals stem (clean monophonic voice)
    #   - instrumental     → other stem  (melody instrument, e.g. bouzouki)
    #   - no Demucs        → full harmonic layer (HPSS fallback)
    if stems:
        vocal_rms = float(np.sqrt(np.mean(stems["vocals"] ** 2))) if "vocals" in stems else 0.0
        other_rms = float(np.sqrt(np.mean(stems["other"]  ** 2))) if "other"  in stems else 0.0
        if vocal_rms >= other_rms and vocal_rms >= STEM_ENERGY_THRESHOLD:
            pitch_source = stems["vocals"]
        elif other_rms >= STEM_ENERGY_THRESHOLD:
            pitch_source = stems["other"]
        else:
            pitch_source = harmonic
    else:
        pitch_source = harmonic
    f0, voiced_flag, voiced_probs = _extract_pitch(pitch_source, sr)
    confident = voiced_flag & ~np.isnan(f0) & (voiced_probs > VOICED_THRESHOLD)
    confident_f0 = f0[confident]

    # Pitch contour for the frontend chart — downsample to ≤600 points so the
    # JSON response stays small. Unvoiced frames become None (chart gaps).
    _CREPE_STEP_S   = 0.01          # 10 ms CREPE step
    _MAX_CHART_PTS  = 600
    _ds  = max(1, len(f0) // _MAX_CHART_PTS)
    _idx = range(0, len(f0), _ds)
    pitch_times  = [round(i * _CREPE_STEP_S, 3) for i in _idx]
    pitch_values = [
        round(float(f0[i]), 1) if (voiced_flag[i] and not np.isnan(f0[i])) else None
        for i in _idx
    ]

    voiced_pct = (
        float(confident.sum()) / float(voiced_flag.size)
        if voiced_flag.size > 0 else 0.0
    )
    microtonal_reliable = voiced_pct >= MICROTONAL_MIN_VOICED

    # Step 5 — Pitch statistics & microtonal deviations
    if confident_f0.size >= 10:
        midi_cents      = 1200.0 * np.log2(confident_f0 / 440.0) + 6900.0
        mean_pitch_hz   = round(float(np.median(confident_f0)), 1)
        pitch_range_cents = _robust_pitch_range(midi_cents)
        deviations      = _microtonal_deviations(midi_cents) if microtonal_reliable else []
        # Kyrios → plagios refinement (Byzantine only — other traditions don't
        # have this authentic/plagal axis)
        if tradition in (None, "byzantine"):
            mode = _refine_kyrios_plagal(mode, tonic_name, midi_cents)

        # Neutral-interval maqam refinement (Arabic only) — resolves pairs
        # that chroma alone cannot distinguish (Kurd/Bayati, Ajam/Rast).
        if tradition == "arabic":
            mode, tonic_name = _refine_arabic_maqam(mode, tonic_name, midi_cents)
    else:
        mean_pitch_hz     = 0.0
        pitch_range_cents = 0
        deviations        = []
        microtonal_reliable = False

    # Tonic reference frequency — find the octave of the detected tonic that
    # sits closest (in pitch) to the median voiced frequency of the recording.
    if confident_f0.size > 0:
        _ti    = TONIC_NAMES.index(tonic_name)
        _med   = 69.0 + 12.0 * np.log2(float(np.median(confident_f0)) / 440.0)
        _oct   = round((_med - (60 + _ti)) / 12.0)
        _tmidi = 60 + _ti + _oct * 12
        tonic_hz = round(440.0 * 2.0 ** ((_tmidi - 69.0) / 12.0), 1)
    else:
        tonic_hz = None

    return {
        "filename":             filename,
        "file_size_kb":         round(len(file_bytes) / 1024, 1),
        "detected_mode":        mode,
        "tonic":                tonic_name,
        "confidence":           mode_confidence,
        "mean_pitch_hz":        mean_pitch_hz,
        "pitch_range_cents":    pitch_range_cents,
        "microtonal_deviations": deviations,
        "microtonal_reliable":  microtonal_reliable,
        "detected_instruments": detected_instruments,
        "duration_seconds":     round(duration, 1),
        "voiced_frames":        int(confident.sum()),
        "total_frames":         int(voiced_flag.size),
        "engine":               "chroma (mode) · CREPE (pitch) · Demucs (separation)" if stems else "chroma (mode) · CREPE (pitch)",
        "status":               "complete",
        "pitch_times":          pitch_times,
        "pitch_contour":        pitch_values,
        "tonic_hz":             tonic_hz,
    }


# ---------------------------------------------------------------------------
# Stage 1 — Pitch extraction (CREPE)
# ---------------------------------------------------------------------------

def _extract_pitch(audio: np.ndarray, sr: int):
    """
    CREPE CNN pitch tracker — cent-level accuracy on monophonic audio.
    model_capacity='medium' balances speed and accuracy well for thesis use.
    Outputs match pYIN's (f0, voiced_flag, voiced_probs) interface.
    """
    import crepe
    _, frequency, confidence, _ = crepe.predict(
        audio,
        sr,
        viterbi=True,
        model_capacity='medium',
        step_size=10,   # ms — roughly matches pYIN hop_length=256 @ 22050 Hz
        verbose=0,
    )
    voiced_flag = confidence > VOICED_THRESHOLD
    f0 = frequency.astype(np.float64)
    f0[~voiced_flag] = np.nan
    return f0, voiced_flag, confidence.astype(np.float64)


# ---------------------------------------------------------------------------
# Stage 2 — Chroma-based mode & tonic detection
# ---------------------------------------------------------------------------

def _detect_mode_chroma(
    audio: np.ndarray,
    sr: int,
    templates: dict[str, np.ndarray] | None = None,
) -> tuple[str, str, float]:
    """
    Joint (tonic, mode) detection — exhaustive search over all 12 tonics × all
    mode templates. Avoids the cascade failure of detect-tonic-first approaches.

    Chroma is time-weighted: the final 25 % of the recording gets 2× weight
    because the Byzantine finalis always appears at cadences, not necessarily
    as the most frequent pitch class overall.
    """
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=36)
    # shape: (12, n_frames)

    # --- ending-weighted mean ---
    n_frames   = chroma.shape[1]
    weights    = np.ones(n_frames)
    end_start  = int(n_frames * 0.75)
    weights[end_start:] = 2.0          # cadence section counts double
    weights   /= weights.sum()
    chroma_mean = (chroma * weights[np.newaxis, :]).sum(axis=1)  # (12,)
    if chroma_mean.sum() > 0:
        chroma_mean /= chroma_mean.sum()

    # --- joint search: 12 tonics × n_modes ---
    active = templates if templates is not None else _CHROMA_TEMPLATES
    all_scores: dict[tuple[int, str], float] = {}
    for tonic_idx in range(12):
        norm = np.roll(chroma_mean, -tonic_idx).astype(float)
        if norm.sum() > 0:
            norm /= norm.sum()
        for mode_name, template in active.items():
            all_scores[(tonic_idx, mode_name)] = _cosine_sim(norm, template)

    best_key                  = max(all_scores, key=all_scores.__getitem__)
    best_tonic_idx, best_mode = best_key
    best_score                = all_scores[best_key]

    scores_list = list(all_scores.values())
    min_s, max_s = min(scores_list), max(scores_list)
    spread = max_s - min_s if max_s > min_s else 1.0
    conf   = float(np.clip(0.50 + 0.48 * (best_score - min_s) / spread, 0.50, 0.98))

    return TONIC_NAMES[best_tonic_idx], best_mode, round(conf, 2)


def _build_chroma_template(intervals: list[int]) -> np.ndarray:
    """Convert cent intervals → 12-bin chroma vector with slight Gaussian blur."""
    chroma = np.zeros(12)
    for iv in intervals:
        semitone = int(round(iv / 100)) % 12
        chroma[semitone] += 1.0
    chroma = gaussian_filter1d(chroma, sigma=0.4)
    return chroma / chroma.sum() if chroma.sum() > 0 else chroma


_CHROMA_TEMPLATES: dict[str, np.ndarray] = {
    name: _build_chroma_template(ivs)
    for name, ivs in SCALE_TEMPLATES.items()
}


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


# ---------------------------------------------------------------------------
# Stage 3 — Microtonal deviation summary (pYIN frames only)
# ---------------------------------------------------------------------------


def _refine_kyrios_plagal(
    mode: str,
    tonic_name: str,
    midi_cents: np.ndarray,
) -> str:
    """
    Byzantine kyrios/plagios disambiguation using absolute pitch.

    Root cause of the pitch-class approach failing: plagal modes descend
    below the tonic (lower tetrachord).  Those sub-tonic notes land at
    relative_pc ≈ 1000–1100 ¢, overlapping the 7th-degree window and
    pulling the median above the 1017 ¢ midpoint even when the true 7th
    is flat (967 ¢).

    Fix: work in absolute pitch space.
      1. Byzantine chant always ends on the finalis (= tonic).  Use the
         last 15 % of voiced frames to pin the tonic's absolute pitch.
      2. Count frames in [tonic+917, tonic+1017) → plagios 7th window
         Count frames in [tonic+1017, tonic+1117) → kyrios 7th window
         (non-overlapping split at the 1017 ¢ midpoint)
      3. Whichever window has more frames wins.
    Sub-tonic notes are at tonic − N ¢ in absolute pitch, so they never
    touch either window.  Octave confusion is eliminated.
    """
    if mode not in KYRIOS_TO_PLAGAL or midi_cents.size < 20:
        return mode

    tonic_idx   = TONIC_NAMES.index(tonic_name)
    expected_pc = tonic_idx * 100.0     # e.g. Bb → 1000 ¢

    # ── Step 1: locate absolute tonic from finalis (ending frames) ──────────
    # Byzantine chant always ends on the finalis (= tonic), so the last 15 %
    # of voiced frames pin the tonic's absolute pitch reliably.
    n         = midi_cents.size
    ending    = midi_cents[int(n * 0.85):]
    tonic_abs = float(np.median(ending)) if ending.size >= 5 else float(np.median(midi_cents))

    actual_pc = tonic_abs % 1200.0
    pc_err    = min(abs(actual_pc - expected_pc), 1200.0 - abs(actual_pc - expected_pc))

    if pc_err > 200:
        # Finalis pitch class doesn't match the chroma tonic — recording
        # probably doesn't end on the tonic (sample/fade-out).
        # Fall back to the pitch-class median heuristic.
        pitch_classes = midi_cents % 1200.0
        relative_pc   = (pitch_classes - expected_pc) % 1200.0
        near_seventh  = (relative_pc >= 900) & (relative_pc <= 1100)
        if near_seventh.sum() < 5:
            return mode
        seventh_median = float(np.median(relative_pc[near_seventh]))
        return KYRIOS_TO_PLAGAL[mode] if seventh_median < 1017.0 else mode

    # ── Step 2: count frames in each 7th-degree window (absolute pitch) ─────
    # Non-overlapping windows split at the 1017 ¢ midpoint between 967 and 1067.
    # Check ±1 octave around the detected tonic to cover all singing registers.
    # Sub-tonic notes (lower tetrachord of plagal modes) live BELOW tonic_abs
    # in absolute pitch — they never touch either window, eliminating the main
    # failure mode of the pitch-class approach.
    count_p = count_k = 0
    for oct_shift in (-1200.0, 0.0, 1200.0):
        base     = tonic_abs + oct_shift
        count_p += int(np.sum((midi_cents >= base + 917.0) & (midi_cents < base + 1017.0)))
        count_k += int(np.sum((midi_cents >= base + 1017.0) & (midi_cents < base + 1117.0)))

    if count_p + count_k < 5:
        return mode     # insufficient 7th-degree evidence

    if count_p > count_k:
        return KYRIOS_TO_PLAGAL[mode]   # flat 7th → plagios
    return mode                         # natural 7th → kyrios


# ---------------------------------------------------------------------------
# Stage 3b — Arabic maqam refinement (neutral-interval disambiguation)
# ---------------------------------------------------------------------------

# Maqam pairs that are ambiguous at 12-TET chroma resolution because one
# member has a neutral (quarter-tone) interval.
#
# Format: detected_mode → (alt_mode, window_lo, window_hi, current_deg, alt_deg)
#   window_lo/hi : cent range (relative to tonic) where the key degree falls
#   current_deg  : expected degree for the *currently detected* mode (cents)
#   alt_deg      : expected degree for the *alternative* mode (cents)
#
# Decision: pick whichever centre is closer to the CREPE median in that window.
_ARABIC_REFINE: dict[str, tuple[str, int, int, int, int]] = {
    # 2nd degree — Kurd = 100 ¢  vs  Bayati = 150 ¢ (neutral 2nd)
    # Bayati's 150 ¢ rounds to 200 ¢ in 12-TET, smearing energy across Eb/E.
    "Maqam Kurd":  ("Maqam Bayati", 50,  240, 100, 150),

    # 3rd degree — Ajam = 400 ¢  vs  Rast = 350 ¢ (neutral 3rd)
    # Both round to 400 ¢ in 12-TET → chroma templates are IDENTICAL.
    # Only CREPE can distinguish them.
    "Maqam Ajam":  ("Maqam Rast",   290, 470, 400, 350),
}


def _refine_arabic_maqam(
    mode: str,
    tonic_name: str,
    midi_cents: np.ndarray,
    min_frames: int = 8,
) -> tuple[str, str]:
    """
    Resolve Arabic maqam pairs that are ambiguous at 12-TET chroma resolution
    because their distinguishing degree is a neutral (quarter-tone) interval.

    Supported pairs:
      • Maqam Kurd  (100 ¢ 2nd)  ↔  Maqam Bayati (150 ¢ neutral 2nd)
      • Maqam Ajam  (400 ¢ 3rd)  ↔  Maqam Rast   (350 ¢ neutral 3rd)

    Strategy (mirrors _refine_kyrios_plagal):
      1. Derive the actual tonic from the ending 15 % of CREPE frames — the
         finalis of a taksim nearly always appears there, and CREPE gives
         better pitch-class accuracy than chroma for short passages.
      2. Compute every frame's interval relative to that tonic (pitch-class,
         mod 1200 ¢).
      3. Gather frames that fall in the ambiguous degree window.
      4. Check whether the median is closer to current_deg or alt_deg.

    Returns (mode, tonic_name) — tonic may also be corrected when the
    chroma search was off by a semitone or two.
    """
    import sys

    if mode not in _ARABIC_REFINE or midi_cents.size < 20:
        return mode, tonic_name

    alt_mode, lo, hi, current_deg, alt_deg = _ARABIC_REFINE[mode]

    # ── Step 1: refine tonic from ending frames ─────────────────────────────
    n      = midi_cents.size
    ending = midi_cents[int(n * 0.85):]
    tonic_abs     = float(np.median(ending)) if ending.size >= 5 else float(np.median(midi_cents))
    tonic_pc_raw  = tonic_abs % 1200.0
    tonic_semitone = int(round(tonic_pc_raw / 100.0)) % 12
    refined_tonic = TONIC_NAMES[tonic_semitone]
    tonic_pc      = float(tonic_semitone * 100)

    # ── Step 2: relative intervals (pitch-class space) ──────────────────────
    rel = (midi_cents % 1200.0 - tonic_pc) % 1200.0

    # ── Step 3: gather frames in the ambiguous degree window ─────────────────
    in_window = rel[(rel >= lo) & (rel <= hi)]
    if in_window.size < min_frames:
        print(
            f"[arabic_refine] {mode}/{tonic_name}: only {in_window.size} frames "
            f"in [{lo},{hi}] ¢ — skipping (tonic_refined={refined_tonic})",
            file=sys.stderr, flush=True,
        )
        return mode, refined_tonic

    median_deg = float(np.median(in_window))

    # ── Step 4: nearest-centre decision ──────────────────────────────────────
    new_mode = (
        alt_mode
        if abs(median_deg - alt_deg) < abs(median_deg - current_deg)
        else mode
    )

    print(
        f"[arabic_refine] {mode}/{tonic_name} → tonic_refined={refined_tonic}  "
        f"degree_window=[{lo},{hi}]  n={in_window.size}  "
        f"median={median_deg:.1f}¢  (current={current_deg}¢, alt={alt_deg}¢)  "
        f"→ {new_mode}/{refined_tonic}",
        file=sys.stderr, flush=True,
    )
    return new_mode, refined_tonic


def _microtonal_deviations(midi_cents: np.ndarray) -> list[str]:
    nearest = np.round(midi_cents / 100.0) * 100.0
    devs    = midi_cents - nearest
    result  = []
    for pct in [10, 25, 50, 75, 90]:
        d    = float(np.percentile(devs, pct))
        sign = "+" if d >= 0 else ""
        result.append(f"p{pct}: {sign}{round(d)}¢")
    return result


def _robust_pitch_range(midi_cents: np.ndarray) -> int:
    """IQR-based range — ignores octave-error outliers from pYIN."""
    p5  = np.percentile(midi_cents, 5)
    p95 = np.percentile(midi_cents, 95)
    return int(p95 - p5)


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _empty_result(filename: str, duration: float, size_bytes: int) -> dict:
    return {
        "filename":             filename,
        "file_size_kb":         round(size_bytes / 1024, 1),
        "detected_mode":        "Undetected",
        "tonic":                "—",
        "confidence":           0.0,
        "mean_pitch_hz":        0.0,
        "pitch_range_cents":    0,
        "microtonal_deviations": [],
        "microtonal_reliable":  False,
        "detected_instruments": [],
        "duration_seconds":     round(duration, 1),
        "voiced_frames":        0,
        "total_frames":         0,
        "engine":               "chroma (mode) · CREPE (pitch) · Demucs (separation)",
        "status":               "no_pitch_detected",
    }
