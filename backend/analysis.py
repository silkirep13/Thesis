"""
DSP/MIR analysis pipeline, split architecture:

  1. HPSS: isolate harmonic content (helps both stages)
  2. Chroma (CQT): mode & tonic detection   [polyphonic-safe]
  3. pYIN: microtonal deviations     [monophonic only, flagged when unreliable]

CREPE slot: replace _extract_pitch() once crepe/torchcrepe supports Python 3.14+.
"""

import io
import sys
from pathlib import Path

import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d

UNIT_CENTS = {"72-EDO": 1200 / 72, "24-EDO": 1200 / 24, "12-TET": 1200 / 12}


def _cents_from_steps(steps: list[int], basis: str) -> list[int]:
    """Cumulative cents from a step pattern, the single source of truth
    cents are derived from, so steps and cents can never silently diverge."""
    unit = UNIT_CENTS[basis]
    cum = 0.0
    out = [0]
    for s in steps:
        cum += s * unit
        out.append(round(cum))
    if abs(sum(steps) * unit - 1200.0) < 0.01:
        out = out[:-1]
    return out


def _template(steps: list[int], basis: str, genus: str | None = None) -> dict:
    """ Package one scale pattern into the structure the detector uses.
    Stores the step pattern, its tuning basis and the genus, and derives the
    interval sizes in cents from the steps rather than storing them
    separately, so the two can never disagree."""
    return {
        "basis": basis,
        "steps": steps,
        "cents": _cents_from_steps(steps, basis),
        "genus": genus,
    }


SCALE_TEMPLATES: dict[str, dict] = {

    "Mode 1 – Protos":     _template([10, 8, 12, 12, 10, 8, 12], "72-EDO", "diatonic"),
    "Mode 2 – Deuteros":   _template([8, 14, 8, 12, 8, 14, 8],   "72-EDO", "soft chromatic"),
    "Mode 3 – Tritos":     _template([12, 12, 6, 12, 12, 12, 6], "72-EDO", "diatonic"),
    "Mode 4 – Tetartos":   _template([12, 10, 8, 12, 10, 8, 12], "72-EDO", "diatonic"),
    "Mode 5 – Plagios A'": _template([10, 8, 12, 12, 10, 8, 12], "72-EDO", "diatonic"),
    "Mode 6 – Plagios B'": _template([6, 20, 4, 12, 6, 20, 4],   "72-EDO", "hard chromatic"),
    "Mode 7 – Barys":      _template([12, 12, 12, 6, 12, 12, 6], "72-EDO", "diatonic"),
    "Mode 8 – Plagios D'": _template([12, 10, 8, 12, 12, 10, 8], "72-EDO", "diatonic"),

    "Greek Major (Matzore)": _template([2, 2, 1, 2, 2, 2, 1], "12-TET"),
    "Greek Minor (Minore)":  _template([2, 1, 2, 2, 1, 3, 1], "12-TET"),
    "Greek Hijaz":           _template([1, 3, 1, 2, 1, 2, 2], "12-TET"),

    "Cypriot Pentachord": _template([3, 3, 4, 4], "24-EDO"),

    "Maqam Rast":     _template([4, 3, 3, 4, 4, 3, 3], "24-EDO"),
    "Maqam Bayati":   _template([3, 3, 4, 4, 2, 4, 4], "24-EDO"),
    "Maqam Hijaz":    _template([2, 6, 2, 4, 2, 4, 4], "24-EDO"),
    "Maqam Nahawand": _template([4, 2, 4, 4, 2, 4, 4], "24-EDO"),
    "Maqam Saba":     _template([3, 3, 2, 4, 4, 4, 4], "24-EDO"),
    "Maqam Kurd":     _template([2, 4, 4, 4, 2, 4, 4], "24-EDO"),
    "Maqam Ajam":     _template([4, 4, 2, 4, 4, 4, 2], "24-EDO"),
    "Maqam Jiharkah": _template([4, 4, 2, 4, 4, 3, 3], "24-EDO"),
}

MODE_INFO: dict[str, dict[str, str]] = {
    "Mode 1 – Protos": {
        "tradition": "Byzantine Chant",
        "alt_name": "Ήχος Πρώτος (First Mode)",
        "description": "Diatonic authentic mode of the Octoechos, built on whole-tone (200¢) "
                        "and three-quarter-tone (167¢) steps. Kyrios (authentic) form of the "
                        "Protos–Plagios A' pair.",
    },
    "Mode 2 – Deuteros": {
        "tradition": "Byzantine Chant",
        "alt_name": "Ήχος Δεύτερος (Second Mode)",
        "description": "Natively in the soft chromatic genus, its 233¢ step is an "
                        "augmented-second leap, giving it a more pungent color than the purely "
                        "diatonic echoi. Kyrios form of the Deuteros–Plagios B' pair.",
    },
    "Mode 3 – Tritos": {
        "tradition": "Byzantine Chant",
        "alt_name": "Ήχος Τρίτος (Third Mode)",
        "description": "Diatonic authentic mode with a 167¢ opening interval, producing a "
                        "bright, direct melodic character. Kyrios form of the Tritos–Barys pair.",
    },
    "Mode 4 – Tetartos": {
        "tradition": "Byzantine Chant",
        "alt_name": "Ήχος Τέταρτος (Fourth Mode)",
        "description": "Diatonic authentic mode with a narrow 133¢ second step, one of the "
                        "most melodically flexible modes of the Octoechos. Kyrios form of the "
                        "Tetartos–Plagios D' pair.",
    },
    "Mode 5 – Plagios A'": {
        "tradition": "Byzantine Chant",
        "alt_name": "Ήχος Πλάγιος Πρώτου (Plagal of the First)",
        "description": "Plagal counterpart of Protos: identical pitch-class scale, but melodic "
                        "movement centers in the lower tetrachord, extending the range below "
                        "the tonic.",
    },
    "Mode 6 – Plagios B'": {
        "tradition": "Byzantine Chant",
        "alt_name": "Ήχος Πλάγιος Δευτέρου (Plagal of the Second)",
        "description": "Natively in the hard chromatic genus, its 333¢ step is a wider "
                        "augmented second than Deuteros's, producing the pungent, Hijaz-like "
                        "color traditionally associated with penitential hymns. Plagal "
                        "counterpart of Deuteros; the melody dwells beneath the finalis.",
    },
    "Mode 7 – Barys": {
        "tradition": "Byzantine Chant",
        "alt_name": "Ήχος Βαρύς (Grave Mode)",
        "description": "Plagal counterpart of Tritos, traditionally named 'Barys' (grave/heavy) "
                        "for its low tessitura, one of only two plagal modes in the Octoechos "
                        "with its own name rather than a numbered 'Plagios' label.",
    },
    "Mode 8 – Plagios D'": {
        "tradition": "Byzantine Chant",
        "alt_name": "Ήχος Πλάγιος Τετάρτου (Plagal of the Fourth)",
        "description": "Plagal counterpart of Tetartos. Its pitch-class content is identical to "
                        "Protos at semitone resolution, so this system reaches it only through "
                        "melodic-range refinement, never chroma alone.",
    },
    "Greek Major (Matzore)": {
        "tradition": "Greek Folk",
        "alt_name": "Ματζόρε",
        "description": "Major scale as used in Greek folk and laïko song, identical in "
                        "interval content to the Western major scale. Very common in "
                        "dance repertoire such as the kalamatianos.",
    },
    "Greek Minor (Minore)": {
        "tradition": "Greek Folk",
        "description": "Common Greek folk/laïko scale resembling the harmonic or natural minor, "
                        "frequently used in rebetiko and demotic song.",
    },
    "Greek Hijaz": {
        "tradition": "Greek Folk",
        "description": "Greek folk adaptation of the Hijaz tetrachord (100¢–400¢ opening), "
                        "widely used in rebetiko and Asia Minor–influenced Greek repertoire.",
    },

    "Cypriot Pentachord": {
        "tradition": "Cypriot Folk",
        "description": "Scale associated with traditional Cypriot folk song, intervallically "
                        "close to the Byzantine diatonic Protos scale, reflecting Cyprus's "
                        "close historical ties to Byzantine musical tradition.",
    },

    "Maqam Rast": {
        "tradition": "Arabic Maqam",
        "description": "One of the foundational maqamat, built on a neutral third (~350¢), "
                        "neither major nor minor, considered the reference scale of the Arabic "
                        "maqam system.",
    },
    "Maqam Bayati": {
        "tradition": "Arabic Maqam",
        "description": "Extremely common maqam built on a neutral second (~150¢) above the "
                        "tonic, a color with no equivalent in 12-TET Western scales.",
    },
    "Maqam Hijaz": {
        "tradition": "Arabic Maqam",
        "description": "Recognizable by its augmented-second interval (100¢–400¢) between the "
                        "2nd and 3rd degrees, the maqam most associated with Middle Eastern "
                        "music in popular perception.",
    },
    "Maqam Nahawand": {
        "tradition": "Arabic Maqam",
        "description": "Arabic maqam closely resembling the Western harmonic minor scale, often "
                        "used as a bridge between Arabic and Western tonal idioms.",
    },
    "Maqam Saba": {
        "tradition": "Arabic Maqam",
        "description": "Features a very narrow interval (100¢) between the 3rd and 4th degrees, "
                        "producing one of the most microtonally striking and melancholic colors "
                        "in the maqam repertoire.",
    },
    "Maqam Kurd": {
        "tradition": "Arabic Maqam",
        "description": "Resembles the Western Phrygian mode, with a semitone (100¢) above the "
                        "tonic; often confused with Maqam Bayati due to a shared lower "
                        "tetrachord shape.",
    },
    "Maqam Ajam": {
        "tradition": "Arabic Maqam",
        "description": "Arabic equivalent of the Western major scale; shares its 12-TET chroma "
                        "profile with Maqam Rast, distinguished only by a non-neutral (400¢) "
                        "third degree.",
    },
    "Maqam Jiharkah": {
        "tradition": "Arabic Maqam",
        "description": "Built on a major-third, augmented-fourth structure, giving it a bright "
                        "color related to but distinct from Maqam Ajam.",
    },
}

TONIC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

KYRIOS_TO_PLAGAL = {
    "Mode 1 – Protos": "Mode 5 – Plagios A'",
}

TRADITION_TEMPLATES: dict[str, list[str]] = {
    "byzantine": [
        "Mode 1 – Protos", "Mode 2 – Deuteros", "Mode 3 – Tritos",
        "Mode 4 – Tetartos", "Mode 5 – Plagios A'", "Mode 6 – Plagios B'",
        "Mode 7 – Barys", "Mode 8 – Plagios D'",
    ],
    "greek": [
        "Greek Major (Matzore)", "Greek Minor (Minore)", "Greek Hijaz",
    ],
    "cypriot": [
        "Cypriot Pentachord",
    ],
    "arabic": [
        "Maqam Rast", "Maqam Bayati", "Maqam Hijaz", "Maqam Nahawand",
        "Maqam Saba", "Maqam Kurd", "Maqam Ajam", "Maqam Jiharkah",
    ],
}

MAX_DURATION        = 240
VOICED_THRESHOLD    = 0.65
MICROTONAL_MIN_VOICED = 0.25
CENTS_PER_BIN       = 10
N_BINS              = 1200 // CENTS_PER_BIN

STEM_ENERGY_THRESHOLD       = 0.005
STEM_ENERGY_THRESHOLD_BASS  = 0.015
STEM_ENERGY_THRESHOLD_DRUMS = 0.020

TRADITION_MELODY_LABEL: dict[str, str] = {
    "byzantine": "Choir / Isocratima",
    "greek":     "Bouzouki / Laouto / Guitar",
    "cypriot":   "Laouto / Violin / Guitar",
    "arabic":    "Oud / Kanun / Ney",
}


_DEMUCS_MODEL = None


def _get_demucs_model():
    """Load htdemucs model once and cache it for the process lifetime."""
    global _DEMUCS_MODEL
    if _DEMUCS_MODEL is None:
        from demucs.pretrained import get_model
        _DEMUCS_MODEL = get_model("htdemucs")
        _DEMUCS_MODEL.eval()
    return _DEMUCS_MODEL


_INSTRUMENT_MODEL: tuple | None = None
_INSTRUMENT_LOAD_FAILED = False

_INSTRUMENT_MODEL_PATH = (
    Path(__file__).resolve().parent / "instrument_classifier" / "models" / "instrument_model.joblib"
)


def _get_instrument_model():
    """ Load the trained instrument classifier once and cache it.
    Returns None if the model file is missing or unloadable, the classifier
    is an enhancement over the RMS heuristic, not a hard dependency, so the
    pipeline degrades to generic melody labels rather than failing the whole
    analysis. The failure is latched so a missing model doesn't re-raise (and
    re-log) on every single upload."""
    global _INSTRUMENT_MODEL, _INSTRUMENT_LOAD_FAILED
    if _INSTRUMENT_MODEL is None and not _INSTRUMENT_LOAD_FAILED:
        try:
            from instrument_classifier.predict import load_model
            _INSTRUMENT_MODEL = load_model(_INSTRUMENT_MODEL_PATH)
        except Exception as e:
            _INSTRUMENT_LOAD_FAILED = True
            print(
                f"[instrument_classifier] model unavailable ({e}), "
                f"falling back to generic melody labels",
                file=sys.stderr, flush=True,
            )
    return _INSTRUMENT_MODEL


def _classify_melody_instrument(
    stems: dict[str, np.ndarray],
    sr: int,
    tradition: str | None,
) -> tuple[str | None, float]:
    """ Name the specific melody instrument (e.g. "Bouzouki") playing in the
    Demucs `other` stem, restricted to instruments valid for `tradition`.
    Returns (label, confidence), or (None, confidence) when the model is
    unavailable, there's no melody stem, or confidence is too low to name
    an instrument honestly, the caller then keeps the generic label."""
    if "other" not in stems:
        return None, 0.0

    bundle = _get_instrument_model()
    if bundle is None:
        return None, 0.0

    model, encoder, feature_set = bundle
    try:
        from instrument_classifier.predict import classify_stem
        name, confidence = classify_stem(
            model, encoder, stems["other"], sr,
            tradition=tradition, feature_set=feature_set,
        )
    except Exception as e:
        print(f"[instrument_classifier] prediction failed ({e})", file=sys.stderr, flush=True)
        return None, 0.0

    return (name.capitalize() if name else None), confidence


def _separate_stems(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """ Run Demucs htdemucs on the audio and return a dict of
    stem_name → mono numpy array resampled to `sr`.

    Stems: 'drums', 'bass', 'other', 'vocals'

    Returns an empty dict if Demucs is not installed or fails, so the
    rest of the pipeline can fall back gracefully to the mixed audio."""
    try:
        import torch
        from demucs.audio import convert_audio
        from demucs.apply import apply_model

        model = _get_demucs_model()

        wav = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

        wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)

        with torch.no_grad():
            sources = apply_model(
                model, wav,
                shifts=1,
                overlap=0.25,
                progress=False,
            )

        stems: dict[str, np.ndarray] = {}
        for i, name in enumerate(model.sources):
            stem_mono = sources[0, i].mean(dim=0).cpu().numpy()
            if model.samplerate != sr:
                stem_mono = librosa.resample(
                    stem_mono, orig_sr=model.samplerate, target_sr=sr
                )
            stem_mono = stem_mono[: int(sr * MAX_DURATION)]
            stems[name] = stem_mono.astype(np.float32)

        return stems

    except Exception:
        return {}


def _detect_instruments(
    stems: dict[str, np.ndarray],
    tradition: str | None,
    melody_instrument: str | None = None,
) -> list[str]:
    """ Decide which instrument categories are audible based on per-stem RMS.
    Returns a list of human-readable labels shown in the UI.

    `melody_instrument`, when given, is the specific instrument named by the
    trained classifier (e.g. "Bouzouki") and replaces the generic
    tradition-wide melody label for the `other` stem. When it is None, model
    unavailable, or confidence too low to name an instrument honestly, the
    generic label is used instead."""
    if not stems:
        return []

    rms = {name: float(np.sqrt(np.mean(stem ** 2))) for name, stem in stems.items()}
    present = {name for name, energy in rms.items()
               if name not in ("drums", "bass") and energy >= STEM_ENERGY_THRESHOLD}

    labels: list[str] = []

    if "vocals" in present:
        labels.append("Voice")

    if "other" in present:
        labels.append(
            melody_instrument
            or TRADITION_MELODY_LABEL.get(tradition or "", "Melody instrument")
        )

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


def _keep_tail(audio: np.ndarray, sr: int) -> np.ndarray:
    """ Enforce MAX_DURATION by keeping the END of the clip, not the start.
    Every tonic-resolution mechanism in this pipeline (end-weighted chroma
    mean, finalis anchoring for kyrios/plagal and Arabic neutral-interval
    refinement, and tonic-by-finalis) depends on the analyzed clip's ending
    being the piece's ACTUAL ending, Byzantine/Arabic melodies reliably
    resolve to their tonic there. Truncating from the start instead would
    silently discard that ending for any recording longer than
    MAX_DURATION, which is the common case for real chant/taksim
    recordings (often several minutes), breaking all of the above."""
    max_samples = int(sr * MAX_DURATION)
    if audio.shape[-1] <= max_samples:
        return audio
    return audio[-max_samples:]


def _load_audio(file_bytes: bytes) -> tuple[np.ndarray, int]:
    """ Load audio from raw bytes.
    Tries soundfile (fast, strict) first, falls back to pydub (ffmpeg-backed,
    handles WebM, WebA, OGG, AAC, and other formats soundfile rejects)."""
    TARGET_SR = 22050
    try:
        audio, sr = librosa.load(io.BytesIO(file_bytes), sr=TARGET_SR, mono=True)
        return _keep_tail(audio, sr), sr
    except Exception:
        pass

    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(file_bytes))
        seg = seg.set_channels(1).set_frame_rate(TARGET_SR)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= float(2 ** (seg.sample_width * 8 - 1))
        return _keep_tail(samples, TARGET_SR), TARGET_SR
    except Exception as e:
        raise ValueError(
            f"Could not decode audio. Make sure ffmpeg is installed for non-MP3/WAV formats. ({e})"
        )


def analyze_audio(file_bytes: bytes, filename: str, tradition: str | None = None) -> dict:
    """ Run the whole analysis for one uploaded recording.
    Separates the audio, detects the mode and tonic, tracks the pitch, names
    the instrument and builds the score, then returns everything the web
    interface needs as a single dictionary."""
    audio, sr = _load_audio(file_bytes)
    duration = float(librosa.get_duration(y=audio, sr=sr))

    stems = _separate_stems(audio, sr)

    melody_instrument, melody_confidence = _classify_melody_instrument(stems, sr, tradition)
    detected_instruments = _detect_instruments(stems, tradition, melody_instrument)

    harmonic, _ = librosa.effects.hpss(audio)

    if tradition and tradition in TRADITION_TEMPLATES:
        active_templates = {k: _CHROMA_TEMPLATES[k] for k in TRADITION_TEMPLATES[tradition]}
    else:
        active_templates = _CHROMA_TEMPLATES
    tonic_name, mode, mode_confidence, tie_group, all_scores = _detect_mode_chroma(harmonic, sr, active_templates)
    tie_group_size = len(tie_group)

    # The voice carries the melody whenever there is one, and the instruments
    # follow it, so the transcription is taken from the vocal stem whenever
    # singing is present at all, not merely when it happens to be the loudest
    # part of the mix. Instrumental stems are used only for pieces with no
    # singing in them.
    if stems:
        vocal_rms = float(np.sqrt(np.mean(stems["vocals"] ** 2))) if "vocals" in stems else 0.0
        other_rms = float(np.sqrt(np.mean(stems["other"]  ** 2))) if "other"  in stems else 0.0
        if vocal_rms >= STEM_ENERGY_THRESHOLD:
            pitch_source = stems["vocals"]
            transcribed_from_voice = True
        elif other_rms >= STEM_ENERGY_THRESHOLD:
            pitch_source = stems["other"]
            transcribed_from_voice = False
        else:
            pitch_source = harmonic
            transcribed_from_voice = False
    else:
        pitch_source = harmonic
        transcribed_from_voice = False
    f0, voiced_flag, voiced_probs = _extract_pitch(pitch_source, sr)
    confident = voiced_flag & ~np.isnan(f0) & (voiced_probs > VOICED_THRESHOLD)
    confident_f0 = f0[confident]

    _CREPE_STEP_S   = 0.01
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

    if confident_f0.size >= 10:
        midi_cents      = 1200.0 * np.log2(confident_f0 / 440.0) + 6900.0
        mean_pitch_hz   = round(float(np.median(confident_f0)), 1)
        pitch_range_cents = _robust_pitch_range(midi_cents)
        deviations      = _microtonal_deviations(midi_cents) if microtonal_reliable else []

        # Byzantine only, and deliberately so. Extending this to Greek folk
        # was tried and withdrawn: chant holds its finalis for a long closing
        # note, but a strophic folk song's last sung seconds simply dwell on
        # whatever the melody dwells on, which is as often the third as the
        # tonic. Measured on two recordings with the answer known, "Μήλο μου
        # κόκκινο", where the ending gives the right tonic and chroma does
        # not, and "Σαμιώτισσα", where the reverse holds, the ending's
        # leading pitch class wins by a ratio of 1.48 and 1.33 respectively,
        # and the chroma gap is 0.036 against 0.055. No threshold separates
        # the two cases; one fitted to them would be memorising them. Greek
        # tonic detection therefore stays with chroma alone, and confusing a
        # tonic with its own dominant remains a known limitation.
        if tradition in (None, "byzantine"):
            tonic_name, mode = _resolve_tonic_by_finalis(all_scores, tonic_name, mode, midi_cents)

        tonic_idx = TONIC_NAMES.index(tonic_name)
        if tradition in (None, "byzantine") and mode in KYRIOS_TO_PLAGAL:
            own_score     = all_scores.get((tonic_idx, mode))
            partner_score = all_scores.get((tonic_idx, KYRIOS_TO_PLAGAL[mode]))
            if (
                own_score is not None and partner_score is not None
                and abs(own_score - partner_score) <= _MODE_TIE_EPSILON
            ):
                mode = _refine_kyrios_plagal(mode, tonic_name, midi_cents)

        if tradition == "arabic":
            mode, tonic_name = _refine_arabic_maqam(mode, tonic_name, midi_cents)
    else:
        mean_pitch_hz     = 0.0
        pitch_range_cents = 0
        deviations        = []
        microtonal_reliable = False

    onset_times  = librosa.onset.onset_detect(
        y=pitch_source, sr=sr, units="time", backtrack=True, delta=0.2,
    )
    onset_frames = {int(round(t / _CREPE_STEP_S)) for t in onset_times}
    notation = _drop_out_of_range_notes(_segment_notes(
        f0, confident, tonic_name, mode, step_s=_CREPE_STEP_S, onset_frames=onset_frames,
    ))

    # Label on the staff: what was actually transcribed. The voice takes
    # priority as the source, so it is named as such; otherwise the staff
    # carries the instrument the classifier identified, and falls back to a
    # neutral word only when neither could be established.
    if transcribed_from_voice:
        staff_label = "Φωνή"
    elif melody_instrument:
        staff_label = melody_instrument
    else:
        staff_label = "Μελωδία"

    score = {}
    try:
        import notation as notation_mod
        # The pulse is measured on the full mix, not on the isolated melodic
        # stem the notes came from. Separation strips out the percussion,
        # which is exactly where the beat lives, so measuring the stem asks
        # about rhythm in the one place it was removed from.
        # Quarter-tone accidentals are written only where they carry meaning.
        # The Greek modes are defined in 12-TET, so every degree of them is a
        # piano key and any microtone in the score is expressive intonation
        # rather than structure, it belongs in the deviation figures, not on
        # the staff, where it prints symbols musicians do not read. The
        # Byzantine and Arabic modes are defined in 72- and 24-EDO, where the
        # neutral intervals ARE the mode, and rounding them away would erase
        # what the analysis exists to find. Keying this on the tuning basis
        # rather than the mode keeps it right even when the mode within a
        # tradition is detected wrongly, since a tradition's modes share one
        # basis.
        semitone_only = SCALE_TEMPLATES.get(mode, {}).get("basis") == "12-TET"
        score = notation_mod.notes_to_musicxml(
            notation, audio, sr,
            mode=mode, tonic=tonic_name,
            instrument_name=staff_label,
            filename=filename,
            semitone_only=semitone_only,
        )
    except Exception as e:
        print(f"[notation] score generation failed ({e})", file=sys.stderr, flush=True)

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
        "mode_info":            MODE_INFO.get(mode),
        "notation":             notation,
        "genus":                SCALE_TEMPLATES.get(mode, {}).get("genus"),
        "tie_group_size":       tie_group_size,
        "melody_instrument":    melody_instrument,
        "melody_instrument_confidence": round(melody_confidence, 2),
        "musicxml":             score.get("musicxml"),
        "tempo_bpm":            score.get("tempo_bpm"),
        "pulse_strength":       score.get("pulse_strength"),
        "metrical":             score.get("metrical"),
    }


def _extract_pitch(audio: np.ndarray, sr: int):
    """ CREPE CNN pitch tracker, cent-level accuracy on monophonic audio.
    model_capacity='medium' balances speed and accuracy well for thesis use.
    Outputs match pYIN's (f0, voiced_flag, voiced_probs) interface."""
    import crepe
    _, frequency, confidence, _ = crepe.predict(
        audio,
        sr,
        viterbi=True,
        model_capacity='medium',
        step_size=10,
        verbose=0,
    )
    voiced_flag = confidence > VOICED_THRESHOLD
    f0 = frequency.astype(np.float64)
    f0[~voiced_flag] = np.nan
    return f0, voiced_flag, confidence.astype(np.float64)


_MODE_TIE_EPSILON = 1e-9


def _detect_mode_chroma(
    audio: np.ndarray,
    sr: int,
    templates: dict[str, np.ndarray] | None = None,
) -> tuple[str, str, float, frozenset[tuple[int, str]], dict[tuple[int, str], float]]:
    """ Joint (tonic, mode) detection, exhaustive search over all 12 tonics × all
    mode templates. Avoids the cascade failure of detect-tonic-first approaches.
    Chroma is time-weighted: the final 25 % of the recording gets 2× weight
    because the Byzantine finalis always appears at cadences, not necessarily
    as the most frequent pitch class overall. In recitative liturgical chant
    this weighting can still be insufficient, a "reciting tone" sung for
    most of the text can dominate the histogram by sheer duration even
    within the weighted final quarter, out-scoring the true tonic's family.
    See _resolve_tonic_by_finalis, which trusts the literal final note over
    this whole-clip shape when they disagree.
    Also returns the full all_scores table (for _resolve_tonic_by_finalis to
    look up the best mode at a tonic outside the winning tie group) and the
    tie group (every (tonic_idx, mode_name) within _MODE_TIE_EPSILON of the
    winning score), so
    callers can tell an unambiguous match from one settled by a tiebreak
    (e.g. Protos/Plagios A', which are identical at any chroma resolution,
    see KYRIOS_TO_PLAGAL)."""
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=36, n_chroma=36)

    n_frames   = chroma.shape[1]
    weights    = np.ones(n_frames)
    end_start  = int(n_frames * 0.75)
    weights[end_start:] = 2.0
    weights   /= weights.sum()
    chroma_mean = (chroma * weights[np.newaxis, :]).sum(axis=1)
    if chroma_mean.sum() > 0:
        chroma_mean /= chroma_mean.sum()

    active = templates if templates is not None else _CHROMA_TEMPLATES
    all_scores: dict[tuple[int, str], float] = {}
    for tonic_idx in range(12):
        norm = np.roll(chroma_mean, -3 * tonic_idx).astype(float)
        if norm.sum() > 0:
            norm /= norm.sum()
        for mode_name, template in active.items():
            all_scores[(tonic_idx, mode_name)] = _cosine_sim(norm, template)

    best_key                  = max(all_scores, key=all_scores.__getitem__)
    best_tonic_idx, best_mode = best_key
    best_score                = all_scores[best_key]

    tie_group = frozenset(
        key for key, score in all_scores.items()
        if best_score - score <= _MODE_TIE_EPSILON
    )

    scores_list = list(all_scores.values())
    min_s, max_s = min(scores_list), max(scores_list)
    spread = max_s - min_s if max_s > min_s else 1.0
    outside_tie = [s for k, s in all_scores.items() if k not in tie_group]
    second_best = max(outside_tie, default=min_s)
    margin = best_score - second_best
    conf   = float(np.clip(0.50 + 0.48 * margin / spread, 0.50, 0.98))

    return TONIC_NAMES[best_tonic_idx], best_mode, round(conf, 2), tie_group, all_scores


def _build_chroma_template(cents: list[int], n_bins: int = 36) -> np.ndarray:
    """ Convert cent intervals → n_bins-bin chroma vector (33.3¢/bin at the
    default 36) with Gaussian blur scaled in cents, not bins: sigma=1.2 bins
    at 36-bin resolution ≈ the same ~13¢ smoothing as the old sigma=0.4 at
    12-bin resolution. Leaving sigma unscaled when the bin count changed
    would re-merge neighbouring bins and undo the resolution gain."""
    chroma = np.zeros(n_bins)
    bin_width = 1200.0 / n_bins
    for c in cents:
        idx = int(round(c / bin_width)) % n_bins
        chroma[idx] += 1.0
    chroma = gaussian_filter1d(chroma, sigma=1.2, mode="wrap")
    return chroma / chroma.sum() if chroma.sum() > 0 else chroma


_POOLED_MODE_NAMES = sorted({name for names in TRADITION_TEMPLATES.values() for name in names})
_CHROMA_TEMPLATES: dict[str, np.ndarray] = {
    name: _build_chroma_template(SCALE_TEMPLATES[name]["cents"])
    for name in _POOLED_MODE_NAMES
}


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """ Measure how similar two pitch profiles are, ignoring loudness.
    Returns a value near 1 when the two have the same shape and near 0 when
    they are unrelated, which is how an observed recording is matched against
    each candidate scale template."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def _resolve_tonic_by_finalis(
    all_scores: dict[tuple[int, str], float],
    tonic_name: str,
    mode: str,
    midi_cents: np.ndarray,
) -> tuple[str, str]:
    """ Prefer the tonic the melody actually ends on over the whole-clip chroma
    match, whenever they disagree, not just when the chroma winner is part
    of an exact tie.
    Two separate reasons chroma-only tonic detection can be wrong even
    without a mathematical tie:
    1)  The diatonic genus is a rotational system (Protos/Plagios A' at one
        tonic are chroma-identical to Tetartos a 4th above and Plagios D' a
        2nd below; Tritos/Barys tie a 4th apart), a real, imperfect
        recording rarely lands on an EXACT tie between family members, but
        can easily score a nearby family highest by a small margin due to
        noise, without the true family entering contention at all.
    2)  Recitative liturgical chant can spend most of its duration on a
        "reciting tone" a step or so above the true tonic, which dominates
        the whole-clip chroma histogram by sheer duration even after
        end-weighting, outscoring the true tonic's family entirely rather
        than merely tying it.
    Byzantine chant reliably resolves to its tonic at the end regardless of
    either effect, a stronger, more theoretically grounded signal than the
    overall chroma shape for TONIC identification specifically. This anchors
    the tonic from the last 15% of confident CREPE frames (mirroring
    _refine_kyrios_plagal / _refine_arabic_maqam) and, if it disagrees with
    the chroma winner, switches to whichever mode scores best at that
    finalis-confirmed tonic, chroma is still trusted to pick the mode/genus
    once the tonic itself is settled."""
    if midi_cents.size < 20:
        return tonic_name, mode

    n = midi_cents.size
    ending = midi_cents[int(n * 0.85):]
    if ending.size < 5:
        return tonic_name, mode

    # The most-held pitch class of the ending, not its median. A median of
    # pitch is only meaningful when the ending sits on one note: where the
    # closing phrase oscillates between two, it lands in the gap between them
    # and names a note nobody sang. Counting frames per pitch class instead
    # picks whichever note was actually dwelt on longest, and rounding each
    # frame individually keeps a recording tuned slightly off A440 on the
    # right degree.
    pitch_classes = np.round(ending / 100.0).astype(int) % 12
    finalis_tonic_idx = int(np.argmax(np.bincount(pitch_classes, minlength=12)))

    original_tonic_idx = TONIC_NAMES.index(tonic_name)
    if finalis_tonic_idx == original_tonic_idx:
        return tonic_name, mode

    candidates = {m: s for (t, m), s in all_scores.items() if t == finalis_tonic_idx}
    if not candidates:
        return tonic_name, mode
    best_mode_at_finalis = max(candidates, key=candidates.get)

    return TONIC_NAMES[finalis_tonic_idx], best_mode_at_finalis


def _refine_kyrios_plagal(
    mode: str,
    tonic_name: str,
    midi_cents: np.ndarray,
) -> str:
    """ Byzantine kyrios/plagios disambiguation using absolute pitch.
    Root cause of the pitch-class approach failing: plagal modes descend
    below the tonic (lower tetrachord).  Those sub-tonic notes land at
    relative_pc ≈ 1000–1100 ¢, overlapping the 7th-degree window and
    pulling the median above the 1017 ¢ midpoint even when the true 7th
    is flat (967 ¢).
    Fix: work in absolute pitch space.
    1)  Byzantine chant always ends on the finalis (= tonic).  Use the
        last 15 % of voiced frames to pin the tonic's absolute pitch.
    2)  Count frames in [tonic+917, tonic+1017) → plagios 7th window
        Count frames in [tonic+1017, tonic+1117) → kyrios 7th window
        (non-overlapping split at the 1017 ¢ midpoint)
    3)  Whichever window has more frames wins.
    Sub-tonic notes are at tonic − N ¢ in absolute pitch, so they never
    touch either window.  Octave confusion is eliminated."""
    if mode not in KYRIOS_TO_PLAGAL or midi_cents.size < 20:
        return mode

    tonic_idx   = TONIC_NAMES.index(tonic_name)
    expected_pc = tonic_idx * 100.0

    n         = midi_cents.size
    ending    = midi_cents[int(n * 0.85):]
    tonic_abs = float(np.median(ending)) if ending.size >= 5 else float(np.median(midi_cents))

    actual_pc = tonic_abs % 1200.0
    pc_err    = min(abs(actual_pc - expected_pc), 1200.0 - abs(actual_pc - expected_pc))

    if pc_err > 200:
        pitch_classes = midi_cents % 1200.0
        relative_pc   = (pitch_classes - expected_pc) % 1200.0
        near_seventh  = (relative_pc >= 900) & (relative_pc <= 1100)
        if near_seventh.sum() < 5:
            return mode
        seventh_median = float(np.median(relative_pc[near_seventh]))
        return KYRIOS_TO_PLAGAL[mode] if seventh_median < 1017.0 else mode

    count_p = count_k = 0
    for oct_shift in (-1200.0, 0.0, 1200.0):
        base     = tonic_abs + oct_shift
        count_p += int(np.sum((midi_cents >= base + 917.0) & (midi_cents < base + 1017.0)))
        count_k += int(np.sum((midi_cents >= base + 1017.0) & (midi_cents < base + 1117.0)))

    if count_p + count_k < 5:
        return mode

    if count_p > count_k:
        return KYRIOS_TO_PLAGAL[mode]
    return mode


_ARABIC_REFINE: dict[str, tuple[str, int, int, int, int]] = {
    "Maqam Kurd":  ("Maqam Bayati", 50,  240, 100, 150),

    "Maqam Ajam":  ("Maqam Rast",   290, 470, 400, 350),
}


def _refine_arabic_maqam(
    mode: str,
    tonic_name: str,
    midi_cents: np.ndarray,
    min_frames: int = 8,
) -> tuple[str, str]:
    """ Resolve Arabic maqam pairs that are ambiguous at 12-TET chroma resolution
    because their distinguishing degree is a neutral (quarter-tone) interval.
    Supported pairs:
        Maqam Kurd  (100 ¢ 2nd)  ↔  Maqam Bayati (150 ¢ neutral 2nd)
        Maqam Ajam  (400 ¢ 3rd)  ↔  Maqam Rast   (350 ¢ neutral 3rd)

    Strategy (mirrors _refine_kyrios_plagal):
    1)  Derive the actual tonic from the ending 15 % of CREPE frames, the
        finalis of a taksim nearly always appears there, and CREPE gives
        better pitch-class accuracy than chroma for short passages.
    2)  Compute every frame's interval relative to that tonic (pitch-class,
        mod 1200 ¢).
    3)  Gather frames that fall in the ambiguous degree window.
    4)  Check whether the median is closer to current_deg or alt_deg.
    Returns (mode, tonic_name), tonic may also be corrected when the
    chroma search was off by a semitone or two."""
    if mode not in _ARABIC_REFINE or midi_cents.size < 20:
        return mode, tonic_name

    alt_mode, lo, hi, current_deg, alt_deg = _ARABIC_REFINE[mode]

    n      = midi_cents.size
    ending = midi_cents[int(n * 0.85):]
    tonic_abs     = float(np.median(ending)) if ending.size >= 5 else float(np.median(midi_cents))
    tonic_pc_raw  = tonic_abs % 1200.0
    tonic_semitone = int(round(tonic_pc_raw / 100.0)) % 12
    refined_tonic = TONIC_NAMES[tonic_semitone]
    tonic_pc      = float(tonic_semitone * 100)

    rel = (midi_cents % 1200.0 - tonic_pc) % 1200.0

    in_window = rel[(rel >= lo) & (rel <= hi)]
    if in_window.size < min_frames:
        print(
            f"[arabic_refine] {mode}/{tonic_name}: only {in_window.size} frames "
            f"in [{lo},{hi}] ¢, skipping (tonic_refined={refined_tonic})",
            file=sys.stderr, flush=True,
        )
        return mode, refined_tonic

    median_deg = float(np.median(in_window))

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


_NOTE_JUMP_CENTS = 60.0


def _nearest_degree(relative_cents: float, template: list[int]) -> tuple[int, float]:
    """ Find the scale degree (0-indexed into `template`) closest to
    `relative_cents` (pitch-class cents relative to the tonic, in [0, 1200)).
    Checks the template value plus/minus one octave to handle wraparound
    near 0¢/1200¢.
    Returns (degree_index, signed_deviation_cents) where deviation is
    relative_cents minus the matched candidate (positive = sharp of the
    degree, negative = flat of it)."""
    best_idx, best_dev, best_dist = 0, 0.0, float("inf")
    for idx, t in enumerate(template):
        for cand in (t, t + 1200.0, t - 1200.0):
            dist = abs(relative_cents - cand)
            if dist < best_dist:
                best_dist, best_idx, best_dev = dist, idx, relative_cents - cand
    return best_idx, best_dev


def _accidental_symbol(deviation_cents: float) -> str:
    """ Classify a degree's deviation into a notation-relevant bucket.
    Thresholds: <=25¢ reads as in-tune, <=90¢ as a quarter-tone (neutral
    interval) shift, beyond that as a full semitone-scale accidental."""
    d = abs(deviation_cents)
    if d <= 25:
        return "natural"
    if d <= 90:
        return "quarter-sharp" if deviation_cents > 0 else "quarter-flat"
    return "sharp" if deviation_cents > 0 else "flat"


def _segment_notes(
    f0: np.ndarray,
    confident: np.ndarray,
    tonic_name: str,
    mode: str,
    step_s: float = 0.01,
    min_note_frames: int = 5,
    onset_frames: set[int] | None = None,
) -> list[dict]:
    """ Segment the confident CREPE pitch contour into discrete notes.
    Within each contiguous run of confident frames, splits at cent jumps
    larger than `_NOTE_JUMP_CENTS` (a new note) AND at any given
    `onset_frames` (attack transients detected independently of pitch,
    catches a re-articulated note at the same pitch, which a pitch-jump
    split alone would miss). Discards segments shorter than
    `min_note_frames` (spurious blips), then for each remaining segment
    computes the median pitch-class relative to the tonic and snaps it to
    the nearest degree of the mode's scale template."""
    template_entry = SCALE_TEMPLATES.get(mode)
    if template_entry is None or f0.size == 0:
        return []
    template = template_entry["cents"]

    tonic_idx = TONIC_NAMES.index(tonic_name)
    tonic_pc  = tonic_idx * 100.0
    n         = f0.size
    onsets    = onset_frames or set()
    notes: list[dict] = []

    i = 0
    while i < n:
        if not confident[i]:
            i += 1
            continue

        j = i
        cents_run = []
        while j < n and confident[j]:
            cents_run.append(1200.0 * np.log2(f0[j] / 440.0) + 6900.0)
            j += 1
        cents_run = np.array(cents_run)

        pitch_jumps  = np.where(np.abs(np.diff(cents_run)) > _NOTE_JUMP_CENTS)[0] + 1
        onset_splits = [k - i for k in onsets if i < k < j]
        bounds = sorted(set([0] + list(pitch_jumps) + onset_splits + [len(cents_run)]))

        for k in range(len(bounds) - 1):
            s, e = bounds[k], bounds[k + 1]
            if e - s < min_note_frames:
                continue
            median_cents   = float(np.median(cents_run[s:e]))
            relative_cents = (median_cents - tonic_pc) % 1200.0
            degree_idx, deviation = _nearest_degree(relative_cents, template)

            notes.append({
                "start_s":          round((i + s) * step_s, 3),
                "end_s":            round((i + e) * step_s, 3),
                "degree":           degree_idx + 1,
                "cents_from_tonic": round(relative_cents, 1),
                "deviation_cents":  round(deviation, 1),
                "accidental":       _accidental_symbol(deviation),
                "midi_cents":       round(median_cents, 1),
            })

        i = j

    return notes


# 1400 cents either side of the melody's centre, so 28 semitones in total.
# A single melodic line, sung or played, does not span more than that inside
# one excerpt; anything further out is a tracking artefact rather than music.
_MELODIC_HALF_RANGE_CENTS = 1400.0


def _drop_out_of_range_notes(notes: list[dict]) -> list[dict]:
    """ Remove notes that lie far outside the melody's own pitch range.
    CREPE sometimes locks onto a subharmonic, most often where another
    instrument bleeds into the separated stem, producing notes one or two
    octaves below everything around them. They are discarded rather than
    transposed back: an octave shift cannot be pinned down unambiguously, and
    guessing one would write a pitch nobody actually performed. The centre is
    a median, so it survives a contaminated minority of notes."""
    if len(notes) < 8:
        return notes

    centre = float(np.median([n["midi_cents"] for n in notes]))
    return [
        n for n in notes
        if abs(n["midi_cents"] - centre) <= _MELODIC_HALF_RANGE_CENTS
    ]


def _microtonal_deviations(midi_cents: np.ndarray) -> list[str]:
    """Summarise how far the sung pitches sit from the piano keys.
    Every detected pitch is compared with the nearest equal-tempered note and
    the differences are reported at five percentiles. A distribution centred
    away from zero is the signature of the microtonal intervals this project
    exists to detect."""
    nearest = np.round(midi_cents / 100.0) * 100.0
    devs    = midi_cents - nearest
    result  = []
    for pct in [10, 25, 50, 75, 90]:
        d    = float(np.percentile(devs, pct))
        sign = "+" if d >= 0 else ""
        result.append(f"p{pct}: {sign}{round(d)}¢")
    return result


def _robust_pitch_range(midi_cents: np.ndarray) -> int:
    """IQR-based range, ignores octave-error outliers from pYIN."""
    p5  = np.percentile(midi_cents, 5)
    p95 = np.percentile(midi_cents, 95)
    return int(p95 - p5)


def _empty_result(filename: str, duration: float, size_bytes: int) -> dict:
    """ Build the response used when no pitch could be detected at all.
    Keeps every field the interface expects but fills it with neutral values,
    so a silent or unusable upload reports honestly instead of failing."""
    return {
        "filename":             filename,
        "file_size_kb":         round(size_bytes / 1024, 1),
        "detected_mode":        "Undetected",
        "tonic":                "-",
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
        "pitch_times":          [],
        "pitch_contour":        [],
        "tonic_hz":             None,
        "mode_info":            None,
        "notation":             [],
        "genus":                None,
        "tie_group_size":       0,
        "melody_instrument":    None,
        "melody_instrument_confidence": 0.0,
        "musicxml":             None,
        "tempo_bpm":            None,
        "pulse_strength":       None,
        "metrical":             None,
    }
