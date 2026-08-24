"""
Tradition-aware inference for the trained instrument classifier.

The model itself is trained across all instruments regardless of tradition
(see train.py) — restriction happens at PREDICTION time, not training time,
using the tradition the user already selected in the app before uploading.
This mirrors TRADITION_TEMPLATES in analysis.py, which restricts the mode
search the same way: a Greek Folk upload can never be classified as "oud"
even though the model technically knows what an oud sounds like, because
oud simply isn't a candidate for that tradition.
"""
import numpy as np

# Feature extraction MUST be the exact same code that produced the training
# features — a mismatch silently yields garbage predictions rather than an
# error. prepare_dataset stays the single source of truth for it.
# Dual import form so this module works both as part of the package (imported
# by analysis.py) and as a plain script run from inside this directory
# (how train.py is invoked).
try:
    from .prepare_dataset import _extract_features
except ImportError:
    from prepare_dataset import _extract_features

# Which traditions each instrument is used in. Kept in sync with
# TRADITION_MELODY_LABEL / the tradition scoping in analysis.py.
TRADITION_INSTRUMENTS: dict[str, set[str]] = {
    "greek":   {"bouzouki", "laouto", "violin", "clarinet", "santouri", "guitar"},
    "cypriot": {"laouto", "violin", "pithkiavlin", "guitar"},
    "arabic":  {"oud", "ney", "kanun", "violin"},
}


def instruments_for_tradition(tradition: str) -> set[str]:
    """The exact set of instruments valid for one known tradition."""
    return set(TRADITION_INSTRUMENTS.get(tradition, set()))


def allowed_candidates_for_instrument(instrument: str) -> set[str]:
    """
    Every instrument that shares at least one tradition with `instrument` —
    the union across all traditions it appears in.

    Used only for OFFLINE EVALUATION (train.py), where the specific
    tradition a given test clip was recorded under isn't tracked in the
    manifest, so the true instrument's full set of possible traditions is
    the best available approximation. Real predictions should use
    instruments_for_tradition() instead, since the deployed app always
    knows the user's selected tradition exactly — no approximation needed.
    """
    allowed = {instrument}
    for members in TRADITION_INSTRUMENTS.values():
        if instrument in members:
            allowed |= members
    return allowed


def predict_instrument(
    model,
    encoder,
    y: np.ndarray,
    sr: int,
    tradition: str | None = None,
) -> tuple[str, float]:
    """
    Predict which instrument is playing in one audio clip.

    If `tradition` is given, every instrument not valid for that tradition
    is excluded before picking the winner, and the returned confidence is
    renormalized to be a genuine probability among just the remaining
    candidates — not the model's raw (and therefore misleadingly diluted-
    by-irrelevant-options) unrestricted probability.

    Returns (instrument_name, confidence).
    """
    features = _extract_features(y, sr).reshape(1, -1)
    proba = model.predict_proba(features)[0]
    class_names = list(encoder.classes_)

    if tradition is not None:
        allowed = instruments_for_tradition(tradition)
        mask = np.array([name in allowed for name in class_names])
        if mask.any():
            proba = np.where(mask, proba, 0.0)
            total = proba.sum()
            if total > 0:
                proba = proba / total

    best_idx = int(np.argmax(proba))
    return class_names[best_idx], float(proba[best_idx])


# Inference-time constants. CLIP_* must match prepare_dataset's training
# clip geometry — the model only ever saw 4 s windows, so it must only ever
# be asked about 4 s windows.
_CLIP_S = 4.0
_HOP_S = 2.0
_SILENCE_RMS = 0.01

# Below this aggregated confidence the caller falls back to a generic label
# rather than naming a specific instrument.
#
# Calibrated on WHOLE-FILE predictions (the actual deployment case), not on
# single clips. That distinction matters: averaging probability vectors over
# ~40 windows smooths toward the mean, so aggregated confidence runs markedly
# LOWER than single-clip confidence — a lone clip can hit 0.9 by luck, the
# average across a whole recording rarely does unless the model is
# consistently sure. A threshold picked from single-clip numbers (where
# correct predictions had median 0.71) is far too aggressive here.
#
# Measured over the 10 held-out recordings, tradition-restricted:
#   0.50 -> names 3/10 files, 3 correct   (100 % precision, poor coverage)
#   0.40 -> names 6/10 files, 5 correct   ( 83 % precision, good coverage)
#   0.00 -> names 10/10 files, 6 correct  ( 60 % precision)
# 0.40 is the best precision/coverage trade for a user-facing app: naming an
# instrument only when reasonably sure, without silently refusing on most
# uploads. Caveat: n=10 files, so this is a coarse operating point, not a
# finely-tuned one.
MIN_CONFIDENCE = 0.40


def load_model(model_path):
    """Load a trained bundle; returns (model, label_encoder)."""
    import joblib
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["label_encoder"]


def classify_stem(
    model,
    encoder,
    y: np.ndarray,
    sr: int,
    tradition: str | None = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[str | None, float]:
    """
    Classify a full-length melody stem by sliding the trained 4 s window
    across it and soft-voting the results.

    Averaging the probability VECTORS (rather than taking a majority vote on
    each window's argmax) keeps the strength of each window's opinion: a run
    of barely-decided windows shouldn't outvote a few highly confident ones.
    Near-silent windows are skipped using the same RMS gate as training, so
    rests and gaps don't dilute the average.

    Returns (instrument_name, confidence), or (None, confidence) when the
    aggregated confidence falls below `min_confidence` — the caller should
    then fall back to a generic label instead of naming an instrument.
    """
    clip_len = int(_CLIP_S * sr)
    hop_len = int(_HOP_S * sr)
    class_names = list(encoder.classes_)

    if y.size < clip_len:
        return None, 0.0

    windows = []
    start = 0
    while start + clip_len <= y.size:
        clip = y[start:start + clip_len]
        if float(np.sqrt(np.mean(clip ** 2))) >= _SILENCE_RMS:
            windows.append(_extract_features(clip, sr))
        start += hop_len

    if not windows:
        return None, 0.0

    proba = model.predict_proba(np.stack(windows)).mean(axis=0)

    if tradition is not None:
        allowed = instruments_for_tradition(tradition)
        mask = np.array([name in allowed for name in class_names])
        if mask.any():
            proba = np.where(mask, proba, 0.0)
            total = proba.sum()
            if total > 0:
                proba = proba / total

    best_idx = int(np.argmax(proba))
    confidence = float(proba[best_idx])
    if confidence < min_confidence:
        return None, confidence
    return class_names[best_idx], confidence
