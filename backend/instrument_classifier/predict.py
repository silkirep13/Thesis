""" Tradition-aware inference for the trained instrument classifier.
The model is trained on all instruments at once, and the restriction to a
single tradition happens at prediction time using the tradition the user
selected before uploading. A Greek folk upload can therefore never come back
as "oud", even though the model knows what an oud sounds like."""
import numpy as np

try:
    from .prepare_dataset import _extract_features, FEATURE_EXTRACTORS
except ImportError:
    from prepare_dataset import _extract_features, FEATURE_EXTRACTORS

TRADITION_INSTRUMENTS: dict[str, set[str]] = {
    "greek":   {"bouzouki", "laouto", "violin", "clarinet", "santouri", "guitar"},
    "cypriot": {"laouto", "violin", "pithkiavlin", "guitar"},
    "arabic":  {"oud", "ney", "kanun", "violin"},
}

_CLIP_S = 4.0
_HOP_S = 2.0
_SILENCE_RMS = 0.01

MIN_CONFIDENCE = 0.50

def instruments_for_tradition(tradition: str) -> set[str]:
    """ Return the instruments that are valid for one tradition.
    This is the set used for real predictions, since the app always knows
    which tradition the user picked."""
    return set(TRADITION_INSTRUMENTS.get(tradition, set()))


def allowed_candidates_for_instrument(instrument: str) -> set[str]:
    """ Return every instrument sharing at least one tradition with this one.
    Used only for offline evaluation, where the tradition of a test clip is
    not recorded and the union of the true instrument's traditions is the
    closest available approximation. This is also why violin gains nothing
    from the restriction: it belongs to all three traditions, so its union is
    nearly every instrument."""
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
    """ Predict the instrument playing in a single short clip.
    When a tradition is given, instruments outside it are zeroed out before
    the winner is chosen and the remaining probabilities are rescaled, so the
    returned confidence stays a genuine probability among real candidates.
    Returns the instrument name and that confidence."""
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


def load_model(model_path):
    """ Load a saved model bundle from disk.
    Returns the classifier, the label encoder and the name of the feature set
    the weights were trained with, so inference can extract features the same
    way and not silently mismatch."""
    import joblib
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["label_encoder"], bundle.get("feature_set", "handcrafted")


def classify_stem(
    model,
    encoder,
    y: np.ndarray,
    sr: int,
    tradition: str | None = None,
    min_confidence: float = MIN_CONFIDENCE,
    feature_set: str = "handcrafted",
) -> tuple[str | None, float]:
    """ Classify a full-length recording by sliding the trained window across it.
    The probability vectors of all windows are averaged rather than majority
    voted, so a few confident windows are not outvoted by many undecided
    ones, and near-silent windows are skipped. Returns the instrument name,
    or None when confidence is too low to name one honestly."""
    clip_len = int(_CLIP_S * sr)
    hop_len = int(_HOP_S * sr)
    class_names = list(encoder.classes_)
    extractor = FEATURE_EXTRACTORS[feature_set]

    if y.size < clip_len:
        return None, 0.0

    windows = []
    start = 0
    while start + clip_len <= y.size:
        clip = y[start:start + clip_len]
        if float(np.sqrt(np.mean(clip ** 2))) >= _SILENCE_RMS:
            windows.append(extractor(clip, sr))
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
