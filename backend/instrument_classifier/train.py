"""
Trains a classical ML instrument classifier (Random Forest) on features
produced by prepare_dataset.py, using a VIDEO-LEVEL train/test split to
avoid data leakage between clips cut from the same recording — clips
from one video share mic/room/player, so a per-clip random split would
report inflated accuracy that doesn't reflect real generalization.

Usage:
    python train.py --data-dir data/processed --test-videos-per-class 2
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from predict import allowed_candidates_for_instrument


def load_manifest(data_dir: Path):
    with open(data_dir / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    features = np.load(data_dir / "features.npy")
    assert len(rows) == features.shape[0], "manifest/features row count mismatch"
    return rows, features


def video_level_split(rows: list[dict], test_videos_per_class: int, seed: int = 42):
    """
    For each instrument, hold out `test_videos_per_class` distinct source
    videos entirely for testing. Never split individual clips randomly —
    clips from the same video leak recording-condition information that
    would let the model "cheat" instead of learning the instrument's timbre.

    Keys are (instrument, source_file) pairs, not bare filenames — source
    files are only unique within their instrument folder (e.g. two
    different instruments' clips both named "v1.wav" are unrelated
    recordings and must not be conflated).
    """
    rng = np.random.default_rng(seed)
    by_instrument: dict[str, set] = {}
    for r in rows:
        by_instrument.setdefault(r["instrument"], set()).add(r["source_file"])

    test_keys: set[tuple[str, str]] = set()
    for instrument, files in by_instrument.items():
        files = sorted(files)
        rng.shuffle(files)
        n_test = min(test_videos_per_class, max(1, len(files) // 4))
        test_keys.update((instrument, f) for f in files[:n_test])

    train_idx = [i for i, r in enumerate(rows) if (r["instrument"], r["source_file"]) not in test_keys]
    test_idx = [i for i, r in enumerate(rows) if (r["instrument"], r["source_file"]) in test_keys]
    return train_idx, test_idx, test_keys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--test-videos-per-class", type=int, default=2)
    parser.add_argument("--out-model", default="instrument_model.joblib")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rows, features = load_manifest(data_dir)

    labels_raw = [r["instrument"] for r in rows]
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels_raw)

    train_idx, test_idx, test_keys = video_level_split(rows, args.test_videos_per_class)
    print(f"Train clips: {len(train_idx)}  Test clips: {len(test_idx)}")
    print(f"Held-out test videos: {sorted(test_keys)}")

    if not test_idx:
        raise SystemExit(
            "No test videos held out — need at least a few videos per instrument "
            "before a meaningful video-level split is possible."
        )

    x_train, y_train = features[train_idx], labels[train_idx]
    x_test, y_test = features[test_idx], labels[test_idx]

    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print("\n=== Held-out video-level evaluation (unrestricted) ===")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(list(encoder.classes_))
    print(confusion_matrix(y_test, y_pred))

    # --- Tradition-restricted evaluation ---
    # Same trained model, same test clips — but at prediction time, only
    # consider instruments that share a tradition with the TRUE instrument,
    # mirroring how the deployed app would only ever compare against the
    # instruments valid for the user-selected tradition. No retraining.
    y_proba = clf.predict_proba(x_test)
    class_names = list(encoder.classes_)
    y_pred_restricted = np.empty_like(y_pred)
    for i in range(len(x_test)):
        true_instrument = class_names[y_test[i]]
        allowed = allowed_candidates_for_instrument(true_instrument)
        mask = np.array([name in allowed for name in class_names])
        masked_proba = np.where(mask, y_proba[i], -1.0)
        y_pred_restricted[i] = np.argmax(masked_proba)

    print("\n=== Held-out video-level evaluation (tradition-restricted) ===")
    print(classification_report(y_test, y_pred_restricted, target_names=encoder.classes_, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(class_names)
    print(confusion_matrix(y_test, y_pred_restricted))

    unrestricted_acc = float(np.mean(y_pred == y_test))
    restricted_acc = float(np.mean(y_pred_restricted == y_test))
    print(f"\nOverall accuracy: unrestricted={unrestricted_acc:.3f}  tradition-restricted={restricted_acc:.3f}")

    joblib.dump({"model": clf, "label_encoder": encoder}, args.out_model)
    print(f"\nSaved model to {args.out_model}")


if __name__ == "__main__":
    main()

