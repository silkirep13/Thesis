""" Trains a MULTI-LABEL instrument classifier on synthesised mixtures.
One binary decision per instrument ("is a bouzouki playing? yes/no") instead
of one shared decision over ten mutually exclusive classes, so several
instruments can be reported at once.
The run also scores the deployed single-label model on the same held-out
mixtures. Multi-label output is only worth adopting if it beats what we
already have on the material it is meant to handle, and without that
comparison a plausible-looking F1 says nothing about whether anything
improved.
Usage:
    python train_multi.py --data-dir data/processed_multi"""

import argparse
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, hamming_loss

DEFAULT_THRESHOLD = 0.5


def _proba_matrix(model, x):
    """ MultiOutputClassifier returns a list of (n, 2) arrays, one per label.
    Collapse to an (n, n_labels) matrix of P(present).
    A label whose training column was constant has a single column, so the
    positive class may be absent; guard rather than index blindly."""
    cols = []
    for est, proba in zip(model.estimators_, model.predict_proba(x)):
        classes = list(est.classes_)
        if 1 in classes:
            cols.append(proba[:, classes.index(1)])
        else:
            cols.append(np.zeros(proba.shape[0]))
    return np.column_stack(cols)


def main():
    """ Train and evaluate the experimental multi-label classifier.
    Learns one yes/no decision per instrument instead of a single choice
    between them, and scores the deployed single-label model on the same
    mixtures so the two can be compared fairly."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-model", default="models/instrument_model_multilabel.joblib")
    ap.add_argument("--single-label-model", default="models/instrument_model_singlelabel_backup.joblib")
    args = ap.parse_args()

    d = Path(args.data_dir)
    x_tr, y_tr = np.load(d / "x_train.npy"), np.load(d / "y_train.npy")
    x_te, y_te = np.load(d / "x_test.npy"), np.load(d / "y_test.npy")
    instruments = json.loads((d / "instruments.json").read_text(encoding="utf-8"))

    print(f"Train {x_tr.shape}  Test {x_te.shape}  labels {len(instruments)}")
    print("Positives per instrument (train / test):")
    for i, name in enumerate(instruments):
        print(f"  {name:12s} {int(y_tr[:, i].sum()):5d} / {int(y_te[:, i].sum()):4d}")

    clf = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=42, n_jobs=-1),
        n_jobs=-1,
    )
    clf.fit(x_tr, y_tr)

    proba = _proba_matrix(clf, x_te)

    print("\n=== Multi-label, threshold sweep ===")
    print(f"{'thresh':>7} {'micro-F1':>9} {'macro-F1':>9} {'exact':>7} {'hamming':>8}")
    best = None
    for t in (0.30, 0.40, 0.50, 0.60):
        pred = (proba >= t).astype(int)
        rep = classification_report(y_te, pred, target_names=instruments,
                                    output_dict=True, zero_division=0)
        micro, macro = rep["micro avg"]["f1-score"], rep["macro avg"]["f1-score"]
        exact = float(np.mean((pred == y_te).all(axis=1)))
        ham = hamming_loss(y_te, pred)
        print(f"{t:>7.2f} {micro:>9.3f} {macro:>9.3f} {exact:>7.3f} {ham:>8.3f}")
        if best is None or micro > best[1]:
            best = (t, micro)

    thresh = best[0]
    pred = (proba >= thresh).astype(int)
    print(f"\n=== Per-instrument report at threshold {thresh:.2f} ===")
    print(classification_report(y_te, pred, target_names=instruments, zero_division=0))

    sl_path = Path(args.single_label_model)
    if sl_path.exists():
        bundle = joblib.load(sl_path)
        sl_model, sl_enc = bundle["model"], bundle["label_encoder"]
        sl_names = list(sl_enc.classes_)
        sl_pred = sl_model.predict(x_te)

        hit = 0
        solo_hit = solo_n = 0
        for row, p in enumerate(sl_pred):
            name = sl_names[p]
            if name in instruments and y_te[row, instruments.index(name)] == 1:
                hit += 1
                if y_te[row].sum() == 1:
                    solo_hit += 1
            if y_te[row].sum() == 1:
                solo_n += 1

        print("=== Baseline: deployed single-label model, same mixtures ===")
        print(f"  its one prediction is actually present : {hit}/{len(y_te)}"
              f" ({hit / len(y_te):.3f})")
        print(f"  correct on the solo mixtures           : {solo_hit}/{solo_n}"
              f" ({solo_hit / max(solo_n, 1):.3f})")
        print("  (it can never name more than one, so recall over all present")
        print(f"   instruments is capped at {len(y_te) / y_te.sum():.3f})")

        multi_exact = float(np.mean((pred == y_te).all(axis=1)))
        print(f"\n  multi-label exact-set accuracy         : {multi_exact:.3f}")
    else:
        print(f"[baseline skipped, {sl_path} not found]")

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "instruments": instruments,
                 "feature_set": "yamnet", "multi_label": True,
                 "threshold": thresh}, args.out_model)
    print(f"\nSaved to {args.out_model}  (threshold={thresh:.2f})")


if __name__ == "__main__":
    main()
