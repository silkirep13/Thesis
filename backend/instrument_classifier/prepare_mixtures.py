"""Builds a MULTI-LABEL dataset by mixing the existing solo recordings.
The deployed classifier answers "which one instrument is this?" because it
was trained on solo material and emits a single label. Recognising several
instruments at once needs training data where several are playing and the
answer is known. Rather than collect new recordings, ensemble material with
reliable per-instrument ground truth barely exists for these traditions, we
synthesise it: mixing two or three solo clips gives a mixture whose contents
are known exactly, because we chose them.
Two constraints make the result trustworthy, and both are easy to get wrong:
SPLIT. Mixtures must be built from recordings on one side of the
train/test split only. A test mixture containing a clip from a training
recording re-introduces exactly the leakage the recording-level split exists
to prevent, the model would recognise the recording, not the instruments.
The split here is computed with the same rule and seed as train.py.
PLAUSIBILITY. Instruments are combined only within a tradition. A bouzouki
and an oud never play together, so teaching the model to expect that
combination spends capacity on a case it will never see and blurs the
tradition boundaries the single-label model exploits so effectively.
Usage:
    python prepare_mixtures.py --raw-dir data/raw --out-dir data/processed_multi"""
import argparse
import json
from pathlib import Path

import numpy as np
import librosa

from prepare_dataset import (
    CLIP_DURATION_S, SR, SILENCE_RMS_THRESHOLD, AUDIO_EXTENSIONS,
    _extract_features_yamnet,
)
from predict import TRADITION_INSTRUMENTS

SOLO_FRACTION = 0.40
DUO_FRACTION = 0.40

ACCOMP_GAIN_DB = (-12.0, -3.0)


def _load_clips(path: Path, max_clips: int, rng) -> list[np.ndarray]:
    """Sample up to `max_clips` non-silent 4 s excerpts from one recording."""
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    clip_len = int(CLIP_DURATION_S * SR)
    if y.size < clip_len:
        return []

    starts = np.arange(0, y.size - clip_len + 1, clip_len // 2)
    rng.shuffle(starts)
    out = []
    for s in starts:
        clip = y[s:s + clip_len]
        if float(np.sqrt(np.mean(clip ** 2))) >= SILENCE_RMS_THRESHOLD:
            out.append(clip)
            if len(out) >= max_clips:
                break
    return out


def _split_recordings(raw_dir: Path, test_per_class: int, seed: int = 42):
    """ Same hold-out rule as train.py's video_level_split, applied to files on
    disk rather than manifest rows, so both datasets hold out the same
    recordings and results stay comparable."""
    rng = np.random.default_rng(seed)
    train, test = {}, {}
    for inst_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        files = sorted(p for p in inst_dir.iterdir()
                       if p.suffix.lower() in AUDIO_EXTENSIONS)
        if not files:
            continue
        names = [f.name for f in files]
        rng.shuffle(names)
        n_test = min(test_per_class, max(1, len(names) // 4))
        held = set(names[:n_test])
        train[inst_dir.name] = [f for f in files if f.name not in held]
        test[inst_dir.name] = [f for f in files if f.name in held]
    return train, test


def _mix(clips: list[np.ndarray], rng) -> np.ndarray:
    """ Sum clips at uneven gains, then normalise to avoid clipping."""
    lead, *rest = clips
    out = lead / (np.max(np.abs(lead)) + 1e-9)
    for c in rest:
        gain_db = rng.uniform(*ACCOMP_GAIN_DB)
        c = c / (np.max(np.abs(c)) + 1e-9) * (10.0 ** (gain_db / 20.0))
        out = out + c
    peak = np.max(np.abs(out))
    return out / peak * 0.95 if peak > 0 else out


def _generate(side: dict, instruments: list[str], n_mixtures: int, rng,
              clip_cache: dict, label: str):
    """ Generate `n_mixtures` (features, multi-hot label) pairs from one side."""
    usable = {
        trad: sorted(members & set(k for k, v in side.items() if v))
        for trad, members in TRADITION_INSTRUMENTS.items()
    }
    usable = {t: m for t, m in usable.items() if len(m) >= 1}
    if not usable:
        raise SystemExit(f"[{label}] no usable traditions, check the raw data layout")

    idx = {name: i for i, name in enumerate(instruments)}
    feats, labels, sizes = [], [], []
    traditions = list(usable)

    leads = [i for i in instruments if any(i in m for m in usable.values())]

    for n in range(n_mixtures):
        lead = leads[n % len(leads)]
        lead_trads = [t for t in traditions if lead in usable[t]]
        trad = lead_trads[rng.integers(len(lead_trads))]
        pool = usable[trad]

        r = rng.random()
        if r < SOLO_FRACTION or len(pool) == 1:
            k = 1
        elif r < SOLO_FRACTION + DUO_FRACTION or len(pool) == 2:
            k = 2
        else:
            k = 3
        k = min(k, len(pool))

        others = [i for i in pool if i != lead]
        rng.shuffle(others)
        chosen = [lead] + others[:k - 1]
        clips = []
        for inst in chosen:
            files = side[inst]
            f = files[rng.integers(len(files))]
            key = str(f)
            if key not in clip_cache:
                clip_cache[key] = _load_clips(f, 40, rng)
            pool_clips = clip_cache[key]
            if not pool_clips:
                continue
            clips.append(pool_clips[rng.integers(len(pool_clips))])

        if not clips:
            continue

        mixture = _mix(clips, rng)
        vec = np.zeros(len(instruments), dtype=np.int8)
        for inst in chosen[:len(clips)]:
            vec[idx[inst]] = 1

        feats.append(_extract_features_yamnet(mixture, SR))
        labels.append(vec)
        sizes.append(int(vec.sum()))

        if (n + 1) % 200 == 0:
            print(f"  [{label}] {n + 1}/{n_mixtures}")

    counts = {s: sizes.count(s) for s in sorted(set(sizes))}
    print(f"  [{label}] built {len(feats)} mixtures, sizes {counts}")
    return np.stack(feats), np.stack(labels)


def main():
    """ Build the synthetic mixture dataset used by the multi-label experiment.
    Splits the recordings first, then creates mixtures separately on each
    side, so no clip from a training recording can ever appear in a test
    mixture."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-mixtures", type=int, default=2400)
    ap.add_argument("--test-mixtures", type=int, default=800)
    ap.add_argument("--test-videos-per-class", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files, test_files = _split_recordings(
        raw_dir, args.test_videos_per_class, args.seed)
    instruments = sorted(train_files)

    print("Recording split (train / held-out):")
    for inst in instruments:
        print(f"  {inst:12s} {len(train_files[inst])} / {len(test_files[inst])}"
              f"   held out: {[f.name for f in test_files[inst]]}")

    rng = np.random.default_rng(args.seed)
    print("\nGenerating training mixtures...")
    x_tr, y_tr = _generate(train_files, instruments, args.train_mixtures, rng, {}, "train")
    print("Generating held-out mixtures...")
    x_te, y_te = _generate(test_files, instruments, args.test_mixtures, rng, {}, "test")

    np.save(out_dir / "x_train.npy", x_tr)
    np.save(out_dir / "y_train.npy", y_tr)
    np.save(out_dir / "x_test.npy", x_te)
    np.save(out_dir / "y_test.npy", y_te)
    (out_dir / "instruments.json").write_text(
        json.dumps(instruments, ensure_ascii=False), encoding="utf-8")
    (out_dir / "feature_set.txt").write_text("yamnet", encoding="utf-8")

    print(f"\nTrain {x_tr.shape}  Test {x_te.shape}")
    print(f"Wrote to {out_dir}")


if __name__ == "__main__":
    main()
