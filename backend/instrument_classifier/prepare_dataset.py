"""Turns raw instrument recordings into the feature table the classifier trains on.
Each recording is cut into fixed-length clips and every clip is described by a
vector of numbers, either hand-designed descriptors or pretrained embeddings.
Expected input layout: one subfolder per instrument, any audio format:
    raw_dir/
      oud/
        video1.mp3
        video2.wav
      ney/
        video1.mp3
      ...
Usage:
    python prepare_dataset.py --raw-dir data/raw --out-dir data/processed"""
import argparse
import csv
from pathlib import Path

import numpy as np
import librosa

CLIP_DURATION_S = 4.0
CLIP_HOP_S = 2.0
SR = 22050
SILENCE_RMS_THRESHOLD = 0.01
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".webm"}

YAMNET_SR = 16000
_YAMNET = None


def _extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """ Describe one clip with 56 hand-designed numbers.
    Combines MFCC coefficients, spectral centroid, rolloff, zero-crossing rate
    and chroma, keeping the mean and the standard deviation of each over time.
    The mean says where the energy sits and the deviation says how much it
    moves, which is what separates a plucked string from a sustained wind."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        centroid.mean(axis=1), centroid.std(axis=1),
        rolloff.mean(axis=1), rolloff.std(axis=1),
        zcr.mean(axis=1), zcr.std(axis=1),
        chroma.mean(axis=1), chroma.std(axis=1),
    ])


def _get_yamnet():
    """ Load the pretrained YAMNet network once and keep it in memory.
    The weights are downloaded from TensorFlow Hub the first time and cached
    on disk afterwards, so only the first call is slow. The cache is kept
    beside the code rather than in the system temp folder, which Windows
    empties periodically and which left an empty directory behind that the
    loader mistook for a valid download."""
    global _YAMNET
    if _YAMNET is None:
        import os
        os.environ.setdefault(
            "TFHUB_CACHE_DIR", str(Path(__file__).resolve().parent / "models" / "tfhub")
        )
        import tensorflow_hub as hub
        _YAMNET = hub.load("https://tfhub.dev/google/yamnet/1")
    return _YAMNET


def _extract_features_yamnet(y: np.ndarray, sr: int) -> np.ndarray:
    """ Describe one clip with 1024 numbers taken from pretrained YAMNet.
    YAMNet was trained by Google on roughly two million labelled sounds; we
    ignore its own output classes and take the internal representation it
    forms just before classifying, which captures far more detail than the
    hand-designed numbers and is what finally separated the plucked strings."""
    model = _get_yamnet()
    if sr != YAMNET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=YAMNET_SR)
    _scores, embeddings, _spec = model(y.astype(np.float32))
    return embeddings.numpy().mean(axis=0)


FEATURE_EXTRACTORS = {
    "handcrafted": _extract_features,
    "yamnet": _extract_features_yamnet,
}


def _segment_and_extract(path: Path, instrument: str, rows: list, feature_rows: list,
                         extractor=_extract_features) -> int:
    """ Cut one recording into overlapping clips and describe each of them.
    Consecutive clips overlap by half their length, which doubles the number
    of samples and stops a note from always being cut at the same point.
    Near-silent clips are dropped so that pauses are not learned as sound."""
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    clip_len = int(CLIP_DURATION_S * sr)
    hop_len = int(CLIP_HOP_S * sr)

    clip_idx = 0
    start = 0
    while start + clip_len <= len(y):
        clip = y[start:start + clip_len]
        rms = float(np.sqrt(np.mean(clip ** 2)))
        if rms >= SILENCE_RMS_THRESHOLD:
            feats = extractor(clip, sr)
            rows.append({
                "instrument": instrument,
                "source_file": path.name,
                "clip_index": clip_idx,
                "start_s": round(start / sr, 2),
                "end_s": round((start + clip_len) / sr, 2),
            })
            feature_rows.append(feats)
            clip_idx += 1
        start += hop_len

    return clip_idx


def main():
    """ Scan every instrument folder and write the training data to disk.
    Produces a manifest listing where each clip came from and a matching array
    of feature vectors, plus a note of which feature set was used so that
    training and prediction can never disagree about it"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True, help="Directory with one subfolder per instrument")
    parser.add_argument("--out-dir", required=True, help="Where to write manifest.csv and features.npy")
    parser.add_argument("--features", choices=sorted(FEATURE_EXTRACTORS), default="handcrafted",
                        help="Feature set: 56 hand-designed descriptors (default) or "
                             "1024-dim pretrained YAMNet embeddings")
    args = parser.parse_args()

    extractor = FEATURE_EXTRACTORS[args.features]
    print(f"Feature set: {args.features}")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    feature_rows: list[np.ndarray] = []

    instrument_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if not instrument_dirs:
        raise SystemExit(f"No instrument subfolders found in {raw_dir}")

    for inst_dir in instrument_dirs:
        instrument = inst_dir.name
        files = sorted(p for p in inst_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
        print(f"[{instrument}] {len(files)} source file(s)")
        for f in files:
            n_clips = _segment_and_extract(f, instrument, rows, feature_rows, extractor)
            print(f"  {f.name}: {n_clips} clips")

    if not rows:
        raise SystemExit("No usable clips extracted, check silence threshold / audio files.")

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["instrument", "source_file", "clip_index", "start_s", "end_s"])
        writer.writeheader()
        writer.writerows(rows)

    features = np.stack(feature_rows)
    np.save(out_dir / "features.npy", features)
    (out_dir / "feature_set.txt").write_text(args.features, encoding="utf-8")

    print(f"\nWrote {len(rows)} clips across {len(instrument_dirs)} instruments")
    print(f"  manifest: {manifest_path}")
    print(f"  features: {out_dir / 'features.npy'}  shape={features.shape}")


if __name__ == "__main__":
    main()
