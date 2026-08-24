"""
Segments raw instrument recordings into fixed-length clips and extracts
classical ML features (MFCC + spectral descriptors) for the instrument
classifier training set.

Expected input layout — one subfolder per instrument, any audio format:
    raw_dir/
      oud/
        video1.mp3
        video2.wav
      ney/
        video1.mp3
      ...

Usage:
    python prepare_dataset.py --raw-dir data/raw --out-dir data/processed
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import librosa

CLIP_DURATION_S = 4.0
CLIP_HOP_S = 2.0  # 50% overlap between consecutive clips
SR = 22050
SILENCE_RMS_THRESHOLD = 0.01
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".webm"}


def _extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compact classical-ML feature vector for one clip: MFCC (13 coeffs,
    mean+std), spectral centroid, rolloff, zero-crossing rate, and chroma
    (mean+std) — a standard timbre feature set that works reasonably well
    with small (tens-to-hundreds of clips per class) training sets.
    """
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


def _segment_and_extract(path: Path, instrument: str, rows: list, feature_rows: list) -> int:
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    clip_len = int(CLIP_DURATION_S * sr)
    hop_len = int(CLIP_HOP_S * sr)

    clip_idx = 0
    start = 0
    while start + clip_len <= len(y):
        clip = y[start:start + clip_len]
        rms = float(np.sqrt(np.mean(clip ** 2)))
        if rms >= SILENCE_RMS_THRESHOLD:
            feats = _extract_features(clip, sr)
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True, help="Directory with one subfolder per instrument")
    parser.add_argument("--out-dir", required=True, help="Where to write manifest.csv and features.npy")
    args = parser.parse_args()

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
            n_clips = _segment_and_extract(f, instrument, rows, feature_rows)
            print(f"  {f.name}: {n_clips} clips")

    if not rows:
        raise SystemExit("No usable clips extracted — check silence threshold / audio files.")

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["instrument", "source_file", "clip_index", "start_s", "end_s"])
        writer.writeheader()
        writer.writerows(rows)

    features = np.stack(feature_rows)
    np.save(out_dir / "features.npy", features)

    print(f"\nWrote {len(rows)} clips across {len(instrument_dirs)} instruments")
    print(f"  manifest: {manifest_path}")
    print(f"  features: {out_dir / 'features.npy'}  shape={features.shape}")


if __name__ == "__main__":
    main()
