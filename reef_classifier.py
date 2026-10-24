"""
Coral Reef Soundscape - Exploratory Auto-Labeling
================================================
Reads audio paths from metadata.csv and exports exploratory acoustic-richness
labels to predictions.csv. These heuristic labels are not trained or validated
and should not be treated as an ecological assessment.

Usage:
    python reef_classifier.py --auto-label

Requirements: numpy, scipy
"""

import argparse
import csv
import math
import os

import numpy as np
from scipy.io import wavfile


# ---------------------------------------------------------------------------
# 1. Loading audio & features
# ---------------------------------------------------------------------------
def load_metadata(path="metadata.csv"):
    """Load file metadata from CSV (with or without labels)."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fpath = r.get("file_path") or r.get("filename")
            fname = r.get("filename") or os.path.basename(fpath) if fpath else None
            site = r.get("site") or "Unknown Site"

            rows.append({
                "file": fpath,
                "filename": fname,
                "site": site
            })

    return rows


def trim_silence(x, sr, top_db=40):
    """Remove leading/trailing silence using RMS threshold."""
    if len(x) == 0:
        return x
    frame = max(1, int(0.1 * sr))
    if len(x) < frame:
        return x
    kernel = np.ones(frame) / frame
    energy = np.convolve(x ** 2, kernel, mode="same")
    rms = np.sqrt(np.maximum(energy, 0.0))
    thr = rms.max() * (10 ** (-top_db / 20))
    idx = np.where(rms > thr)[0]
    return x[idx[0]:idx[-1]+1] if len(idx) else x


def load_audio(path):
    """Load audio and convert to mono if needed."""
    try:
        sr, x = wavfile.read(path)
    except Exception as e:
        print(f"WARNING: Could not read {path}: {e}")
        return None

    x = x.astype(np.float64)

    # Stereo -> mono
    if x.ndim > 1:
        x = x.mean(axis=1)

    # Resample to 32kHz if needed (crude linear interpolation)
    if sr != 32000 and len(x) > 0:
        target_sr = 32000
        n = int(round(len(x) * target_sr / sr))
        if n > 0:
            x = np.interp(np.linspace(0, len(x)-1, n), np.arange(len(x)), x)
        sr = target_sr

    x = trim_silence(x, sr)
    return sr, x


# ---------------------------------------------------------------------------
# 2. Bioacoustic feature extraction (same as before)
# ---------------------------------------------------------------------------
def stft(x, sr, win_ms=50, hop_ms=12.5, nfft=1024):
    """Short-time Fourier transform."""
    win = int(win_ms / 1000 * sr)
    hop = int(hop_ms / 1000 * sr)
    if win < 8:
        win = 8
    if hop < 1:
        hop = 1

    n = len(x)
    if n == 0:
        S = np.zeros((nfft // 2 + 1, 1))
        freqs = np.fft.rfftfreq(nfft, 1 / sr)
        return S, freqs

    if n < win:
        x = np.pad(x, (0, win - n))
        n = win

    n_frames = 1 + (n - win) // hop
    starts = np.arange(n_frames) * hop
    window = np.hanning(win)
    idx = np.arange(win)[None, :] + starts[:, None]
    segs = x[idx].T * window[:, None]
    S = np.abs(np.fft.rfft(segs, n=nfft, axis=0))
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    return S, freqs


def band_power(S, freqs, f_lo, f_hi, sr):
    """Average power in a frequency band."""
    f_hi = min(f_hi, sr / 2)
    if f_lo >= f_hi:
        return 0.0
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        return 0.0
    return float(np.mean(S[mask, :] ** 2))


def aci(S):
    """Acoustic Complexity Index."""
    if S.shape[1] < 2:
        return 0.0
    d = np.abs(np.diff(S, axis=1))
    return float(d.sum() / (S.shape[0] * S.shape[1]))


def extract_features(path):
    """Extract all 12 bioacoustic features."""
    sr, x = load_audio(path)
    if sr is None:
        return {"rms_db": -999.0, "centroid": 0, 
                "rolloff": 0, "flatness": 0, "spectral_entropy": 0,
                "band_shrimp": 0, "band_fish": 0, "band_low": 0,
                "band_high": 0, "shrimp_fish_ratio": 0, "aci": 0, "zcr": 0}

    S, freqs = stft(x, sr)
    P = np.mean(S ** 2, axis=1) + 1e-12
    total_power = P.sum()

    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    rms_db = 20 * math.log10(rms) if rms > 0 else -999.0

    centroid = float(np.sum(freqs * P) / total_power)
    cum = np.cumsum(P) / total_power
    roll_idx = int(np.searchsorted(cum, 0.85))
    roll_idx = min(roll_idx, len(freqs) - 1)
    rolloff = float(freqs[roll_idx])

    flatness = float(
        np.exp(np.mean(np.log(P+1e-12))) / (np.mean(P)+1e-12)
    )

    p = P / total_power
    spectral_entropy = float(-np.sum(p * np.log(p + 1e-12))) if len(p) > 0 else 0.0

    shrimp = band_power(S, freqs, 2000, 10000, sr) / total_power
    fish = band_power(S, freqs, 100, 2000, sr) / total_power
    low = band_power(S, freqs, 20, 100, sr) / total_power
    high = band_power(S, freqs, 10000, 20000, sr) / total_power
    ratio = shrimp / (fish + 1e-12)

    zcr_val = 0.0
    if len(x) > 1:
        signs = np.sign(x)
        signs[signs == 0] = 1
        zcr_val = float(np.mean(signs[:-1] != signs[1:]))
    complexity = aci(S)

    return {
        "rms_db": rms_db if not math.isnan(rms_db) else -999.0,
        "centroid": centroid,
        "rolloff": rolloff,
        "flatness": flatness,
        "spectral_entropy": spectral_entropy,
        "band_shrimp": shrimp,
        "band_fish": fish,
        "band_low": low,
        "band_high": high,
        "shrimp_fish_ratio": ratio,
        "aci": complexity,
        "zcr": zcr_val
    }


# ---------------------------------------------------------------------------
# 3. Classifier (Logistic Regression)
# ---------------------------------------------------------------------------
class LogisticRegression:
    def __init__(self, lr=0.2, epochs=3000, l2=0.05):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w = None
        self.b = 0.0
        self.means = None
        self.stds = None

    def fit(self, X, y):
        n_samples = len(X)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        n_features = X.shape[1] if hasattr(X, 'shape') else 1
        
        means = X.mean(axis=0).flatten()
        stds = np.sqrt(np.var(X, axis=0)) + 1e-8
        self.means = means
        self.stds = stds
        X_std = (X - means) / stds
        
        n_iters = self.epochs
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(n_iters):
            z = np.clip(X_std @ self.w + self.b, -50, 50)
            p = 1.0 / (1.0 + np.exp(-z))
            
            grad_w = X_std.T @ (p - y) / n_samples + self.l2 * self.w
            grad_b = (p - y).mean()

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

        return self

    def predict_proba(self, X):
        """Return probability of Healthy (1) vs Degraded (0)."""
        if np.array(X).shape[0] == 0:
            return np.zeros(len(X))
        
        means = self.means
        stds = self.stds
        X_std = (np.array(X) - means) / stds

        z = np.clip(X_std @ self.w + self.b, -50, 50)
        p = 1.0 / (1.0 + np.exp(-z))
        return p[:, 0] if p.ndim > 1 else float(p)


# ---------------------------------------------------------------------------
# 4. Auto-Classify All Files & Save to CSV
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Coral Reef Soundscape Classifier")
    parser.add_argument("--auto-label", action="store_true", 
                        help="Create exploratory labels from acoustic richness features")
    args = parser.parse_args()

    feature_names = [
        "rms_db", "centroid", "rolloff", "flatness",
        "spectral_entropy", "band_shrimp", "band_fish",
        "band_low", "band_high", "shrimp_fish_ratio",
        "aci", "zcr"
    ]

    print("Loading metadata from metadata.csv...")
    if not args.auto_label:
        parser.error("No supervised training/evaluation mode is configured. Use --auto-label.")

    print("=" * 60)
    print("EXPLORATORY AUTO-LABELING")
    print("These are acoustic-richness heuristics, not trained or validated labels.")
    print("=" * 60)

    rows = load_metadata("metadata.csv")
    valid_rows = []
    feature_rows = []
    for row in rows:
        fpath = row["file"]
        if not fpath:
            print(f"WARNING: Skipping metadata row without a file path: {row}")
            continue

        features = extract_features(fpath)
        if features is None:
            continue
        valid_rows.append(row)
        feature_rows.append([features[name] for name in feature_names])

    if not feature_rows:
        parser.error("No readable audio files were found using paths in metadata.csv.")

    X = np.asarray(feature_rows, dtype=float)
    # Reef soundscapes with greater acoustic complexity, entropy, and biological
    # band energy score higher. This is an exploratory proxy, not a validated model.
    richness_indices = [
        feature_names.index(name)
        for name in ("aci", "spectral_entropy", "band_shrimp", "band_fish")
    ]
    richness_features = X[:, richness_indices]
    means = richness_features.mean(axis=0)
    stds = richness_features.std(axis=0)
    stds[stds == 0] = 1.0
    richness_score = ((richness_features - means) / stds).mean(axis=1)
    threshold = float(np.median(richness_score))
    preds = richness_score >= threshold
    probs = 1.0 / (1.0 + np.exp(-np.clip(richness_score - threshold, -50, 50)))

    output_path = "predictions.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_label", "probability_healthy"])
        for row, pred, prob in zip(valid_rows, preds, probs):
            writer.writerow([
                row["filename"] or os.path.basename(row["file"]),
                "Healthy" if pred else "Degraded",
                f"{float(prob):.3f}",
            ])

    print(f"\nExtracted features from {len(valid_rows)} files.")
    print(f"Saved exploratory predictions to: {output_path}")
    print("\nExample predictions:")
    for row, pred, prob in list(zip(valid_rows, preds, probs))[:6]:
        fname = row["filename"] or os.path.basename(row["file"])
        site = row["site"]
        label = "Healthy" if pred else "Degraded"
        print(f"  {fname:30s} -> {label:9s} (score={float(prob):.3f}) [Site: {site}]")


if __name__ == "__main__":
    main()