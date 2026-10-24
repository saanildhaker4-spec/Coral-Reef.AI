"""
Shrink reef recordings to 16-bit PCM WAV (half the size of 32-bit float) so
they fit in the GitHub repo. reef_classifier.load_audio scales 16-bit back to
[-1, 1], so features change by well under 1%.

Usage:
    python shrink_audio.py SOURCE_FOLDER [DEST_FOLDER]   (default DEST_FOLDER: data)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def shrink(src, dest):
    sr, x = wavfile.read(src)
    if np.issubdtype(x.dtype, np.integer):
        if x.dtype.itemsize <= 2:
            return False
        x = x / -np.iinfo(x.dtype).min
    pcm = np.round(np.clip(x, -1, 1) * 32767).astype(np.int16)
    wavfile.write(dest, sr, pcm)
    return True


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src_dir = Path(sys.argv[1])
    dest_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() == ".wav")
    if not files:
        sys.exit(f"No .wav files found in {src_dir}")
    for f in files:
        dest = dest_dir / f.name  # keep the exact name so metadata.csv matches
        if shrink(f, dest):
            print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB -> {dest.stat().st_size / 1e6:.1f} MB")
        else:
            print(f"{f.name}: already 16-bit or smaller, skipped")


if __name__ == "__main__":
    main()
