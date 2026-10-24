from scipy.io import wavfile
import numpy as np, glob, os

for p in sorted(glob.glob('data/*.WAV')):
    sr, x = wavfile.read(p)
    x = x.astype(float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    print(f"{p:40s} sr={sr} dur={x.shape[0]/sr:6.2f}s rms={rms:.4f} peak={peak:.4f}")
