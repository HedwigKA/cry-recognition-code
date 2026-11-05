"""Baseline after Fahmi et al. (2020): short-time energy statistics + spectral centroid/bandwidth, k-NN.

Clip-level features on the 8 kHz signal (energy-gated frames are kept, as in the original).
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import SR, load_audio

FRAME, HOP = 200, 80


def clip_features(path: str) -> np.ndarray:
    x = load_audio(path)
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    fr = np.pad(x, (0, max(0, idx.max() + 1 - len(x))))[idx] * np.hanning(FRAME)
    energy = (fr ** 2).sum(1)
    mag = np.abs(np.fft.rfft(fr, 256))
    freqs = np.fft.rfftfreq(256, 1 / SR)
    p = mag / (mag.sum(1, keepdims=True) + 1e-10)
    cen = (p * freqs).sum(1)
    bw = np.sqrt((p * (freqs - cen[:, None]) ** 2).sum(1))
    st = lambda v: [v.mean(), v.std(), v.min(), v.max()]
    return np.array(st(energy) + [energy.sum()] + st(cen) + st(bw), dtype=np.float32)


def knn(k: int = 5):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
