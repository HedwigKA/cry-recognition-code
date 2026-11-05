"""Audio loading, windowing and log-mel features.

Parameters follow the paper (Ch. 3.4.1): mono 8 kHz, 1 s windows with 0.5 s hop,
40-band log-mel, 25 ms frames, 10 ms stride.
"""
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 8000
WIN = SR            # 1 s
HOP = SR // 2       # 0.5 s
FRAME = 200         # 25 ms
STRIDE = 80         # 10 ms
N_FFT = 256
N_MELS = 40
FMIN, FMAX = 20.0, SR / 2
SILENCE_DBFS = -50.0  # windows quieter than this are treated as silence


def load_audio(path: str) -> np.ndarray:
    """Read a file, mix to mono, resample to 8 kHz, return float32 in [-1, 1]."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def windows(x: np.ndarray) -> np.ndarray:
    """Cut a signal into 1 s windows with 0.5 s hop (last partial window zero-padded)."""
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    n = 1 + int(np.ceil((len(x) - WIN) / HOP))
    x = np.pad(x, (0, (n - 1) * HOP + WIN - len(x)))
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def rms_dbfs(w: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.sqrt(np.mean(w ** 2, axis=-1)) + 1e-10)


def _mel_filterbank() -> np.ndarray:
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(FMIN), hz2mel(FMAX), N_MELS + 2)
    bins = np.floor((N_FFT + 1) * mel2hz(mels) / SR).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


_FB = _mel_filterbank()
_HANN = np.hanning(FRAME).astype(np.float32)


def logmel(w: np.ndarray) -> np.ndarray:
    """Batch of 1 s windows (n, 8000) -> log-mel in dB, shape (n, frames=98, 40)."""
    n_frames = 1 + (w.shape[1] - FRAME) // STRIDE
    idx = np.arange(FRAME)[None, :] + STRIDE * np.arange(n_frames)[:, None]
    frames = w[:, idx] * _HANN                         # (n, T, 200)
    spec = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2   # (n, T, 129)
    mel = spec @ _FB.T                                 # (n, T, 40)
    return (10 * np.log10(mel + 1e-10)).astype(np.float32)
