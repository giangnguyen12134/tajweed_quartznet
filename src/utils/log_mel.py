from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

SR = 16000
N_FFT = 512          # 32 ms
WIN = 400            # 25 ms
HOP = 160            # 10 ms  -> 100 fps input, 50 fps after the stride-2 prologue
N_MELS = 80
FMIN, FMAX = 20.0, 7600.0


# ---------------------------------------------------------------------------
# Mel filterbank
# ---------------------------------------------------------------------------
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    n_bins = n_fft // 2 + 1
    pts = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        r = min(r, n_bins - 1)
        c = min(c, r - 1) if r - 1 > l else c
        if l < c:
            fb[m, l:c] = (np.arange(l, c) - l) / float(c - l)
        if c < r:
            fb[m, c:r] = (r - np.arange(c, r)) / float(r - c)
    # Slaney-style area normalisation keeps low mels from dominating the scale
    enorm = 2.0 / (pts[2:n_mels + 2] - pts[:n_mels])
    return fb * enorm[:, None]


_FB: Optional[np.ndarray] = None
_WINDOW = np.hanning(WIN).astype(np.float32)


def log_mel(wav: np.ndarray, sr: int = SR, dither: float = 1e-5) -> np.ndarray:
    """
    wav: float32 mono in [-1, 1]. Returns (N_MELS, T) float32.

    Per-utterance mean/variance normalisation (CMVN) is applied at the end.
    On a device this is the right choice over global CMVN: it absorbs the
    microphone's frequency response and the room, for free, per recording.
    """
    global _FB
    if _FB is None:
        _FB = mel_filterbank(sr)
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x - x.mean()
    # pre-emphasis lifts the high frequencies that carry sibilant/emphatic cues
    x = np.concatenate([x[:1], x[1:] - 0.97 * x[:-1]])
    if dither:
        x = x + dither * np.random.randn(len(x)).astype(np.float32)

    n_frames = 1 + max(0, (len(x) - WIN) // HOP)
    if n_frames <= 0:
        return np.zeros((N_MELS, 0), dtype=np.float32)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * _WINDOW
    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)).astype(np.float32) ** 2
    mel = spec @ _FB.T
    feats = np.log(mel + 1e-6).T.astype(np.float32)          # (N_MELS, T)
    feats -= feats.mean(axis=1, keepdims=True)
    feats /= (feats.std(axis=1, keepdims=True) + 1e-5)
    return feats