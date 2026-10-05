"""Audio loading and resampling: ONE canonical preprocessing path for all models.

Every model backend (ASR, DNSMOS, ECAPA) receives mono float32 audio at 16 kHz produced
by `load_mono(path, 16000)`.  [ASSUMPTION] Resampling is done here (scipy polyphase) so the
result does not depend on each library's internal resampler. Amplitude is NOT normalised.

Requires: numpy, soundfile (+ scipy only when resampling is actually needed).
"""
from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    try:
        from scipy.signal import resample_poly
    except ImportError as e:  # pragma: no cover
        raise ImportError("scipy is required to resample audio: pip install scipy") from e
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, int(sr_out) // g, int(sr_in) // g).astype(np.float32)


def load_mono(path: Union[str, Path], target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Read an audio file as mono float32 in [-1, 1] (channels averaged), optionally resampled."""
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if x.shape[0] == 0:
        raise ValueError(f"empty audio: {path}")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if not np.all(np.isfinite(x)):
        raise ValueError(f"audio contains NaN/Inf: {path}")
    if target_sr is not None and sr != target_sr:
        x = resample(x, sr, target_sr)
        sr = target_sr
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)
