"""Hand-crafted acoustic features (numpy only) for window-level NV detection.

Frame features are computed ONCE per utterance (25 ms Hann frames, 10 ms hop, 16 kHz) and
aggregated over any window. Loudness is also expressed relative to the utterance's own
level so the detector is less sensitive to recording gain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

FEATURE_VERSION = "1"
SR = 16000
WIN = 400
HOP = 160
N_FFT = 512
N_AC = 1024
EPS = 1e-10
BAND_EDGES = (0, 250, 500, 1000, 2000, 3000, 4000, 5500, 8000)
FEATURE_NAMES: List[str] = (
    ["dur", "e_mean", "e_std", "e_min", "e_max", "e_p10", "e_p90", "e_rel_mean", "e_rel_max"]
    + [f"band{i}_mean" for i in range(8)] + [f"band{i}_std" for i in range(8)]
    + ["flat_mean", "flat_std", "cent_mean", "cent_std", "zcr_mean", "zcr_std", "hf_mean", "lf_mean",
       "voic_mean", "voic_max", "mod_peak_hz", "mod_peak_ratio", "silence_frac"]
)
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class FrameFeatures:
    e_db: np.ndarray        # (n,) total log energy in dB
    band_rel: np.ndarray    # (n, 8) band energy relative to total, dB
    flat: np.ndarray
    cent: np.ndarray
    zcr: np.ndarray
    hf: np.ndarray
    lf: np.ndarray
    voic: np.ndarray
    p10: float
    p50: float
    p90: float

    @property
    def n_frames(self) -> int:
        return len(self.e_db)


def compute_frame_features(wav: np.ndarray, sr: int = SR) -> FrameFeatures:
    if sr != SR:
        raise ValueError(f"features expect {SR} Hz audio, got {sr}")
    x = np.asarray(wav, dtype=np.float64)
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, WIN)[::HOP]
    hann = np.hanning(WIN)
    spec = np.fft.rfft(frames * hann, n=N_FFT, axis=1)
    power = spec.real ** 2 + spec.imag ** 2
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    total = power.sum(axis=1)
    e_db = 10 * np.log10(total + EPS)

    bands = np.stack([power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
                      for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])], axis=1)
    band_rel = 10 * np.log10(bands + EPS) - e_db[:, None]

    sel = (freqs >= 100) & (freqs <= 8000)
    p = power[:, sel] + EPS
    flat = np.exp(np.log(p).mean(axis=1)) / p.mean(axis=1)
    cent = (power * freqs).sum(axis=1) / (total + EPS)
    zcr = (np.diff(np.signbit(frames), axis=1) != 0).mean(axis=1)
    hf = power[:, freqs >= 4000].sum(axis=1) / (total + EPS)
    lf = power[:, freqs < 500].sum(axis=1) / (total + EPS)

    ac_spec = np.fft.rfft(frames * hann, n=N_AC, axis=1)
    ac = np.fft.irfft(ac_spec.real ** 2 + ac_spec.imag ** 2, n=N_AC, axis=1)
    voic = ac[:, 40:268].max(axis=1) / (ac[:, 0] + EPS)

    return FrameFeatures(e_db, band_rel, flat, cent, zcr, hf, lf, voic,
                         float(np.percentile(e_db, 10)), float(np.median(e_db)), float(np.percentile(e_db, 90)))


def _frame_slice(ff: FrameFeatures, t0: float, t1: float) -> slice:
    lo = int(np.ceil((t0 * SR - WIN / 2) / HOP))
    hi = int(np.floor((t1 * SR - WIN / 2) / HOP)) + 1
    lo, hi = max(lo, 0), min(hi, ff.n_frames)
    if hi <= lo:                                    # window shorter than a frame: use the nearest frame
        k = int(np.clip(round(((t0 + t1) / 2 * SR - WIN / 2) / HOP), 0, ff.n_frames - 1))
        lo, hi = k, k + 1
    return slice(lo, hi)


def window_features(ff: FrameFeatures, t0: float, t1: float) -> np.ndarray:
    s = _frame_slice(ff, t0, t1)
    e = ff.e_db[s]
    b = ff.band_rel[s]
    mod_hz, mod_ratio = 0.0, 0.0
    if len(e) >= 16:
        env = 10 ** (e / 20)
        env = (env - env.mean()) * np.hanning(len(env))
        sp = np.abs(np.fft.rfft(env)) ** 2
        fr = np.fft.rfftfreq(len(env), HOP / SR)
        band = (fr >= 2) & (fr <= 10)
        wide = (fr >= 0.5) & (fr <= 20)
        if band.any() and sp[wide].sum() > 0:
            k = int(np.argmax(np.where(band, sp, -1)))
            mod_hz, mod_ratio = float(fr[k]), float(sp[k] / (sp[wide].sum() + EPS))
    f = [t1 - t0, e.mean(), e.std(), e.min(), e.max(), np.percentile(e, 10), np.percentile(e, 90),
         e.mean() - ff.p50, e.max() - ff.p90]
    f += list(b.mean(axis=0)) + list(b.std(axis=0))
    f += [ff.flat[s].mean(), ff.flat[s].std(), ff.cent[s].mean(), ff.cent[s].std(),
          ff.zcr[s].mean(), ff.zcr[s].std(), ff.hf[s].mean(), ff.lf[s].mean(),
          ff.voic[s].mean(), ff.voic[s].max(), mod_hz, mod_ratio, float((e < ff.p50 - 25).mean())]
    out = np.asarray(f, dtype=np.float32)
    assert out.shape == (N_FEATURES,)
    return out
