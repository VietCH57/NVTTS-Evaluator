"""Pure scoring functions. No I/O, no model code.

[OFFICIAL]   weights of the Track A / Track B formulas (task description).
[ASSUMPTION] how raw values are normalised to [0,1] (the organizers only say
             "normalized to [0,1]"): see ScoringConfig.

Rules enforced here:
* `auto_score` uses only NVPA, WER, pMOS, SS, with the official weights and NO
  renormalisation, so its maximum is 0.70. It is never marked official.
* `final_score` requires SN and Q. If either is None the result is None ("N/A").
  There is deliberately no fallback to pMOS.
* Out-of-range inputs raise ValueError instead of being silently fixed.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Dict, Optional

# [OFFICIAL] ------------------------------------------------------------------
WEIGHTS: Dict[str, Dict[str, float]] = {
    "A": {"NVPA": 0.30, "SN": 0.15, "Q": 0.15, "1-WER": 0.15, "pMOS": 0.15, "SS": 0.10},
    "B": {"NVPA": 0.30, "SN": 0.15, "Q": 0.15, "1-WER": 0.10, "pMOS": 0.10, "SS": 0.20},
}
AUTOMATIC_COMPONENTS = ("NVPA", "1-WER", "pMOS", "SS")
HUMAN_COMPONENTS = ("SN", "Q")


@dataclass(frozen=True)
class ScoringConfig:
    """[ASSUMPTION] normalisation choices; all recorded in reports."""
    mos_min: float = 1.0          # SN, Q, pMOS: (x - mos_min) / (mos_max - mos_min)
    mos_max: float = 5.0
    clip_wer: bool = True         # use min(WER, 1) in (1 - WER); raw WER is still reported
    clip_ss: bool = True          # use max(SS, 0) when cosine similarity is negative


@dataclass(frozen=True)
class Component:
    raw: float
    normalized: float
    weight: float

    @property
    def contribution(self) -> float:
        return self.weight * self.normalized


@dataclass(frozen=True)
class ScoreResult:
    track: str
    kind: str                       # "auto" | "final"
    value: float
    max_possible: float             # 0.70 for "auto", 1.0 for "final"
    components: Dict[str, Component]

    @property
    def official(self) -> bool:
        """True only for the final score computed with human SN and Q."""
        return self.kind == "final"

    @property
    def renormalized(self) -> float:
        """value / max_possible. Convenience only; never present as official."""
        return self.value / self.max_possible


# ----------------------------------------------------------------------------- helpers
def _check_track(track: str) -> None:
    if track not in WEIGHTS:
        raise ValueError(f"track must be 'A' or 'B', got {track!r}")


def _finite(name: str, x: float) -> float:
    if x is None or isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number, got {x!r}")
    return float(x)


def _in_range(name: str, x: float, lo: float, hi: float) -> float:
    x = _finite(name, x)
    if not lo <= x <= hi:
        raise ValueError(f"{name}={x} outside [{lo}, {hi}]")
    return x


def normalize_mos(x: float, cfg: ScoringConfig = ScoringConfig()) -> float:
    x = _in_range("MOS-like score", x, cfg.mos_min, cfg.mos_max)
    return (x - cfg.mos_min) / (cfg.mos_max - cfg.mos_min)


def one_minus_wer(wer: float, cfg: ScoringConfig = ScoringConfig()) -> float:
    wer = _finite("WER", wer)
    if wer < 0:
        raise ValueError(f"WER={wer} must be >= 0")
    return 1.0 - (min(wer, 1.0) if cfg.clip_wer else wer)


def normalize_ss(ss: float, cfg: ScoringConfig = ScoringConfig()) -> float:
    ss = _in_range("SS (cosine)", ss, -1.0, 1.0)
    return max(ss, 0.0) if cfg.clip_ss else ss


def _automatic_components(track, nvpa, wer, pmos, ss, cfg) -> Dict[str, Component]:
    w = WEIGHTS[track]
    nvpa = _in_range("NVPA", nvpa, 0.0, 1.0)
    return {
        "NVPA": Component(nvpa, nvpa, w["NVPA"]),
        "1-WER": Component(_finite("WER", wer), one_minus_wer(wer, cfg), w["1-WER"]),
        "pMOS": Component(_finite("pMOS", pmos), normalize_mos(pmos, cfg), w["pMOS"]),
        "SS": Component(_finite("SS", ss), normalize_ss(ss, cfg), w["SS"]),
    }


# ----------------------------------------------------------------------------- scores
def auto_score(track: str, nvpa: float, wer: float, pmos: float, ss: float,
               cfg: ScoringConfig = ScoringConfig()) -> ScoreResult:
    """Automatic score (Fast mode). Maximum is 0.70 by construction. Not official."""
    _check_track(track)
    comps = _automatic_components(track, nvpa, wer, pmos, ss, cfg)
    return ScoreResult(
        track=track,
        kind="auto",
        value=sum(c.contribution for c in comps.values()),
        max_possible=sum(WEIGHTS[track][k] for k in AUTOMATIC_COMPONENTS),
        components=comps,
    )


def final_score(track: str, nvpa: float, wer: float, pmos: float, ss: float,
                sn: Optional[float], q: Optional[float],
                cfg: ScoringConfig = ScoringConfig()) -> Optional[ScoreResult]:
    """Official-formula score. Returns None ("N/A") unless BOTH human SN and Q are given."""
    _check_track(track)
    if sn is None or q is None:
        return None
    comps = _automatic_components(track, nvpa, wer, pmos, ss, cfg)
    w = WEIGHTS[track]
    comps["SN"] = Component(_finite("SN", sn), normalize_mos(sn, cfg), w["SN"])
    comps["Q"] = Component(_finite("Q", q), normalize_mos(q, cfg), w["Q"])
    return ScoreResult(
        track=track,
        kind="final",
        value=sum(c.contribution for c in comps.values()),
        max_possible=sum(w.values()),
        components=comps,
    )
