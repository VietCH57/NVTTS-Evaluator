"""Map each gap between reference words to an audio window of the generated audio."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..metrics.wer import TextNormConfig, normalize_text
from .timing import HypWord


@dataclass(frozen=True)
class Window:
    t0: float
    t1: float
    widened: bool    # True if a neighbouring reference word was not recognised, so the window spans more audio

    @property
    def center(self) -> float:
        return 0.5 * (self.t0 + self.t1)


def norm_gap_map(raw_words: Sequence[str], cfg: TextNormConfig = TextNormConfig()) -> List[int]:
    """Gap index in RAW whitespace words (as stored in the manifest) -> gap index in NORMALISED
    words (what ASR alignment uses). Normalisation can drop a word (pure punctuation) or split one."""
    out = [0]
    for w in raw_words:
        out.append(out[-1] + len(normalize_text(w, cfg).split()))
    return out


def ref_to_hyp_map(pairs) -> Dict[int, int]:
    return {i: j for i, j in pairs if i is not None and j is not None}


def build_gap_windows(
    n_ref: int,
    ref_to_hyp: Dict[int, int],
    hyp: Sequence[HypWord],
    duration: float,
    min_len: float = 0.2,
    max_len: float = 3.0,
) -> List[Optional[Window]]:
    """One entry per gap 0..n_ref. None = no reliable window (no aligned neighbour, or too long)."""
    out: List[Optional[Window]] = []
    for g in range(n_ref + 1):
        prev_i = g - 1
        while prev_i >= 0 and prev_i not in ref_to_hyp:
            prev_i -= 1
        next_i = g
        while next_i < n_ref and next_i not in ref_to_hyp:
            next_i += 1
        if prev_i < 0 and next_i >= n_ref:
            out.append(None)
            continue
        t0 = hyp[ref_to_hyp[prev_i]].last_start if prev_i >= 0 else 0.0
        t1 = hyp[ref_to_hyp[next_i]].start if next_i < n_ref else duration
        widened = (g > 0 and prev_i != g - 1) or (g < n_ref and next_i != g)
        t1 = max(t1, t0)
        if t1 - t0 < min_len:
            t1 = min(duration, t0 + min_len)
            if t1 - t0 < min_len:
                t0 = max(0.0, t1 - min_len)
        if t1 - t0 > max_len:
            out.append(None)
            continue
        out.append(Window(t0, t1, widened))
    return out
