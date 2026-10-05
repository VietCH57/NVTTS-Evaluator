"""Recognised words with timing, built from ASR token timestamps.

sherpa-onnx tokens carry a leading space on the first piece of each word (e.g. ' MI','LA','N').
For every word we keep the time of its first token (`start`) and of its last token
(`last_start`). A transducer only gives token emission times, so the true end of a word is
unknown; the window used for a gap therefore begins at `last_start` of the previous word
(the final syllable's onset), and training and inference use exactly the same definition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..metrics.wer import TextNormConfig, normalize_text

WORD_START_MARKERS = (" ", "\u2581")


@dataclass(frozen=True)
class HypWord:
    text: str          # normalised
    start: float       # seconds, first token of the word
    last_start: float  # seconds, last token of the word


def hyp_words_from_asr(rec: Dict[str, Any], cfg: TextNormConfig = TextNormConfig()) -> Optional[List[HypWord]]:
    """None when the ASR record has no usable token timestamps."""
    toks, ts = rec.get("tokens"), rec.get("timestamps")
    if not toks or not ts or len(toks) != len(ts):
        return None
    groups: List[list] = []
    for tok, t in zip(toks, ts):
        piece = tok.lstrip("".join(WORD_START_MARKERS))
        if not groups or tok[:1] in WORD_START_MARKERS:
            groups.append([[piece], float(t), float(t)])
        else:
            groups[-1][0].append(piece)
            groups[-1][2] = float(t)
    out: List[HypWord] = []
    for pieces, start, last in groups:
        for w in normalize_text("".join(pieces), cfg).split():   # a token-word may normalise to 0 or several words
            out.append(HypWord(w, start, last))
    return out
