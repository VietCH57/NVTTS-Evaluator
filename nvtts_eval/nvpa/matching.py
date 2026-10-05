"""One-to-one matching of gold NV events to detections, with a failure taxonomy.

Positions are gap indices in NORMALISED reference words. A gold event (type t, gap g) is
realised if some detection of type t sits at a gap g' with |g - g'| <= tolerance, and each
detection can realise only one gold event (greedy, closest first, then most confident).

Reasons for a miss:
  alignment_unreliable  no usable window at the gold gap (and nothing matched nearby)
  wrong_type            a different NV type was detected within tolerance
  wrong_position        the right type was detected, but farther than the tolerance
  missing               nothing relevant was detected
Same-gap events of the same type (e.g. two breaths in one gap) cannot both be matched, because
the detector reports presence per window; this is rare in the data (same-gap pairs are mostly
sniff + breathing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

Gold = Tuple[str, int]                 # (type, normalised gap)


@dataclass
class EventResult:
    type: str
    gap: int
    hit: bool
    reason: str                        # "ok" or a miss reason
    matched_gap: Optional[int] = None
    offset_words: Optional[int] = None
    prob: Optional[float] = None       # detector probability of this type at the matched (or gold) gap


def match_events(
    gold: Sequence[Gold],
    pred: Dict[int, Set[str]],
    probs: Dict[int, Dict[str, float]],
    valid_gaps: Set[int],
    tolerance: int,
) -> Tuple[List[EventResult], List[Tuple[str, int]]]:
    """Return (one result per gold event, unmatched detections = spurious)."""
    cands = []
    for e, (t, g) in enumerate(gold):
        for g2, types in pred.items():
            if t in types and abs(g2 - g) <= tolerance:
                cands.append((abs(g2 - g), -probs.get(g2, {}).get(t, 0.0), e, g2))
    cands.sort()
    used_event: Dict[int, int] = {}
    used_pred: Set[Tuple[str, int]] = set()
    for _, _, e, g2 in cands:
        t = gold[e][0]
        if e in used_event or (t, g2) in used_pred:
            continue
        used_event[e] = g2
        used_pred.add((t, g2))

    results: List[EventResult] = []
    for e, (t, g) in enumerate(gold):
        if e in used_event:
            g2 = used_event[e]
            results.append(EventResult(t, g, True, "ok", g2, g2 - g, probs.get(g2, {}).get(t)))
            continue
        near_other = any(t2 != t and abs(g2 - g) <= tolerance for g2, types in pred.items() for t2 in types)
        far_same = any(t in types and abs(g2 - g) > tolerance for g2, types in pred.items())
        if g not in valid_gaps and not near_other:
            reason = "alignment_unreliable"
        elif near_other:
            reason = "wrong_type"
        elif far_same:
            reason = "wrong_position"
        else:
            reason = "missing"
        results.append(EventResult(t, g, False, reason, None, None, probs.get(g, {}).get(t)))

    spurious = [(t, g2) for g2, types in sorted(pred.items()) for t in sorted(types) if (t, g2) not in used_pred]
    return results, spurious
