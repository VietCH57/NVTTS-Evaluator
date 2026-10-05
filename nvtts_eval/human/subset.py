"""Deterministic, stratified selection of the samples that humans will rate.

Goals (spec v2, section 15): cover rare NV types, keep tail speakers visible (the data is dominated
by a few speakers), and mix short and long utterances. Same seed + same manifest -> same subset.

Algorithm
  1. Rare NV types (<= `rare_type_share` of all events) get priority: utterances containing them are
     taken first, up to `max_rare_fraction` of the subset.
  2. The remaining slots are filled over strata (head/tail speaker x short/long). Quotas are
     proportional to sqrt(stratum size) so that small strata are not drowned out; inside a stratum
     speakers are visited round-robin so one speaker cannot take all slots.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

from ..data.manifest import Manifest, Sample


def _duration(manifest: Manifest, s: Sample) -> Optional[float]:
    p = manifest.resolve_generated(s) or manifest.resolve_ground_truth(s)
    try:
        import soundfile as sf
        i = sf.info(str(p))
        return i.frames / i.samplerate
    except Exception:
        return None


def _round_robin(samples: List[Sample], k: int, rng: random.Random) -> List[Sample]:
    by_spk: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        by_spk[s.speaker_id].append(s)
    spks = sorted(by_spk)
    rng.shuffle(spks)
    for v in by_spk.values():
        rng.shuffle(v)
    out: List[Sample] = []
    while len(out) < k and any(by_spk.values()):
        for sp in spks:
            if by_spk[sp] and len(out) < k:
                out.append(by_spk[sp].pop())
    return out


def _quotas(sizes: Dict[Any, int], total: int) -> Dict[Any, int]:
    """Largest-remainder allocation proportional to sqrt(size), capped by each stratum's size."""
    quota = {k: 0 for k in sizes}
    left = min(total, sum(sizes.values()))
    while left > 0:
        open_ = {k: v for k, v in sizes.items() if quota[k] < v}
        if not open_:
            break
        w = {k: math.sqrt(v - quota[k]) for k, v in open_.items()}
        z = sum(w.values())
        raw = {k: left * w[k] / z for k in open_}
        add = {k: min(int(raw[k]), open_[k] - quota[k]) for k in open_}
        given = sum(add.values())
        if given == 0:                                   # hand out the remainder one by one
            for k in sorted(open_, key=lambda k: -raw[k])[:left]:
                add[k] = 1
            given = sum(add.values())
        for k, v in add.items():
            quota[k] += v
        left -= given
    return quota


def select_subset(manifest: Manifest, size: int, seed: int = 0, rare_type_share: float = 0.05,
                  max_rare_fraction: float = 0.5, long_utt_seconds: float = 15.0,
                  head_speaker_min_utts: int = 20) -> Dict[str, Any]:
    rng = random.Random(seed)
    samples = list(manifest.samples)
    if size >= len(samples):
        chosen = samples
        rare_types: List[str] = []
    else:
        ev_count = Counter(e["type"] for s in samples for e in s.nv_events)
        total_ev = sum(ev_count.values()) or 1
        rare_types = sorted(t for t, c in ev_count.items() if c / total_ev <= rare_type_share)
        chosen: List[Sample] = []
        taken = set()
        cap = int(size * max_rare_fraction)
        rare = [s for s in samples if any(e["type"] in rare_types for e in s.nv_events)]
        rng.shuffle(rare)
        rare.sort(key=lambda s: -sum(e["type"] in rare_types for e in s.nv_events))
        for t in sorted(rare_types, key=lambda t: ev_count[t]):          # rarest type first, guarantee coverage
            for s in rare:
                if len(chosen) >= cap:
                    break
                if s.sample_id not in taken and any(e["type"] == t for e in s.nv_events):
                    chosen.append(s)
                    taken.add(s.sample_id)
                    break
        for s in rare:
            if len(chosen) >= cap:
                break
            if s.sample_id not in taken:
                chosen.append(s)
                taken.add(s.sample_id)

        spk_n = Counter(s.speaker_id for s in samples)
        def stratum(s: Sample):
            d = _duration(manifest, s)
            long_ = (d >= long_utt_seconds) if d is not None else (len(s.clean_text.split()) >= 50)
            return ("head" if spk_n[s.speaker_id] >= head_speaker_min_utts else "tail", "long" if long_ else "short")
        pool: Dict[Any, List[Sample]] = defaultdict(list)
        for s in samples:
            if s.sample_id not in taken:
                pool[stratum(s)].append(s)
        quota = _quotas({k: len(v) for k, v in pool.items()}, size - len(chosen))
        for k in sorted(pool):
            chosen += _round_robin(pool[k], quota[k], rng)

    ids = sorted(s.sample_id for s in chosen)
    spk_n = Counter(s.speaker_id for s in samples)
    return {
        "sample_ids": ids,
        "params": {"size": size, "seed": seed, "rare_type_share": rare_type_share,
                   "max_rare_fraction": max_rare_fraction, "long_utt_seconds": long_utt_seconds,
                   "head_speaker_min_utts": head_speaker_min_utts},
        "rare_types": rare_types,
        "composition": {
            "n": len(ids), "n_manifest": len(samples),
            "speakers": len({s.speaker_id for s in chosen}),
            "head_speaker_samples": sum(1 for s in chosen if spk_n[s.speaker_id] >= head_speaker_min_utts),
            "events_by_type": dict(Counter(e["type"] for s in chosen for e in s.nv_events)),
        },
    }
