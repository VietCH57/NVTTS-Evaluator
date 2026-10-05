"""Aggregate NVPA records into the headline value, diagnostics and the shuffle baseline."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence

from ..stats import NAN, bootstrap_ci, bootstrap_ratio_ci, group_stats
from .matching import Tolerance, match_events, tolerance_bounds, tolerance_label


def _flatten(records, speaker_of, policy):
    """Events of successful samples; with policy 'exclude', unreliable-alignment events are dropped."""
    out = []
    for r in records:
        if "error" in r:
            continue
        for e in r["events"]:
            if policy == "exclude" and e["reason"] == "alignment_unreliable":
                continue
            out.append({**e, "sample_id": r["sample_id"], "speaker_id": speaker_of.get(r["sample_id"]),
                        "n_ref_words": r["n_ref_words"]})
    return out


def _pos_class(e) -> str:
    return "start" if e["gap"] == 0 else "end" if e["gap"] >= e["n_ref_words"] else "mid"


def shuffle_baseline(records: Sequence[Dict[str, Any]], tolerance: Tolerance, n_reps: int = 50, seed: int = 0,
                     policy: str = "miss") -> Dict[str, Any]:
    """NVPA obtained when each utterance's gold events keep their types but get uniformly random gaps,
    against the SAME detections. This is the score a position-blind system could reach by chance
    at this tolerance."""
    ok = [r for r in records if "error" not in r and r["events"]]
    if not ok:
        return {"n_reps": 0}
    prepared = []
    for r in ok:
        pred = {g: set(ts) for g, ts in r["pred"]}
        probs = {g: p for g, p in r["probs"]}
        prepared.append(([e["type"] for e in r["events"]], pred, probs, set(r["valid_gaps"]), r["n_gaps"]))
    vals: List[float] = []
    for rep in range(n_reps):
        rng = random.Random(seed * 100003 + rep)
        hits = n = 0
        for types, pred, probs, valid, n_gaps in prepared:
            gold = [(t, rng.randrange(n_gaps)) for t in types]
            res, _ = match_events(gold, pred, probs, valid, tolerance)
            for e in res:
                if policy == "exclude" and e.reason == "alignment_unreliable":
                    continue
                n += 1
                hits += e.hit
        vals.append(hits / n if n else NAN)
    vals = [v for v in vals if v == v]
    mean = sum(vals) / len(vals) if vals else NAN
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if vals else NAN
    return {"n_reps": len(vals), "mean": mean, "std": std, "min": min(vals, default=NAN), "max": max(vals, default=NAN)}


def summarize_nvpa(records: Sequence[Dict[str, Any]], speaker_of: Dict[str, str], tolerance: Tolerance,
                   policy: str = "miss", n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                   min_n: int = 10, shuffle_reps: int = 50) -> Dict[str, Any]:
    ok = [r for r in records if "error" not in r]
    ev = _flatten(records, speaker_of, policy)
    n = len(ev)
    hits = sum(e["hit"] for e in ev)
    kw = dict(n_boot=n_boot, alpha=alpha, seed=seed)

    per_sample: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])      # hits, events, speaker index holder
    spk_of_s = {}
    for e in ev:
        per_sample[e["sample_id"]][0] += int(e["hit"])
        per_sample[e["sample_id"]][1] += 1
        spk_of_s[e["sample_id"]] = e["speaker_id"]
    ids = sorted(per_sample)
    nums = [per_sample[i][0] for i in ids]
    dens = [per_sample[i][1] for i in ids]
    cl = [spk_of_s[i] for i in ids]

    by_type = group_stats([{**e, "v": 1.0 if e["hit"] else 0.0} for e in ev], "type", "v", min_n=min_n,
                          cluster_key="speaker_id", n_boot=min(n_boot, 500), seed=seed)
    by_pos = group_stats([{**e, "pos": _pos_class(e), "v": 1.0 if e["hit"] else 0.0} for e in ev],
                         "pos", "v", min_n=min_n, n_boot=min(n_boot, 500), seed=seed)
    spk_rates = []
    spk_hits: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for e in ev:
        spk_hits[e["speaker_id"]][0] += int(e["hit"])
        spk_hits[e["speaker_id"]][1] += 1
    spk_rates = [h / c for h, c in spk_hits.values() if c]

    reasons = Counter(e["reason"] for e in ev)
    n_wrong_type = reasons["wrong_type"]
    hit_offsets = [e["offset_words"] for e in ev if e["hit"] and e["offset_words"] is not None]
    hit_secs = [abs(e["sec_offset"]) for e in ev if e["hit"] and e.get("sec_offset") is not None]
    total_words = sum(r["n_ref_words"] for r in ok)
    n_spur = sum(len(r["spurious"]) for r in ok)
    macro_type = [v["mean"] for v in by_type.values() if v["n"] > 0]

    return {
        "value": hits / n if n else NAN,
        "definition": "micro: realised gold NV events / gold NV events",
        "unreliable_policy": policy, "tolerance": tolerance_label(tolerance),
        "tolerance_before": tolerance_bounds(tolerance)[0], "tolerance_after": tolerance_bounds(tolerance)[1],
        "n_events": n, "n_hits": hits, "n_samples": len(ok), "n_failed": len(records) - len(ok),
        "n_no_timing_samples": sum(1 for r in ok if r.get("no_timing")),
        "ci_utterance": bootstrap_ratio_ci(nums, dens, **kw),
        "ci_speaker": bootstrap_ratio_ci(nums, dens, clusters=cl, **kw),
        "macro_type": sum(macro_type) / len(macro_type) if macro_type else NAN,
        "macro_speaker": sum(spk_rates) / len(spk_rates) if spk_rates else NAN,
        "by_type": by_type, "by_position": by_pos,
        "reasons": dict(reasons),
        "detection_rate": (hits + n_wrong_type) / n if n else NAN,
        "type_accuracy_given_detected": hits / (hits + n_wrong_type) if (hits + n_wrong_type) else NAN,
        "placement_error_words": dict(Counter(hit_offsets)),
        "mean_abs_placement_error_sec": sum(hit_secs) / len(hit_secs) if hit_secs else NAN,
        "spurious_total": n_spur, "spurious_per_100_words": 100 * n_spur / total_words if total_words else NAN,
        "shuffle_baseline": shuffle_baseline(records, tolerance, shuffle_reps, seed, policy),
    }
