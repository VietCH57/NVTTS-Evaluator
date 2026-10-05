"""Re-evaluate NVPA from a stored "nvpa" artifact under different tolerances and detector thresholds,
WITHOUT re-running any model (the artifact keeps the detector probabilities of every gap).

Use it on the ground-truth calibration run to see
  * how many misses are only "wrong_position" (a detection of the right type exists, but farther
    than the tolerance) and whether detections sit systematically before/after the gold gap
    (signed offset histogram);
  * how NVPA and the random-placement baseline move together. Read the LIFT (NVPA minus baseline),
    not NVPA alone: a looser tolerance or a lower threshold raises both.

Tolerances are `n` (symmetric +-n gaps) or `before:after` (e.g. `0:1` = the detection may be one gap later but
never earlier).

Per-type thresholds from training stay as they are (rare types have too few dev events to tune);
`scale` multiplies all enabled thresholds (types disabled with threshold > 1 stay disabled).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .matching import Tolerance, match_events, tolerance_bounds, tolerance_label
from .summary import shuffle_baseline


def rethreshold(records: Sequence[Dict[str, Any]], thresholds: Dict[str, float], scale: float = 1.0
                ) -> List[Dict[str, Any]]:
    thr = {t: (v if v > 1.0 else min(v * scale, 1.0)) for t, v in thresholds.items()}
    out = []
    for r in records:
        if "error" in r:
            continue
        pred = []
        for g, p in r["probs"]:
            ts = sorted(t for t, v in p.items() if v >= thr.get(t, 1.01))
            if ts:
                pred.append([g, ts])
        out.append({**r, "pred": pred})
    return out


def evaluate(records: Sequence[Dict[str, Any]], tolerance: Tolerance, policy: str = "miss") -> Dict[str, Any]:
    hits = n = spur = words = 0
    per: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    reasons: Counter = Counter()
    for r in records:
        gold = [(e["type"], e["gap"]) for e in r["events"]]
        pred = {g: set(ts) for g, ts in r["pred"]}
        probs = {g: p for g, p in r["probs"]}
        res, sp = match_events(gold, pred, probs, set(r["valid_gaps"]), tolerance)
        spur += len(sp)
        words += r["n_ref_words"]
        for e in res:
            if policy == "exclude" and e.reason == "alignment_unreliable":
                continue
            n += 1
            hits += e.hit
            per[e.type][0] += e.hit
            per[e.type][1] += 1
            reasons[e.reason] += 1
    return {"nvpa": hits / n if n else float("nan"), "n_events": n,
            "spurious_per_100_words": 100 * spur / words if words else float("nan"),
            "per_type": {t: (h / c if c else float("nan")) for t, (h, c) in sorted(per.items())},
            "reasons": dict(reasons)}


def offset_histogram(records: Sequence[Dict[str, Any]], max_offset: int = 5) -> Dict[str, Any]:
    """Signed distance (detected gap - gold gap, in words) from every gold event to the NEAREST detection of
    the same type within +-max_offset gaps; events with none are counted under 'none'. Ties prefer the earlier gap."""
    hist: Counter = Counter()
    by_type: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        pred: Dict[int, set] = {g: set(ts) for g, ts in r["pred"]}
        for e in r["events"]:
            best: Optional[int] = None
            for g2, ts in pred.items():
                d = g2 - e["gap"]
                if e["type"] in ts and abs(d) <= max_offset and (best is None or (abs(d), d) < (abs(best), best)):
                    best = d
            key = "none" if best is None else best
            hist[key] += 1
            by_type[e["type"]][key] += 1
    return {"all": dict(hist), "by_type": {t: dict(c) for t, c in by_type.items()}}


def sweep(records: Sequence[Dict[str, Any]], thresholds: Dict[str, float], tolerances: Sequence[Tolerance] = (0, 1, 2, 3),
          scales: Sequence[float] = (0.5, 0.75, 1.0, 1.25), policy: str = "miss", reps: int = 20, seed: int = 0
          ) -> Dict[str, Any]:
    rows = []
    for scale in scales:
        rt = rethreshold(records, thresholds, scale)
        for tol in tolerances:
            ev = evaluate(rt, tol, policy)
            base = shuffle_baseline(rt, tol, reps, seed, policy)
            b, a = tolerance_bounds(tol)
            rows.append({"tolerance": tolerance_label(tol), "before": b, "after": a, "scale": scale, **ev,
                         "shuffle": base.get("mean"),
                         "lift": ev["nvpa"] - base["mean"] if base.get("n_reps") else None})
    return {"rows": rows, "offsets_at_scale_1": offset_histogram(rethreshold(records, thresholds, 1.0))}


def format_sweep(res: Dict[str, Any]) -> str:
    def f(x, nd=3):
        return "  -  " if x is None or x != x else f"{x:.{nd}f}"
    types = sorted({t for r in res["rows"] for t in r["per_type"]})
    head = f"{'tol':>4} {'scale':>5} {'NVPA':>6} {'random':>6} {'lift':>6} {'spur/100w':>9}  " + " ".join(f"{t[:6]:>6}" for t in types)
    lines = [head, "-" * len(head)]
    for r in res["rows"]:
        lines.append(f"{r['tolerance']:>4} {r['scale']:>5.2f} {f(r['nvpa']):>6} {f(r['shuffle']):>6} {f(r['lift']):>6} "
                     f"{f(r['spurious_per_100_words'], 2):>9}  " + " ".join(f"{f(r['per_type'].get(t)):>6}" for t in types))
    h = res["offsets_at_scale_1"]["all"]
    keys = sorted((k for k in h if k != "none"), key=int)
    lines.append("\nSigned offset of the nearest same-type detection (detected gap - gold gap, scale 1):")
    lines.append("  " + "  ".join(f"{k:+d}:{h[k]}" for k in keys) + f"  none:{h.get('none', 0)}")
    return "\n".join(lines)
