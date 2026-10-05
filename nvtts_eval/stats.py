"""Shared descriptive statistics: summaries, bootstrap CIs, per-group breakdowns.

Design notes
* Deterministic: bootstrap uses `random.Random(seed)`.
* Missing values (None / NaN / records without the field, e.g. failed samples) are
  never silently dropped: they are counted as `n_missing` next to every result.
* Cluster bootstrap: the dataset is dominated by a few speakers (spk_0000 is ~37% of
  dev). Passing `clusters` (e.g. speaker ids) resamples whole clusters, giving an
  interval that reflects uncertainty about *new speakers*, not just new utterances.
  Which of the two a report uses must be stated in the report.
* Percentiles use linear interpolation.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

NAN = float("nan")


def _valid(x: Any) -> bool:
    return x is not None and not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return NAN
    k = (len(sorted_vals) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def describe(values: Iterable[Any], percentiles: Sequence[int] = (5, 25, 50, 75, 95)) -> Dict[str, float]:
    vals = [x for x in values if _valid(x)]
    if not vals:
        return {"n": 0}
    v = sorted(float(x) for x in vals)
    mean = sum(v) / len(v)
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / len(v))     # population std
    out: Dict[str, float] = {"n": len(v), "mean": mean, "std": std, "min": v[0], "max": v[-1]}
    for q in percentiles:
        out[f"p{q}"] = percentile(v, q)
    out["median"] = percentile(v, 50)
    return out


def bootstrap_ci(
    values: Sequence[Any],
    clusters: Optional[Sequence[Hashable]] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean. With `clusters`, resamples whole clusters."""
    if clusters is not None and len(clusters) != len(values):
        raise ValueError("clusters must have the same length as values")
    pairs = [(float(v), (clusters[i] if clusters is not None else None))
             for i, v in enumerate(values) if _valid(v)]
    if len(pairs) < 2:
        return NAN, NAN
    rng = random.Random(seed)
    means: List[float] = []
    if clusters is None:
        vals = [p[0] for p in pairs]
        n = len(vals)
        for _ in range(n_boot):
            means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    else:
        agg: Dict[Hashable, List[float]] = defaultdict(lambda: [0.0, 0])
        for v, c in pairs:
            agg[c][0] += v
            agg[c][1] += 1
        if len(agg) < 2:
            return NAN, NAN
        cl = list(agg.values())
        k = len(cl)
        for _ in range(n_boot):
            picks = [cl[rng.randrange(k)] for _ in range(k)]
            means.append(sum(p[0] for p in picks) / sum(p[1] for p in picks))
    means.sort()
    return percentile(means, 100 * alpha / 2), percentile(means, 100 * (1 - alpha / 2))


def group_stats(
    records: Iterable[Dict[str, Any]],
    group_key: str,
    value_key: str,
    min_n: int = 10,
    cluster_key: Optional[str] = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Mean, CI, n, n_missing per group. `low_n` flags groups with fewer than `min_n` valid values."""
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[r.get(group_key)].append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for g in sorted(groups, key=lambda x: str(x)):
        rs = groups[g]
        vals = [r.get(value_key) for r in rs]
        valid = [v for v in vals if _valid(v)]
        cl = [r.get(cluster_key) for r in rs] if cluster_key else None
        lo, hi = bootstrap_ci(vals, cl, n_boot=n_boot, seed=seed)
        out[str(g)] = {
            "n": len(valid),
            "n_missing": len(rs) - len(valid),
            "mean": (sum(valid) / len(valid)) if valid else NAN,
            "ci_low": lo,
            "ci_high": hi,
            "low_n": len(valid) < min_n,
        }
    return out


def macro_mean(records: Iterable[Dict[str, Any]], group_key: str, value_key: str) -> float:
    """Unweighted mean of per-group means (e.g. per-speaker macro average)."""
    sums: Dict[Any, List[float]] = defaultdict(list)
    for r in records:
        v = r.get(value_key)
        if _valid(v):
            sums[r.get(group_key)].append(float(v))
    if not sums:
        return NAN
    return sum(sum(v) / len(v) for v in sums.values()) / len(sums)


def bootstrap_ratio_ci(
    numerators: Sequence[Any],
    denominators: Sequence[Any],
    clusters: Optional[Sequence[Hashable]] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI of sum(num)/sum(den), e.g. corpus-level WER
    (total errors / total reference words). With `clusters`, resamples whole clusters."""
    if len(numerators) != len(denominators):
        raise ValueError("numerators and denominators must have the same length")
    if clusters is not None and len(clusters) != len(numerators):
        raise ValueError("clusters must have the same length as numerators")
    rows = [(float(n), float(d), (clusters[i] if clusters is not None else None))
            for i, (n, d) in enumerate(zip(numerators, denominators)) if _valid(n) and _valid(d)]
    if len(rows) < 2:
        return NAN, NAN
    rng = random.Random(seed)
    ratios: List[float] = []
    if clusters is None:
        k = len(rows)
        for _ in range(n_boot):
            picks = [rows[rng.randrange(k)] for _ in range(k)]
            den = sum(p[1] for p in picks)
            if den > 0:
                ratios.append(sum(p[0] for p in picks) / den)
    else:
        agg: Dict[Hashable, List[float]] = defaultdict(lambda: [0.0, 0.0])
        for n, d, c in rows:
            agg[c][0] += n
            agg[c][1] += d
        if len(agg) < 2:
            return NAN, NAN
        cl = list(agg.values())
        k = len(cl)
        for _ in range(n_boot):
            picks = [cl[rng.randrange(k)] for _ in range(k)]
            den = sum(p[1] for p in picks)
            if den > 0:
                ratios.append(sum(p[0] for p in picks) / den)
    if not ratios:
        return NAN, NAN
    ratios.sort()
    return percentile(ratios, 100 * alpha / 2), percentile(ratios, 100 * (1 - alpha / 2))


def _ranks(x: Sequence[float]) -> List[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1          # average rank for ties
        i = j + 1
    return r


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (NaN if fewer than 3 pairs or no variation)."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 3:
        return NAN
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sx == 0 or sy == 0:
        return NAN
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)
