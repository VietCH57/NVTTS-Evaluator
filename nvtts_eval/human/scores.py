"""Import rater sheets, check raters against the hidden anchors, aggregate to per-sample scores."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import BootstrapConfig, HumanConfig
from ..stats import bootstrap_ci

FIELDS = ("SN", "Q", "NV_naturalness", "NV_placement")
REQUIRED = ("SN", "Q")


def read_rating_sheet(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Rows as {item_id, SN, Q, NV_naturalness, NV_placement}; values are float or None. Returns (rows, problems)."""
    rows, problems = [], []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for ln, r in enumerate(csv.DictReader(f), start=2):
            item = (r.get("item_id") or "").strip()
            if not item:
                continue
            row: Dict[str, Any] = {"item_id": item}
            for k in FIELDS:
                raw = (r.get(k) or "").strip().replace(",", ".")
                if not raw:
                    row[k] = None
                    continue
                try:
                    v = float(raw)
                except ValueError:
                    problems.append(f"{Path(path).name}:{ln} {item} {k}={raw!r} is not a number")
                    row[k] = None
                    continue
                if not 1.0 <= v <= 5.0:
                    problems.append(f"{Path(path).name}:{ln} {item} {k}={v} outside [1, 5]")
                    row[k] = None
                else:
                    row[k] = v
            rows.append(row)
    return rows, problems


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 5:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def aggregate_ratings(sheets: Dict[str, List[Dict[str, Any]]], key: Dict[str, Any], hcfg: HumanConfig,
                      bcfg: BootstrapConfig, problems: Optional[List[str]] = None) -> Dict[str, Any]:
    """sheets: rater id -> rows. key: contents of PRIVATE_key.json."""
    kinds = {i["item_id"]: i for i in key["items"]}
    problems = list(problems or [])

    # 1) rater sanity from anchors, on the mean of SN and Q
    raters: Dict[str, Dict[str, Any]] = {}
    for rater, rows in sheets.items():
        gt, dg, n_items = [], [], 0
        for r in rows:
            k = kinds.get(r["item_id"])
            if k is None:
                problems.append(f"{rater}: unknown item_id {r['item_id']}")
                continue
            if r["SN"] is None and r["Q"] is None:
                continue
            n_items += 1
            score = _mean([r["SN"], r["Q"]])
            if k["kind"] == "anchor_gt":
                gt.append(score)
            elif k["kind"] == "anchor_degraded":
                dg.append(score)
        gm, dm = _mean(gt), _mean(dg)
        reasons = []
        if gm is not None and gm < hcfg.anchor_gt_min:
            reasons.append(f"ground-truth anchors rated {gm:.2f} < {hcfg.anchor_gt_min}")
        if gm is not None and dm is not None and gm - dm < hcfg.anchor_gap_min:
            reasons.append(f"ground-truth minus degraded anchors = {gm - dm:.2f} < {hcfg.anchor_gap_min}")
        raters[rater] = {"n_items": n_items, "anchor_gt_mean": gm, "anchor_degraded_mean": dm,
                         "n_anchor_gt": len(gt), "n_anchor_degraded": len(dg),
                         "flagged": bool(reasons), "reasons": reasons,
                         "accepted": not (reasons and hcfg.exclude_flagged)}
    accepted = [r for r, v in raters.items() if v["accepted"]]

    # 2) per-sample means over accepted raters (model items only)
    per: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_rater_sn: Dict[str, Dict[str, float]] = defaultdict(dict)
    by_rater_q: Dict[str, Dict[str, float]] = defaultdict(dict)
    for rater in accepted:
        for r in sheets[rater]:
            k = kinds.get(r["item_id"])
            if k is None or k["kind"] != "model":
                continue
            for f in FIELDS:
                if r[f] is not None:
                    per[k["sample_id"]][f].append(r[f])
            if r["SN"] is not None:
                by_rater_sn[rater][k["sample_id"]] = r["SN"]
            if r["Q"] is not None:
                by_rater_q[rater][k["sample_id"]] = r["Q"]
    speaker_of = {i["sample_id"]: i["speaker_id"] for i in key["items"] if i["kind"] == "model"}
    records = []
    for sid in sorted(per):
        rec: Dict[str, Any] = {"sample_id": sid, "speaker_id": speaker_of.get(sid)}
        for f in FIELDS:
            rec[f] = _mean(per[sid][f]) if per[sid][f] else None
            rec[f"n_{f}"] = len(per[sid][f])
        records.append(rec)
    both = [r for r in records if r["SN"] is not None and r["Q"] is not None]

    def summarize(f: str) -> Optional[Dict[str, Any]]:
        vals = [(r[f], r["speaker_id"]) for r in records if r[f] is not None]
        if not vals:
            return None
        v = [x[0] for x in vals]
        kw = dict(n_boot=bcfg.n_boot, alpha=bcfg.alpha, seed=bcfg.seed)
        return {"mean": sum(v) / len(v), "n": len(v), "ci_utterance": bootstrap_ci(v, **kw),
                "ci_speaker": bootstrap_ci(v, clusters=[x[1] for x in vals], **kw)}

    # 3) inter-rater agreement (accepted raters, items rated by both, >= 5 items)
    def agreement(tbl: Dict[str, Dict[str, float]]):
        rs, out = sorted(tbl), []
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                common = sorted(set(tbl[rs[i]]) & set(tbl[rs[j]]))
                r = _pearson([tbl[rs[i]][c] for c in common], [tbl[rs[j]][c] for c in common])
                if r is not None:
                    out.append(r)
        return {"mean_pairwise_pearson": sum(out) / len(out) if out else None, "n_pairs": len(out)}

    return {
        "sample_ids": [r["sample_id"] for r in both],
        "n_rated_samples": len(both),
        "n_raters_total": len(raters), "n_raters_accepted": len(accepted),
        "raters": raters,
        "metrics": {"SN": summarize("SN"), "Q": summarize("Q"),
                    "NV_naturalness": summarize("NV_naturalness"), "NV_placement": summarize("NV_placement")},
        "agreement": {"SN": agreement(by_rater_sn), "Q": agreement(by_rater_q)},
        "records": records,
        "problems": problems,
    }
