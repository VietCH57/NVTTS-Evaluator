"""Build summary.json / console report from cached artifacts. Reads artifacts only; never runs a model.

Rules (spec v2): no fabricated numbers. AutoScore needs all four automatic components
(NVPA, WER, pMOS, SS); until every one is available it is reported as null with the list
of what is missing. Human SN/Q are not produced here, so final_score is always null.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional

from ..config import EvalConfig
from ..core.artifacts import ArtifactStore
from ..data.manifest import Manifest
from ..metrics.wer import summarize_wer, wer_records
from ..scoring.formulas import (WEIGHTS, ScoreResult, auto_score, normalize_mos, normalize_ss,
                                one_minus_wer)
from ..stats import bootstrap_ci, describe, group_stats, macro_mean


def _scrub(obj: Any) -> Any:
    """Make JSON-safe: NaN/Inf -> None, tuples -> lists."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def _score_to_dict(r: Optional[ScoreResult]) -> Optional[Dict[str, Any]]:
    if r is None:
        return None
    return {"track": r.track, "kind": r.kind, "value": r.value, "max_possible": r.max_possible,
            "renormalized": r.renormalized, "official": r.official,
            "components": {k: {**dataclasses.asdict(c), "contribution": c.contribution}
                           for k, c in r.components.items()}}


def _summarize_field(records: List[Dict[str, Any]], field: str, speaker_of: Dict[str, str],
                     cfg: EvalConfig) -> Dict[str, Any]:
    recs = [{**r, "speaker_id": speaker_of.get(r["sample_id"])} for r in records]
    ok = [r for r in recs if "error" not in r and r.get(field) is not None]
    b = cfg.bootstrap
    vals = [r[field] for r in ok]
    kw = dict(n_boot=b.n_boot, alpha=b.alpha, seed=b.seed)
    d = describe(vals)
    return {
        "field": field, "n": len(ok), "n_failed": len(recs) - len(ok),
        "mean": d.get("mean"), "median": d.get("median"), "std": d.get("std"),
        "min": d.get("min"), "max": d.get("max"),
        "ci_utterance": bootstrap_ci(vals, **kw),
        "ci_speaker": bootstrap_ci(vals, clusters=[r["speaker_id"] for r in ok], **kw),
        "macro_speaker_mean": macro_mean(ok, "speaker_id", field),
        "by_speaker": group_stats(ok, "speaker_id", field, min_n=b.min_n,
                                  n_boot=min(b.n_boot, 500), seed=b.seed),
    }


def build_summary(manifest: Manifest, store: ArtifactStore, cfg: EvalConfig,
                  nvpa: Optional[float] = None) -> Dict[str, Any]:
    track = manifest.header.track
    speaker_of = {s.sample_id: s.speaker_id for s in manifest}
    b = cfg.bootstrap
    metrics: Dict[str, Any] = {}
    values: Dict[str, Optional[float]] = {"NVPA": nvpa, "WER": None, "pMOS": None, "SS": None}

    if store.read_meta("asr") is not None:
        recs = wer_records(manifest, store.read_records("asr"), cfg.text_norm)
        w = summarize_wer(recs, cfg.wer.aggregation, b.n_boot, b.alpha, b.seed, b.min_n)
        metrics["wer"] = w
        values["WER"] = w["value"]
    if store.read_meta("pmos") is not None:
        p = _summarize_field(store.read_records("pmos"), cfg.pmos.output, speaker_of, cfg)
        p["output_used"] = cfg.pmos.output
        metrics["pmos"] = p
        values["pMOS"] = p["mean"]
    if store.read_meta("ss") is not None:
        s = _summarize_field(store.read_records("ss"), "cosine", speaker_of, cfg)
        metrics["ss"] = s
        values["SS"] = s["mean"]

    missing = [k for k, v in values.items() if v is None or (isinstance(v, float) and not math.isfinite(v))]
    auto: Optional[ScoreResult] = None
    partial: Dict[str, Any] = {}
    if not missing:
        auto = auto_score(track, values["NVPA"], values["WER"], values["pMOS"], values["SS"], cfg.scoring)
    else:
        w_ = WEIGHTS[track]
        norm = {"WER": lambda v: one_minus_wer(v, cfg.scoring), "pMOS": lambda v: normalize_mos(v, cfg.scoring),
                "SS": lambda v: normalize_ss(v, cfg.scoring), "NVPA": lambda v: v}
        key = {"WER": "1-WER", "pMOS": "pMOS", "SS": "SS", "NVPA": "NVPA"}
        for k, v in values.items():
            if k not in missing:
                nv = norm[k](v)
                partial[k] = {"raw": v, "normalized": nv, "weight": w_[key[k]], "contribution": w_[key[k]] * nv}

    return _scrub({
        "track": track,
        "data_counts": {"n_samples": len(manifest), "n_speakers": len(set(speaker_of.values())),
                        "manifest_source": manifest.header.source, "split": manifest.header.split},
        "automatic_metrics": metrics,
        "automatic_score": _score_to_dict(auto),
        "automatic_score_missing_components": missing,
        "automatic_score_partial_components": partial,
        "human_metrics": {"SN": None, "Q": None},
        "final_score": None,
        "active_assumptions": cfg.to_dict(),
    })


def format_summary(s: Dict[str, Any]) -> str:
    def f(x, nd=3):
        return "N/A" if x is None else f"{x:.{nd}f}"

    def ci(c):
        return "" if not c or c[0] is None else f" [{c[0]:.3f}, {c[1]:.3f}]"

    m = s["automatic_metrics"]
    lines = [f"Track {s['track']} | {s['data_counts']['n_samples']} samples, "
             f"{s['data_counts']['n_speakers']} speakers | source={s['data_counts']['manifest_source']}"]
    if "wer" in m:
        w = m["wer"]
        lines.append(f"WER ({w['aggregation_used']}): {f(w['value'])}  95% CI utt{ci(w['corpus_wer_ci_utterance'])} "
                     f"spk{ci(w['corpus_wer_ci_speaker'])}  failed={w['n_failed']}  "
                     f"[S={w['sub']} D={w['del']} I={w['ins']} / {w['total_ref_words']} words]"
                     + (f"  WARNING: {w['n_hyp_with_digits']} hypotheses contain digits" if w["n_hyp_with_digits"] else ""))
    if "pmos" in m:
        p = m["pmos"]
        lines.append(f"pMOS ({p['output_used']}): {f(p['mean'], 2)}  95% CI utt{ci(p['ci_utterance'])} "
                     f"spk{ci(p['ci_speaker'])}  failed={p['n_failed']}")
    if "ss" in m:
        q = m["ss"]
        lines.append(f"SS (cosine): {f(q['mean'])}  95% CI utt{ci(q['ci_utterance'])} spk{ci(q['ci_speaker'])}  "
                     f"macro-speaker={f(q['macro_speaker_mean'])}  failed={q['n_failed']}")
    a = s["automatic_score"]
    if a:
        lines.append(f"AutoScore: {f(a['value'])} (max {a['max_possible']:.2f}; renormalized {f(a['renormalized'])})  [not official]")
    else:
        lines.append(f"AutoScore: N/A  (missing: {', '.join(s['automatic_score_missing_components'])})")
    lines.append("Human metrics: SN N/A | Q N/A      Final score: N/A")
    return "\n".join(lines)
