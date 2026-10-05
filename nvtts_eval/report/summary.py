"""Build summary.json / console report from cached artifacts. Reads artifacts only; never runs a model.

Rules (spec v2): no fabricated numbers.
* AutoScore needs all four automatic components (NVPA, WER, pMOS, SS); otherwise it is null and the
  missing components are listed.
* The final score needs human SN and Q (see human/). When they come from a rated SUBSET, the
  automatic components of the final score are computed on that same subset
  (`human.auto_on_subset`, default true); full-set automatic results are reported separately.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Set

from ..config import EvalConfig
from ..core.artifacts import ArtifactStore
from ..data.manifest import Manifest
from ..metrics.wer import summarize_wer, wer_records
from ..nvpa.summary import summarize_nvpa
from ..scoring.formulas import (WEIGHTS, ScoreResult, auto_score, final_score, normalize_mos, normalize_ss,
                                one_minus_wer)
from ..stats import bootstrap_ci, describe, group_stats, macro_mean, spearman


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


def _filter(records: List[Dict[str, Any]], ids: Optional[Set[str]]) -> List[Dict[str, Any]]:
    return records if ids is None else [r for r in records if r["sample_id"] in ids]


def summarize_field(records: List[Dict[str, Any]], field: str, speaker_of: Dict[str, str],
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
        "by_speaker": group_stats(ok, "speaker_id", field, min_n=b.min_n, n_boot=min(b.n_boot, 500), seed=b.seed),
    }


def automatic_metrics(manifest: Manifest, store: ArtifactStore, cfg: EvalConfig,
                      ids: Optional[Set[str]] = None, nvpa_override: Optional[float] = None):
    """Summaries of every available automatic metric over `ids` (None = all samples).
    Returns (metrics dict, values dict with keys NVPA/WER/pMOS/SS -> float or None)."""
    m = manifest if ids is None else manifest.subset(ids)
    speaker_of = {s.sample_id: s.speaker_id for s in m}
    b = cfg.bootstrap
    metrics: Dict[str, Any] = {}
    values: Dict[str, Optional[float]] = {"NVPA": nvpa_override, "WER": None, "pMOS": None, "SS": None}

    if store.read_meta("asr") is not None:
        recs = wer_records(m, _filter(store.read_records("asr"), set(speaker_of)), cfg.text_norm)
        w = summarize_wer(recs, cfg.wer.aggregation, b.n_boot, b.alpha, b.seed, b.min_n)
        metrics["wer"] = w
        values["WER"] = w["value"]
    if store.read_meta("pmos") is not None:
        p = summarize_field(_filter(store.read_records("pmos"), set(speaker_of)), cfg.pmos.output, speaker_of, cfg)
        p["output_used"] = cfg.pmos.output
        metrics["pmos"] = p
        values["pMOS"] = p["mean"]
    if store.read_meta("ss") is not None:
        s = summarize_field(_filter(store.read_records("ss"), set(speaker_of)), "cosine", speaker_of, cfg)
        metrics["ss"] = s
        values["SS"] = s["mean"]
    meta = store.read_meta("nvpa")
    if meta is not None:
        params = meta.config.get("params", {})
        n = summarize_nvpa(_filter(store.read_records("nvpa"), set(speaker_of)), speaker_of,
                           params.get("tolerance_words", cfg.nvpa.tolerance_words), cfg.nvpa.unreliable_policy,
                           b.n_boot, b.alpha, b.seed, b.min_n, cfg.nvpa.shuffle_reps)
        det = meta.config.get("detector", {})
        n["detector"] = {"backend": det.get("backend"), "thresholds": det.get("thresholds"),
                         "training": det.get("train")}
        metrics["nvpa"] = n
        if nvpa_override is None:
            values["NVPA"] = n["value"]
    return metrics, values


def _missing(values: Dict[str, Optional[float]]) -> List[str]:
    return [k for k, v in values.items() if v is None or (isinstance(v, float) and not math.isfinite(v))]


def _partial(track: str, values: Dict[str, Optional[float]], cfg: EvalConfig) -> Dict[str, Any]:
    w = WEIGHTS[track]
    norm = {"WER": lambda v: one_minus_wer(v, cfg.scoring), "pMOS": lambda v: normalize_mos(v, cfg.scoring),
            "SS": lambda v: normalize_ss(v, cfg.scoring), "NVPA": lambda v: v}
    key = {"WER": "1-WER", "pMOS": "pMOS", "SS": "SS", "NVPA": "NVPA"}
    miss = _missing(values)
    out = {}
    for k, v in values.items():
        if k not in miss:
            nv = norm[k](v)
            out[k] = {"raw": v, "normalized": nv, "weight": w[key[k]], "contribution": w[key[k]] * nv}
    return out


def build_summary(manifest: Manifest, store: ArtifactStore, cfg: EvalConfig, nvpa: Optional[float] = None,
                  human: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`human` is the content of human_scores.json (see human/scores.py), or None."""
    track = manifest.header.track
    speaker_of = {s.sample_id: s.speaker_id for s in manifest}
    metrics, values = automatic_metrics(manifest, store, cfg, None, nvpa)
    missing = _missing(values)
    auto = None if missing else auto_score(track, values["NVPA"], values["WER"], values["pMOS"], values["SS"], cfg.scoring)

    human_block: Dict[str, Any] = {"SN": None, "Q": None}
    final: Optional[ScoreResult] = None
    if human:
        ids = set(human["sample_ids"]) & set(speaker_of)
        sub_metrics, sub_values = automatic_metrics(manifest, store, cfg, ids, nvpa) if cfg.human.auto_on_subset \
            else (metrics, values)
        sn_m, q_m = human["metrics"].get("SN"), human["metrics"].get("Q")
        sn, q = (sn_m or {}).get("mean"), (q_m or {}).get("mean")
        sub_missing = _missing(sub_values)
        human_block = {**human["metrics"], "n_rated_samples": human["n_rated_samples"],
                       "n_raters_accepted": human["n_raters_accepted"], "raters": human["raters"],
                       "agreement": human.get("agreement"), "subset_size": len(ids),
                       "auto_values_used_for_final": {k: v for k, v in sub_values.items()},
                       "auto_on_subset": cfg.human.auto_on_subset}
        # validation of the NVPA detector against humans: per-sample NVPA vs the raters' NV_placement score
        corr = None
        if store.read_meta("nvpa") is not None:
            rate = {r["sample_id"]: sum(e["hit"] for e in r["events"]) / len(r["events"])
                    for r in store.read_records("nvpa") if "error" not in r and r["events"]}
            pairs = [(rate[r["sample_id"]], r["NV_placement"]) for r in human.get("records", [])
                     if r.get("NV_placement") is not None and r["sample_id"] in rate]
            if len(pairs) >= 5:
                corr = {"spearman": spearman([p[0] for p in pairs], [p[1] for p in pairs]), "n": len(pairs)}
        human_block["nvpa_vs_human_placement"] = corr
        if sn is not None and q is not None and not sub_missing:
            final = final_score(track, sub_values["NVPA"], sub_values["WER"], sub_values["pMOS"], sub_values["SS"],
                                sn, q, cfg.scoring)
        elif sub_missing:
            human_block["final_score_missing_components"] = sub_missing

    return _scrub({
        "track": track,
        "data_counts": {"n_samples": len(manifest), "n_speakers": len(set(speaker_of.values())),
                        "manifest_source": manifest.header.source, "split": manifest.header.split},
        "automatic_metrics": metrics,
        "automatic_score": _score_to_dict(auto),
        "automatic_score_missing_components": missing,
        "automatic_score_partial_components": _partial(track, values, cfg) if missing else {},
        "human_metrics": human_block,
        "final_score": _score_to_dict(final),
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
    if "nvpa" in m:
        n = m["nvpa"]
        lines.append(f"NVPA (micro, tol={n['tolerance_words']} word): {f(n['value'])}  95% CI utt{ci(n['ci_utterance'])} "
                     f"spk{ci(n['ci_speaker'])}  [{n['n_hits']}/{n['n_events']} events; failed samples={n['n_failed']}]")
        lines.append("  per type: " + "  ".join(f"{t.strip('[]')}={f(v['mean'])} (n={v['n']}{', LOW n' if v['low_n'] else ''})"
                                                 for t, v in n["by_type"].items()))
        sh = n["shuffle_baseline"]
        lines.append(f"  macro-type={f(n['macro_type'])} macro-speaker={f(n['macro_speaker'])}  "
                     f"detection={f(n['detection_rate'])} type-acc|detected={f(n['type_accuracy_given_detected'])}  "
                     f"misses={ {k: v for k, v in n['reasons'].items() if k != 'ok'} }")
        lines.append(f"  random-placement baseline at this tolerance: {f(sh.get('mean'))} (std {f(sh.get('std'))})  "
                     f"spurious NVs/100 words: {f(n['spurious_per_100_words'], 2)}")
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
    h = s["human_metrics"]
    if h.get("SN") and isinstance(h["SN"], dict):
        lines.append(f"Human (n={h['n_rated_samples']} rated, {h['n_raters_accepted']} raters accepted): "
                     f"SN {f(h['SN']['mean'], 2)}{ci(h['SN'].get('ci_utterance'))}  Q {f(h['Q']['mean'], 2)}{ci(h['Q'].get('ci_utterance'))}"
                     + (f"  NV-nat {f(h['NV_naturalness']['mean'], 2)}" if h.get("NV_naturalness") else "")
                     + (f"  NV-place {f(h['NV_placement']['mean'], 2)}" if h.get("NV_placement") else ""))
        c = h.get("nvpa_vs_human_placement")
        if c and c.get("spearman") is not None:
            lines.append(f"  NVPA vs human NV_placement (per sample): Spearman {c['spearman']:.2f} (n={c['n']})")
        flagged = [r for r, v in h["raters"].items() if v.get("flagged")]
        if flagged:
            lines.append(f"  WARNING: raters flagged by anchor checks: {flagged}")
    else:
        lines.append("Human metrics: SN N/A | Q N/A")
    fs = s["final_score"]
    lines.append(f"Final score: {f(fs['value'])}  (official formula; automatic parts on the rated subset)" if fs else "Final score: N/A")
    return "\n".join(lines)
