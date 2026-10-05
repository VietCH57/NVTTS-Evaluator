"""Command line interface (all commands take --manifest, --run-dir and optionally --config).

  run              compute metrics (cached) and write RUN/summary.json
  summary          re-read cached artifacts (and human scores, if imported) and rewrite the summary
  train-detector   train the NVPA window detector on a ground-truth TRAIN manifest
  human-subset     choose the stratified subset that humans will rate
  human-export     write the anonymised rating package (audio + sheet + private key)
  human-import     import rater sheets, check anchors, aggregate, then rewrite the summary
  nvpa-sweep       re-evaluate NVPA from the stored artifact under other tolerances / thresholds (no models run)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import EvalConfig, load_config
from .core.artifacts import ArtifactStore
from .core.metric import Metric, run_metric
from .data.manifest import Manifest
from .report.summary import build_summary, format_summary

ALL_METRICS = ("asr", "pmos", "ss", "nvpa")


def build_metrics(cfg: EvalConfig, names: Sequence[str], store: Optional[ArtifactStore] = None) -> Dict[str, Metric]:
    """Construct metric objects. Cheap: models are loaded lazily, only if a metric must recompute."""
    from .metrics.asr_metric import AsrMetric
    from .metrics.pmos import PMosMetric
    from .metrics.speaker_sim import SpeakerSimMetric

    out: Dict[str, Metric] = {}
    for n in names:
        if n == "asr":
            from .models.asr import SherpaOnnxOfflineASR
            if not cfg.asr.model_dir:
                raise ValueError("config asr.model_dir is required to run the ASR")
            out[n] = AsrMetric(SherpaOnnxOfflineASR(Path(cfg.asr.model_dir), cfg.asr.use_int8, cfg.asr.num_threads,
                                                    cfg.asr.provider, cfg.asr.decoding_method))
        elif n == "pmos":
            from .models.dnsmos import DnsmosTorchNative
            out[n] = PMosMetric(DnsmosTorchNative(cfg.pmos.repo_id, cfg.pmos.revision, cfg.pmos.device))
        elif n == "ss":
            from .models.ecapa import SpeechBrainEcapa
            out[n] = SpeakerSimMetric(SpeechBrainEcapa(cfg.ss.source, cfg.ss.savedir, cfg.ss.device),
                                      cfg.ss.ref_mode, cfg.ss.max_reference_clips)
        elif n == "nvpa":
            from .nvpa.detector import GapDetector
            from .nvpa.metric import NvpaMetric, NvpaParams
            if store is None:
                raise ValueError("the nvpa metric needs the artifact store of the run")
            if not cfg.nvpa.detector_path:
                raise ValueError("config nvpa.detector_path is required (train one with `train-detector`)")
            out[n] = NvpaMetric(store, GapDetector.load(Path(cfg.nvpa.detector_path)), cfg.text_norm,
                                NvpaParams(tolerance_words=cfg.nvpa.tolerance_words, min_window=cfg.nvpa.min_window,
                                       max_window=cfg.nvpa.max_window, tolerance_before=cfg.nvpa.tolerance_before,
                                       tolerance_after=cfg.nvpa.tolerance_after,
                                       threshold_scale=cfg.nvpa.threshold_scale))
        else:
            raise ValueError(f"unknown metric {n!r}; choose from {ALL_METRICS}")
    return out


def _load_manifest(path: Path, limit: Optional[int]) -> Manifest:
    m = Manifest.load(path)
    if limit:
        m = m.subset([s.sample_id for s in m.samples[:limit]])
    return m


def _human_path(run_dir: Path) -> Path:
    return run_dir / "human" / "human_scores.json"


def _load_human(run_dir: Path) -> Optional[dict]:
    p = _human_path(run_dir)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _write_summary(manifest: Manifest, store: ArtifactStore, cfg: EvalConfig, run_dir: Path) -> None:
    summary = build_summary(manifest, store, cfg, human=_load_human(run_dir))
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + format_summary(summary))
    print(f"\nwrote {run_dir / 'summary.json'}")


def _progress(name):
    def f(i, n):
        if i == n or i % 25 == 0:
            print(f"  [{name}] {i}/{n}", file=sys.stderr)
    return f


def _cmd_train_detector(a, cfg: EvalConfig) -> int:
    from .nvpa.detector import train_gap_detector
    from .nvpa.training import WindowParams, build_training_set

    manifest = _load_manifest(a.manifest, a.limit)
    if manifest.header.split == "dev":
        raise SystemExit("refusing to train on the dev split: dev is reserved for calibration. Use the train split.")
    store = ArtifactStore(a.run_dir)
    r = run_metric(build_metrics(cfg, ["asr"])["asr"], manifest, store, on_progress=_progress("asr"))
    print(f"asr: {'cache hit' if r.from_cache else 'computed'}  records={r.meta.n_records} errors={r.meta.n_errors}")
    nv_types = tuple(manifest.header.nv_types)
    X, Y, groups, info = build_training_set(manifest, r.records, nv_types, cfg.text_norm,
                                            WindowParams(cfg.nvpa.min_window, cfg.nvpa.max_window),
                                            progress=_progress("windows"))
    print(f"training set: {info}")
    det = train_gap_detector(X, Y, groups, nv_types, seed=cfg.bootstrap.seed)
    det.meta["training_set"] = info
    det.save(a.out)
    print(f"\nwrote {a.out}")
    for t in nv_types:
        m = det.meta[t]
        print(f"  {t:16s} pos={m['n_pos']:5d} " + (f"thr={m['threshold']:.3f} oof P={m['oof_precision']:.2f} "
              f"R={m['oof_recall']:.2f} AUC={m['oof_auc']:.3f}" if "threshold" in m else m.get("note", "")))
    return 0


def _cmd_human(a, cfg: EvalConfig, manifest: Manifest, store: ArtifactStore) -> int:
    from .human.package import export_package
    from .human.scores import aggregate_ratings, read_rating_sheet
    from .human.subset import select_subset

    hdir = a.run_dir / "human"
    h = cfg.human
    if a.cmd == "human-subset":
        sub = select_subset(manifest, a.size or h.subset_size, h.seed, h.rare_type_share, h.max_rare_fraction,
                            h.long_utt_seconds, h.head_speaker_min_utts)
        hdir.mkdir(parents=True, exist_ok=True)
        (hdir / "subset.json").write_text(json.dumps(sub, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {hdir / 'subset.json'}\n{json.dumps(sub['composition'], ensure_ascii=False)}  rare types: {sub['rare_types']}")
    elif a.cmd == "human-export":
        sp = hdir / "subset.json"
        if not sp.is_file():
            raise SystemExit("run `human-subset` first")
        ids = json.loads(sp.read_text(encoding="utf-8"))["sample_ids"]
        meta = export_package(manifest, ids, a.out, h.n_gt_anchors, h.n_degraded_anchors, h.degrade_snr_db, h.seed)
        kinds = {}
        for i in meta["items"]:
            kinds[i["kind"]] = kinds.get(i["kind"], 0) + 1
        print(f"wrote package to {a.out}  items={kinds}  skipped (no audio)={len(meta['skipped_missing_audio'])}")
        print(f"give raters ONLY: {Path(a.out) / 'for_raters'}   keep private: {Path(a.out) / 'PRIVATE_key.json'}")
    else:  # human-import
        key = json.loads((Path(a.package) / "PRIVATE_key.json").read_text(encoding="utf-8"))
        sheets, problems = {}, []
        for p in a.ratings:
            rater = Path(p).stem
            if rater in sheets:
                raise SystemExit(f"two rating files share the name {rater!r}; rename each file after its rater "
                                 f"(e.g. alice.csv, bob.csv): the file name is the rater id")
            rows, pr = read_rating_sheet(Path(p))
            sheets[rater] = rows
            problems += pr
        agg = aggregate_ratings(sheets, key, h, cfg.bootstrap, problems)
        hdir.mkdir(parents=True, exist_ok=True)
        _human_path(a.run_dir).write_text(json.dumps(agg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"raters: {agg['n_raters_accepted']}/{agg['n_raters_total']} accepted; rated samples (SN+Q): {agg['n_rated_samples']}")
        for r, v in agg["raters"].items():
            if v["flagged"]:
                print(f"  flagged {r}: {'; '.join(v['reasons'])}")
        for pr in problems[:10]:
            print("  problem:", pr)
        _write_summary(manifest, store, cfg, a.run_dir)
    return 0


def _cmd_nvpa_sweep(a, cfg: EvalConfig) -> int:
    from .nvpa.sweep import format_sweep, sweep

    store = ArtifactStore(a.run_dir)
    meta = store.read_meta("nvpa")
    if meta is None:
        raise SystemExit("no nvpa artifact in this run dir: run the nvpa metric first")
    thr = meta.config.get("detector", {}).get("thresholds")
    if not thr:
        raise SystemExit("the nvpa artifact does not record detector thresholds")
    tols = [tuple(int(x) for x in t.split(":")) if ":" in t else int(t) for t in a.tolerances]
    res = sweep(store.read_records("nvpa"), thr, tols, a.scales, cfg.nvpa.unreliable_policy, a.reps,
                cfg.bootstrap.seed)
    (a.run_dir / "nvpa_sweep.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"artifact used params={meta.config.get('params')}  base thresholds={thr}\n")
    print(format_sweep(res))
    print(f"\nwrote {a.run_dir / 'nvpa_sweep.json'}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="nvtts_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("nvpa-sweep")
    sw.add_argument("--run-dir", required=True, type=Path)
    sw.add_argument("--config", type=Path)
    sw.add_argument("--tolerances", nargs="+", default=["0", "1", "2", "3"],
                    help="n (symmetric) or before:after, e.g. 0 1 0:1 2")
    sw.add_argument("--scales", nargs="+", type=float, default=[0.5, 0.75, 1.0, 1.25])
    sw.add_argument("--reps", type=int, default=20, help="shuffle repetitions per setting")
    for name in ("run", "summary", "train-detector", "human-subset", "human-export", "human-import"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True, type=Path)
        p.add_argument("--run-dir", required=True, type=Path)
        p.add_argument("--config", type=Path)
        p.add_argument("--limit", type=int, help="only the first N samples (smoke tests; use a separate run dir)")
        if name == "run":
            p.add_argument("--metrics", nargs="+", choices=ALL_METRICS,
                           help="default: asr pmos ss, plus nvpa when nvpa.detector_path is configured")
            p.add_argument("--force", action="store_true", help="recompute even if cached")
        if name == "train-detector":
            p.add_argument("--out", required=True, type=Path)
        if name == "human-subset":
            p.add_argument("--size", type=int)
        if name == "human-export":
            p.add_argument("--out", required=True, type=Path)
        if name == "human-import":
            p.add_argument("--package", required=True, type=Path, help="folder created by human-export")
            p.add_argument("--ratings", required=True, nargs="+", help="one filled rating_sheet.csv per rater "
                                                                       "(rater id = file name)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    if a.cmd == "nvpa-sweep":
        return _cmd_nvpa_sweep(a, cfg)
    a.run_dir.mkdir(parents=True, exist_ok=True)
    if a.cmd == "train-detector":
        return _cmd_train_detector(a, cfg)
    manifest = _load_manifest(a.manifest, a.limit)
    store = ArtifactStore(a.run_dir)
    if a.cmd.startswith("human-"):
        return _cmd_human(a, cfg, manifest, store)

    if a.cmd == "run":
        names: List[str] = a.metrics or ["asr", "pmos", "ss"] + (["nvpa"] if cfg.nvpa.detector_path else [])
        if "nvpa" in names and "asr" in names:
            names = [n for n in names if n != "nvpa"] + ["nvpa"]       # nvpa reads the asr artifact
        for name in names:
            metric = build_metrics(cfg, [name], store)[name]
            r = run_metric(metric, manifest, store, force=a.force, on_progress=_progress(name))
            print(f"{name}: {'cache hit' if r.from_cache else 'computed'}  "
                  f"records={r.meta.n_records} errors={r.meta.n_errors}")
    _write_summary(manifest, store, cfg, a.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
