"""Command line interface.

    python -m nvtts_eval.cli run     --manifest M.jsonl --run-dir RUN --config configs\\default.yaml
    python -m nvtts_eval.cli summary --manifest M.jsonl --run-dir RUN --config configs\\default.yaml

`run` computes the requested metrics (cached; unchanged inputs are not recomputed) and then
writes RUN/summary.json. `summary` only re-reads cached artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from .config import EvalConfig, load_config
from .core.artifacts import ArtifactStore
from .core.metric import Metric, run_metric
from .data.manifest import Manifest
from .report.summary import build_summary, format_summary

ALL_METRICS = ("asr", "pmos", "ss")


def build_metrics(cfg: EvalConfig, names: Sequence[str]) -> Dict[str, Metric]:
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
        else:
            raise ValueError(f"unknown metric {n!r}; choose from {ALL_METRICS}")
    return out


def _load_manifest(path: Path, limit: Optional[int]) -> Manifest:
    m = Manifest.load(path)
    if limit:
        m = m.subset([s.sample_id for s in m.samples[:limit]])
    return m


def _write_summary(manifest: Manifest, store: ArtifactStore, cfg: EvalConfig, run_dir: Path) -> None:
    summary = build_summary(manifest, store, cfg)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + format_summary(summary))
    print(f"\nwrote {run_dir / 'summary.json'}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="nvtts_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True, type=Path)
        p.add_argument("--run-dir", required=True, type=Path)
        p.add_argument("--config", type=Path)
        p.add_argument("--limit", type=int, help="only the first N samples (smoke tests; use a separate run dir)")
        if name == "run":
            p.add_argument("--metrics", nargs="+", default=list(ALL_METRICS), choices=ALL_METRICS)
            p.add_argument("--force", action="store_true", help="recompute even if cached")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    manifest = _load_manifest(a.manifest, a.limit)
    a.run_dir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(a.run_dir)

    if a.cmd == "run":
        for name, metric in build_metrics(cfg, a.metrics).items():
            def progress(i, n, _name=name):
                if i == n or i % 25 == 0:
                    print(f"  [{_name}] {i}/{n}", file=sys.stderr)
            r = run_metric(metric, manifest, store, force=a.force, on_progress=progress)
            print(f"{name}: {'cache hit' if r.from_cache else 'computed'}  "
                  f"records={r.meta.n_records} errors={r.meta.n_errors}")
    _write_summary(manifest, store, cfg, a.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
