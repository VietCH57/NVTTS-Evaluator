"""Metric interface and the generic runner.

A metric implements `compute(sample, manifest) -> dict` for ONE sample. The runner
handles caching, missing inputs and per-sample failures uniformly, so a single bad
file never aborts a run and never disappears silently: it becomes a record with an
"error" field and is counted in the artifact meta.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..data.manifest import Manifest, Sample
from .artifacts import ArtifactMeta, ArtifactStore
from .hashing import fingerprint_manifest, hash_json

RESERVED_KEYS = ("sample_id", "error")


@dataclass(frozen=True)
class Needs:
    """Which manifest inputs a metric reads; drives both validation and cache fingerprint."""
    generated: bool = True
    references: bool = False
    ground_truth: bool = False
    text: bool = False


class Metric(ABC):
    name: str = ""
    version: str = "1"                 # bump when the computation changes
    needs: Needs = Needs()

    def config(self) -> Dict[str, Any]:
        """Everything that changes the result (checkpoint, tolerance, ...). Hashed for caching."""
        return {}

    def setup(self) -> None:
        """Lazy, heavy initialisation (load models). Called only if recomputation is needed."""

    @abstractmethod
    def compute(self, sample: Sample, manifest: Manifest) -> Dict[str, Any]:
        """Return metric fields for one sample (must not use keys 'sample_id' or 'error')."""


@dataclass
class MetricRun:
    meta: ArtifactMeta
    records: List[Dict[str, Any]]
    from_cache: bool


def _missing_input(metric: Metric, sample: Sample, manifest: Manifest) -> Optional[str]:
    n = metric.needs
    if n.generated:
        p = manifest.resolve_generated(sample)
        if p is None or not Path(p).is_file():
            return "missing_generated_audio"
    if n.references:
        refs = manifest.resolve_references(sample)
        if not refs or not all(Path(p).is_file() for p in refs):
            return "missing_reference_audio"
    if n.ground_truth:
        p = manifest.resolve_ground_truth(sample)
        if p is None or not Path(p).is_file():
            return "missing_ground_truth_audio"
    return None


def run_metric(
    metric: Metric,
    manifest: Manifest,
    store: ArtifactStore,
    force: bool = False,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> MetricRun:
    if not metric.name:
        raise ValueError("metric.name must be set")
    n = metric.needs
    input_hash = fingerprint_manifest(
        manifest, generated=n.generated, references=n.references,
        ground_truth=n.ground_truth, text=n.text,
    )
    config = metric.config()
    config_hash = hash_json(config)

    if not force and store.is_fresh(metric.name, metric.version, input_hash, config_hash):
        return MetricRun(store.read_meta(metric.name), store.read_records(metric.name), from_cache=True)

    metric.setup()
    records: List[Dict[str, Any]] = []
    total = len(manifest)
    for i, s in enumerate(manifest, 1):
        rec: Dict[str, Any] = {"sample_id": s.sample_id}
        reason = _missing_input(metric, s, manifest)
        if reason:
            rec["error"] = reason
        else:
            try:
                out = metric.compute(s, manifest)
                bad = [k for k in RESERVED_KEYS if k in out]
                if bad:
                    raise ValueError(f"compute() returned reserved keys {bad}")
                rec.update(out)
            except Exception as e:  # per-sample failure is recorded, not fatal
                rec = {"sample_id": s.sample_id, "error": f"{type(e).__name__}: {e}"}
        records.append(rec)
        if on_progress:
            on_progress(i, total)

    meta = store.write(
        metric.name, records,
        dict(metric_version=metric.version, input_hash=input_hash, config_hash=config_hash, config=config),
    )
    return MetricRun(meta, records, from_cache=False)
