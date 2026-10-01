"""Per-metric artifact store.

Layout:  <run_dir>/artifacts/<metric>/per_sample.jsonl  +  meta.json

An artifact is "fresh" only if the metric name/version, input fingerprint and config
hash all match AND the stored records file still has the recorded checksum. The
`score` stage reads artifacts only; it never re-runs a model.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = 1


class ArtifactError(RuntimeError):
    pass


@dataclass
class ArtifactMeta:
    metric: str
    metric_version: str
    input_hash: str
    config_hash: str
    config: Dict[str, Any]
    n_records: int
    n_errors: int
    records_sha256: str
    schema_version: int = SCHEMA_VERSION


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class ArtifactStore:
    def __init__(self, run_dir: Path):
        self.root = Path(run_dir) / "artifacts"

    def _dir(self, metric: str) -> Path:
        return self.root / metric

    def read_meta(self, metric: str) -> Optional[ArtifactMeta]:
        p = self._dir(metric) / "meta.json"
        if not p.is_file():
            return None
        try:
            return ArtifactMeta(**json.loads(p.read_text(encoding="utf-8")))
        except (ValueError, TypeError):
            return None

    def write(self, metric: str, records: Iterable[Dict[str, Any]], meta_fields: Dict[str, Any]) -> ArtifactMeta:
        records = list(records)
        d = self._dir(metric)
        d.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
        meta = ArtifactMeta(
            metric=metric,
            n_records=len(records),
            n_errors=sum(1 for r in records if "error" in r),
            records_sha256=_sha256_bytes(payload),
            **meta_fields,
        )
        rec_tmp, meta_tmp = d / "per_sample.jsonl.tmp", d / "meta.json.tmp"
        rec_tmp.write_bytes(payload)
        meta_tmp.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(rec_tmp, d / "per_sample.jsonl")
        os.replace(meta_tmp, d / "meta.json")   # meta last; checksum guards a crash in between
        return meta

    def read_records(self, metric: str, verify: bool = True) -> List[Dict[str, Any]]:
        meta = self.read_meta(metric)
        p = self._dir(metric) / "per_sample.jsonl"
        if meta is None or not p.is_file():
            raise ArtifactError(f"no artifact for metric {metric!r} in {self.root}")
        raw = p.read_bytes()
        if verify and _sha256_bytes(raw) != meta.records_sha256:
            raise ArtifactError(f"artifact {metric!r} is corrupted (checksum mismatch)")
        return [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]

    def is_fresh(self, metric: str, metric_version: str, input_hash: str, config_hash: str) -> bool:
        meta = self.read_meta(metric)
        if meta is None or meta.schema_version != SCHEMA_VERSION:
            return False
        if (meta.metric_version, meta.input_hash, meta.config_hash) != (metric_version, input_hash, config_hash):
            return False
        try:
            self.read_records(metric, verify=True)
        except ArtifactError:
            return False
        return True
