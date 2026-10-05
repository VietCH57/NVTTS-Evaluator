"""Speaker similarity artifact ("ss"): cosine similarity between the generated audio's
embedding and the reference embedding.

[ASSUMPTION] reference = centroid of the L2-normalised embeddings of all reference clips
(`ref_mode="centroid"`), or the mean of per-clip cosines (`"mean_cosine"`). Embeddings are
cached per clip and per reference set for the duration of the run, because many samples
share the same speaker's reference clips.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..audio import load_mono
from ..core.metric import Metric, Needs
from ..data.manifest import Manifest, Sample

SAMPLE_RATE = 16000


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("embedding has zero or non-finite norm")
    return v / n


class SpeakerSimMetric(Metric):
    name = "ss"
    version = "1"
    needs = Needs(generated=True, references=True)

    def __init__(self, backend, ref_mode: str = "centroid", max_reference_clips: Optional[int] = None):
        if ref_mode not in ("centroid", "mean_cosine"):
            raise ValueError("ref_mode must be 'centroid' or 'mean_cosine'")
        self.backend, self.ref_mode, self.max_reference_clips = backend, ref_mode, max_reference_clips
        self._clip_cache: Dict[str, Tuple[np.ndarray, float]] = {}

    def config(self) -> Dict[str, Any]:
        return {"backend": self.backend.describe(), "sample_rate": SAMPLE_RATE,
                "ref_mode": self.ref_mode, "max_reference_clips": self.max_reference_clips}

    def setup(self) -> None:
        self.backend.setup()

    def _clip(self, path: str) -> Tuple[np.ndarray, float]:
        if path not in self._clip_cache:
            wav, sr = load_mono(path, SAMPLE_RATE)
            self._clip_cache[path] = (_unit(self.backend.embed(wav)), len(wav) / sr)
        return self._clip_cache[path]

    def compute(self, sample: Sample, manifest: Manifest) -> Dict[str, Any]:
        refs = [str(p) for p in manifest.resolve_references(sample)]
        if self.max_reference_clips is not None:
            refs = refs[: self.max_reference_clips]
        clips = [self._clip(p) for p in refs]
        embs: List[np.ndarray] = [c[0] for c in clips]
        wav, sr = load_mono(manifest.resolve_generated(sample), SAMPLE_RATE)
        g = _unit(self.backend.embed(wav))
        if self.ref_mode == "centroid":
            cos = float(np.dot(g, _unit(np.mean(embs, axis=0))))
        else:
            cos = float(np.mean([np.dot(g, e) for e in embs]))
        return {"cosine": cos, "n_ref_clips": len(embs), "ref_seconds": float(sum(c[1] for c in clips)),
                "ref_mode": self.ref_mode}
