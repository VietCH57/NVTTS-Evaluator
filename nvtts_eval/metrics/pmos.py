"""Predicted MOS artifact ("pmos"): all four DNSMOS outputs are stored; which one is
reported as pMOS is a config choice (summary layer)."""
from __future__ import annotations

from typing import Any, Dict

from ..audio import load_mono
from ..core.metric import Metric, Needs
from ..data.manifest import Manifest, Sample

SAMPLE_RATE = 16000


class PMosMetric(Metric):
    name = "pmos"
    version = "1"
    needs = Needs(generated=True)

    def __init__(self, backend):
        self.backend = backend

    def config(self) -> Dict[str, Any]:
        return {"backend": self.backend.describe(), "sample_rate": SAMPLE_RATE}

    def setup(self) -> None:
        self.backend.setup()

    def compute(self, sample: Sample, manifest: Manifest) -> Dict[str, Any]:
        wav, sr = load_mono(manifest.resolve_generated(sample), SAMPLE_RATE)
        scores = self.backend.score(wav, sr)
        return {**scores, "duration": len(wav) / sr}
