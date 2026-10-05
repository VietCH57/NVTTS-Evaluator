"""ASR transcription as a cached artifact ("asr").

WER is computed from this artifact, and NVPA will reuse its token timestamps, so the ASR
runs once per generated-audio set.
"""
from __future__ import annotations

from typing import Any, Dict

from ..audio import load_mono
from ..core.metric import Metric, Needs
from ..data.manifest import Manifest, Sample

SAMPLE_RATE = 16000


class AsrMetric(Metric):
    name = "asr"
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
        r = self.backend.transcribe(wav, sr)
        return {"text": r.text, "tokens": r.tokens, "timestamps": r.timestamps, "duration": len(wav) / sr}
