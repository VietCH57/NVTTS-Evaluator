"""DNSMOS backend: prj-beatrice/dnsmos-torch-native (torch + transformers, remote code).

Usage documented on the model card:
    model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval().to(device)
    scores = model(wav_tensor_1d, sampling_rate=sr)   # .sig .bak .ovrl .p808

SECURITY: trust_remote_code=True executes code from the repo. Inspect it once and pin a
commit via `revision`. Windowing of audio longer than DNSMOS's analysis window is done
inside the model code; verify with scripts/probe_models.py on a long (~25 s) clip.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

OUTPUTS = ("sig", "bak", "ovrl", "p808")


def _scalar(x: Any) -> float:
    if hasattr(x, "reshape"):
        x = x.reshape(-1)[0]
    if hasattr(x, "item"):
        x = x.item()
    return float(x)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


class DnsmosTorchNative:
    def __init__(self, repo_id: str = "prj-beatrice/dnsmos-torch-native",
                 revision: Optional[str] = None, device: str = "auto"):
        self.repo_id, self.revision, self.device_name = repo_id, revision, device
        self._model = None
        self._device = None

    def describe(self) -> Dict[str, Any]:
        return {"backend": "dnsmos-torch-native", "repo_id": self.repo_id,
                "revision": self.revision, "input_sample_rate": 16000}

    def setup(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel  # lazy

        self._device = resolve_device(self.device_name)
        kwargs = {"trust_remote_code": True}
        if self.revision:
            kwargs["revision"] = self.revision
        self._model = AutoModel.from_pretrained(self.repo_id, **kwargs).eval().to(self._device)

    def score(self, wav: np.ndarray, sample_rate: int) -> Dict[str, float]:
        self.setup()
        import torch

        with torch.inference_mode():
            out = self._model(torch.from_numpy(wav).to(self._device), sampling_rate=sample_rate)
        return {k: _scalar(getattr(out, k)) for k in OUTPUTS}
