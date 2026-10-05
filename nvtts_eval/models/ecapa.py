"""Speaker embedding backend: speechbrain/spkrec-ecapa-voxceleb (ECAPA-TDNN, 16 kHz mono).

Model card usage:
    from speechbrain.inference.speaker import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    embeddings = classifier.encode_batch(signal)

Windows note: SpeechBrain may try to symlink downloaded files, which can fail without
Developer Mode/admin rights; if so, enable Developer Mode or pre-download into `savedir`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .dnsmos import resolve_device


class SpeechBrainEcapa:
    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb",
                 savedir: Optional[str] = None, device: str = "auto"):
        self.source, self.savedir, self.device_name = source, savedir, device
        self._clf = None
        self._device = None

    def describe(self) -> Dict[str, Any]:
        return {"backend": "speechbrain-ecapa", "source": self.source, "input_sample_rate": 16000}

    def setup(self) -> None:
        if self._clf is not None:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # older SpeechBrain
            from speechbrain.pretrained import EncoderClassifier
        self._device = resolve_device(self.device_name)
        kwargs: Dict[str, Any] = {"source": self.source, "run_opts": {"device": self._device}}
        if self.savedir:
            kwargs["savedir"] = self.savedir
        self._clf = EncoderClassifier.from_hparams(**kwargs)

    def embed(self, wav: np.ndarray) -> np.ndarray:
        """wav: mono float32 at 16 kHz -> embedding vector (float64)."""
        self.setup()
        import torch

        sig = torch.from_numpy(wav).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._clf.encode_batch(sig)
        return emb.reshape(-1).detach().cpu().numpy().astype(np.float64)
