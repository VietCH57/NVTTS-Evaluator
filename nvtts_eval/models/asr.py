"""ASR backend: hynt/Zipformer-30M-RNNT-6000h (offline transducer) through sherpa-onnx.

The Hugging Face repo ships ONNX files (fp32 and int8) plus bpe.model, but NO tokens.txt,
which sherpa-onnx needs. Either generate it with `scripts/make_tokens.py` from bpe.model,
or use sherpa-onnx's packaged release of the same model (it includes tokens.txt).

Input contract: mono float32 audio (the evaluator always passes 16 kHz).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np


@dataclass
class ASRResult:
    text: str
    tokens: Optional[List[str]] = None
    timestamps: Optional[List[float]] = None    # seconds, one per token, if the backend provides them


class ASRBackend(Protocol):
    def describe(self) -> Dict[str, Any]: ...
    def setup(self) -> None: ...
    def transcribe(self, wav: np.ndarray, sample_rate: int) -> ASRResult: ...


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick(model_dir: Path, stem: str, use_int8: bool) -> Path:
    cands = sorted(model_dir.glob(f"{stem}*.onnx"))
    if not cands:
        raise FileNotFoundError(f"no {stem}*.onnx in {model_dir}")
    matching = [p for p in cands if (".int8." in p.name) == use_int8]
    pool = matching or cands          # e.g. packaged int8 releases ship a non-int8 decoder
    if len(pool) > 1:
        raise ValueError(f"ambiguous {stem} files in {model_dir}: {[p.name for p in pool]}")
    return pool[0]


def resolve_sherpa_files(model_dir: Path, use_int8: bool) -> Dict[str, Path]:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"ASR model_dir not found: {model_dir}")
    tokens = model_dir / "tokens.txt"
    if not tokens.is_file():
        raise FileNotFoundError(
            f"{tokens} is missing. Generate it from bpe.model:\n"
            f"  python scripts/make_tokens.py --bpe {model_dir / 'bpe.model'} --out {tokens}"
        )
    return {
        "encoder": _pick(model_dir, "encoder", use_int8),
        "decoder": _pick(model_dir, "decoder", use_int8),
        "joiner": _pick(model_dir, "joiner", use_int8),
        "tokens": tokens,
    }


class SherpaOnnxOfflineASR:
    def __init__(self, model_dir: Path, use_int8: bool = False, num_threads: int = 2,
                 provider: str = "cpu", decoding_method: str = "greedy_search"):
        self.model_dir = Path(model_dir)
        self.use_int8 = use_int8
        self.num_threads = num_threads
        self.provider = provider
        self.decoding_method = decoding_method
        self._recognizer = None
        self._files: Optional[Dict[str, Path]] = None
        self._hashes: Optional[Dict[str, str]] = None

    def _resolve(self) -> Dict[str, Path]:
        if self._files is None:
            self._files = resolve_sherpa_files(self.model_dir, self.use_int8)
        return self._files

    def describe(self) -> Dict[str, Any]:
        """Everything that changes the transcript; hashed for cache validity. No heavy imports."""
        files = self._resolve()
        if self._hashes is None:
            self._hashes = {k: _sha256(p) for k, p in files.items()}
        try:
            from importlib.metadata import version
            sherpa_version = version("sherpa-onnx")
        except Exception:
            sherpa_version = "unknown"
        return {
            "backend": "sherpa-onnx-offline-transducer",
            "sherpa_onnx": sherpa_version,
            "files": {k: {"name": files[k].name, "sha256": h} for k, h in self._hashes.items()},
            "decoding_method": self.decoding_method,
            "feature_dim": 80,
        }

    def setup(self) -> None:
        if self._recognizer is not None:
            return
        import sherpa_onnx  # lazy

        f = self._resolve()
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(f["encoder"]), decoder=str(f["decoder"]), joiner=str(f["joiner"]),
            tokens=str(f["tokens"]), num_threads=self.num_threads, sample_rate=16000,
            feature_dim=80, decoding_method=self.decoding_method, provider=self.provider,
        )

    def transcribe(self, wav: np.ndarray, sample_rate: int) -> ASRResult:
        self.setup()
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, wav)
        self._recognizer.decode_stream(stream)
        r = stream.result
        tokens = [str(t) for t in getattr(r, "tokens", [])] or None
        ts = [float(t) for t in getattr(r, "timestamps", [])] or None
        return ASRResult(text=str(r.text), tokens=tokens, timestamps=ts)
