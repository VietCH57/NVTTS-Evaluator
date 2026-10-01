"""Content hashing used to decide whether a cached metric artifact is still valid."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..data.manifest import Manifest


def sha256_file(path: Union[str, Path], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_json(obj: Any) -> str:
    """Canonical (key-order independent) hash of a JSON-serialisable object."""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fingerprint_manifest(
    manifest: Manifest,
    *,
    generated: bool = False,
    references: bool = False,
    ground_truth: bool = False,
    text: bool = False,
) -> str:
    """Hash of exactly the inputs a metric depends on.

    Audio is hashed by CONTENT, so replacing a file changes the fingerprint even if the
    path is the same. A missing file contributes the marker "MISSING", so adding it later
    also invalidates the cache.
    """
    cache: Dict[str, str] = {}

    def fh(p: Optional[Path]) -> str:
        if p is None:
            return "NONE"
        key = str(p)
        if key not in cache:
            cache[key] = sha256_file(p) if Path(p).is_file() else "MISSING"
        return cache[key]

    entries = []
    for s in manifest:
        e: Dict[str, Any] = {"id": s.sample_id}
        if text:
            e["text"] = s.text
        if generated:
            e["gen"] = fh(manifest.resolve_generated(s))
        if references:
            e["ref"] = [fh(p) for p in manifest.resolve_references(s)]
        if ground_truth:
            e["gt"] = fh(manifest.resolve_ground_truth(s))
        entries.append(e)
    flags = dict(generated=generated, references=references, ground_truth=ground_truth, text=text)
    return hash_json({"flags": flags, "entries": entries})
