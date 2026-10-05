"""Build the detector's training set from a ground-truth (train) manifest and its ASR artifact.

Uses exactly the same alignment / window code as inference, so training and evaluation
windows are defined identically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..alignment import align
from ..audio import load_mono
from ..data.manifest import Manifest
from ..data.nv_parser import NVParser
from ..metrics.wer import TextNormConfig, normalize_text
from .features import N_FEATURES, compute_frame_features, window_features
from .timing import hyp_words_from_asr
from .windows import build_gap_windows, norm_gap_map, ref_to_hyp_map


@dataclass
class WindowParams:
    min_len: float = 0.2
    max_len: float = 3.0


def build_training_set(manifest: Manifest, asr_records: Sequence[Dict[str, Any]], nv_types: Sequence[str],
                       norm: TextNormConfig = TextNormConfig(), params: WindowParams = WindowParams(),
                       parser: Optional[NVParser] = None,
                       progress: Optional[Callable[[int, int], None]] = None
                       ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    parser = parser or NVParser(nv_types)
    by_id = {r["sample_id"]: r for r in asr_records}
    type_idx = {t: k for k, t in enumerate(nv_types)}
    X: List[np.ndarray] = []
    Y: List[np.ndarray] = []
    groups: List[str] = []
    info = {"samples_used": 0, "skipped_no_asr": 0, "skipped_no_timing": 0, "windows": 0, "invalid_windows": 0}
    for n, s in enumerate(manifest, 1):
        rec = by_id.get(s.sample_id)
        if rec is None or "error" in rec:
            info["skipped_no_asr"] += 1
            continue
        hyp = hyp_words_from_asr(rec, norm)
        if hyp is None:
            info["skipped_no_timing"] += 1
            continue
        parsed = parser.parse(s.text)
        gmap = norm_gap_map(parsed.words, norm)
        ref = normalize_text(s.clean_text, norm).split()
        pairs = align(ref, [h.text for h in hyp])
        windows = build_gap_windows(len(ref), ref_to_hyp_map(pairs), hyp, rec["duration"],
                                    params.min_len, params.max_len)
        wav, sr = load_mono(manifest.resolve_generated(s), 16000)
        ff = compute_frame_features(wav, sr)
        labels: Dict[int, set] = {}
        for e in parsed.events:
            labels.setdefault(gmap[e.gap_index], set()).add(e.type)
        for g, w in enumerate(windows):
            if w is None:
                info["invalid_windows"] += 1
                continue
            y = np.zeros(len(nv_types), dtype=np.int8)
            for t in labels.get(g, ()):
                if t in type_idx:
                    y[type_idx[t]] = 1
            X.append(window_features(ff, w.t0, w.t1))
            Y.append(y)
            groups.append(s.speaker_id)
        info["samples_used"] += 1
        if progress:
            progress(n, len(manifest))
    info["windows"] = len(X)
    Xa = np.stack(X) if X else np.zeros((0, N_FEATURES), dtype=np.float32)
    Ya = np.stack(Y) if Y else np.zeros((0, len(nv_types)), dtype=np.int8)
    info["positives"] = {t: int(Ya[:, k].sum()) for t, k in type_idx.items()}
    return Xa, Ya, groups, info
