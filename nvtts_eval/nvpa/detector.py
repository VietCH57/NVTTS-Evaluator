"""Window-level NV detector: one gradient-boosting classifier per NV type on hand-crafted features.

Trained on the TRAIN split only (windows come from ASR alignment of ground-truth audio),
with per-type decision thresholds picked on speaker-grouped out-of-fold predictions. The dev
split is never used for training; it is reserved for the ground-truth calibration run.

The evaluator only needs `predict_windows` and `thresholds`; any object with those (and
`describe()`, `nv_types`) can replace this class.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .features import FEATURE_VERSION, N_FEATURES, compute_frame_features, window_features
from .windows import Window


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class GapDetector:
    def __init__(self, models: Dict[str, Any], thresholds: Dict[str, float], nv_types: Sequence[str],
                 meta: Optional[Dict[str, Any]] = None, source_path: Optional[Path] = None):
        self.models, self.thresholds, self.nv_types = models, dict(thresholds), tuple(nv_types)
        self.meta = meta or {}
        self.source_path = Path(source_path) if source_path else None
        self._sha: Optional[str] = None

    # ---- inference -------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        for t in self.nv_types:
            m = self.models.get(t)
            out[t] = np.zeros(len(X)) if m is None else m.predict_proba(X)[:, 1]
        return out

    def predict_windows(self, wav: np.ndarray, sr: int, windows: Sequence[Optional[Window]]
                        ) -> List[Optional[Dict[str, float]]]:
        ff = compute_frame_features(wav, sr)
        idx = [i for i, w in enumerate(windows) if w is not None]
        res: List[Optional[Dict[str, float]]] = [None] * len(windows)
        if not idx:
            return res
        X = np.stack([window_features(ff, windows[i].t0, windows[i].t1) for i in idx])
        probs = self.predict_proba(X)
        for k, i in enumerate(idx):
            res[i] = {t: float(probs[t][k]) for t in self.nv_types}
        return res

    def describe(self) -> Dict[str, Any]:
        if self._sha is None and self.source_path is not None:
            self._sha = _sha256(self.source_path)
        return {"backend": "gap-gbm", "feature_version": FEATURE_VERSION, "model_sha256": self._sha,
                "thresholds": {t: round(float(v), 6) for t, v in sorted(self.thresholds.items())},
                "train": self.meta}

    # ---- persistence -----------------------------------------------------------------
    def save(self, path: Path) -> None:
        import joblib
        import sklearn
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "thresholds": self.thresholds, "nv_types": list(self.nv_types),
                     "meta": self.meta, "feature_version": FEATURE_VERSION, "sklearn": sklearn.__version__},
                    path)

    @classmethod
    def load(cls, path: Path) -> "GapDetector":
        import joblib
        d = joblib.load(path)
        try:
            import sklearn
            trained, current = d.get("sklearn"), sklearn.__version__
            if trained and trained.split(".")[:2] != current.split(".")[:2]:
                warnings.warn(f"detector {Path(path).name} was trained with scikit-learn {trained} but scikit-learn "
                              f"{current} is installed: predictions may differ. Install scikit-learn=={trained} or retrain.",
                              RuntimeWarning, stacklevel=2)
        except ImportError:                                  # pragma: no cover
            pass
        if d.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"detector was trained with feature_version {d.get('feature_version')!r}, "
                             f"code is {FEATURE_VERSION!r}; retrain it")
        return cls(d["models"], d["thresholds"], d["nv_types"], d["meta"], source_path=Path(path))


def _best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(y.sum()), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    k = int(np.argmax(f1))
    return float(p[order][k])


def train_gap_detector(X: np.ndarray, Y: np.ndarray, groups: Sequence[str], nv_types: Sequence[str],
                       seed: int = 0, n_splits: int = 5, max_iter: int = 200,
                       max_pos_weight: float = 20.0) -> GapDetector:
    """X: (n, N_FEATURES); Y: (n, n_types) 0/1 window labels; groups: speaker id per window."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    if X.shape[1] != N_FEATURES or Y.shape != (len(X), len(nv_types)):
        raise ValueError("bad shapes for X / Y")
    groups = np.asarray(groups)
    n_groups = len(set(groups.tolist()))
    folds = list(GroupKFold(n_splits=min(n_splits, n_groups)).split(X, Y[:, 0], groups)) if n_groups >= 2 else []

    def make():
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=max_iter,
                                              random_state=seed)

    def weights(y):
        pos = max(int(y.sum()), 1)
        return np.where(y == 1, min((len(y) - pos) / pos, max_pos_weight), 1.0)

    models, thresholds, meta = {}, {}, {}
    for k, t in enumerate(nv_types):
        y = Y[:, k].astype(int)
        n_pos = int(y.sum())
        if n_pos < 5 or n_pos == len(y):
            models[t], thresholds[t] = None, 1.01       # never fires
            meta[t] = {"n_pos": n_pos, "n_neg": int(len(y) - n_pos), "note": "too few positives; detector disabled"}
            continue
        oof = np.zeros(len(y))
        for tr, te in folds:
            if y[tr].sum() == 0:
                continue
            m = make().fit(X[tr], y[tr], sample_weight=weights(y[tr]))
            oof[te] = m.predict_proba(X[te])[:, 1]
        thr = _best_f1_threshold(y, oof) if folds else 0.5
        pred = oof >= thr
        tp = int((pred & (y == 1)).sum())
        meta[t] = {"n_pos": n_pos, "n_neg": int(len(y) - n_pos), "threshold": thr,
                   "oof_precision": tp / max(int(pred.sum()), 1), "oof_recall": tp / n_pos,
                   "oof_auc": float(roc_auc_score(y, oof)) if folds else None}
        models[t] = make().fit(X, y, sample_weight=weights(y))
        thresholds[t] = thr
    meta["_n_windows"] = int(len(X))
    meta["_n_groups"] = n_groups
    return GapDetector(models, thresholds, nv_types, meta)
