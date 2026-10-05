"""WER: text normalisation, alignment, per-sample records and summary.

[OFFICIAL]   NV tags are excluded from the reference; ASR is Zipformer.
[ASSUMPTION] normalisation (NFC, lowercase, punctuation/symbols -> space), whitespace
             tokens (Vietnamese syllables), and corpus-level aggregation
             (sum of errors / sum of reference words) as the headline value.

WER is a pure function of the ASR artifact and the manifest, so changing normalisation
never re-runs the ASR.
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from ..alignment import align, counts
from ..data.manifest import Manifest
from ..stats import NAN, bootstrap_ci, bootstrap_ratio_ci, group_stats, macro_mean


@dataclass(frozen=True)
class TextNormConfig:
    nfc: bool = True
    lowercase: bool = True
    strip_punct: bool = True       # Unicode categories P* and S* become spaces


def normalize_text(text: str, cfg: TextNormConfig = TextNormConfig()) -> str:
    if cfg.nfc:
        text = unicodedata.normalize("NFC", text)
    if cfg.lowercase:
        text = text.lower()
    if cfg.strip_punct:
        text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def edit_counts(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int]:
    """Minimum edit distance split into (substitutions, deletions, insertions)."""
    return counts(align(ref, hyp), ref, hyp)


def wer_records(manifest: Manifest, asr_records: Sequence[Dict[str, Any]],
                cfg: TextNormConfig = TextNormConfig()) -> List[Dict[str, Any]]:
    """One record per manifest sample. Failed/missing ASR results stay visible as error records."""
    by_id = {r["sample_id"]: r for r in asr_records}
    out: List[Dict[str, Any]] = []
    for s in manifest:
        base = {"sample_id": s.sample_id, "speaker_id": s.speaker_id}
        a = by_id.get(s.sample_id)
        if a is None:
            out.append({**base, "error": "missing_asr_record"})
            continue
        if "error" in a:
            out.append({**base, "error": a["error"]})
            continue
        ref = normalize_text(s.clean_text, cfg).split()
        hyp_text = normalize_text(a["text"], cfg)
        hyp = hyp_text.split()
        sub, dele, ins = edit_counts(ref, hyp)
        n = len(ref)
        out.append({
            **base, "ref_words": n, "hyp_words": len(hyp), "sub": sub, "del": dele, "ins": ins,
            "errors": sub + dele + ins, "wer": (sub + dele + ins) / n if n else None,
            "ref": " ".join(ref), "hyp": hyp_text,
            "hyp_has_digits": any(c.isdigit() for c in hyp_text),
        })
    return out


def summarize_wer(records: Sequence[Dict[str, Any]], aggregation: str = "corpus",
                  n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, min_n: int = 10) -> Dict[str, Any]:
    ok = [r for r in records if "error" not in r]
    tot_ref = sum(r["ref_words"] for r in ok)
    tot_err = sum(r["errors"] for r in ok)
    errs, refs = [r["errors"] for r in ok], [r["ref_words"] for r in ok]
    spk = [r["speaker_id"] for r in ok]
    sample_wers = [r["wer"] for r in ok]
    kw = dict(n_boot=n_boot, alpha=alpha, seed=seed)
    corpus = tot_err / tot_ref if tot_ref else NAN
    valid = [w for w in sample_wers if w is not None]
    mean_sample = sum(valid) / len(valid) if valid else NAN
    return {
        "aggregation_used": aggregation,
        "value": corpus if aggregation == "corpus" else mean_sample,
        "n_samples": len(ok), "n_failed": len(records) - len(ok),
        "total_ref_words": tot_ref, "sub": sum(r["sub"] for r in ok),
        "del": sum(r["del"] for r in ok), "ins": sum(r["ins"] for r in ok),
        "corpus_wer": corpus,
        "corpus_wer_ci_utterance": bootstrap_ratio_ci(errs, refs, **kw),
        "corpus_wer_ci_speaker": bootstrap_ratio_ci(errs, refs, clusters=spk, **kw),
        "mean_sample_wer": mean_sample,
        "mean_sample_wer_ci_utterance": bootstrap_ci(sample_wers, **kw),
        "macro_speaker_mean_wer": macro_mean(ok, "speaker_id", "wer"),
        "n_hyp_with_digits": sum(1 for r in ok if r["hyp_has_digits"]),
        "by_speaker": group_stats(ok, "speaker_id", "wer", min_n=min_n, n_boot=min(n_boot, 500), seed=seed),
    }
