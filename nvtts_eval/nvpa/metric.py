"""NVPA as a cached artifact ("nvpa"): per-sample event results plus the detections needed for
diagnostics and the shuffle baseline. Requires the "asr" artifact of the same run."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from ..alignment import align
from ..audio import load_mono
from ..core.artifacts import ArtifactStore
from ..core.metric import Metric, Needs
from ..data.manifest import Manifest, Sample
from ..data.nv_parser import NVParser
from ..metrics.wer import TextNormConfig, normalize_text
from .matching import EventResult, effective_tolerance, match_events
from .timing import hyp_words_from_asr
from .windows import Window, build_gap_windows, norm_gap_map, ref_to_hyp_map


@dataclass(frozen=True)
class NvpaParams:
    tolerance_words: int = 1        # [ASSUMPTION] symmetric position tolerance in reference words; tune with nvpa-sweep
    min_window: float = 0.2
    max_window: float = 3.0
    tolerance_before: Optional[int] = None   # override: how many gaps EARLIER than the gold gap a detection may be
    tolerance_after: Optional[int] = None    # override: how many gaps LATER
    threshold_scale: float = 1.0    # [ASSUMPTION] multiplies every enabled detector threshold (calibrate with nvpa-sweep)

    @property
    def tolerance(self):
        return effective_tolerance(asdict(self))


class NvpaMetric(Metric):
    name = "nvpa"
    version = "1"
    needs = Needs(generated=True)

    def __init__(self, store: ArtifactStore, detector, norm: TextNormConfig = TextNormConfig(),
                 params: NvpaParams = NvpaParams(), parser: Optional[NVParser] = None):
        self.store, self.detector, self.norm, self.params = store, detector, norm, params
        self.parser = parser or NVParser(detector.nv_types)
        self._asr: Optional[Dict[str, Dict[str, Any]]] = None

    def config(self) -> Dict[str, Any]:
        meta = self.store.read_meta("asr")
        if meta is None:
            raise RuntimeError("NVPA needs the 'asr' artifact: run the 'asr' metric first")
        return {"detector": self.detector.describe(), "asr_records_sha256": meta.records_sha256,
                "params": asdict(self.params), "text_norm": asdict(self.norm)}

    def _asr_records(self) -> Dict[str, Dict[str, Any]]:
        if self._asr is None:
            self._asr = {r["sample_id"]: r for r in self.store.read_records("asr")}
        return self._asr

    def setup(self) -> None:
        self._asr_records()

    def compute(self, sample: Sample, manifest: Manifest) -> Dict[str, Any]:
        rec = self._asr_records().get(sample.sample_id)
        if rec is None:
            raise RuntimeError("no ASR record for this sample")
        if "error" in rec:
            raise RuntimeError(f"ASR failed: {rec['error']}")
        parsed = self.parser.parse(sample.text)
        gmap = norm_gap_map(parsed.words, self.norm)
        ref = normalize_text(sample.clean_text, self.norm).split()
        gold = [(e.type, gmap[e.gap_index]) for e in parsed.events]
        base = {"n_ref_words": len(ref), "n_gaps": len(ref) + 1}

        hyp = hyp_words_from_asr(rec, self.norm)
        if hyp is None:
            ev = [EventResult(t, g, False, "alignment_unreliable") for t, g in gold]
            return {**base, "no_timing": True, "events": [asdict(e) for e in ev], "spurious": [],
                    "valid_gaps": [], "pred": [], "probs": [], "n_widened": 0}

        pairs = align(ref, [h.text for h in hyp])
        windows: List[Optional[Window]] = build_gap_windows(
            len(ref), ref_to_hyp_map(pairs), hyp, rec["duration"], self.params.min_window, self.params.max_window)
        wav, sr = load_mono(manifest.resolve_generated(sample), 16000)
        gp = self.detector.predict_windows(wav, sr, windows)
        probs = {g: p for g, p in enumerate(gp) if p is not None}
        scale = self.params.threshold_scale
        thr = {t: (v if v > 1.0 else min(v * scale, 1.0)) for t, v in self.detector.thresholds.items()}    # disabled types (>1) stay disabled
        pred = {g: {t for t, v in p.items() if v >= thr.get(t, 1.01)} for g, p in probs.items()}
        pred = {g: ts for g, ts in pred.items() if ts}
        results, spurious = match_events(gold, pred, probs, set(probs), self.params.tolerance)

        events = []
        for r in results:
            d = asdict(r)
            wg, wm = windows[r.gap], (windows[r.matched_gap] if r.matched_gap is not None else None)
            d["sec_offset"] = (wm.center - wg.center) if (wg is not None and wm is not None) else None
            events.append(d)
        return {**base, "no_timing": False, "events": events, "spurious": [[t, g] for t, g in spurious],
                "valid_gaps": sorted(probs), "n_widened": sum(1 for w in windows if w is not None and w.widened),
                "pred": [[g, sorted(ts)] for g, ts in sorted(pred.items())],
                "probs": [[g, {t: round(v, 4) for t, v in p.items()}] for g, p in sorted(probs.items())]}
