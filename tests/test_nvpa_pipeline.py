import math

import numpy as np
import pytest

from nvtts_eval.core import ArtifactStore, run_metric
from nvtts_eval.nvpa.metric import NvpaMetric, NvpaParams
from nvtts_eval.nvpa.summary import shuffle_baseline, summarize_nvpa
from nvtts_eval.nvpa.training import build_training_set

META = dict(metric_version="1", input_hash="h", config_hash="c", config={})
TYPES = ("laughter", "breathing", "sniff", "throatclearing")
TEXT = "một hai [breathing] ba bốn [laughter] năm"          # words: 5; gaps: breathing@2, laughter@4
ASR_OK = {"tokens": [" MỘT", " HAI", " BA", " BỐN", " NĂM"], "timestamps": [0.0, 0.4, 1.2, 1.6, 2.4],
          "text": "MỘT HAI BA BỐN NĂM", "duration": 3.0}
# windows (min_len .2): g0 [0,.2] g1 [0,.4] g2 [.4,1.2] g3 [1.2,1.6] g4 [1.6,2.4] g5 [2.4,3.0]


class FakeDetector:
    nv_types = TYPES
    thresholds = {t: 0.5 for t in TYPES}

    def __init__(self, by_start):
        self.by_start = by_start                           # window start time -> {type: prob}

    def describe(self):
        return {"backend": "fake", "by_start": {str(k): v for k, v in self.by_start.items()}}

    def predict_windows(self, wav, sr, windows):
        return [None if w is None else {t: self.by_start.get(round(w.t0, 3), {}).get(t, 0.0) for t in TYPES}
                for w in windows]


def run(make_manifest, tmp_path, items, asr_records, det, **params):
    m = make_manifest(items)
    store = ArtifactStore(tmp_path / "run")
    store.write("asr", asr_records, META)
    r = run_metric(NvpaMetric(store, det, params=NvpaParams(**params)), m, store)
    return m, store, r


def asr(sid, **kw):
    return {"sample_id": sid, **{**ASR_OK, **kw}}


def test_perfect_detection_hits_all_events(make_manifest, tmp_path):
    det = FakeDetector({0.4: {"breathing": 0.9}, 1.6: {"laughter": 0.8}})
    _, _, r = run(make_manifest, tmp_path, [("s1", "spk_a", TEXT, 3, [1])], [asr("s1")], det)
    rec = r.records[0]
    assert [e["hit"] for e in rec["events"]] == [True, True] and rec["spurious"] == []
    assert [(e["gap"], e["matched_gap"], e["offset_words"]) for e in rec["events"]] == [(2, 2, 0), (4, 4, 0)]
    assert rec["n_gaps"] == 6 and rec["valid_gaps"] == [0, 1, 2, 3, 4, 5]
    assert rec["pred"] == [[2, ["breathing"]], [4, ["laughter"]]]


def test_offset_wrong_type_missing_and_spurious(make_manifest, tmp_path):
    # breathing detected one gap late (tolerated), laughter replaced by a sniff, plus a spurious throatclearing
    det = FakeDetector({1.2: {"breathing": 0.9}, 1.6: {"sniff": 0.7}, 2.4: {"throatclearing": 0.6}})
    _, _, r = run(make_manifest, tmp_path, [("s1", "spk_a", TEXT, 3, [1])], [asr("s1")], det)
    rec = r.records[0]
    e1, e2 = rec["events"]
    assert e1["hit"] and e1["matched_gap"] == 3 and e1["offset_words"] == 1
    assert e1["sec_offset"] == pytest.approx(1.4 - 0.8)          # window centres: gap 3 = [1.2,1.6], gap 2 = [0.4,1.2]
    assert not e2["hit"] and e2["reason"] == "wrong_type"
    assert sorted(map(tuple, rec["spurious"])) == [("sniff", 4), ("throatclearing", 5)]


def test_nvpa_requires_asr_artifact(make_manifest, tmp_path):
    m = make_manifest([("s1", "spk_a", TEXT, 3, [1])])
    with pytest.raises(RuntimeError, match="asr"):
        run_metric(NvpaMetric(ArtifactStore(tmp_path / "empty"), FakeDetector({})), m, ArtifactStore(tmp_path / "empty"))


def test_asr_failure_and_missing_timing(make_manifest, tmp_path):
    items = [("s1", "spk_a", TEXT, 3, [1]), ("s2", "spk_a", TEXT, 3, [1]), ("s3", "spk_a", TEXT, 3, [1])]
    recs = [{"sample_id": "s1", "error": "boom"}, asr("s2", tokens=None, timestamps=None), asr("s3")]
    _, _, r = run(make_manifest, tmp_path, items, recs, FakeDetector({}))
    assert "ASR failed" in r.records[0]["error"]
    assert r.records[1]["no_timing"] and [e["reason"] for e in r.records[1]["events"]] == ["alignment_unreliable"] * 2
    assert not r.records[2]["no_timing"]


def test_cache_invalidated_by_asr_or_detector_change(make_manifest, tmp_path):
    m = make_manifest([("s1", "spk_a", TEXT, 3, [1])])
    store = ArtifactStore(tmp_path / "run")
    store.write("asr", [asr("s1")], META)
    det = FakeDetector({0.4: {"breathing": 0.9}})
    assert not run_metric(NvpaMetric(store, det), m, store).from_cache
    assert run_metric(NvpaMetric(store, det), m, store).from_cache
    assert not run_metric(NvpaMetric(store, FakeDetector({0.4: {"breathing": 0.2}})), m, store).from_cache
    store.write("asr", [asr("s1", timestamps=[0.0, 0.5, 1.2, 1.6, 2.4])], META)
    assert not run_metric(NvpaMetric(store, FakeDetector({0.4: {"breathing": 0.2}})), m, store).from_cache


def _records(make_manifest, tmp_path):
    """3 samples / 2 speakers. Gold events: 2 per sample (breathing@2, laughter@4)."""
    items = [("s1", "spk_a", TEXT, 3, [1]), ("s2", "spk_a", TEXT, 3, [1]), ("s3", "spk_b", TEXT, 3, [1])]
    store = ArtifactStore(tmp_path / "run")
    store.write("asr", [asr(i[0]) for i in items], META)
    m = make_manifest(items)
    det = FakeDetector({0.4: {"breathing": 0.9}, 1.6: {"laughter": 0.8}})          # perfect on all three
    recs = run_metric(NvpaMetric(store, det), m, store).records
    # degrade: s2 loses its laughter, s3 gets wrong-type laughter
    recs[1] = {**recs[1], "events": [recs[1]["events"][0], {**recs[1]["events"][1], "hit": False, "reason": "missing",
                                                            "matched_gap": None, "offset_words": None}]}
    recs[2] = {**recs[2], "events": [recs[2]["events"][0], {**recs[2]["events"][1], "hit": False, "reason": "wrong_type",
                                                            "matched_gap": None, "offset_words": None}]}
    return recs, {"s1": "spk_a", "s2": "spk_a", "s3": "spk_b"}


def test_summary_hand_computed(make_manifest, tmp_path):
    recs, spk = _records(make_manifest, tmp_path)
    s = summarize_nvpa(recs, spk, tolerance=1, n_boot=200, shuffle_reps=5)
    assert (s["n_events"], s["n_hits"]) == (6, 4) and math.isclose(s["value"], 4 / 6)
    assert s["by_type"]["breathing"]["mean"] == 1.0 and s["by_type"]["laughter"]["mean"] == pytest.approx(1 / 3)
    assert math.isclose(s["macro_type"], (1.0 + 1 / 3) / 2)
    assert s["reasons"] == {"ok": 4, "missing": 1, "wrong_type": 1}
    assert math.isclose(s["detection_rate"], 5 / 6) and math.isclose(s["type_accuracy_given_detected"], 4 / 5)
    assert math.isclose(s["macro_speaker"], (3 / 4 + 1 / 2) / 2)                   # spk_a: 3/4, spk_b: 1/2
    assert s["by_position"]["mid"]["n"] == 6 and s["placement_error_words"] == {0: 4}
    assert s["n_failed"] == 0 and s["spurious_total"] == 0


def test_summary_counts_failures_and_unreliable_policy(make_manifest, tmp_path):
    recs, spk = _records(make_manifest, tmp_path)
    recs = recs + [{"sample_id": "sx", "error": "ASR failed"}]
    unreliable = {**recs[0], "sample_id": "sy", "events": [{**e, "hit": False, "reason": "alignment_unreliable",
                                                            "matched_gap": None, "offset_words": None} for e in recs[0]["events"]]}
    spk = {**spk, "sy": "spk_b"}
    miss = summarize_nvpa(recs + [unreliable], spk, 1, policy="miss", n_boot=50, shuffle_reps=2)
    excl = summarize_nvpa(recs + [unreliable], spk, 1, policy="exclude", n_boot=50, shuffle_reps=2)
    assert miss["n_events"] == 8 and miss["n_hits"] == 4 and miss["n_failed"] == 1
    assert excl["n_events"] == 6 and excl["n_hits"] == 4
    assert miss["reasons"]["alignment_unreliable"] == 2


def test_shuffle_baseline_bounds(make_manifest, tmp_path):
    recs, _ = _records(make_manifest, tmp_path)
    full = [{**r, "pred": [[g, list(TYPES)] for g in range(6)]} for r in recs]       # every type everywhere
    none = [{**r, "pred": []} for r in recs]
    assert shuffle_baseline(full, 0, n_reps=5)["mean"] == 1.0
    assert shuffle_baseline(none, 1, n_reps=5)["mean"] == 0.0
    sparse = shuffle_baseline(recs, 0, n_reps=300, seed=1)                            # detections only at gaps 2 (breathing), 4 (laughter)
    assert 0.05 < sparse["mean"] < 0.40                                                # ~ chance: 1/6 per event
    assert shuffle_baseline(recs, 0, n_reps=20, seed=1) == shuffle_baseline(recs, 0, n_reps=20, seed=1)
    assert shuffle_baseline([], 1)["n_reps"] == 0


def test_build_training_set_labels_windows(make_manifest, tmp_path):
    m = make_manifest([("s1", "spk_a", TEXT, 3, [1]), ("s2", "spk_b", "một hai ba", 3, [1])])
    recs = [asr("s1"), {"sample_id": "s2", "error": "x"}]
    X, Y, groups, info = build_training_set(m, recs, TYPES)
    assert X.shape[0] == 6 and Y.shape == (6, 4) and groups == ["spk_a"] * 6
    assert info["samples_used"] == 1 and info["skipped_no_asr"] == 1
    assert Y[:, 1].tolist() == [0, 0, 1, 0, 0, 0] and Y[:, 0].tolist() == [0, 0, 0, 0, 1, 0]
    assert info["positives"] == {"laughter": 1, "breathing": 1, "sniff": 0, "throatclearing": 0}
