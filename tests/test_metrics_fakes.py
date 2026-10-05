import math

import numpy as np
import pytest

from nvtts_eval.core import ArtifactStore, run_metric
from nvtts_eval.metrics.asr_metric import AsrMetric
from nvtts_eval.metrics.pmos import PMosMetric
from nvtts_eval.metrics.speaker_sim import SpeakerSimMetric
from nvtts_eval.models.asr import ASRResult


class FakeASR:
    """Transcript chosen by audio length in seconds."""
    def __init__(self, table):
        self.table, self.setups, self.calls = table, 0, 0

    def describe(self):
        return {"backend": "fake-asr", "table": self.table}

    def setup(self):
        self.setups += 1

    def transcribe(self, wav, sr):
        self.calls += 1
        return ASRResult(text=self.table[round(len(wav) / sr)], tokens=["a"], timestamps=[0.1])


class FakeDnsmos:
    def describe(self):
        return {"backend": "fake-dnsmos"}

    def setup(self):
        pass

    def score(self, wav, sr):
        assert sr == 16000 and wav.dtype == np.float32
        return {"sig": 3.0, "bak": 4.0, "ovrl": 2.5, "p808": 3.5}


class FakeEmbedder:
    """Embedding chosen by audio length in seconds."""
    def __init__(self, table):
        self.table, self.calls = table, 0

    def describe(self):
        return {"backend": "fake-ecapa", "table": {str(k): list(map(float, v)) for k, v in self.table.items()}}

    def setup(self):
        pass

    def embed(self, wav):
        self.calls += 1
        return np.array(self.table[round(len(wav) / 16000)], dtype=np.float64)


def test_asr_metric_records_and_cache(make_manifest, tmp_path):
    m = make_manifest([("s1", "spk_a", "a b", 2, [1]), ("s2", "spk_a", "c d", 3, [1])])
    store = ArtifactStore(tmp_path / "run")
    be = FakeASR({2: "hai", 3: "ba"})
    r = run_metric(AsrMetric(be), m, store)
    assert [x["text"] for x in r.records] == ["hai", "ba"]
    assert r.records[0]["duration"] == 2.0 and r.records[0]["timestamps"] == [0.1]
    be2 = FakeASR({2: "hai", 3: "ba"})
    r2 = run_metric(AsrMetric(be2), m, store)
    assert r2.from_cache and be2.setups == 0 and be2.calls == 0           # no model load on cache hit
    r3 = run_metric(AsrMetric(FakeASR({2: "hai", 3: "bốn"})), m, store)   # different backend config
    assert not r3.from_cache


def test_pmos_metric_stores_all_four_outputs(make_manifest, tmp_path):
    m = make_manifest([("s1", "spk_a", "a b", 2, [1])])
    r = run_metric(PMosMetric(FakeDnsmos()), m, ArtifactStore(tmp_path / "run"))
    rec = r.records[0]
    assert (rec["sig"], rec["bak"], rec["ovrl"], rec["p808"], rec["duration"]) == (3.0, 4.0, 2.5, 3.5, 2.0)


def test_speaker_sim_centroid_and_mean_cosine_hand_computed(make_manifest, tmp_path):
    # reference clips: 1 s -> e1=[1,0], 2 s -> e2=[0,1];  generated 5 s -> g=[1,0] (norm 1)
    m = make_manifest([("s1", "spk_a", "a b", 5, [1, 2]), ("s2", "spk_a", "a b", 6, [1, 2])])
    table = {1: [1, 0], 2: [0, 1], 5: [1, 0], 6: [3, 4]}
    emb = FakeEmbedder(table)
    r = run_metric(SpeakerSimMetric(emb), m, ArtifactStore(tmp_path / "run1"))
    c0, c1 = r.records[0], r.records[1]
    assert math.isclose(c0["cosine"], math.sqrt(0.5))                      # centroid [√.5, √.5] . [1,0]
    assert math.isclose(c1["cosine"], (3 + 4) / 5 / math.sqrt(2))          # g=[.6,.8]: (.6+.8)/√2
    assert c0["n_ref_clips"] == 2 and math.isclose(c0["ref_seconds"], 3.0)
    assert emb.calls == 4                                                  # 2 ref clips (cached) + 2 generated
    emb2 = FakeEmbedder(table)
    r2 = run_metric(SpeakerSimMetric(emb2, ref_mode="mean_cosine"), m, ArtifactStore(tmp_path / "run2"))
    assert math.isclose(r2.records[0]["cosine"], 0.5)                      # (1 + 0)/2
    assert math.isclose(r2.records[1]["cosine"], (0.6 + 0.8) / 2)


def test_speaker_sim_max_reference_clips_changes_result_and_cache_key(make_manifest, tmp_path):
    m = make_manifest([("s1", "spk_a", "a b", 5, [1, 2])])
    table = {1: [1, 0], 2: [0, 1], 5: [1, 0]}
    store = ArtifactStore(tmp_path / "run")
    full = run_metric(SpeakerSimMetric(FakeEmbedder(table)), m, store).records[0]["cosine"]
    capped = run_metric(SpeakerSimMetric(FakeEmbedder(table), max_reference_clips=1), m, store)
    assert not capped.from_cache and math.isclose(capped.records[0]["cosine"], 1.0) and full < 1.0


def test_speaker_sim_errors_are_recorded_not_raised(make_manifest, tmp_path):
    m = make_manifest([("s1", "spk_a", "a b", 5, []), ("s2", "spk_a", "a b", 6, [1])])
    table = {1: [1, 0], 5: [1, 0], 6: [0, 0]}                              # s2: zero embedding
    r = run_metric(SpeakerSimMetric(FakeEmbedder(table)), m, ArtifactStore(tmp_path / "run"))
    assert r.records[0]["error"] == "missing_reference_audio"
    assert "zero or non-finite" in r.records[1]["error"]


def test_speaker_sim_rejects_bad_mode():
    with pytest.raises(ValueError):
        SpeakerSimMetric(FakeEmbedder({}), ref_mode="nope")
