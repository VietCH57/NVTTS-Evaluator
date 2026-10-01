import json

import pytest

from nvtts_eval.core import (ArtifactError, ArtifactStore, Metric, Needs, fingerprint_manifest,
                             hash_json, run_metric)
from nvtts_eval.data import Manifest, ManifestHeader, NVParser, Sample


def make_manifest(tmp_path, n=3, with_audio=True):
    gen = tmp_path / "gen"
    gen.mkdir(exist_ok=True)
    parser = NVParser()
    samples = []
    for i in range(n):
        sid = f"spk_0000_{i:04d}"
        if with_audio:
            (gen / f"{sid}.wav").write_bytes(b"x" * (i + 1))
        samples.append(Sample.from_text(sid, "spk_0000", f"xin chào [laughter] bạn {i}", parser,
                                        generated_audio=f"{sid}.wav"))
    return Manifest(ManifestHeader(track="A", generated_root=gen.as_posix()), samples)


class SizeMetric(Metric):
    name, version = "size", "1"
    needs = Needs(generated=True)

    def __init__(self, scale=1):
        self.scale, self.calls, self.setups = scale, 0, 0

    def config(self):
        return {"scale": self.scale}

    def setup(self):
        self.setups += 1

    def compute(self, sample, manifest):
        self.calls += 1
        return {"size": manifest.resolve_generated(sample).stat().st_size * self.scale}


# ---- hashing ---------------------------------------------------------------------
def test_hash_json_is_key_order_independent():
    assert hash_json({"a": 1, "b": [1, 2]}) == hash_json({"b": [1, 2], "a": 1})
    assert hash_json({"a": 1}) != hash_json({"a": 2})


def test_fingerprint_tracks_audio_content_and_missing(tmp_path):
    m = make_manifest(tmp_path)
    f1 = fingerprint_manifest(m, generated=True)
    assert f1 == fingerprint_manifest(m, generated=True)
    (tmp_path / "gen" / "spk_0000_0001.wav").write_bytes(b"changed")
    assert fingerprint_manifest(m, generated=True) != f1
    (tmp_path / "gen" / "spk_0000_0002.wav").unlink()
    assert fingerprint_manifest(m, generated=True) != f1


def test_fingerprint_only_covers_requested_inputs(tmp_path):
    m = make_manifest(tmp_path)
    f_text = fingerprint_manifest(m, text=True)
    (tmp_path / "gen" / "spk_0000_0000.wav").write_bytes(b"different audio")
    assert fingerprint_manifest(m, text=True) == f_text      # audio change doesn't affect a text-only metric


# ---- runner / cache --------------------------------------------------------------
def test_run_then_cache_hit(tmp_path):
    m, store = make_manifest(tmp_path), ArtifactStore(tmp_path / "run")
    met = SizeMetric()
    r1 = run_metric(met, m, store)
    assert not r1.from_cache and met.calls == 3 and met.setups == 1
    assert [r["size"] for r in r1.records] == [1, 2, 3]
    r2 = run_metric(SizeMetric(), m, store)
    assert r2.from_cache and r2.records == r1.records


def test_cache_not_used_after_audio_or_config_change(tmp_path):
    m, store = make_manifest(tmp_path), ArtifactStore(tmp_path / "run")
    run_metric(SizeMetric(), m, store)
    assert not run_metric(SizeMetric(scale=2), m, store).from_cache          # config changed
    (tmp_path / "gen" / "spk_0000_0000.wav").write_bytes(b"new")
    assert not run_metric(SizeMetric(scale=2), m, store).from_cache          # audio changed


def test_cache_hit_does_not_call_setup(tmp_path):
    m, store = make_manifest(tmp_path), ArtifactStore(tmp_path / "run")
    run_metric(SizeMetric(), m, store)
    met = SizeMetric()
    assert run_metric(met, m, store).from_cache and met.setups == 0 and met.calls == 0


def test_force_recomputes(tmp_path):
    m, store = make_manifest(tmp_path), ArtifactStore(tmp_path / "run")
    run_metric(SizeMetric(), m, store)
    met = SizeMetric()
    assert not run_metric(met, m, store, force=True).from_cache and met.calls == 3


def test_missing_audio_and_exceptions_become_error_records(tmp_path):
    m = make_manifest(tmp_path)
    (tmp_path / "gen" / "spk_0000_0001.wav").unlink()

    class Boom(SizeMetric):
        def compute(self, sample, manifest):
            if sample.sample_id.endswith("0002"):
                raise RuntimeError("bad file")
            return super().compute(sample, manifest)

    r = run_metric(Boom(), m, ArtifactStore(tmp_path / "run"))
    assert [("error" in x) for x in r.records] == [False, True, True]
    assert r.records[1]["error"] == "missing_generated_audio"
    assert r.records[2]["error"] == "RuntimeError: bad file"
    assert r.meta.n_errors == 2 and r.meta.n_records == 3


def test_reserved_keys_rejected(tmp_path):
    class Bad(SizeMetric):
        def compute(self, sample, manifest):
            return {"sample_id": "oops"}

    r = run_metric(Bad(), make_manifest(tmp_path, n=1), ArtifactStore(tmp_path / "run"))
    assert "reserved" in r.records[0]["error"]


def test_missing_reference_is_reported(tmp_path):
    class NeedsRef(SizeMetric):
        name = "needs_ref"
        needs = Needs(generated=True, references=True)

    r = run_metric(NeedsRef(), make_manifest(tmp_path, n=1), ArtifactStore(tmp_path / "run"))
    assert r.records[0]["error"] == "missing_reference_audio"


def test_corrupted_artifact_is_not_fresh_and_raises_on_read(tmp_path):
    m, store = make_manifest(tmp_path), ArtifactStore(tmp_path / "run")
    run_metric(SizeMetric(), m, store)
    p = tmp_path / "run" / "artifacts" / "size" / "per_sample.jsonl"
    p.write_text(json.dumps({"sample_id": "x", "size": 999}) + "\n", encoding="utf-8")
    with pytest.raises(ArtifactError):
        store.read_records("size")
    assert not run_metric(SizeMetric(), m, store).from_cache                  # silently recomputed
    assert [r["size"] for r in store.read_records("size")] == [1, 2, 3]


def test_unicode_records_roundtrip(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    store.write("u", [{"sample_id": "a", "asr": "xin chào các bạn"}],
                dict(metric_version="1", input_hash="h", config_hash="c", config={}))
    assert store.read_records("u")[0]["asr"] == "xin chào các bạn"
