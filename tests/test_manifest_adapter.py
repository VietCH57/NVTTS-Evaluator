import json

import pytest

from nvtts_eval.data import Manifest, ManifestError, NVParser
from nvtts_eval.data.adapter_vinv import DatasetFormatError, build_manifest


def _write(root, split, spk, items, audio=True):
    d = root / split / spk
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{spk}.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    if audio:
        for it in items:
            (d / it["audio"]).write_bytes(b"")


@pytest.fixture
def dataset(tmp_path):
    train = [
        {"audio": "0001.flac", "text": "xin chào [laughter] các bạn", "language_id": "vi"},
        {"audio": "0002.flac", "text": "tôi [breathing] đây [sniff]", "language_id": "vi"},
    ]
    dev = [{"audio": "0001.flac", "text": "hôm nay [breathing] trời đẹp", "language_id": "vi"}]
    _write(tmp_path, "train", "spk_0000", train)
    _write(tmp_path, "dev", "spk_0000", dev)
    _write(tmp_path, "train", "spk_0001", train[:1])            # in train only
    _write(tmp_path, "dev", "spk_0002", dev)                    # dev only -> no reference
    return tmp_path


def test_ground_truth_manifest_track_a(dataset):
    m, rep = build_manifest(dataset, "dev", track="A")
    assert [s.sample_id for s in m] == ["spk_0000_0001", "spk_0002_0001"]
    s = m.samples[0]
    assert s.clean_text == "hôm nay trời đẹp"
    assert s.nv_events == [{"type": "breathing", "gap_index": 2}]
    assert s.reference_audio == ["train/spk_0000/0001.flac", "train/spk_0000/0002.flac"]
    assert s.generated_audio == s.ground_truth_audio == "dev/spk_0000/0001.flac"
    assert rep.speakers_without_reference == ["spk_0002"]
    assert rep.n_missing_ground_truth == 0
    assert m.resolve_generated(s) == dataset / "dev/spk_0000/0001.flac"


def test_max_reference_clips_and_self_exclusion(dataset):
    m, _ = build_manifest(dataset, "dev", max_reference_clips=1)
    assert len(m.samples[0].reference_audio) == 1
    # calibrating on the reference split itself must not use the utterance as its own reference
    m2, _ = build_manifest(dataset, "train", reference_split="train")
    s = next(x for x in m2 if x.sample_id == "spk_0000_0001")
    assert s.reference_audio == ["train/spk_0000/0002.flac"]


def test_model_source_marks_missing_generated(dataset, tmp_path_factory):
    gen = tmp_path_factory.mktemp("gen")
    (gen / "spk_0000_0001.wav").write_bytes(b"")
    m, rep = build_manifest(dataset, "dev", source="model", generated_dir=gen)
    by = {s.sample_id: s for s in m}
    assert by["spk_0000_0001"].generated_audio == "spk_0000_0001.wav"
    assert by["spk_0002_0001"].generated_audio is None
    assert rep.n_missing_generated == 1
    assert m.resolve_generated(by["spk_0000_0001"]) == gen / "spk_0000_0001.wav"


def test_track_b_requires_reference_map(dataset):
    with pytest.raises(ValueError):
        build_manifest(dataset, "dev", track="B")
    m, _ = build_manifest(dataset, "dev", track="B",
                          reference_map={"spk_0000_0001": ["ref/a.wav"], "spk_0002_0001": ["ref/b.wav"]})
    assert m.header.track == "B"
    with pytest.raises(ManifestError):          # Track B samples need references
        build_manifest(dataset, "dev", track="B", reference_map={"spk_0000_0001": ["ref/a.wav"]})


def test_save_load_roundtrip(dataset, tmp_path):
    m, _ = build_manifest(dataset, "dev")
    out = tmp_path / "m.jsonl"
    m.save(out)
    m2 = Manifest.load(out)
    assert m2.header == m.header and m2.samples == m.samples


def test_load_detects_tampered_events(dataset, tmp_path):
    m, _ = build_manifest(dataset, "dev")
    out = tmp_path / "m.jsonl"
    m.save(out)
    lines = out.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[1])
    rec["nv_events"] = [{"type": "laughter", "gap_index": 0}]
    lines[1] = json.dumps(rec, ensure_ascii=False)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        Manifest.load(out)


def test_bad_inputs(dataset, tmp_path):
    with pytest.raises(DatasetFormatError):
        build_manifest(dataset, "test")
    _write(tmp_path, "dev", "spk_9", [{"audio": "1.flac", "text": "a [cough] b"}], audio=False)
    with pytest.raises(Exception):               # unknown tag -> parser error under default policy
        build_manifest(tmp_path, "dev", track="A")
    with pytest.raises(ValueError):
        build_manifest(dataset, "dev", source="model")


def test_speaker_filter(dataset):
    m, _ = build_manifest(dataset, "dev", speakers=["spk_0000"])
    assert [s.speaker_id for s in m] == ["spk_0000"]
