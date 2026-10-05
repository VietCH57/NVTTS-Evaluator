import json
import math

import pytest

from nvtts_eval.cli import build_metrics, main
from nvtts_eval.config import config_from_dict, load_config
from nvtts_eval.core import ArtifactStore
from nvtts_eval.models.asr import resolve_sherpa_files
from nvtts_eval.models.dnsmos import _scalar
from nvtts_eval.models.tokens import write_tokens
from nvtts_eval.report.summary import build_summary, format_summary

META = dict(metric_version="1", input_hash="h", config_hash="c", config={})


# ---- config --------------------------------------------------------------------------
def test_config_defaults_and_overrides():
    c = config_from_dict({"pmos": {"output": "p808"}, "wer": {"aggregation": "sample_mean"},
                          "scoring": {"clip_wer": False}, "text_norm": {"lowercase": False}})
    assert c.pmos.output == "p808" and c.wer.aggregation == "sample_mean"
    assert c.scoring.clip_wer is False and c.text_norm.lowercase is False
    assert c.asr.use_int8 is False and c.ss.ref_mode == "centroid"


@pytest.mark.parametrize("bad", [{"asr": {"model_dirr": "x"}}, {"nope": 1}, {"pmos": {"output": "mos"}},
                                 {"ss": {"ref_mode": "x"}}, {"wer": {"aggregation": "x"}}, {"asr": 5}])
def test_config_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        config_from_dict(bad)


def test_default_yaml_loads(tmp_path):
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert cfg.pmos.repo_id == "prj-beatrice/dnsmos-torch-native" and cfg.ss.source.endswith("ecapa-voxceleb")
    (tmp_path / "c.json").write_text(json.dumps({"bootstrap": {"seed": 7}}), encoding="utf-8")
    assert load_config(tmp_path / "c.json").bootstrap.seed == 7


# ---- backends (file resolution / construction without heavy imports) -------------------
def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"x")


HF_FILES = ["encoder-epoch-20-avg-10.onnx", "encoder-epoch-20-avg-10.int8.onnx",
            "decoder-epoch-20-avg-10.onnx", "decoder-epoch-20-avg-10.int8.onnx",
            "joiner-epoch-20-avg-10.onnx", "joiner-epoch-20-avg-10.int8.onnx", "tokens.txt", "bpe.model"]


def test_sherpa_file_resolution_hf_layout(tmp_path):
    _touch(tmp_path, *HF_FILES)
    fp32 = resolve_sherpa_files(tmp_path, use_int8=False)
    int8 = resolve_sherpa_files(tmp_path, use_int8=True)
    assert fp32["encoder"].name == "encoder-epoch-20-avg-10.onnx"
    assert int8["joiner"].name == "joiner-epoch-20-avg-10.int8.onnx"
    assert fp32["tokens"].name == "tokens.txt"


def test_sherpa_file_resolution_packaged_int8_layout(tmp_path):
    _touch(tmp_path, "encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt")
    f = resolve_sherpa_files(tmp_path, use_int8=True)          # decoder.onnx used as the only decoder
    assert f["decoder"].name == "decoder.onnx" and f["encoder"].name == "encoder.int8.onnx"


def test_sherpa_resolution_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_sherpa_files(tmp_path / "missing", False)
    _touch(tmp_path, *[f for f in HF_FILES if f != "tokens.txt"])
    with pytest.raises(FileNotFoundError, match="make_tokens"):
        resolve_sherpa_files(tmp_path, False)
    _touch(tmp_path, "tokens.txt", "encoder-extra.onnx")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_sherpa_files(tmp_path, False)


def test_build_metrics_is_lazy_and_describes_asr(tmp_path):
    _touch(tmp_path, *HF_FILES)
    cfg = config_from_dict({"asr": {"model_dir": str(tmp_path)}})
    ms = build_metrics(cfg, ["asr", "pmos", "ss"])             # must work without torch / sherpa installed
    d = ms["asr"].config()["backend"]
    assert d["files"]["encoder"]["name"] == "encoder-epoch-20-avg-10.onnx" and len(d["files"]["tokens"]["sha256"]) == 64
    assert ms["pmos"].backend.describe()["repo_id"] == "prj-beatrice/dnsmos-torch-native"
    assert ms["ss"].config()["ref_mode"] == "centroid"
    with pytest.raises(ValueError):
        build_metrics(config_from_dict({}), ["asr"])           # asr.model_dir missing
    with pytest.raises(ValueError):
        build_metrics(cfg, ["nvpa"])


def test_scalar_helper():
    import numpy as np
    assert _scalar(np.array([3.25])) == 3.25 and _scalar(2) == 2.0 and _scalar(np.float32(1.5)) == 1.5


def test_write_tokens_roundtrip(tmp_path):
    import sentencepiece as spm
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join(["xin chào các bạn", "hôm nay trời đẹp", "tôi rất vui khi gặp bạn"] * 30), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / "bpe"), vocab_size=40,
                                   model_type="bpe", user_defined_symbols=["<blk>", "<sos/eos>"], unk_id=2,
                                   bos_id=-1, eos_id=-1, pad_id=-1, character_coverage=1.0)
    n = write_tokens(tmp_path / "bpe.model", tmp_path / "tokens.txt")
    lines = (tmp_path / "tokens.txt").read_text(encoding="utf-8").splitlines()
    assert n == len(lines) == 40
    assert lines[0].endswith(" 0") and lines[1].endswith(" 1")
    assert [ln.rsplit(" ", 1)[1] for ln in lines] == [str(i) for i in range(40)]


# ---- summary -------------------------------------------------------------------------
def _fill_store(store, ids_speakers):
    store.write("asr", [{"sample_id": "s1", "text": "một hai ba bốn"}, {"sample_id": "s2", "text": "một hai ba năm"}], META)
    store.write("pmos", [{"sample_id": "s1", "ovrl": 3.0, "sig": 1, "bak": 1, "p808": 1},
                         {"sample_id": "s2", "ovrl": 4.0, "sig": 1, "bak": 1, "p808": 1}], META)
    store.write("ss", [{"sample_id": "s1", "cosine": 0.7}, {"sample_id": "s2", "cosine": 0.9}], META)


def test_summary_without_nvpa_has_no_auto_score(make_manifest, tmp_path):
    m = make_manifest([("s1", "spk_a", "một hai [breathing] ba bốn", 1, [1]), ("s2", "spk_b", "một hai ba bốn", 1, [1])])
    store = ArtifactStore(tmp_path / "run")
    _fill_store(store, None)
    s = build_summary(m, store, config_from_dict({"bootstrap": {"n_boot": 100}}))
    assert s["automatic_score"] is None and s["automatic_score_missing_components"] == ["NVPA"]
    assert s["final_score"] is None and s["human_metrics"] == {"SN": None, "Q": None}
    assert math.isclose(s["automatic_metrics"]["wer"]["value"], 1 / 8)
    assert math.isclose(s["automatic_metrics"]["pmos"]["mean"], 3.5) and math.isclose(s["automatic_metrics"]["ss"]["mean"], 0.8)
    part = s["automatic_score_partial_components"]
    assert math.isclose(part["WER"]["contribution"], 0.15 * 0.875)
    json.dumps(s)                                                          # fully JSON-serialisable
    assert "AutoScore: N/A" in format_summary(s)


def test_summary_auto_score_hand_computed(make_manifest, tmp_path):
    m = make_manifest([("s1", "spk_a", "một hai [breathing] ba bốn", 1, [1]), ("s2", "spk_b", "một hai ba bốn", 1, [1])])
    store = ArtifactStore(tmp_path / "run")
    _fill_store(store, None)
    s = build_summary(m, store, config_from_dict({"bootstrap": {"n_boot": 100}}), nvpa=0.5)
    # Track A: .3*.5 + .15*(1-.125) + .15*((3.5-1)/4) + .10*.8 = .15 + .13125 + .09375 + .08 = .455
    a = s["automatic_score"]
    assert math.isclose(a["value"], 0.455) and math.isclose(a["max_possible"], 0.70)
    assert math.isclose(a["renormalized"], 0.455 / 0.70) and a["official"] is False
    assert "not official" in format_summary(s)


def test_summary_pmos_output_is_configurable(make_manifest, tmp_path):
    m = make_manifest([("s1", "spk_a", "một", 1, [1])])
    store = ArtifactStore(tmp_path / "run")
    store.write("pmos", [{"sample_id": "s1", "ovrl": 3.0, "sig": 4.5, "bak": 1, "p808": 2}], META)
    assert build_summary(m, store, config_from_dict({"pmos": {"output": "sig"}}))["automatic_metrics"]["pmos"]["mean"] == 4.5


def test_cli_summary_command_reads_cache_only(make_manifest, tmp_path, capsys):
    m = make_manifest([("s1", "spk_a", "một hai [breathing] ba bốn", 1, [1]), ("s2", "spk_b", "một hai ba bốn", 1, [1])])
    mp = tmp_path / "m.jsonl"
    m.save(mp)
    run = tmp_path / "run"
    _fill_store(ArtifactStore(run), None)
    assert main(["summary", "--manifest", str(mp), "--run-dir", str(run)]) == 0
    out = capsys.readouterr().out
    assert "WER (corpus): 0.125" in out and "AutoScore: N/A" in out
    data = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert data["data_counts"]["n_samples"] == 2
