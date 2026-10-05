import json

import numpy as np
import soundfile as sf

import nvtts_eval.cli as cli
from nvtts_eval.data.adapter_vinv import build_manifest
from nvtts_eval.metrics.asr_metric import AsrMetric
from nvtts_eval.metrics.pmos import PMosMetric
from nvtts_eval.metrics.speaker_sim import SpeakerSimMetric
from tests.test_metrics_fakes import FakeASR, FakeDnsmos, FakeEmbedder


def _dataset(root):
    def write(split, spk, items):
        d = root / split / spk
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{spk}.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        for k, it in enumerate(items, start=1):
            sf.write(str(d / it["audio"]), np.full(24000 * (k + 1), 0.05, dtype="float32"), 24000)  # 24 kHz, 2..s
    write("train", "spk_0000", [{"audio": "0001.flac", "text": "a b", "language_id": "vi"},
                               {"audio": "0002.flac", "text": "c d", "language_id": "vi"}])
    write("dev", "spk_0000", [{"audio": "0001.flac", "text": "xin chào [laughter] các bạn", "language_id": "vi"}])


def test_run_then_cached_rerun_then_summary(tmp_path, monkeypatch, capsys):
    root = tmp_path / "data"
    _dataset(root)
    manifest, _ = build_manifest(root, "dev", track="A")        # ground-truth calibration manifest
    mp = tmp_path / "m.jsonl"
    manifest.save(mp)
    # dev clip is 2 s at 24 kHz -> 2 s after resampling; train refs are 2 s and 3 s
    made = {}

    def fake_build(cfg, names, store=None):
        made["asr"] = FakeASR({2: "xin chào các bạn"})
        made["emb"] = FakeEmbedder({2: [1, 0], 3: [0, 1]})
        all_metrics = {"asr": AsrMetric(made["asr"]), "pmos": PMosMetric(FakeDnsmos()),
                       "ss": SpeakerSimMetric(made["emb"])}
        return {n: all_metrics[n] for n in names}

    monkeypatch.setattr(cli, "build_metrics", fake_build)
    run = tmp_path / "run"
    assert cli.main(["run", "--manifest", str(mp), "--run-dir", str(run)]) == 0
    out = capsys.readouterr().out
    assert "asr: computed" in out and "pmos: computed" in out and "ss: computed" in out
    assert "WER (corpus): 0.000" in out and "AutoScore: N/A" in out
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert s["automatic_metrics"]["ss"]["mean"] is not None and s["track"] == "A"

    assert cli.main(["run", "--manifest", str(mp), "--run-dir", str(run)]) == 0       # unchanged inputs
    out2 = capsys.readouterr().out
    assert out2.count("cache hit") == 3
    assert cli.main(["run", "--manifest", str(mp), "--run-dir", str(run), "--metrics", "asr", "--force"]) == 0
    assert "asr: computed" in capsys.readouterr().out
