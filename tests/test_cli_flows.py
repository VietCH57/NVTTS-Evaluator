import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import nvtts_eval.cli as cli
from nvtts_eval.core import ArtifactStore, Metric, Needs
from nvtts_eval.data import Manifest, ManifestHeader, NVParser, Sample
from nvtts_eval.data.adapter_vinv import main as adapter_main
from tests.test_human import big_manifest
from tests.test_metrics_fakes import FakeDnsmos, FakeEmbedder
from nvtts_eval.metrics.pmos import PMosMetric
from nvtts_eval.metrics.speaker_sim import SpeakerSimMetric

TEXT = "một hai [breathing] ba bốn [laughter] năm"


# ---- human-evaluation commands end to end ---------------------------------------------
def test_human_commands_flow(tmp_path, capsys):
    m = big_manifest(tmp_path)
    mp = tmp_path / "m.jsonl"
    m.save(mp)
    run, pkg = tmp_path / "run", tmp_path / "pkg"
    base = ["--manifest", str(mp), "--run-dir", str(run)]
    assert cli.main(["human-subset", *base, "--size", "12"]) == 0
    sub = json.loads((run / "human" / "subset.json").read_text(encoding="utf-8"))
    assert len(sub["sample_ids"]) == 12
    assert cli.main(["human-export", *base, "--out", str(pkg)]) == 0
    key = json.loads((pkg / "PRIVATE_key.json").read_text(encoding="utf-8"))
    kind = {i["item_id"]: i["kind"] for i in key["items"]}

    def fill(name, gt, deg, model):
        rows = list(csv.DictReader(open(pkg / "for_raters" / "rating_sheet.csv", encoding="utf-8-sig")))
        for r in rows:
            v = {"anchor_gt": gt, "anchor_degraded": deg, "model": model}[kind[r["item_id"]]]
            r["SN"], r["Q"] = str(v), str(v)
        p = tmp_path / f"{name}.csv"
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return str(p)

    sheets = [fill("alice", 5, 2, 4), fill("bob", 4, 2, 3), fill("careless", 3.2, 3.0, 1)]
    capsys.readouterr()
    assert cli.main(["human-import", *base, "--package", str(pkg), "--ratings", *sheets]) == 0
    out = capsys.readouterr().out
    assert "2/3 accepted" in out and "flagged careless" in out
    h = json.loads((run / "human" / "human_scores.json").read_text(encoding="utf-8"))
    assert h["n_rated_samples"] == 12 and abs(h["metrics"]["SN"]["mean"] - 3.5) < 1e-9     # mean of alice(4) and bob(3)
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert s["human_metrics"]["SN"]["mean"] == 3.5 and s["final_score"] is None          # no automatic artifacts yet
    assert "Human (n=12" in out
    assert cli.main(["summary", *base]) == 0                                              # summary picks the scores up too
    assert "Human (n=12" in capsys.readouterr().out


def test_human_import_rejects_duplicate_rater_names(tmp_path):
    m = big_manifest(tmp_path)
    mp = tmp_path / "m.jsonl"
    m.save(mp)
    run, pkg = tmp_path / "run", tmp_path / "pkg"
    base = ["--manifest", str(mp), "--run-dir", str(run)]
    cli.main(["human-subset", *base, "--size", "6"])
    cli.main(["human-export", *base, "--out", str(pkg)])
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for d in ("a", "b"):
        (tmp_path / d / "rating_sheet.csv").write_bytes((pkg / "for_raters" / "rating_sheet.csv").read_bytes())
    with pytest.raises(SystemExit, match="rename"):
        cli.main(["human-import", *base, "--package", str(pkg), "--ratings",
                  str(tmp_path / "a" / "rating_sheet.csv"), str(tmp_path / "b" / "rating_sheet.csv")])


def test_human_export_needs_subset(tmp_path):
    m = big_manifest(tmp_path)
    mp = tmp_path / "m.jsonl"
    m.save(mp)
    with pytest.raises(SystemExit):
        cli.main(["human-export", "--manifest", str(mp), "--run-dir", str(tmp_path / "run"), "--out", str(tmp_path / "p")])


# ---- train-detector, then NVPA through `run` ------------------------------------------
class FakeAsrMetric(Metric):
    """Words evenly spaced every 0.5 s; matches TEXT (5 words)."""
    name, version = "asr", "1"
    needs = Needs(generated=True)

    def config(self):
        return {"fake": "asr"}

    def compute(self, sample, manifest):
        words = NVParser().parse(sample.text).words
        toks = [" " + w.upper() for w in words]
        return {"text": " ".join(w.upper() for w in words), "tokens": toks,
                "timestamps": [0.5 * i for i in range(len(toks))], "duration": 3.0}


def make_train_manifest(tmp_path, split="train", n=14):
    gen = tmp_path / "gen"
    gen.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    p, samples = NVParser(), []
    for i in range(n):
        sid = f"spk_{i % 7}_{i:03d}"
        sf.write(str(gen / f"{sid}.wav"), (0.05 * rng.standard_normal(48000)).astype("float32"), 16000)
        samples.append(Sample.from_text(sid, f"spk_{i % 7}", TEXT, p, generated_audio=f"{sid}.wav",
                                        ground_truth_audio=f"gen/{sid}.wav"))
    hdr = ManifestHeader(track="A", source="ground_truth", split=split, audio_root=tmp_path.as_posix(),
                         generated_root=gen.as_posix())
    return Manifest(hdr, samples)


def patch_asr(monkeypatch):
    real = cli.build_metrics

    def fake(cfg, names, store=None):
        out = {}
        for n in names:
            out[n] = FakeAsrMetric() if n == "asr" else real(cfg, [n], store)[n]
        return out
    monkeypatch.setattr(cli, "build_metrics", fake)


def test_train_detector_refuses_dev_split(tmp_path, monkeypatch):
    patch_asr(monkeypatch)
    mp = tmp_path / "dev.jsonl"
    make_train_manifest(tmp_path, split="dev").save(mp)
    with pytest.raises(SystemExit, match="dev"):
        cli.main(["train-detector", "--manifest", str(mp), "--run-dir", str(tmp_path / "r"), "--out", str(tmp_path / "d.joblib")])


def test_train_detector_then_run_nvpa(tmp_path, monkeypatch, capsys):
    patch_asr(monkeypatch)
    train_mp, dev_mp = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    make_train_manifest(tmp_path, "train", 14).save(train_mp)
    make_train_manifest(tmp_path, "dev", 4).save(dev_mp)
    det = tmp_path / "models" / "gap.joblib"
    assert cli.main(["train-detector", "--manifest", str(train_mp), "--run-dir", str(tmp_path / "r_train"),
                     "--out", str(det)]) == 0
    out = capsys.readouterr().out
    assert det.is_file() and "laughter" in out and "disabled" in out         # sniff / throatclearing have no positives

    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(f"nvpa:\n  detector_path: {det.as_posix()}\n  shuffle_reps: 5\nbootstrap:\n  n_boot: 50\n", encoding="utf-8")
    run = tmp_path / "r_dev"
    assert cli.main(["run", "--manifest", str(dev_mp), "--run-dir", str(run), "--config", str(cfgp),
                     "--metrics", "nvpa", "asr"]) == 0                         # order given reversed: asr must still go first
    out = capsys.readouterr().out
    assert out.index("asr: computed") < out.index("nvpa: computed")
    assert "NVPA (micro, tol=1 word)" in out and "random-placement baseline" in out
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    n = s["automatic_metrics"]["nvpa"]
    assert n["n_events"] == 8 and 0.0 <= s["automatic_metrics"]["nvpa"]["value"] <= 1.0
    assert n["detector"]["thresholds"]["sniff"] > 1 and n["shuffle_baseline"]["n_reps"] == 5
    assert s["automatic_score"] is None and set(s["automatic_score_missing_components"]) == {"pMOS", "SS"}   # asr ran, so WER exists
    # second run hits the cache for both metrics
    assert cli.main(["run", "--manifest", str(dev_mp), "--run-dir", str(run), "--config", str(cfgp),
                     "--metrics", "asr", "nvpa"]) == 0
    assert capsys.readouterr().out.count("cache hit") == 2


def test_nvpa_without_detector_path_is_a_clear_error(tmp_path, monkeypatch):
    mp = tmp_path / "dev.jsonl"
    make_train_manifest(tmp_path, "dev", 2).save(mp)
    with pytest.raises(ValueError, match="detector_path"):
        cli.build_metrics(cli.load_config(None), ["nvpa"], ArtifactStore(tmp_path / "r"))


# ---- adapter: Track B through the CLI -------------------------------------------------
def test_adapter_cli_track_b_with_reference_map(tmp_path, capsys):
    root = tmp_path / "data"
    d = root / "dev" / "spk_0001"
    d.mkdir(parents=True)
    (d / "spk_0001.json").write_text(json.dumps([{"audio": "0001.flac", "text": "a [sniff] b", "language_id": "vi"}]), encoding="utf-8")
    (d / "0001.flac").write_bytes(b"")
    rm = tmp_path / "refs.json"
    rm.write_text(json.dumps({"spk_0001_0001": ["refs/x.wav"]}), encoding="utf-8")
    out = tmp_path / "m.jsonl"
    adapter_main(["--root", str(root), "--split", "dev", "--track", "B", "--reference-map", str(rm), "--out", str(out)])
    m = Manifest.load(out)
    assert m.header.track == "B" and m.samples[0].reference_audio == ["refs/x.wav"]
    with pytest.raises(ValueError, match="reference_map"):
        adapter_main(["--root", str(root), "--split", "dev", "--track", "B", "--out", str(out)])
    adapter_main(["--root", str(root), "--split", "dev", "--track", "A", "--no-references", "--out", str(out)])
    assert Manifest.load(out).samples[0].reference_audio == []
