import csv
import json
import math

import numpy as np
import pytest
import soundfile as sf

from nvtts_eval.config import BootstrapConfig, HumanConfig, config_from_dict
from nvtts_eval.core import ArtifactStore
from nvtts_eval.data import Manifest, ManifestHeader, NVParser, Sample
from nvtts_eval.human.package import SHEET_COLUMNS, degrade_snr, export_package
from nvtts_eval.human.scores import aggregate_ratings, read_rating_sheet
from nvtts_eval.human.subset import _quotas, select_subset
from nvtts_eval.report.summary import build_summary, format_summary

META = dict(metric_version="1", input_hash="h", config_hash="c", config={})


def big_manifest(tmp_path, n_head=40, n_tail=30):
    """Head speaker spk_h (n_head utts), many tail speakers; a few rare NV (sniff / throatclearing)."""
    gen = tmp_path / "gen"
    gen.mkdir(exist_ok=True)
    p = NVParser()
    samples = []
    def add(i, spk, text, secs):
        sid = f"{spk}_{i:04d}"
        sf.write(str(gen / f"{sid}.wav"), np.full(int(16000 * secs), 0.1, dtype="float32"), 16000)
        samples.append(Sample.from_text(sid, spk, text, p, generated_audio=f"{sid}.wav",
                                        ground_truth_audio=f"gen/{sid}.wav"))
    for i in range(n_head):
        add(i, "spk_h", "a [breathing] b c [laughter] d", 5 if i % 2 else 20)
    for i in range(n_tail):
        add(i, f"spk_t{i % 10}", "a b [breathing] c", 5 if i % 3 else 20)
    add(900, "spk_t1", "a [sniff] b", 5)
    add(901, "spk_t2", "a [throatclearing] b [sniff] c", 5)
    hdr = ManifestHeader(track="A", source="ground_truth", split="dev", audio_root=tmp_path.as_posix(),
                         generated_root=gen.as_posix())
    return Manifest(hdr, samples)


# ---- subset --------------------------------------------------------------------------
def test_quotas_sum_and_caps():
    q = _quotas({"a": 100, "b": 4, "c": 1}, 20)
    assert sum(q.values()) == 20 and q["c"] <= 1 and q["b"] <= 4 and q["a"] > q["b"]
    assert _quotas({"a": 3, "b": 2}, 50) == {"a": 3, "b": 2}
    assert sum(_quotas({"a": 10, "b": 10, "c": 10}, 7).values()) == 7


def test_subset_deterministic_covers_rare_types_and_tail_speakers(tmp_path):
    m = big_manifest(tmp_path)
    s1 = select_subset(m, 20, seed=3)
    s2 = select_subset(m, 20, seed=3)
    assert s1 == s2 and len(s1["sample_ids"]) == 20 == len(set(s1["sample_ids"]))
    assert select_subset(m, 20, seed=4)["sample_ids"] != s1["sample_ids"]
    assert set(s1["rare_types"]) == {"sniff", "throatclearing"}
    chosen = {s.sample_id: s for s in m.samples if s.sample_id in set(s1["sample_ids"])}
    types = {e["type"] for s in chosen.values() for e in s.nv_events}
    assert {"sniff", "throatclearing"} <= types                          # rare types are always included
    spk = {s.speaker_id for s in chosen.values()}
    assert len(spk) >= 6                                                   # tail speakers are visible
    assert s1["composition"]["head_speaker_samples"] < 20                  # head speaker does not take everything
    assert select_subset(m, 10_000)["composition"]["n"] == len(m)          # size >= manifest -> everything


# ---- package -------------------------------------------------------------------------
def test_degrade_snr_is_deterministic_and_noisier():
    x = np.full(16000, 0.1, dtype=np.float32)
    a = degrade_snr(x, 5.0, np.random.default_rng(0))
    b = degrade_snr(x, 5.0, np.random.default_rng(0))
    assert np.array_equal(a, b) and not np.array_equal(a, x)
    noise_p = float(np.mean((a - x) ** 2))
    assert 0.5 < (0.1 ** 2 / noise_p) / (10 ** 0.5) < 2.0              # SNR ~ 5 dB (clipping aside)


def test_export_package(tmp_path):
    m = big_manifest(tmp_path)
    ids = select_subset(m, 12, seed=0)["sample_ids"]
    out = tmp_path / "pkg"
    key = export_package(m, ids, out, n_gt_anchors=2, n_degraded_anchors=2, snr_db=5, seed=0)
    kinds = [i["kind"] for i in key["items"]]
    assert kinds.count("model") == 12 and kinds.count("anchor_gt") == 2 and kinds.count("anchor_degraded") == 2
    assert len(set(i["item_id"] for i in key["items"])) == 16
    assert kinds[:12] != ["model"] * 12                                    # shuffled: anchors are mixed in
    wavs = sorted((out / "for_raters" / "audio").glob("*.wav"))
    assert len(wavs) == 16 and all(w.name.startswith("item_") for w in wavs)
    rows = list(csv.DictReader(open(out / "for_raters" / "rating_sheet.csv", encoding="utf-8-sig")))
    assert list(rows[0].keys()) == SHEET_COLUMNS and len(rows) == 16 and rows[0]["SN"] == ""
    assert (out / "for_raters" / "README.md").is_file() and (out / "PRIVATE_key.json").is_file()
    assert not any("PRIVATE" in p.name or "sample" in p.name for p in (out / "for_raters").rglob("*"))
    anchor_samples = {i["sample_id"] for i in key["items"] if i["kind"] != "model"}
    assert anchor_samples.isdisjoint(ids)                                  # anchors use samples outside the subset
    deg = next(i for i in key["items"] if i["kind"] == "anchor_degraded")
    gt = next(i for i in key["items"] if i["kind"] == "anchor_gt")
    wd, _ = sf.read(out / "for_raters" / "audio" / f"{deg['item_id']}.wav")
    wg, _ = sf.read(out / "for_raters" / "audio" / f"{gt['item_id']}.wav")
    assert np.std(wd) > np.std(wg)                                          # degraded one has added noise


# ---- import & aggregation (hand-computed) --------------------------------------------
def write_sheet(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({**{c: "" for c in SHEET_COLUMNS}, **r})


KEY = {"items": [
    {"item_id": "item_1", "kind": "model", "sample_id": "s1", "speaker_id": "spk_a"},
    {"item_id": "item_2", "kind": "model", "sample_id": "s2", "speaker_id": "spk_b"},
    {"item_id": "item_3", "kind": "anchor_gt", "sample_id": "g1", "speaker_id": "spk_a"},
    {"item_id": "item_4", "kind": "anchor_degraded", "sample_id": "g2", "speaker_id": "spk_a"},
]}


def sheet(tmp_path, name, vals):
    p = tmp_path / f"{name}.csv"
    write_sheet(p, [{"item_id": k, **v} for k, v in vals.items()])
    return read_rating_sheet(p)


def test_read_rating_sheet_validates(tmp_path):
    p = tmp_path / "r.csv"
    write_sheet(p, [{"item_id": "item_1", "SN": "4,5", "Q": "6"}, {"item_id": "item_2", "SN": "abc", "Q": " 3 "},
                    {"item_id": "", "SN": "1"}])
    rows, problems = read_rating_sheet(p)
    assert rows[0]["SN"] == 4.5 and rows[0]["Q"] is None and rows[1]["SN"] is None and rows[1]["Q"] == 3.0
    assert len(rows) == 2 and len(problems) == 2


def test_aggregate_with_anchor_check(tmp_path):
    good = {"item_1": dict(SN="4", Q="5", NV_naturalness="4"), "item_2": dict(SN="3", Q="3"),
            "item_3": dict(SN="5", Q="5"), "item_4": dict(SN="2", Q="2")}
    lazy = {"item_1": dict(SN="1", Q="1"), "item_2": dict(SN="1", Q="1"),
            "item_3": dict(SN="3.2", Q="3.2"), "item_4": dict(SN="3", Q="3")}      # GT - degraded = 0.2 < 0.5
    sheets, problems = {}, []
    for n, v in (("alice", good), ("bob", lazy)):
        rows, pr = sheet(tmp_path, n, v)
        sheets[n] = rows
        problems += pr
    agg = aggregate_ratings(sheets, KEY, HumanConfig(), BootstrapConfig(n_boot=100))
    assert agg["raters"]["bob"]["flagged"] and not agg["raters"]["alice"]["flagged"]
    assert agg["n_raters_accepted"] == 1 and agg["n_rated_samples"] == 2
    assert math.isclose(agg["metrics"]["SN"]["mean"], (4 + 3) / 2) and math.isclose(agg["metrics"]["Q"]["mean"], (5 + 3) / 2)
    assert agg["metrics"]["NV_naturalness"]["mean"] == 4 and agg["metrics"]["NV_placement"] is None
    assert math.isclose(agg["raters"]["alice"]["anchor_gt_mean"], 5.0) and math.isclose(agg["raters"]["alice"]["anchor_degraded_mean"], 2.0)
    keep = aggregate_ratings(sheets, KEY, HumanConfig(exclude_flagged=False), BootstrapConfig(n_boot=100))
    assert keep["n_raters_accepted"] == 2 and math.isclose(keep["metrics"]["SN"]["mean"], (4 + 1 + 3 + 1) / 4)
    assert agg["sample_ids"] == ["s1", "s2"] and agg["problems"] == []


def test_aggregate_reports_unknown_items_and_agreement(tmp_path):
    key = {"items": [{"item_id": f"item_{i}", "kind": "model", "sample_id": f"s{i}", "speaker_id": "spk_a"} for i in range(8)]}
    a = {f"item_{i}": dict(SN=str(1 + i % 5), Q="4") for i in range(8)}
    b = {f"item_{i}": dict(SN=str(1 + i % 5), Q="3") for i in range(8)}
    sheets = {"a": sheet(tmp_path, "a", a)[0], "b": sheet(tmp_path, "b", b)[0],
              "c": sheet(tmp_path, "c", {"item_999": dict(SN="3", Q="3")})[0]}
    agg = aggregate_ratings(sheets, key, HumanConfig(), BootstrapConfig(n_boot=50))
    assert math.isclose(agg["agreement"]["SN"]["mean_pairwise_pearson"], 1.0)
    assert agg["agreement"]["SN"]["n_pairs"] == 1 and agg["agreement"]["Q"]["mean_pairwise_pearson"] is None   # Q constant
    assert any("unknown item_id" in p for p in agg["problems"])
    assert aggregate_ratings({}, key, HumanConfig(), BootstrapConfig())["n_rated_samples"] == 0


# ---- final score on the rated subset (hand-computed) ----------------------------------
def test_final_score_uses_subset_automatic_metrics(tmp_path):
    from nvtts_eval.data import Manifest as M
    p = NVParser()
    samples = [Sample.from_text(f"s{i}", "spk_a" if i < 2 else "spk_b", "một hai ba bốn", p) for i in range(1, 5)]
    m = M(ManifestHeader(track="A", source="model", split="dev"), samples)
    store = ArtifactStore(tmp_path / "run")
    # ASR: s1 exact (0/4), s2 exact, s3 4 errors, s4 4 errors  -> full-set WER = 8/16 = 0.5 ; subset {s1,s2} WER = 0
    store.write("asr", [{"sample_id": "s1", "text": "một hai ba bốn"}, {"sample_id": "s2", "text": "một hai ba bốn"},
                        {"sample_id": "s3", "text": "x y z w"}, {"sample_id": "s4", "text": "x y z w"}], META)
    store.write("pmos", [{"sample_id": f"s{i}", "ovrl": 3.0 if i < 3 else 1.0} for i in range(1, 5)], META)
    store.write("ss", [{"sample_id": f"s{i}", "cosine": 0.8 if i < 3 else 0.0} for i in range(1, 5)], META)
    cfg = config_from_dict({"bootstrap": {"n_boot": 50}})
    human = {"sample_ids": ["s1", "s2"], "n_rated_samples": 2, "n_raters_accepted": 1, "raters": {"a": {"flagged": False}},
             "agreement": None, "metrics": {"SN": {"mean": 4.0, "ci_utterance": [3, 5]}, "Q": {"mean": 3.0, "ci_utterance": [2, 4]},
                                            "NV_naturalness": None, "NV_placement": None}}
    s = build_summary(m, store, cfg, nvpa=0.5, human=human)
    # full set: WER .5, pMOS 2.0, SS .4  -> AutoScore = .15 + .15*.5 + .15*.25 + .1*.4 = .3025
    assert math.isclose(s["automatic_score"]["value"], 0.3025)
    # subset {s1,s2}: WER 0, pMOS 3.0, SS .8, SN 4 -> .75, Q 3 -> .5
    # final = .3*.5 + .15*.75 + .15*.5 + .15*1 + .15*.5 + .1*.8 = .15 + .1125 + .075 + .15 + .075 + .08 = .6425
    assert math.isclose(s["final_score"]["value"], 0.6425) and s["final_score"]["official"] is True
    assert s["human_metrics"]["subset_size"] == 2 and s["human_metrics"]["auto_values_used_for_final"]["WER"] == 0
    full = build_summary(m, store, config_from_dict({"bootstrap": {"n_boot": 50}, "human": {"auto_on_subset": False}}),
                         nvpa=0.5, human=human)
    # automatic parts on the FULL set: .3*.5 + .15*.75 + .15*.5 + .15*.5 + .15*.25 + .1*.4 = .15+.1125+.075+.075+.0375+.04
    assert math.isclose(full["final_score"]["value"], 0.49) and "Final score" in format_summary(s)
    assert build_summary(m, store, cfg, nvpa=0.5)["final_score"] is None
    nope = build_summary(m, store, cfg, human={**human, "metrics": {"SN": None, "Q": None}}, nvpa=0.5)
    assert nope["final_score"] is None
