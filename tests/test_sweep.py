import json
import math

import pytest

import nvtts_eval.cli as cli
from nvtts_eval.config import config_from_dict
from nvtts_eval.core import ArtifactStore
from nvtts_eval.data import Manifest, ManifestHeader, NVParser, Sample
from nvtts_eval.nvpa.sweep import evaluate, format_sweep, offset_histogram, rethreshold, sweep
from nvtts_eval.report.summary import build_summary
from nvtts_eval.stats import spearman

TYPES = ("laughter", "breathing", "sniff", "throatclearing")
THR = {"laughter": 0.5, "breathing": 0.5, "sniff": 1.01, "throatclearing": 1.01}


def P(**kw):
    return {t: kw.get(t, 0.0) for t in TYPES}


def rec(sample_id="a", events=(("breathing", 5), ("laughter", 12)), probs=None, n_words=20):
    probs = probs if probs is not None else [[5, P(breathing=0.2)], [6, P(breathing=0.9)], [13, P(laughter=0.7)]]
    ev = [{"type": t, "gap": g, "hit": False, "reason": "missing", "matched_gap": None, "offset_words": None}
          for t, g in events]
    return {"sample_id": sample_id, "n_ref_words": n_words, "n_gaps": n_words + 1, "no_timing": False,
            "events": ev, "spurious": [], "valid_gaps": list(range(n_words + 1)), "pred": [], "probs": probs}


def test_rethreshold_scale_and_disabled_types():
    r = rec(probs=[[3, P(sniff=0.9)], [5, P(breathing=0.2)], [6, P(breathing=0.9)]])
    base = rethreshold([r], THR, 1.0)[0]["pred"]
    assert base == [[6, ["breathing"]]]
    low = rethreshold([r], THR, 0.3)[0]["pred"]                       # breathing threshold 0.15
    assert low == [[5, ["breathing"]], [6, ["breathing"]]]            # disabled sniff (thr 1.01) never fires
    assert rethreshold([r], THR, 0.01)[0]["pred"][0] != [3, ["sniff"]]
    assert rethreshold([{"sample_id": "x", "error": "boom"}], THR) == []


def test_evaluate_tolerance_hand_computed():
    r1 = rethreshold([rec()], THR, 1.0)
    e0, e1 = evaluate(r1, 0), evaluate(r1, 1)
    assert e0["nvpa"] == 0.0 and e0["reasons"] == {"wrong_position": 2}
    assert e1["nvpa"] == 1.0 and e1["per_type"] == {"breathing": 1.0, "laughter": 1.0}
    assert e1["spurious_per_100_words"] == 0.0
    low = rethreshold([rec()], THR, 0.3)                              # also detects breathing at gap 5
    e = evaluate(low, 0)
    assert e["nvpa"] == 0.5 and e["per_type"] == {"breathing": 1.0, "laughter": 0.0}
    assert e["spurious_per_100_words"] == pytest.approx(10.0)         # (breathing,6) and (laughter,13): 2 per 20 words


def test_offset_histogram():
    r = rec(events=(("breathing", 5), ("laughter", 12), ("sniff", 3), ("breathing", 17)))
    h = offset_histogram(rethreshold([r], THR, 1.0))
    assert h["all"] == {1: 2, "none": 2}                              # breathing +1, laughter +1, sniff none, 17 none (6 is 11 away)
    assert h["by_type"]["breathing"] == {1: 1, "none": 1}
    far = offset_histogram(rethreshold([rec()], THR, 1.0), max_offset=0)
    assert far["all"] == {"none": 2}


def test_sweep_grid_monotonic_and_lift():
    recs = [rec("a"), rec("b", events=(("breathing", 6), ("laughter", 13)))]
    res = sweep(recs, THR, tolerances=(0, 1, 2), scales=(0.3, 1.0), reps=10, seed=1)
    assert len(res["rows"]) == 6
    by = {(r["scale"], r["tolerance"]): r for r in res["rows"]}
    assert by[(1.0, 0)]["nvpa"] == 0.5 and by[(1.0, 1)]["nvpa"] == 1.0           # record b is exact, a is off by one
    for scale in (0.3, 1.0):
        nv = [by[(scale, t)]["nvpa"] for t in (0, 1, 2)]
        sh = [by[(scale, t)]["shuffle"] for t in (0, 1, 2)]
        assert nv == sorted(nv) and sh == sorted(sh)                              # looser tolerance never lowers either
    assert by[(1.0, 1)]["lift"] == pytest.approx(by[(1.0, 1)]["nvpa"] - by[(1.0, 1)]["shuffle"])
    assert by[(0.3, 0)]["spurious_per_100_words"] > by[(1.0, 0)]["spurious_per_100_words"]
    assert "tol" in format_sweep(res) and "Signed offset" in format_sweep(res)


def test_sweep_cli_reads_artifact_only(tmp_path, capsys):
    store = ArtifactStore(tmp_path / "run")
    store.write("nvpa", [rec("a"), {"sample_id": "bad", "error": "x"}],
                dict(metric_version="1", input_hash="h", config_hash="c",
                     config={"params": {"tolerance_words": 1}, "detector": {"thresholds": THR}}))
    assert cli.main(["nvpa-sweep", "--run-dir", str(tmp_path / "run"), "--tolerances", "0", "1", "--scales", "1.0",
                     "--reps", "3"]) == 0
    out = capsys.readouterr().out
    assert "Signed offset" in out and "+1:2" in out
    data = json.loads((tmp_path / "run" / "nvpa_sweep.json").read_text(encoding="utf-8"))
    assert [(r["tolerance"], r["nvpa"]) for r in data["rows"]] == [(0, 0.0), (1, 1.0)]
    with pytest.raises(SystemExit, match="nvpa"):
        cli.main(["nvpa-sweep", "--run-dir", str(tmp_path / "empty")])


# ---- Spearman ------------------------------------------------------------------------
def test_spearman_values():
    assert spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) == pytest.approx(8 / math.sqrt(10 * 9.5))   # ties: average ranks
    assert math.isnan(spearman([1, 2], [1, 2])) and math.isnan(spearman([1, 1, 1], [1, 2, 3]))


# ---- NVPA detector validated against human NV_placement in the summary ---------------------
def test_summary_reports_nvpa_vs_human_placement(tmp_path):
    parser = NVParser()
    samples = [Sample.from_text(f"s{i}", f"spk_{i % 2}", "a [breathing] b c [laughter] d", parser) for i in range(1, 7)]
    m = Manifest(ManifestHeader(track="A", source="model", split="dev"), samples)
    store = ArtifactStore(tmp_path / "run")
    hits = [2, 2, 1, 1, 0, 0]                                         # per-sample hit counts out of 2 events
    recs = []
    for s, h in zip(samples, hits):
        r = rec(s.sample_id, events=(("breathing", 1), ("laughter", 3)), n_words=4)
        for k, e in enumerate(r["events"]):
            e["hit"], e["reason"] = k < h, "ok" if k < h else "missing"
        recs.append(r)
    store.write("nvpa", recs, dict(metric_version="1", input_hash="h", config_hash="c",
                                   config={"params": {"tolerance_words": 1}, "detector": {"thresholds": THR}}))
    place = [5, 4, 3, 3, 2, 1]
    human = {"sample_ids": [s.sample_id for s in samples], "n_rated_samples": 6, "n_raters_accepted": 1, "raters": {},
             "agreement": None, "metrics": {"SN": {"mean": 4.0}, "Q": {"mean": 4.0}},
             "records": [{"sample_id": s.sample_id, "NV_placement": p} for s, p in zip(samples, place)]}
    s = build_summary(m, store, config_from_dict({"bootstrap": {"n_boot": 20}, "nvpa": {"shuffle_reps": 2}}), human=human)
    c = s["human_metrics"]["nvpa_vs_human_placement"]
    assert c["n"] == 6 and c["spearman"] == pytest.approx(16 / math.sqrt(16 * 17))      # hand-computed 0.9701
    few = build_summary(m, store, config_from_dict({"bootstrap": {"n_boot": 20}, "nvpa": {"shuffle_reps": 2}}),
                        human={**human, "records": human["records"][:3]})
    assert few["human_metrics"]["nvpa_vs_human_placement"] is None
