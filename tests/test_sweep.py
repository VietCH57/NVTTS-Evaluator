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
    by = {(r["scale"], int(r["tolerance"])): r for r in res["rows"]}
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
    assert [(r["tolerance"], r["nvpa"]) for r in data["rows"]] == [("0", 0.0), ("1", 1.0)]
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


# ---- asymmetric tolerance and threshold scale ----------------------------------------------
def test_asymmetric_tolerance_matching():
    from nvtts_eval.nvpa.matching import effective_tolerance, match_events, tolerance_bounds, tolerance_label
    late = ({6: {"breathing"}}, {6: {"breathing": 0.9}})
    early = ({4: {"breathing"}}, {4: {"breathing": 0.9}})
    V = set(range(20))
    gold = [("breathing", 5)]
    assert match_events(gold, *late, V, (0, 1))[0][0].hit                       # one gap LATE is accepted by 0:1
    r = match_events(gold, *early, V, (0, 1))[0][0]
    assert not r.hit and r.reason == "wrong_position"                           # one gap EARLY is not
    assert match_events(gold, *early, V, (1, 0))[0][0].hit and not match_events(gold, *late, V, (1, 0))[0][0].hit
    assert match_events(gold, *early, V, 1)[0][0].hit and match_events(gold, *late, V, 1)[0][0].hit      # symmetric int
    assert match_events(gold, *late, V, (0, 1))[0][0].offset_words == 1
    assert tolerance_bounds(2) == (2, 2) and tolerance_label((0, 1)) == "0:1" and tolerance_label((1, 1)) == "1"
    assert effective_tolerance({"tolerance_words": 1}) == (1, 1)
    assert effective_tolerance({"tolerance_words": 1, "tolerance_before": 0, "tolerance_after": None}) == (0, 1)
    assert effective_tolerance({"tolerance_words": 2, "tolerance_after": 3}) == (2, 3)
    with pytest.raises(ValueError):
        tolerance_bounds((-1, 1))


def test_sweep_with_asymmetric_pairs_and_cli_parsing(tmp_path, capsys):
    # detections one gap LATE for both events (gap 6 vs 5, gap 13 vs 12): 0:1 accepts them, 1:0 does not
    res = sweep([rec()], THR, tolerances=(0, (0, 1), (1, 0), 1), scales=(1.0,), reps=5, seed=0)
    by = {r["tolerance"]: r for r in res["rows"]}
    assert by["0"]["nvpa"] == 0.0 and by["0:1"]["nvpa"] == 1.0 and by["1:0"]["nvpa"] == 0.0 and by["1"]["nvpa"] == 1.0
    assert by["0:1"]["shuffle"] <= by["1"]["shuffle"]                           # narrower window: chance can only drop
    store = ArtifactStore(tmp_path / "run")
    store.write("nvpa", [rec()], dict(metric_version="1", input_hash="h", config_hash="c",
                                      config={"params": {"tolerance_words": 1}, "detector": {"thresholds": THR}}))
    assert cli.main(["nvpa-sweep", "--run-dir", str(tmp_path / "run"), "--tolerances", "0", "0:1", "--scales", "1.0",
                     "--reps", "3"]) == 0
    out = capsys.readouterr().out
    assert " 0:1 " in out or "0:1" in out


def test_nvpa_metric_threshold_scale_and_asymmetric_tolerance(make_manifest, tmp_path):
    from nvtts_eval.core import run_metric
    from nvtts_eval.nvpa.metric import NvpaMetric, NvpaParams
    text = "một hai [breathing] ba bốn [laughter] năm"
    asr = {"sample_id": "s1", "tokens": [" MỘT", " HAI", " BA", " BỐN", " NĂM"], "timestamps": [0.0, 0.4, 1.2, 1.6, 2.4],
           "text": "x", "duration": 3.0}
    m = make_manifest([("s1", "spk_a", text, 3, [1])])

    class Det:
        nv_types = TYPES
        thresholds = {"laughter": 0.5, "breathing": 0.5, "sniff": 1.01, "throatclearing": 1.01}

        def describe(self):
            return {"backend": "fake"}

        def predict_windows(self, wav, sr, windows):      # breathing p=0.4 at gap 3 (window starting 1.2): 1 gap LATE
            return [None if w is None else {t: (0.4 if (t == "breathing" and round(w.t0, 3) == 1.2) else 0.0) for t in TYPES}
                    for w in windows]

    def go(**kw):
        store = ArtifactStore(tmp_path / f"run_{len(kw)}_{sorted(kw.items())}".replace(" ", ""))
        store.write("asr", [asr], dict(metric_version="1", input_hash="h", config_hash="c", config={}))
        return run_metric(NvpaMetric(store, Det(), params=NvpaParams(**kw)), m, store).records[0]
    r = go()                                                        # scale 1: 0.4 < 0.5 -> nothing detected
    assert r["pred"] == [] and [e["hit"] for e in r["events"]] == [False, False]
    r = go(threshold_scale=0.75)                                    # 0.5 * 0.75 = 0.375 <= 0.4 -> breathing detected at gap 3
    assert r["pred"] == [[3, ["breathing"]]] and r["events"][0]["hit"] and r["events"][0]["offset_words"] == 1
    r = go(threshold_scale=0.75, tolerance_before=1, tolerance_after=0)          # late by one is not allowed now
    assert not r["events"][0]["hit"] and r["events"][0]["reason"] == "wrong_position"
    r = go(threshold_scale=0.75, tolerance_before=0, tolerance_after=1)
    assert r["events"][0]["hit"]


def test_config_asymmetric_and_scale_validation():
    c = config_from_dict({"nvpa": {"tolerance_before": 0, "tolerance_after": 1, "threshold_scale": 0.75}})
    assert (c.nvpa.tolerance_before, c.nvpa.tolerance_after, c.nvpa.threshold_scale) == (0, 1, 0.75)
    for bad in ({"tolerance_before": -1}, {"threshold_scale": 0}, {"threshold_scale": -1}):
        with pytest.raises(ValueError):
            config_from_dict({"nvpa": bad})


def test_summary_tolerance_check_and_labels(tmp_path):
    parser = NVParser()
    samples = [Sample.from_text(f"s{i}", "spk_a", "a [breathing] b c [laughter] d", parser) for i in (1, 2)]
    m = Manifest(ManifestHeader(track="A", source="model", split="dev"), samples)
    store = ArtifactStore(tmp_path / "run")
    recs = []
    for s in samples:
        r = rec(s.sample_id, events=(("breathing", 1), ("laughter", 3)), n_words=4,
                probs=[[2, P(breathing=0.9)], [3, P(laughter=0.9)]])          # breathing late by one, laughter exact
        r["pred"] = [[2, ["breathing"]], [3, ["laughter"]]]
        for e in r["events"]:                                                    # what the 0:1 run would have stored
            e["hit"], e["reason"] = True, "ok"
        recs.append(r)
    store.write("nvpa", recs, dict(metric_version="1", input_hash="h", config_hash="c",
                                   config={"params": {"tolerance_words": 1, "tolerance_before": 0, "tolerance_after": 1},
                                           "detector": {"thresholds": THR}}))
    from nvtts_eval.report.summary import format_summary
    s = build_summary(m, store, config_from_dict({"bootstrap": {"n_boot": 20}, "nvpa": {"shuffle_reps": 3}}))
    n = s["automatic_metrics"]["nvpa"]
    assert (n["tolerance"], n["tolerance_before"], n["tolerance_after"]) == ("0:1", 0, 1) and n["value"] == 1.0
    checks = {c["tolerance"]: c["nvpa"] for c in n["tolerance_check"]}
    assert checks == {"0": 0.5, "0:1": 1.0, "2": 1.0}
    assert "tol=-0/+1 word" in format_summary(s) and "tolerance check" in format_summary(s)
