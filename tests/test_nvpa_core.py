import math

import numpy as np
import pytest

from nvtts_eval.alignment import align, counts
from nvtts_eval.nvpa.detector import GapDetector, train_gap_detector
from nvtts_eval.nvpa.features import (FEATURE_NAMES, N_FEATURES, compute_frame_features, window_features)
from nvtts_eval.nvpa.matching import match_events
from nvtts_eval.nvpa.timing import HypWord, hyp_words_from_asr
from nvtts_eval.nvpa.windows import Window, build_gap_windows, norm_gap_map, ref_to_hyp_map


# ---- alignment -------------------------------------------------------------------
def test_align_pairs_and_counts():
    ref, hyp = "a b c d".split(), "a x c".split()
    pairs = align(ref, hyp)
    assert pairs == [(0, 0), (1, 1), (2, 2), (3, None)]
    assert counts(pairs, ref, hyp) == (1, 1, 0)
    assert align("a".split(), "a b".split()) == [(0, 0), (None, 1)]
    assert align([], []) == [] and align(["a"], []) == [(0, None)]


# ---- timing ----------------------------------------------------------------------
def test_hyp_words_group_tokens_and_normalise():
    rec = {"tokens": [" MI", "LA", "N", " NHƯNG", " 5%", " ,"], "timestamps": [0.0, 0.08, 0.2, 0.5, 0.9, 1.0]}
    w = hyp_words_from_asr(rec)
    assert [x.text for x in w] == ["milan", "nhưng", "5"]               # ',' alone vanishes, '%' stripped
    assert (w[0].start, w[0].last_start) == (0.0, 0.2) and (w[1].start, w[1].last_start) == (0.5, 0.5)
    rec2 = {"tokens": ["▁XIN", "▁CHÀO"], "timestamps": [0.1, 0.4]}
    assert [x.text for x in hyp_words_from_asr(rec2)] == ["xin", "chào"]


@pytest.mark.parametrize("rec", [{}, {"tokens": None, "timestamps": None}, {"tokens": [" A"], "timestamps": []},
                                 {"tokens": [" A", " B"], "timestamps": [0.1]}])
def test_hyp_words_none_without_usable_timestamps(rec):
    assert hyp_words_from_asr(rec) is None


# ---- gap windows (hand-computed) ---------------------------------------------------
def hw(starts, lasts=None):
    lasts = lasts or starts
    return [HypWord(f"w{i}", s, l) for i, (s, l) in enumerate(zip(starts, lasts))]


def test_norm_gap_map_handles_dropped_and_split_words():
    assert norm_gap_map(["a", "-", "đô-la", "b"]) == [0, 1, 1, 3, 4]


def test_gap_windows_basic_and_edges():
    hyp = hw([0.0, 0.4, 1.2, 1.6, 2.4])
    m = ref_to_hyp_map([(i, i) for i in range(5)])
    w = build_gap_windows(5, m, hyp, duration=3.0)
    assert (w[2].t0, w[2].t1, w[2].widened) == (0.4, 1.2, False)
    assert (w[4].t0, w[4].t1) == (1.6, 2.4)
    assert (w[0].t0, w[0].t1) == (0.0, 0.2)                 # before first word: [0, start] but at least min_len
    assert (w[5].t0, w[5].t1) == (2.4, 3.0)                 # after last word: up to the end of the audio
    assert len(w) == 6


def test_gap_windows_widen_over_unrecognised_words_and_unreliable_cases():
    hyp = hw([0.0, 1.0, 3.0])                               # ref words 0,2,4 recognised; 1 and 3 deleted by ASR
    m = {0: 0, 2: 1, 4: 2}
    w = build_gap_windows(5, m, hyp, duration=4.0, max_len=3.0)
    assert (w[2].t0, w[2].t1, w[2].widened) == (0.0, 1.0, True)   # gap 2: prev word 1 missing -> uses word 0
    assert w[3].widened and (w[3].t0, w[3].t1) == (1.0, 3.0)
    assert build_gap_windows(3, {}, [], 2.0) == [None] * 4        # nothing aligned
    long = build_gap_windows(2, {0: 0, 1: 1}, hw([0.0, 9.0]), duration=10.0, max_len=3.0)
    assert long[1] is None                                         # window longer than max_len is unusable


def test_gap_windows_min_length_near_end():
    hyp = hw([0.0, 0.9])
    w = build_gap_windows(2, {0: 0, 1: 1}, hyp, duration=1.0, min_len=0.2)
    assert w[2].t1 == 1.0 and w[2].t1 - w[2].t0 >= 0.2 - 1e-9


# ---- matching ---------------------------------------------------------------------
V = set(range(10))


def test_match_hit_within_tolerance_and_offset():
    res, sp = match_events([("breathing", 2)], {3: {"breathing"}}, {3: {"breathing": 0.9}}, V, tolerance=1)
    assert res[0].hit and res[0].matched_gap == 3 and res[0].offset_words == 1 and sp == []
    res0, sp0 = match_events([("breathing", 2)], {3: {"breathing"}}, {}, V, tolerance=0)
    assert not res0[0].hit and res0[0].reason == "wrong_position" and sp0 == [("breathing", 3)]


def test_match_reasons():
    gold = [("laughter", 5)]
    assert match_events(gold, {}, {}, V, 1)[0][0].reason == "missing"
    assert match_events(gold, {5: {"sniff"}}, {}, V, 1)[0][0].reason == "wrong_type"
    assert match_events(gold, {9: {"laughter"}}, {}, V, 1)[0][0].reason == "wrong_position"
    assert match_events(gold, {}, {}, V - {5}, 1)[0][0].reason == "alignment_unreliable"
    r = match_events(gold, {5: {"laughter"}}, {}, V - {5}, 1)[0][0]       # detected despite flagged window -> hit
    assert r.hit


def test_match_is_one_to_one_and_reports_spurious():
    res, sp = match_events([("breathing", 2), ("breathing", 3)], {2: {"breathing"}}, {}, V, 1)
    assert [r.hit for r in res] == [True, False] and res[1].reason == "missing" and sp == []
    res, sp = match_events([("breathing", 3)], {2: {"breathing"}, 4: {"breathing"}},
                           {2: {"breathing": 0.6}, 4: {"breathing": 0.9}}, V, 1)
    assert res[0].matched_gap == 4 and sp == [("breathing", 2)]           # tie in distance -> most confident
    res, sp = match_events([("breathing", 3)], {3: {"breathing"}, 7: {"sniff"}}, {}, V, 1)
    assert res[0].hit and sp == [("sniff", 7)]


def test_match_same_gap_different_types_both_hit():
    res, _ = match_events([("sniff", 4), ("breathing", 4)], {4: {"sniff", "breathing"}}, {}, V, 0)
    assert all(r.hit for r in res)


# ---- features -----------------------------------------------------------------------
def _sig(kind, n=16000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    if kind == "noise":
        return (0.1 * rng.standard_normal(n)).astype(np.float32)
    if kind == "tone":      # harmonic, voiced-like
        return sum(0.1 / k * np.sin(2 * np.pi * 150 * k * t) for k in range(1, 8)).astype(np.float32)
    return np.zeros(n, dtype=np.float32) + 1e-5 * rng.standard_normal(n).astype(np.float32)


def test_features_separate_noise_from_voiced_and_are_finite():
    fn, ft, fs = (compute_frame_features(_sig(k)) for k in ("noise", "tone", "silence"))
    wn, wt, ws = (window_features(f, 0.2, 0.8) for f in (fn, ft, fs))
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    assert wn.shape == (N_FEATURES,) and np.isfinite(wn).all() and np.isfinite(ws).all()
    assert wn[idx["flat_mean"]] > wt[idx["flat_mean"]]              # noise is spectrally flatter
    assert wt[idx["voic_mean"]] > wn[idx["voic_mean"]]              # harmonic tone is more periodic
    assert wn[idx["hf_mean"]] > wt[idx["hf_mean"]]
    assert wn[idx["e_mean"]] > ws[idx["e_mean"]] + 30               # loud vs near-silent, dB
    assert wn[idx["dur"]] == pytest.approx(0.6)


def test_features_short_windows_and_short_audio():
    f = compute_frame_features(_sig("noise", n=200))                # shorter than one frame
    assert f.n_frames == 1
    assert np.isfinite(window_features(f, 0.0, 0.01)).all()
    f2 = compute_frame_features(_sig("noise"))
    assert np.isfinite(window_features(f2, 0.5, 0.505)).all() and np.isfinite(window_features(f2, 0.9, 5.0)).all()
    with pytest.raises(ValueError):
        compute_frame_features(_sig("noise"), sr=8000)


# ---- detector ---------------------------------------------------------------------
def _toy_xy(n=900, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
    Y = np.stack([(X[:, 0] > 1.0), (X[:, 3] + X[:, 4] > 2.2), np.zeros(n, bool), (X[:, 7] < -1.5)], axis=1).astype(int)
    groups = [f"spk{i % 6}" for i in range(n)]
    return X, Y, groups


def test_train_save_load_and_predict(tmp_path, monkeypatch):
    X, Y, groups = _toy_xy()
    types = ("laughter", "breathing", "sniff", "throatclearing")
    det = train_gap_detector(X, Y, groups, types, max_iter=60)
    assert det.meta["laughter"]["oof_auc"] > 0.9 and det.meta["laughter"]["oof_recall"] > 0.5
    assert det.models["sniff"] is None and det.thresholds["sniff"] > 1 and "disabled" in det.meta["sniff"]["note"]
    p = det.predict_proba(X[:20])
    assert set(p) == set(types) and all(((v >= 0) & (v <= 1)).all() for v in p.values())
    assert (p["sniff"] == 0).all()
    path = tmp_path / "d.joblib"
    det.save(path)
    d2 = GapDetector.load(path)
    assert d2.thresholds == det.thresholds and np.allclose(d2.predict_proba(X[:20])["laughter"], p["laughter"])
    assert len(d2.describe()["model_sha256"]) == 64 and d2.describe()["train"]["_n_windows"] == len(X)
    import nvtts_eval.nvpa.detector as dm
    monkeypatch.setattr(dm, "FEATURE_VERSION", "999")
    with pytest.raises(ValueError, match="retrain"):
        GapDetector.load(path)


def test_predict_windows_handles_none_windows():
    X, Y, groups = _toy_xy()
    det = train_gap_detector(X, Y, groups, ("laughter", "breathing", "sniff", "throatclearing"), max_iter=30)
    out = det.predict_windows(_sig("noise"), 16000, [Window(0.1, 0.5, False), None, Window(0.6, 0.9, True)])
    assert out[1] is None and set(out[0]) == set(det.nv_types) and all(0 <= v <= 1 for v in out[2].values())
    assert det.predict_windows(_sig("noise"), 16000, [None, None]) == [None, None]


def test_load_warns_on_scikit_learn_version_mismatch(tmp_path):
    import joblib
    X, Y, groups = _toy_xy()
    det = train_gap_detector(X, Y, groups, ("laughter", "breathing", "sniff", "throatclearing"), max_iter=20)
    path = tmp_path / "d.joblib"
    det.save(path)
    d = joblib.load(path)
    d["sklearn"] = "0.20.0"                                   # pretend it was trained with another version
    joblib.dump(d, path)
    with pytest.warns(RuntimeWarning, match="scikit-learn 0.20.0"):
        GapDetector.load(path)
    import warnings
    det.save(path)                                           # same version: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        GapDetector.load(path)
