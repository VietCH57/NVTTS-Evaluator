import math

import numpy as np
import pytest
import soundfile as sf

from nvtts_eval.audio import load_mono, resample
from nvtts_eval.metrics.wer import (TextNormConfig, edit_counts, normalize_text, summarize_wer,
                                    wer_records)
from nvtts_eval.stats import bootstrap_ratio_ci


# ---- audio ---------------------------------------------------------------------------
def test_load_mono_resamples_and_keeps_duration(tmp_path):
    t = np.arange(24000) / 24000
    sf.write(str(tmp_path / "a.flac"), (0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32"), 24000)
    x, sr = load_mono(tmp_path / "a.flac", 16000)
    assert sr == 16000 and x.dtype == np.float32 and x.ndim == 1 and len(x) == 16000
    spec = np.abs(np.fft.rfft(x))
    assert abs(np.argmax(spec) * 16000 / len(x) - 440) < 2          # tone frequency preserved


def test_load_mono_averages_channels_and_no_resample_when_same_rate(tmp_path):
    stereo = np.stack([np.full(1600, 0.2), np.full(1600, 0.4)], axis=1).astype("float32")
    sf.write(str(tmp_path / "s.wav"), stereo, 16000)
    x, sr = load_mono(tmp_path / "s.wav", 16000)
    assert sr == 16000 and len(x) == 1600 and np.allclose(x, 0.3, atol=1e-3)
    assert resample(x, 16000, 16000) is x


def test_load_mono_rejects_empty(tmp_path):
    sf.write(str(tmp_path / "e.wav"), np.zeros(0, dtype="float32"), 16000)
    with pytest.raises(ValueError):
        load_mono(tmp_path / "e.wav")


# ---- normalisation -------------------------------------------------------------------
def test_normalize_text():
    assert normalize_text("Chứ, Đã xong! (ok) — 50%.") == "chứ đã xong ok 50"
    assert normalize_text("A\u0301") == normalize_text("\u00c1") == "á"      # NFC
    assert normalize_text("Hôm NAY", TextNormConfig(lowercase=False)) == "Hôm NAY"
    assert normalize_text("a, b", TextNormConfig(strip_punct=False)) == "a, b"
    assert normalize_text("  a   b ") == "a b"


# ---- edit distance (hand-computed) ---------------------------------------------------
@pytest.mark.parametrize("ref,hyp,expected", [
    ("a b c d", "a b c d", (0, 0, 0)),
    ("a b c d", "a x c", (1, 1, 0)),            # b->x substitution, d deleted
    ("a b", "a b c d", (0, 0, 2)),
    ("a b c", "", (0, 3, 0)),
    ("", "a b", (0, 0, 2)),
    ("a b c", "c b a", (2, 0, 0)),
    ("", "", (0, 0, 0)),
])
def test_edit_counts(ref, hyp, expected):
    assert edit_counts(ref.split(), hyp.split()) == expected


# ---- WER records & summary -----------------------------------------------------------
def asr(sid, text):
    return {"sample_id": sid, "text": text}


def test_wer_ignores_nv_tags_in_reference(make_manifest):
    m = make_manifest([("s1", "spk_a", "xin chào [laughter] các bạn", 1, [1])])
    r = wer_records(m, [asr("s1", "xin chào các bạn")])[0]
    assert r["errors"] == 0 and r["ref"] == "xin chào các bạn" and r["wer"] == 0


def test_wer_records_and_corpus_summary_hand_computed(make_manifest):
    m = make_manifest([
        ("s1", "spk_a", "một hai [breathing] ba bốn", 1, [1]),     # exact
        ("s2", "spk_a", "một hai ba bốn", 1, [1]),                 # 1 substitution
        ("s3", "spk_b", "năm sáu", 1, [1]),                        # ASR failure
        ("s4", "spk_b", "bảy tám chín", 1, [1]),                   # no ASR record at all
    ])
    recs = wer_records(m, [asr("s1", "Một hai ba bốn."), asr("s2", "một hai ba năm"),
                           {"sample_id": "s3", "error": "boom"}])
    assert [("error" in r) for r in recs] == [False, False, True, True]
    assert recs[3]["error"] == "missing_asr_record"
    s = summarize_wer(recs, n_boot=200)
    assert s["n_samples"] == 2 and s["n_failed"] == 2 and s["total_ref_words"] == 8
    assert math.isclose(s["corpus_wer"], 1 / 8) and math.isclose(s["value"], 1 / 8)
    assert math.isclose(s["mean_sample_wer"], (0 + 0.25) / 2)
    assert (s["sub"], s["del"], s["ins"]) == (1, 0, 0)
    assert s["by_speaker"]["spk_a"]["n"] == 2
    assert summarize_wer(recs, aggregation="sample_mean", n_boot=50)["value"] == s["mean_sample_wer"]


def test_wer_can_exceed_one_and_flags_digits(make_manifest):
    m = make_manifest([("s1", "spk_a", "một", 1, [1])])
    r = wer_records(m, [asr("s1", "1 2 3 4")])[0]
    assert r["wer"] == 4.0 and r["hyp_has_digits"] is True
    assert summarize_wer([r], n_boot=10)["n_hyp_with_digits"] == 1


def test_bootstrap_ratio_ci_brackets_point_estimate():
    errs = [0, 1, 0, 2, 1, 0, 0, 3, 1, 0] * 5
    refs = [10] * 50
    pt = sum(errs) / sum(refs)
    lo, hi = bootstrap_ratio_ci(errs, refs, seed=3)
    assert lo < pt < hi
    clu = bootstrap_ratio_ci(errs, refs, clusters=[i % 5 for i in range(50)], seed=3)
    assert clu[0] <= pt <= clu[1]
    assert all(math.isnan(x) for x in bootstrap_ratio_ci([1], [10]))
