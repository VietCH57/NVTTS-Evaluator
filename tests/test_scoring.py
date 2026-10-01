import math

import pytest

from nvtts_eval.scoring import (WEIGHTS, ScoringConfig, auto_score, final_score,
                                normalize_mos, normalize_ss, one_minus_wer)

# Hand-computed (see spec v2, section 6): NVPA .713, WER .043, pMOS 4.12, SS .871
KW = dict(nvpa=0.713, wer=0.043, pmos=4.12, ss=0.871)


def test_official_weights_sum_to_one():
    for t in "AB":
        assert math.isclose(sum(WEIGHTS[t].values()), 1.0)


def test_auto_score_track_a_hand_computed():
    r = auto_score("A", **KW)
    assert math.isclose(r.value, 0.56155, abs_tol=1e-9)
    assert math.isclose(r.max_possible, 0.70)
    assert math.isclose(r.renormalized, 0.56155 / 0.70)
    assert round(r.renormalized, 3) == 0.802
    assert r.official is False


def test_auto_score_track_b_hand_computed():
    r = auto_score("B", **KW)
    assert math.isclose(r.value, 0.5618, abs_tol=1e-9)
    assert math.isclose(r.max_possible, 0.70)


def test_final_score_hand_computed():
    a = final_score("A", **KW, sn=4.10, q=4.25)
    b = final_score("B", **KW, sn=4.10, q=4.25)
    assert math.isclose(a.value, 0.799675, abs_tol=1e-9)
    assert math.isclose(b.value, 0.799925, abs_tol=1e-9)
    assert a.official and a.max_possible == 1.0 and a.renormalized == a.value


def test_final_score_is_na_without_human_metrics():
    assert final_score("A", **KW, sn=None, q=4.0) is None
    assert final_score("A", **KW, sn=4.0, q=None) is None
    assert final_score("B", **KW, sn=None, q=None) is None


def test_final_score_has_no_default_for_human_metrics():
    with pytest.raises(TypeError):
        final_score("A", **KW)          # sn / q must be passed explicitly (None allowed)


def test_perfect_and_worst():
    assert math.isclose(final_score("A", 1, 0, 5, 1, 5, 5).value, 1.0)
    assert math.isclose(auto_score("A", 1, 0, 5, 1).value, 0.70)
    assert math.isclose(auto_score("B", 0, 5, 1, 0).value, 0.0)   # WER 5 clipped -> 1-WER = 0
    assert math.isclose(final_score("B", 0, 5, 1, 0, 1, 1).value, 0.0)


def test_wer_above_one_is_clipped_but_configurable():
    assert one_minus_wer(1.7) == 0.0
    assert math.isclose(one_minus_wer(1.7, ScoringConfig(clip_wer=False)), -0.7)


def test_negative_ss_is_clipped():
    assert normalize_ss(-0.2) == 0.0
    assert normalize_ss(-0.2, ScoringConfig(clip_ss=False)) == -0.2


def test_mos_normalisation():
    assert normalize_mos(1) == 0.0 and normalize_mos(5) == 1.0 and math.isclose(normalize_mos(4.12), 0.78)


@pytest.mark.parametrize("kw", [
    dict(nvpa=1.2), dict(nvpa=-0.1), dict(wer=-0.01), dict(pmos=0.5), dict(pmos=5.5),
    dict(ss=1.2), dict(nvpa=float("nan")), dict(wer=float("inf")), dict(pmos=None),
])
def test_out_of_range_inputs_raise(kw):
    with pytest.raises(ValueError):
        auto_score("A", **{**KW, **kw})


def test_invalid_human_scores_and_track():
    with pytest.raises(ValueError):
        final_score("A", **KW, sn=6, q=4)
    with pytest.raises(ValueError):
        auto_score("C", **KW)


def test_components_expose_breakdown():
    r = final_score("A", **KW, sn=4.10, q=4.25)
    assert set(r.components) == {"NVPA", "1-WER", "pMOS", "SS", "SN", "Q"}
    assert math.isclose(sum(c.contribution for c in r.components.values()), r.value)
