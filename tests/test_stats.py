import math

from nvtts_eval.stats import bootstrap_ci, describe, group_stats, macro_mean, percentile


def test_percentile_and_describe_hand_computed():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    d = describe([1, 2, 3, 4, None, float("nan")])
    assert d["n"] == 4 and d["mean"] == 2.5 and d["median"] == 2.5 and d["min"] == 1 and d["max"] == 4
    assert math.isclose(d["std"], math.sqrt(1.25))
    assert describe([None]) == {"n": 0}


def test_bootstrap_is_deterministic_and_brackets_mean():
    vals = [(i * 37 % 41) / 41 for i in range(40)]          # continuous-ish values
    mean = sum(vals) / len(vals)
    a, b = bootstrap_ci(vals, seed=1), bootstrap_ci(vals, seed=1)
    assert a == b
    assert a[0] < mean < a[1]
    assert bootstrap_ci(vals, seed=2) != a
    assert all(math.isnan(x) for x in bootstrap_ci([1.0]))


def test_cluster_bootstrap_is_wider_when_clusters_differ():
    vals = [1.0] * 30 + [0.0] * 30                      # cluster 0 all ones, cluster 1 all zeros ...
    clusters = [0] * 30 + [1] * 30
    # ... plus a few more clusters so resampling is possible
    vals += [1.0, 0.0, 1.0, 0.0]
    clusters += [2, 3, 4, 5]
    iid = bootstrap_ci(vals, seed=0)
    clu = bootstrap_ci(vals, clusters=clusters, seed=0)
    assert (clu[1] - clu[0]) > (iid[1] - iid[0])
    assert all(math.isnan(x) for x in bootstrap_ci([1.0, 0.0], clusters=[0, 0]))   # one cluster only


def test_group_stats_counts_missing_and_low_n():
    recs = [
        {"t": "breathing", "v": 1.0}, {"t": "breathing", "v": 0.0}, {"t": "breathing", "v": None},
        {"t": "breathing", "error": "x"}, {"t": "sniff", "v": 1.0},
    ]
    g = group_stats(recs, "t", "v", min_n=2, n_boot=50)
    assert g["breathing"]["n"] == 2 and g["breathing"]["n_missing"] == 2 and g["breathing"]["mean"] == 0.5
    assert not g["breathing"]["low_n"]
    assert g["sniff"]["n"] == 1 and g["sniff"]["low_n"]


def test_macro_vs_micro_mean():
    recs = [{"s": "a", "v": 1.0}] * 9 + [{"s": "b", "v": 0.0}]
    micro = sum(r["v"] for r in recs) / len(recs)
    assert micro == 0.9 and macro_mean(recs, "s", "v") == 0.5
    assert math.isnan(macro_mean([], "s", "v"))
