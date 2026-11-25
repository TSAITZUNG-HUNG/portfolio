import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from slotlab.balance import InfeasibleError, Targets, balance, to_integer_weights, total_variation  # noqa: E402
from slotlab.mc import RunningStats, ci95, n_required  # noqa: E402
from slotlab.toy_slot import Game, exact_distribution, rtp, stats, simulate_payouts  # noqa: E402

V, P = exact_distribution(Game())
TG = Targets(rtp=0.96, hit_lo=0.30, hit_hi=0.45, big_threshold_x100=10_000, big_cap=1e-4, min_ratio=0.25)


def test_exact_distribution_is_a_probability_distribution():
    assert abs(P.sum() - 1) < 1e-12 and (P >= 0).all()
    assert (np.diff(V) > 0).all()                    # 原子已排序且不重複


def test_exact_rtp_matches_large_monte_carlo():
    x = simulate_payouts(Game(), 600_000, np.random.default_rng(3)) / 100.0
    se = x.std(ddof=1) / len(x) ** 0.5
    assert abs(x.mean() - rtp(V, P)) < 4 * se


def test_balance_hits_target_rtp_and_constraints():
    w = balance(V, P, TG)
    s = stats(V, w)
    assert abs(s["rtp"] - 0.96) < 1e-9
    assert 0.30 <= s["hit_rate"] <= 0.45
    assert s["p_ge_100x"] <= 1e-4 + 1e-12
    assert (w >= TG.min_ratio * P - 1e-15).all()     # 每個結果都保留
    assert abs(w.sum() - 1) < 1e-9


def test_balance_never_touches_the_payout_table():
    before = V.copy()
    balance(V, P, TG)
    assert np.array_equal(V, before)


def test_big_win_floor_is_honoured_and_costs_little():
    nat_big = stats(V, P)["p_ge_100x"]
    w_a = balance(V, P, TG)
    w_b = balance(V, P, Targets(**{**TG.__dict__, "big_floor": 0.7 * nat_big}))
    assert stats(V, w_b)["p_ge_100x"] >= 0.7 * nat_big - 1e-12
    assert stats(V, w_b)["p_ge_100x"] > stats(V, w_a)["p_ge_100x"]
    assert total_variation(w_b, P) - total_variation(w_a, P) < 0.005   # 代價 < 0.5 個百分點


def test_lower_target_costs_more_distortion():
    tvs = [total_variation(balance(V, P, Targets(**{**TG.__dict__, "rtp": r})), P) for r in (0.99, 0.96, 0.92)]
    assert tvs[0] < tvs[1] < tvs[2]


def test_infeasible_targets_raise():
    with pytest.raises(InfeasibleError):
        balance(V, P, Targets(rtp=0.96, min_ratio=0.99, big_cap=1e-9))


def test_integer_weights_positive_exact_sum_small_rtp_error():
    w = balance(V, P, TG)
    iw = to_integer_weights(w)
    assert iw.min() >= 1 and iw.dtype == np.int64 and int(iw.sum()) == 10**12
    assert abs(rtp(V, iw / iw.sum()) - 0.96) < 2e-6


def test_running_stats_equals_numpy_over_batches():
    x = np.random.default_rng(0).normal(5, 2, 10_000)
    rs = RunningStats()
    for chunk in np.array_split(x, 7):
        rs.update(chunk)
    assert abs(rs.mean - x.mean()) < 1e-12
    assert abs(rs.var - x.var(ddof=1)) < 1e-9


def test_ci_and_required_n_formulas():
    lo, hi = ci95(1.0, 0.01)
    assert abs((hi - lo) - 2 * 1.959963984540054 * 0.01) < 1e-12
    n = n_required(3.5, 0.00576)
    assert 1_900_000 < n < 2_100_000
