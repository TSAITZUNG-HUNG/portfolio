import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from slotlab import jackpot as J  # noqa: E402


def block_se(x, blocks=100):
    """分塊平均的標準誤：滾存時相鄰獎金相關，iid 標準誤會低估。"""
    m = np.array([b.mean() for b in np.array_split(x, blocks)])
    return m.std(ddof=1) / blocks ** 0.5


def test_conservation_without_carry_iid():
    """守恆：長期平均獎金 = c/p + S（κ=0 時各次獨立，可用一般標準誤）。"""
    pl = J.Pool(c=0.02, p=1 / 2_000, seed=50)
    x = J.simulate_prizes(pl, 400_000, np.random.default_rng(1))
    se = x.std(ddof=1) / len(x) ** 0.5
    assert abs(x.mean() - pl.mean_prize) < 4 * se


def test_conservation_with_carry_uses_block_standard_error():
    pl = J.Pool(c=0.04, p=1 / 500, carry=0.5)
    x = J.simulate_prizes(pl, 600_000, np.random.default_rng(2))
    assert abs(x.mean() - pl.mean_prize) < 4 * block_se(x)


def test_carry_makes_successive_prizes_correlated():
    """這是為什麼不能用 iid 標準誤：滾存讓相鄰獎金正相關。"""
    pl = J.Pool(c=0.04, p=1 / 500, carry=0.5)
    x = J.simulate_prizes(pl, 200_000, np.random.default_rng(3))
    r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert r1 > 0.1
    x0 = J.simulate_prizes(J.Pool(c=0.04, p=1 / 500), 200_000, np.random.default_rng(3))
    assert abs(np.corrcoef(x0[:-1], x0[1:])[0, 1]) < 0.02


def test_bet_stream_money_identity_is_exact():
    """逐注模擬：派出 + 剩餘餘額 == 全部提撥（一分不差）。"""
    pl = J.Pool(c=0.04, p=1 / 50, carry=0.5, seed=1.0)
    bal, hit, paid, final = J.simulate_bet_stream(pl, 50_000, np.random.default_rng(4))
    contributions = pl.c * len(hit)
    n_hits = int(hit.sum())
    # seed 每次命中後補入，屬營運方出資
    assert abs(paid + final - (contributions + pl.seed * (n_hits + 1))) < 1e-6


def test_event_driven_matches_bet_by_bet_distribution_shape():
    pl = J.Pool(c=0.04, p=1 / 50, carry=0.5)
    bal, hit, paid, final = J.simulate_bet_stream(pl, 600_000, np.random.default_rng(5))
    a = (1 - pl.carry) * bal[hit]
    b = J.simulate_prizes(pl, len(a), np.random.default_rng(6))
    assert abs(a.std() / a.mean() - b.std() / b.mean()) < 0.03       # 變異係數一致
    assert abs(a.mean() - b.mean()) < 4 * max(block_se(a), block_se(b))


def test_guarantee_reserve_is_self_funding_exactly():
    c, p = 0.02, 1 / 200_000
    mu = c / p
    for gf in (0.05, 0.1, 0.3, 0.5):
        G = gf * mu
        s = J.guarantee_reserve_share(c, p, G)
        assert s is not None and 0 < s < 1
        assert abs(s * c / p - J._shortfall(G, (1 - s) * c, p)) < 1e-6 * mu


def test_guarantee_share_is_monotone_and_zero_at_zero():
    c, p = 0.02, 1 / 200_000
    assert J.guarantee_reserve_share(c, p, 0) == 0.0
    ss = [J.guarantee_reserve_share(c, p, f * c / p) for f in (0.05, 0.1, 0.2, 0.4)]
    assert all(a < b for a, b in zip(ss, ss[1:]))


def test_guarantee_keeps_mean_prize_and_shrinks_big_prizes_by_reserve_share():
    c, p = 0.02, 1 / 20_000
    mu = c / p
    G = 0.3 * mu
    s = J.guarantee_reserve_share(c, p, G)
    K = np.random.default_rng(7).geometric(p, size=2_000_000)
    prize = np.maximum((1 - s) * c * K, G)
    se = prize.std(ddof=1) / len(prize) ** 0.5
    assert abs(prize.mean() - mu) < 4 * se                              # 平均獎金不變（只是重新分配）
    q_base = J.prize_quantile(c, p, 0.0, 0.99)
    q_g = J.prize_quantile((1 - s) * c, p, G, 0.99)
    assert abs(q_g / q_base - (1 - s)) < 0.01                           # 大獎等比例縮 (1−s)


def test_hunter_formula_matches_exact_sum_over_geometric():
    """獨立於模擬：對幾何分布直接加總，與解析公式比對。"""
    c, p = 0.02, 1 / 500
    k_grid = np.arange(1, 400_000)
    pmf = p * (1 - p) ** (k_grid - 1)
    for k in (0.5, 1, 2, 3):
        theta = k * c / p
        m = c * k_grid >= theta
        exact = p * (c * k_grid[m] * pmf[m]).sum() / pmf[m].sum()
        assert abs(J.hunter_jackpot_rtp(c, p, k) - exact) / exact < 2e-3


def test_hunter_rtp_baseline_and_monotone():
    c, p = 0.02, 1 / 200_000
    assert J.hunter_jackpot_rtp(c, p, 0) == c
    v = [J.hunter_jackpot_rtp(c, p, k) for k in (0.5, 1, 2, 3)]
    assert all(a < b for a, b in zip(v, v[1:])) and v[0] > c
    assert abs(J.hunter_jackpot_rtp(c, p, 1) / c - 2.0) < 0.01          # k=1 → 約 2 倍


def test_hunter_formula_agrees_with_bet_by_bet_simulation():
    pl = J.Pool(c=0.02, p=1 / 500)
    bal, hit, paid, final = J.simulate_bet_stream(pl, 1_500_000, np.random.default_rng(9))
    for k in (1, 2):
        a = J.hunter_jackpot_rtp(pl.c, pl.p, k)
        b = J.hunter_jackpot_rtp_from_stream(bal, pl.p, pl.mean_prize, k)
        assert abs(a - b) / a < 0.05


def test_top_share_known_values():
    assert J.top_share(np.array([0, 0, 0, 100.0]), 0.25) == 1.0
    assert abs(J.top_share(np.array([10, 10, 10, 10.0]), 0.25) - 0.25) < 1e-12
    assert J.top_share(np.zeros(5)) == 0.0


def test_player_winnings_hit_rate_and_mean_rtp():
    pl = J.Pool(c=0.04, p=1 / 500, carry=0.5)
    pool = J.simulate_prizes(pl, 200_000, np.random.default_rng(10))
    w = J.player_winnings(pl, pool, 10_000, 300, np.random.default_rng(11))
    assert abs((w > 0).mean() - (1 - (1 - pl.p) ** 300)) < 0.02
    assert abs(w.mean() / 300 - pl.c) < 0.002                           # 即時池：約 4%（誤差遠小於 0.2 個百分點）
