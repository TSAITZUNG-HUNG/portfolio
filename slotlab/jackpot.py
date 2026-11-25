"""
樂透式 Jackpot 獎池（虛構玩法）：結構、模擬與設計取捨。

單位：以「每注 = 1」計，所有獎金為「幾倍投注」。

模型（每一筆投注依序發生）：
- 每注提撥 c 進獎池；
- 每注以機率 p 命中，命中者拿走 (1−κ)·餘額；κ 為「滾存比例」（保留給下一期的比例）；
- 命中後餘額 = κ·B + S，S 為營運方出的基礎獎金（seed）。

關鍵性質（讓模擬有標準答案可以檢驗）：
1. 兩次命中之間的投注數 K 服從幾何分布，P(K=k)=p(1−p)^(k−1)，平均 1/p。
2. 守恆：長期平均每次命中的獎金 = c/p + S（進去多少、最終就派出多少）。
3. 隨機取一筆投注，當下獎池大小的分布，與「命中那一刻的獎池」分布相同
   （幾何分布的無記憶性）。這就是「看得到獎池的玩家只在獎池高時才玩」之所以有利可圖的原因。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.signal import lfilter


@dataclass(frozen=True)
class Pool:
    c: float                 # 每注提撥
    p: float                 # 每注命中機率
    seed: float = 0.0        # 命中後營運方補入的基礎獎金
    carry: float = 0.0       # κ：命中後保留給下一期的比例（0 = 全額派出並歸零）

    @property
    def mean_prize(self) -> float:
        """守恆：長期平均每次命中的獎金。"""
        return self.c / self.p + self.seed


# ------------------------------- 模擬 -------------------------------

def simulate_prizes(pool: Pool, n_hits: int, rng: np.random.Generator, burn_in: int = 2_000) -> np.ndarray:
    """
    連續 n_hits 次命中的獎金序列（事件驅動：只模擬命中，不逐注模擬）。
    遞迴：post_k = κ·(post_{k−1} + c·K_k) + S；prize_k = (1−κ)·(post_{k−1} + c·K_k)。
    用 lfilter 以 AR(1) 一次算完。
    """
    n = n_hits + burn_in
    K = rng.geometric(pool.p, size=n).astype(float)           # 支撐 k >= 1
    x = pool.carry * pool.c * K + pool.seed
    post = lfilter([1.0], [1.0, -pool.carry], x)               # post_k
    pre = np.empty(n)
    pre[0] = pool.seed + pool.c * K[0]                         # 起點：只有基礎獎金的空池
    pre[1:] = post[:-1] + pool.c * K[1:]
    prize = (1.0 - pool.carry) * pre
    return prize[burn_in:]


def simulate_bet_stream(pool: Pool, n_bets: int, rng: np.random.Generator):
    """
    逐注模擬（慢，但最不容易寫錯）：回傳 (每注當下的獎池餘額 pre-hit, 命中旗標, 累計派出)。
    用來獨立驗證事件驅動版本與解析結果。
    """
    hit = rng.random(n_bets) < pool.p
    balance_before = np.empty(n_bets)
    paid = 0.0
    bal = pool.seed
    for t in range(n_bets):
        bal += pool.c
        balance_before[t] = bal
        if hit[t]:
            prize = (1 - pool.carry) * bal
            paid += prize
            bal = pool.carry * bal + pool.seed
    return balance_before, hit, paid, bal


# ------------------------------- 保底的取捨 -------------------------------

def _shortfall(G: float, c_grow: float, p: float) -> float:
    """E[(G − c'·K)+]，K 為幾何分布（精確加總）。"""
    if G <= 0:
        return 0.0
    kmax = int(np.ceil(G / c_grow))
    k = np.arange(1, kmax + 1)
    pmf = p * (1 - p) ** (k - 1)
    return float((pmf * np.maximum(G - c_grow * k, 0)).sum())


def guarantee_reserve_share(c: float, p: float, G: float) -> float | None:
    """
    終極池設「保底 G」：命中時獎金 = max(餘額, G)。差額由「保底準備金」補。
    準備金來源：每注提撥 c 中抽出比例 s，其餘 (1−s)·c 進獎池。
    求最小的 s，使準備金「平均而言」剛好付得起保底差額：
        s·c/p  =  E[(G − (1−s)·c·K)+]
    無解（G 太大，抽再多也補不起）回傳 None。
    """
    if G <= 0:
        return 0.0
    f = lambda s: s * c / p - _shortfall(G, (1 - s) * c, p)
    lo, hi = 1e-9, 0.999
    if f(hi) < 0:
        return None
    return float(brentq(f, lo, hi, xtol=1e-12))


def prize_quantile(c_grow: float, p: float, G: float, q: float) -> float:
    """獎金 = max(c'·K, G) 的 q 分位數（精確）。"""
    kq = np.ceil(np.log1p(-q) / np.log1p(-p))
    return float(max(c_grow * kq, G))


def guarantee_table(c: float, p: float, G_fracs, qs=(0.5, 0.9, 0.99, 0.999)):
    """G 以「平均獎金 c/p」的比例表示。回傳 list[dict]。"""
    mu = c / p
    rows = []
    for gf in G_fracs:
        G = gf * mu
        s = guarantee_reserve_share(c, p, G)
        if s is None:
            rows.append({"G_frac": gf, "G": G, "s": None})
            continue
        cg = (1 - s) * c
        row = {"G_frac": gf, "G": G, "s": s,
               "p_floor_binds": float(1 - (1 - p) ** np.floor(G / cg)) if G > 0 else 0.0}
        for q in qs:
            row[f"q{q}"] = prize_quantile(cg, p, G, q)
        rows.append(row)
    return rows


# ------------------------------- 玩家策略：只在獎池高時才玩 -------------------------------

def hunter_jackpot_rtp(c: float, p: float, k: float) -> float:
    """
    「看得到獎池、只在餘額 ≥ k·平均獎金 時才下注」的玩家，其獎池部分的 RTP（解析解）。
    隨機一注當下獎池 = c·K，K 幾何分布；每注期望回收 = p·c·K。
        RTP_h = p·c·E[K | c·K ≥ θ]，θ = k·c/p
    幾何分布無記憶：E[K | K ≥ k*] = k* − 1 + 1/p。
    k=0 即「照常玩」，得 RTP = c（守恆）。
    """
    if k <= 0:
        return c
    kstar = int(np.ceil(k / p))
    return float(p * c * (kstar - 1 + 1 / p))


def hunter_jackpot_rtp_from_stream(balance_before: np.ndarray, p: float, mean_prize: float, k: float) -> float:
    """用逐注模擬出的獎池序列，估計同一個策略的 RTP（獨立於解析解）。"""
    theta = k * mean_prize
    m = balance_before >= theta
    if m.sum() == 0:
        return float("nan")
    return float(p * balance_before[m].mean())


# ------------------------------- 兩種視角：整體 vs 單一玩家 -------------------------------

def player_winnings(pool: Pool, prize_pool: np.ndarray, n_players: int, rounds: int,
                    rng: np.random.Generator) -> np.ndarray:
    """
    單一玩家玩 `rounds` 局的總中獎金額（只計這個獎池）。
    命中次數 ~ Binomial(rounds, p)；每次獎金從穩態獎金樣本中抽（近似獨立）。
    回傳每位玩家的總獎金（長度 n_players）。
    """
    hits = rng.binomial(rounds, pool.p, size=n_players)
    out = np.zeros(n_players)
    idx = np.flatnonzero(hits)
    for i in idx:
        out[i] = rng.choice(prize_pool, size=hits[i]).sum()
    return out


def top_share(winnings: np.ndarray, top_frac: float = 0.01) -> float:
    """最會中的前 top_frac 的玩家，拿走全部獎金的多少比例。"""
    w = np.sort(winnings)[::-1]
    k = max(1, int(round(len(w) * top_frac)))
    tot = w.sum()
    return float(w[:k].sum() / tot) if tot > 0 else 0.0
