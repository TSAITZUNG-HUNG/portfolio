"""
虛構的「任意位置計數付費」老虎機（5x3 = 15 格）。

規則（刻意簡化，與任何實際產品無關）：
- 每格獨立抽一個符號：6 種付費符號 A–F 或空白。
- 某付費符號在 15 格中出現 c >= 3 個，就依賠付表給 pay[sym][c] 倍。
- 一轉的總賠付 = 各符號賠付加總，並以 MAX_WIN 封頂。

關鍵性質：15 格的符號數量向量服從多項分布，只有 C(21,6)=54,264 種組合，
所以「整個結果分布」可以**精確枚舉**，不需要靠模擬。這讓我們有標準答案可以
檢驗 Monte Carlo 與 LP 配平的正確性。

賠付以「整數百分之一倍」(uint，payout×100) 儲存，避免浮點誤差，
這和實務上查找表要求整數賠付的做法一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import lgamma, log

import numpy as np

N_CELLS = 15
SYMBOLS = ["A", "B", "C", "D", "E", "F"]      # 由低階到高階
BLANK = len(SYMBOLS)                          # 第 7 類：空白
MIN_COUNT = 3
MAX_COUNT = N_CELLS
MAX_WIN_X100 = 500_000                        # 封頂 5,000x（以百分之一倍計）


@dataclass(frozen=True)
class Game:
    """遊戲參數。frozen：配平只能動「權重」，不能偷改這份賠付表。"""
    probs: tuple = (0.11, 0.085, 0.065, 0.045, 0.03, 0.015, 0.65)
    base_x100: tuple = (50, 100, 200, 500, 1500, 5000)   # 每個符號 c=3 時的賠付（×100）
    growth: float = 2.5                            # 每多一個符號，賠付乘上的倍數
    name: str = "toy"

    def with_changes(self, **kw) -> "Game":
        d = {**self.__dict__, **kw}
        return Game(**d)


def pay_table_x100(game: Game) -> np.ndarray:
    """shape (6, N_CELLS+1)：pay[k][c] = 該符號出現 c 個時的賠付（×100，整數）。"""
    t = np.zeros((len(SYMBOLS), N_CELLS + 1), dtype=np.int64)
    for k, base in enumerate(game.base_x100):
        for c in range(MIN_COUNT, MAX_COUNT + 1):
            t[k, c] = int(round(base * game.growth ** (c - MIN_COUNT)))
    return t


def payout_from_counts(counts: np.ndarray, table: np.ndarray) -> np.ndarray:
    """counts: (..., 7) 符號數量；回傳 (...,) 賠付（×100，已封頂）。"""
    total = np.zeros(counts.shape[:-1], dtype=np.int64)
    for k in range(len(SYMBOLS)):
        total += table[k][counts[..., k]]
    return np.minimum(total, MAX_WIN_X100)


def _compositions(n: int, parts: int):
    """所有「n 個格子分到 parts 類」的數量向量（stars and bars）。"""
    if parts == 1:
        yield (n,)
        return
    for first in range(n + 1):
        for rest in _compositions(n - first, parts - 1):
            yield (first,) + rest


def exact_distribution(game: Game):
    """
    精確結果分布。回傳 (payouts_x100 遞增, probs)，相同賠付已合併成一個「原子」。
    """
    comps = np.array(list(_compositions(N_CELLS, len(SYMBOLS) + 1)), dtype=np.int64)
    logp = np.array([log(p) if p > 0 else -np.inf for p in game.probs])
    lg = np.vectorize(lgamma)
    log_coef = lgamma(N_CELLS + 1) - lg(comps + 1).sum(axis=1)
    with np.errstate(invalid="ignore"):
        log_pmf = log_coef + np.where(comps > 0, comps * logp, 0.0).sum(axis=1)
    pmf = np.exp(log_pmf)
    assert abs(pmf.sum() - 1) < 1e-12, "多項分布機率總和應為 1"
    payout = payout_from_counts(comps, pay_table_x100(game))
    order = np.unique(payout)
    probs = np.array([pmf[payout == v].sum() for v in order])
    return order, probs


def rtp(payouts_x100: np.ndarray, probs: np.ndarray) -> float:
    return float((payouts_x100 * probs).sum() / 100.0)


def stats(payouts_x100: np.ndarray, probs: np.ndarray) -> dict:
    x = payouts_x100 / 100.0
    mean = float((x * probs).sum())
    var = float(((x - mean) ** 2 * probs).sum())
    return {
        "rtp": mean,
        "std": var ** 0.5,
        "hit_rate": float(probs[payouts_x100 > 0].sum()),
        "p_ge_10x": float(probs[payouts_x100 >= 1000].sum()),
        "p_ge_100x": float(probs[payouts_x100 >= 10000].sum()),
        "max_payout_x": float(x.max()),
        "n_atoms": int(len(x)),
    }


def simulate_counts(game: Game, n: int, rng: np.random.Generator) -> np.ndarray:
    """n 轉的符號數量（多項分布），shape (n, 7)。"""
    return rng.multinomial(N_CELLS, game.probs, size=n)


def simulate_payouts(game: Game, n: int, rng: np.random.Generator) -> np.ndarray:
    return payout_from_counts(simulate_counts(game, n, rng), pay_table_x100(game))
