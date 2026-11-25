"""Monte Carlo 估計：一律「點估計＋標準誤＋信賴區間」，不單報一個數字。"""
from __future__ import annotations

import numpy as np

Z95 = 1.959963984540054


class RunningStats:
    """Welford 串流估計：記憶體與模擬次數脫鉤，可分批累積。"""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, batch: np.ndarray):
        b = np.asarray(batch, dtype=float)
        nb = b.size
        if nb == 0:
            return
        mb = b.mean()
        m2b = ((b - mb) ** 2).sum()
        delta = mb - self.mean
        tot = self.n + nb
        self.mean += delta * nb / tot
        self.m2 += m2b + delta ** 2 * self.n * nb / tot
        self.n = tot

    @property
    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else float("nan")

    @property
    def se(self) -> float:
        return (self.var / self.n) ** 0.5


def ci95(mean: float, se: float) -> tuple[float, float]:
    return mean - Z95 * se, mean + Z95 * se


def n_required(sigma: float, delta: float, z: float = 1.6448536269514722) -> int:
    """
    要在單尾 95% 信心下，分辨兩個「各自獨立模擬」的估計量相差 delta，
    每個遊戲各需要的模擬次數：n = (z · σ · √2 / δ)²。
    """
    return int(np.ceil((z * sigma * 2 ** 0.5 / delta) ** 2))
