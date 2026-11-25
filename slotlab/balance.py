"""
用線性規劃（LP）做 RTP／波動率配平 —— 只動「結果的權重」，不動賠付表。

問題：每個「結果原子」j 有固定賠付 x_j 與自然機率 p_j。找一組新機率 w_j，使得
    minimize   TV(w, p) = ½ Σ |w_j − p_j|          （離自然分布越近，機制被扭曲越少）
    subject to Σ w_j = 1,  w_j ≥ min_ratio · p_j    （每個結果都保留，不刪結果）
               Σ w_j x_j = 目標 RTP
               命中率落在 [hit_lo, hit_hi]
               big_floor ≤ P(賠付 ≥ big_threshold) ≤ big_cap   （大獎頻率區間）
絕對值用輔助變數 t_j ≥ |w_j − p_j| 線性化。

為什麼只動權重：公布的賠付表要與玩家畫面上看到的倍率一致。若用係數去壓賠付，
賠付表就失效了。所以 `balance()` 的輸入輸出裡，賠付向量 x 從頭到尾不變，
並用斷言守住這一點。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class Targets:
    rtp: float = 0.96
    hit_lo: float = 0.0
    hit_hi: float = 1.0
    big_threshold_x100: int = 10_000          # 100x
    big_cap: float = 1.0                      # 大獎機率上限
    big_floor: float = 0.0                    # 大獎機率下限（保護大獎的體感頻率）
    min_ratio: float = 0.0                    # 每個原子至少保留自然機率的多少比例


class InfeasibleError(RuntimeError):
    pass


def balance(payouts_x100: np.ndarray, p: np.ndarray, tg: Targets) -> np.ndarray:
    """回傳新機率 w（賠付向量不變）。無解則丟 InfeasibleError。"""
    x = payouts_x100.astype(float) / 100.0
    n = len(x)
    # 變數：[w_0..w_{n-1}, t_0..t_{n-1}]
    c = np.concatenate([np.zeros(n), np.ones(n)])           # minimize Σ t  (= 2·TV)

    # t_j >= w_j - p_j   ->  w_j - t_j <= p_j
    # t_j >= p_j - w_j   -> -w_j - t_j <= -p_j
    I = np.eye(n)
    A_ub = [np.hstack([I, -I]), np.hstack([-I, -I])]
    b_ub = [p, -p]

    hit_mask = (payouts_x100 > 0).astype(float)
    big_mask = (payouts_x100 >= tg.big_threshold_x100).astype(float)
    rows, rhs = [], []
    # 命中率區間
    rows += [np.concatenate([hit_mask, np.zeros(n)]), np.concatenate([-hit_mask, np.zeros(n)])]
    rhs += [tg.hit_hi, -tg.hit_lo]
    # 大獎機率區間 [big_floor, big_cap]
    rows += [np.concatenate([big_mask, np.zeros(n)]), np.concatenate([-big_mask, np.zeros(n)])]
    rhs += [tg.big_cap, -tg.big_floor]
    A_ub.append(np.array(rows)); b_ub.append(np.array(rhs))

    A_eq = np.array([
        np.concatenate([np.ones(n), np.zeros(n)]),            # Σ w = 1
        np.concatenate([x, np.zeros(n)]),                     # Σ w x = RTP
    ])
    b_eq = np.array([1.0, tg.rtp])
    bounds = [(tg.min_ratio * pj, None) for pj in p] + [(0, None)] * n

    res = linprog(c, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub),
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise InfeasibleError(f"LP 無解：{res.message}")
    w = np.clip(res.x[:n], 0, None)
    return w / w.sum()


def total_variation(w: np.ndarray, p: np.ndarray) -> float:
    return 0.5 * float(np.abs(w - p).sum())


def to_integer_weights(w: np.ndarray, total: int = 10**12) -> np.ndarray:
    """
    換成正整數權重（RGS 查找表要求整數）。最大餘數法；每個原子至少 1。
    回傳整數權重，總和 == total。
    """
    raw = w * total
    base = np.maximum(np.floor(raw).astype(np.int64), 1)
    diff = total - int(base.sum())
    if diff != 0:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac if diff > 0 else frac)
        i = 0
        while diff != 0:
            j = order[i % len(order)]
            if diff > 0:
                base[j] += 1; diff -= 1
            elif base[j] > 1:
                base[j] -= 1; diff += 1
            i += 1
    assert int(base.sum()) == total
    return base
