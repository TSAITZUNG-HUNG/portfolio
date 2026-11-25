"""示範一：用 LP 把 RTP 配平到目標值，只動權重、不動賠付。

執行：python demos/01_rtp_volatility_balancing/run.py
產出：results/*.png、results/results.json
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from slotlab.balance import Targets, balance, to_integer_weights, total_variation  # noqa: E402
from slotlab.toy_slot import Game, exact_distribution, rtp, stats  # noqa: E402

OUT = pathlib.Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

TARGET = Targets(rtp=0.96, hit_lo=0.30, hit_hi=0.45,
                 big_threshold_x100=10_000, big_cap=1e-4, min_ratio=0.25)

game = Game()
payouts, p = exact_distribution(game)          # 賠付向量 x：此後絕不改動
payouts_before = payouts.copy()

w = balance(payouts, p, TARGET)
assert np.array_equal(payouts, payouts_before), "配平不得改動賠付表"

# 方案 B：同樣的目標，但要求 P(>=100x) 不低於自然值的 70%（保護大獎體感）
big_nat = stats(payouts, p)["p_ge_100x"]
TARGET_B = Targets(**{**TARGET.__dict__, "big_floor": 0.7 * big_nat})
w_b = balance(payouts, p, TARGET_B)
assert np.array_equal(payouts, payouts_before)

iw = to_integer_weights(w)                      # 正整數權重（總和 1e12）
w_int = iw / iw.sum()

nat, bal, fin = stats(payouts, p), stats(payouts, w), stats(payouts, w_int)
tv = total_variation(w, p)

bal_b = stats(payouts, w_b)
tv_b = total_variation(w_b, p)
result = {
    "target": TARGET.__dict__,
    "natural": nat, "balanced_lp": bal, "balanced_integer": fin,
    "tv_distance": tv,
    "scenario_b_protect_big_wins": {"target": TARGET_B.__dict__, "stats": bal_b, "tv_distance": tv_b},
    "integer_rtp_error": abs(rtp(payouts, w_int) - TARGET.rtp),
    "payout_table_unchanged": True,
}
(OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

# ---- 圖 1：賠付區間機率（對數軸）與各結果被重新加權的方向 ----
x = payouts
# 賠付以 ×100 計：0 為未中獎，其餘依區間分桶
edges = [1, 100, 200, 500, 1000, 2000, 5000, 10_000, 50_000, 500_001]
labels = ["0", "0–1x", "1–2x", "2–5x", "5–10x", "10–20x", "20–50x", "50–100x", "100–500x", "500x+"]
def bucket(prob):
    out = [prob[x == 0].sum()]
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(prob[(x >= lo) & (x < hi)].sum())
    return np.array(out)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
idx = np.arange(len(labels))
ax[0].bar(idx - 0.27, bucket(p), 0.27, label="natural", color="#9ca3af")
ax[0].bar(idx, bucket(w), 0.27, label="A: balanced (LP)", color="#2563eb")
ax[0].bar(idx + 0.27, bucket(w_b), 0.27, label="B: balanced, big wins protected", color="#16a34a")
ax[0].set_yscale("log"); ax[0].set_xticks(idx); ax[0].set_xticklabels(labels, rotation=35, ha="right")
ax[0].set_ylabel("probability (log)"); ax[0].set_title("Payout distribution: natural vs balanced")
ax[0].legend()

with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.where(p > 0, w / p, np.nan)
with np.errstate(divide="ignore", invalid="ignore"):
    ratio_b = np.where(p > 0, w_b / p, np.nan)
ax[1].scatter(np.maximum(payouts / 100, 0.05), ratio, s=7, alpha=0.55, color="#2563eb", label="A")
ax[1].scatter(np.maximum(payouts / 100, 0.05), ratio_b, s=7, alpha=0.55, color="#16a34a", label="B")
ax[1].legend()
ax[1].axhline(1, color="#6b7280", lw=1)
ax[1].set_xscale("log"); ax[1].set_xlabel("payout (x bet, log)"); ax[1].set_ylabel("new weight / natural probability")
ax[1].set_title("Which outcomes get re-weighted (payout table untouched)")
fig.tight_layout(); fig.savefig(OUT / "distribution_and_reweighting.png", dpi=140); plt.close(fig)

# ---- 圖 2：目標 RTP 越低，離自然分布越遠（代價可量化） ----
targets = np.linspace(0.90, 1.04, 15)
tvs = []
for r in targets:
    ww = balance(payouts, p, Targets(**{**TARGET.__dict__, "rtp": float(r)}))
    tvs.append(total_variation(ww, p))
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(targets * 100, np.array(tvs) * 100, "o-", color="#2563eb")
ax.axvline(nat["rtp"] * 100, color="#6b7280", ls="--", lw=1)
ax.text(nat["rtp"] * 100 + 0.2, max(tvs) * 100 * 0.9, "natural RTP", color="#6b7280")
ax.set_xlabel("target RTP (%)"); ax.set_ylabel("TV distance to natural distribution (%)")
ax.set_title("Cost of balancing: distortion vs target RTP")
fig.tight_layout(); fig.savefig(OUT / "tv_cost_vs_target_rtp.png", dpi=140); plt.close(fig)

fmt = lambda s: (f"RTP {s['rtp']*100:.3f}% | std {s['std']:.3f} | hit {s['hit_rate']*100:.2f}% | "
                 f"P(>=100x) {s['p_ge_100x']:.2e} | max {s['max_payout_x']:.0f}x")
print("natural :", fmt(nat))
print("balanced:", fmt(bal))
print("integer :", fmt(fin))
print(f"TV distance = {tv*100:.3f}%   integer RTP error = {result['integer_rtp_error']:.2e}")
print("B (protect big wins):", fmt(bal_b), f"| TV {tv_b*100:.3f}%")
print("payout table unchanged:", result["payout_table_unchanged"])
