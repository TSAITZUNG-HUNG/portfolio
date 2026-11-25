"""示範二：Monte Carlo 的統計紀律。

(a) 收斂：RTP 估計值與 95% 信賴區間隨模擬次數收斂到「精確值」。
(b) 覆蓋率：重複實驗，95% 信賴區間是不是真的 ~95% 包到精確值？
(c) 雜訊陷阱：兩個 RTP 只差 δ 的設定，在有限模擬下排序會不會排反？

標準答案（精確 RTP）來自 slotlab.toy_slot.exact_distribution —— 這是這個虛構遊戲
才有的奢侈；真實遊戲沒有，才更需要標準誤。
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
from slotlab.mc import RunningStats, ci95, n_required  # noqa: E402
from slotlab.toy_slot import Game, exact_distribution, rtp, simulate_payouts, stats  # noqa: E402

OUT = pathlib.Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
SEED = 20261125

game = Game()
v, p = exact_distribution(game)
true_rtp = rtp(v, p)
sigma = stats(v, p)["std"]

# ---------- (a) 收斂 ----------
rng = np.random.default_rng(SEED)
checkpoints = np.unique(np.logspace(3, 6.6, 28).astype(int))
rs = RunningStats()
rows = []
done = 0
for cp in checkpoints:
    need = cp - done
    while need > 0:
        b = min(need, 250_000)
        rs.update(simulate_payouts(game, b, rng) / 100.0)
        need -= b; done += b
    rows.append((rs.n, rs.mean, rs.se))
conv = np.array(rows)

# ---------- (b) 覆蓋率 ----------
REPS, N_COV = 400, 20_000
cover = 0
widths = []
rng_b = np.random.default_rng(SEED + 1)
for _ in range(REPS):
    x = simulate_payouts(game, N_COV, rng_b) / 100.0
    m, se = x.mean(), x.std(ddof=1) / N_COV ** 0.5
    lo, hi = ci95(m, se)
    cover += (lo <= true_rtp <= hi); widths.append(hi - lo)
coverage = cover / REPS

# ---------- (c) 雜訊陷阱 ----------
# 變體 B：最高階符號的基礎賠付 +8%，精確 RTP 只差約 0.2–0.3 個百分點
base_b = list(game.base_x100); base_b[-1] = int(round(base_b[-1] * 1.08))
game_b = game.with_changes(base_x100=tuple(base_b), name="toy-B")
vb, pb = exact_distribution(game_b)
delta = rtp(vb, pb) - true_rtp
N_TRAP, REPS_C = 100_000, 300
rng_c = np.random.default_rng(SEED + 2)
diffs = []
for _ in range(REPS_C):
    a = simulate_payouts(game, N_TRAP, rng_c).mean() / 100.0
    b = simulate_payouts(game_b, N_TRAP, rng_c).mean() / 100.0
    diffs.append(b - a)                       # 真實 B 比 A 高 delta
diffs = np.array(diffs)
wrong = float((diffs <= 0).mean())
n95 = n_required(sigma, delta)

result = {
    "exact_rtp": true_rtp, "sigma_per_spin": sigma,
    "convergence_last": {"n": int(conv[-1, 0]), "mean": conv[-1, 1], "se": conv[-1, 2],
                         "abs_error": abs(conv[-1, 1] - true_rtp)},
    "coverage": {"reps": REPS, "n_per_rep": N_COV, "nominal": 0.95, "observed": coverage,
                 "mean_ci_width": float(np.mean(widths))},
    "noise_trap": {"delta_exact": delta, "n_per_game": N_TRAP, "reps": REPS_C,
                   "share_ordered_wrong": wrong, "n_required_95pct": n95},
}
(OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

# ---------- 圖 ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
n, m, se = conv[:, 0], conv[:, 1], conv[:, 2]
ax[0].fill_between(n, m - 1.96 * se, m + 1.96 * se, color="#93c5fd", alpha=0.6, label="95% CI")
ax[0].plot(n, m, "o-", ms=3, color="#2563eb", label="MC estimate")
ax[0].axhline(true_rtp, color="#dc2626", ls="--", label=f"exact RTP {true_rtp:.4f}")
ax[0].set_xscale("log"); ax[0].set_xlabel("simulated spins"); ax[0].set_ylabel("RTP")
ax[0].set_title("(a) Convergence with standard error"); ax[0].legend()
ax[0].set_ylim(true_rtp - 0.12, true_rtp + 0.12)

ax[1].hist(diffs * 100, bins=30, color="#93c5fd", edgecolor="white")
ax[1].axvline(0, color="#6b7280", lw=1)
ax[1].axvline(delta * 100, color="#dc2626", ls="--", label=f"true gap = {delta*100:.2f} pp")
ax[1].set_xlabel("estimated RTP(B) - RTP(A)  (percentage points)"); ax[1].set_ylabel("repeats")
ax[1].set_title(f"(c) Noise trap: ordered wrong {wrong*100:.0f}% at N={N_TRAP:,}")
ax[1].legend()
fig.tight_layout(); fig.savefig(OUT / "convergence_and_noise_trap.png", dpi=140); plt.close(fig)

print(f"exact RTP = {true_rtp:.5f}   per-spin sigma = {sigma:.3f}")
c = result["convergence_last"]
print(f"(a) N={c['n']:,}: {c['mean']:.5f} ± {1.96*c['se']:.5f} (95% CI)  |error|={c['abs_error']:.5f}")
print(f"(b) coverage over {REPS} repeats of N={N_COV:,}: {coverage*100:.1f}% (nominal 95%)")
print(f"(c) true gap {delta*100:.3f} pp; N={N_TRAP:,} → ordered wrong {wrong*100:.1f}%; "
      f"N needed for 95% correct ≈ {n95:,} per game")
