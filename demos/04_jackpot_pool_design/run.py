"""示範四：樂透式 Jackpot 獎池設計與模擬（虛構玩法）。

執行：python demos/04_jackpot_pool_design/run.py
產出：results/*.png、results/results.json

虛構參數（每注 = 1）：
  即時獎共池：每注提撥 4%，命中機率 1/500，命中後保留 50% 滾入下一期
  終極彩池  ：每注提撥 2%，命中機率 1/200,000，命中後歸零
  基礎遊戲  ：RTP 90%（不模擬，只用來算整體 RTP）→ 整體 = 90% + 4% + 2% = 96%
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
from slotlab.jackpot import (Pool, guarantee_table, hunter_jackpot_rtp,  # noqa: E402
                             hunter_jackpot_rtp_from_stream, player_winnings,
                             simulate_bet_stream, simulate_prizes, top_share)

OUT = pathlib.Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
SEED = 20261126
BASE_RTP = 0.90

IP = Pool(c=0.04, p=1 / 500, carry=0.5)
UP = Pool(c=0.02, p=1 / 200_000)
N_PLAYERS, ROUNDS = 10_000, 300
rng = np.random.default_rng(SEED)
res = {"params": {"IP": IP.__dict__, "UP": UP.__dict__, "base_rtp": BASE_RTP,
                  "n_players": N_PLAYERS, "rounds": ROUNDS}}

# ---------------- A. 保底的取捨（終極池） ----------------
G_FRACS = [0, 0.05, 0.10, 0.20, 0.30, 0.50]
gt = guarantee_table(UP.c, UP.p, G_FRACS)
res["guarantee_table"] = gt
base_q = gt[0]

# 基礎獎金（seed，營運方出）：另一條路
S = 100.0
p_small_no_seed = float(1 - (1 - UP.p) ** np.ceil(S / UP.c - 1))        # 獎金 < S 的機率（無 seed）
res["seed"] = {"S": S, "p_prize_below_S_without_seed": p_small_no_seed,
               "operator_rtp_cost": S * UP.p, "operator_rtp_cost_pp": S * UP.p * 100}

# ---------------- B. 即時池滾存的效果 ----------------
carry_rows = []
carry_prizes = {}
for kappa in (0.0, 0.3, 0.5, 0.7):
    pool = Pool(c=IP.c, p=IP.p, carry=kappa)
    x = simulate_prizes(pool, 600_000, rng)
    mu = pool.mean_prize
    carry_prizes[kappa] = x
    carry_rows.append({"carry": kappa, "mean": float(x.mean()), "theory_mean": mu,
                       "cv": float(x.std() / x.mean()),
                       "p_below_quarter_mean": float((x < 0.25 * mu).mean()),
                       "p99_over_mean": float(np.quantile(x, 0.99) / mu)})
res["carry"] = carry_rows

# ---------------- C. 看得到獎池的玩家策略 ----------------
ks = np.array([0, 0.5, 1, 2, 3, 5])
rtp_h = np.array([hunter_jackpot_rtp(UP.c, UP.p, k) for k in ks])
# 逐注模擬（p 放大 → 能在合理時間內驗證同一個推導）
sim_pool = Pool(c=0.02, p=1 / 500)
bal, hit, paid, final = simulate_bet_stream(sim_pool, 2_000_000, np.random.default_rng(SEED + 1))
rtp_sim = np.array([sim_pool.p * bal.mean() if k == 0 else
                    hunter_jackpot_rtp_from_stream(bal, sim_pool.p, sim_pool.mean_prize, k) for k in ks])
rtp_ana_sim = np.array([hunter_jackpot_rtp(sim_pool.c, sim_pool.p, k) for k in ks])
other = IP.c                                                           # 即時池照常玩
total_h = BASE_RTP + other + rtp_h
breakeven_k = (1 - BASE_RTP - other) / UP.c - 1                        # 整體 RTP = 100% 的 k
res["hunter"] = {"k": ks.tolist(), "up_rtp": rtp_h.tolist(), "total_rtp": total_h.tolist(),
                 "sim_check_p_1_over_500": {"analytic": rtp_ana_sim.tolist(), "bet_by_bet": rtp_sim.tolist()},
                 "breakeven_k_total_rtp_100pct": breakeven_k}

# ---------------- D. 兩種視角：1 萬位玩家 × 300 局 ----------------
ip_pool = simulate_prizes(IP, 400_000, rng)
up_pool = simulate_prizes(UP, 400_000, rng)
w_ip = player_winnings(IP, ip_pool, N_PLAYERS, ROUNDS, rng)
w_up = player_winnings(UP, up_pool, N_PLAYERS, ROUNDS, rng)
w_all = w_ip + w_up
total_bets = N_PLAYERS * ROUNDS
rep_ip, rep_up = [], []
for r in range(40):
    rr = np.random.default_rng(SEED + 100 + r)
    rep_ip.append(player_winnings(IP, ip_pool, N_PLAYERS, ROUNDS, rr).mean() / ROUNDS)
    rep_up.append(player_winnings(UP, up_pool, N_PLAYERS, ROUNDS, rr).mean() / ROUNDS)
rep_ip, rep_up = np.array(rep_ip), np.array(rep_up)
res["players"] = {
    "total_bets": total_bets,
    "system_expected_UP_hits": total_bets * UP.p, "system_expected_IP_hits": total_bets * IP.p,
    "p_player_wins_IP_theory": float(1 - (1 - IP.p) ** ROUNDS),
    "p_player_wins_IP_sim": float((w_ip > 0).mean()),
    "p_player_wins_UP_theory": float(1 - (1 - UP.p) ** ROUNDS),
    "p_player_wins_UP_sim": float((w_up > 0).mean()),
    "up_winners_in_sample": int((w_up > 0).sum()),
    "mean_jackpot_rtp_per_player": float(w_all.mean() / ROUNDS), "theory_rtp": IP.c + UP.c,
    "median_winnings": float(np.median(w_all)), "p90_winnings": float(np.quantile(w_all, 0.9)),
    "p99_winnings": float(np.quantile(w_all, 0.99)),
    "top1pct_share_of_all_jackpot_money": top_share(w_all, 0.01),
    "top1pct_share_UP_only": top_share(w_up, 0.01),
    "share_players_with_no_jackpot": float((w_all == 0).mean()),
    "repeat40": {"IP_rtp_mean": float(rep_ip.mean()), "IP_rtp_sd": float(rep_ip.std()),
                 "UP_rtp_mean": float(rep_up.mean()), "UP_rtp_sd": float(rep_up.std()),
                 "total_rtp_sd_of_single_run": float((rep_ip + rep_up).std())},
}
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))

# ---------------- 圖 ----------------
fig, ax = plt.subplots(2, 2, figsize=(13, 9.2))

a = ax[0, 0]
gx = [r["G_frac"] * 100 for r in gt]
a.plot(gx, [r["q0.99"] / base_q["q0.99"] * 100 for r in gt], "o-", color="#dc2626",
       label="big prizes (every quantile above G) vs no guarantee")
a.set_xlabel("guaranteed minimum prize G (% of mean prize)"); a.set_ylabel("big prize vs no-guarantee (%)")
a.set_title("(A) Cost of a guarantee: big prizes shrink by the reserve share"); a.legend(loc="lower left")
a2 = a.twinx(); a2.plot(gx, [r["p_floor_binds"] * 100 for r in gt], "s--", color="#6b7280")
a2.set_ylabel("jackpots actually topped up by the guarantee (%)", color="#6b7280")

a = ax[0, 1]
cols = {0.0: "#9ca3af", 0.3: "#93c5fd", 0.5: "#2563eb", 0.7: "#1e3a8a"}
for kappa, x in carry_prizes.items():
    mu = IP.c / IP.p
    s_ = np.sort(x / mu)
    a.plot(s_, np.linspace(0, 1, len(s_)), color=cols[kappa], label=f"carry {kappa:.0%} (CV {x.std()/x.mean():.2f})")
a.set_xlim(0, 4); a.set_xlabel("instant-pool prize / mean prize"); a.set_ylabel("cumulative probability")
a.set_title("(B) Rollover smooths the instant pool"); a.legend(loc="lower right")

a = ax[1, 0]
kk = np.linspace(0, 5, 51)
a.plot(kk, [BASE_RTP + other + hunter_jackpot_rtp(UP.c, UP.p, k) for k in kk], color="#2563eb",
       label="selective player: total RTP (analytic)")
a.plot(ks, total_h, "o", color="#2563eb")
a.axhline(1.0, color="#dc2626", ls="--", label="100% RTP"); a.axhline(0.96, color="#6b7280", ls=":", label="advertised 96%")
a.set_xlabel("play only when ultimate pool ≥ k × mean prize"); a.set_ylabel("total RTP")
a.set_title(f"(C) Visible pool is exploitable: 100% RTP at k ≈ {breakeven_k:.1f}"); a.legend(loc="upper left")

a = ax[1, 1]
win = w_all[w_all > 0]
a.hist(np.log10(win), bins=40, color="#93c5fd", edgecolor="white")
a.set_xlabel("log10(total jackpot winnings over 300 rounds), winners only"); a.set_ylabel("players")
pz = res["players"]
a.set_title(f"(D) Player view: {pz['share_players_with_no_jackpot']*100:.0f}% win nothing; top 1% take {pz['top1pct_share_of_all_jackpot_money']*100:.0f}%")
a.text(0.98, 0.95, f"{pz['up_winners_in_sample']} players hit the ultimate pool\n(far right, log10 > 3)", transform=a.transAxes, ha="right", va="top", color="#6b7280")
fig.tight_layout(); fig.savefig(OUT / "jackpot_overview.png", dpi=140); plt.close(fig)

# ---------------- 文字摘要 ----------------
print("A. 保底（終極池，平均獎金 %.0f）" % UP.mean_prize)
for r in gt:
    print(f"   G={r['G_frac']:>4.0%}μ  準備金抽 {r['s']*100:5.2f}%  保底生效 {r['p_floor_binds']*100:4.1f}%  "
          f"中位 {r['q0.5']:.0f}  P99 {r['q0.99']:.0f} ({r['q0.99']/base_q['q0.99']*100:.1f}%)  P99.9 {r['q0.999']:.0f} ({r['q0.999']/base_q['q0.999']*100:.1f}%)")
print("   基礎獎金 S=%.0f：無 seed 時 %.1f%% 的獎金低於 S；營運成本 %.3f 個百分點 RTP" % (S, p_small_no_seed * 100, S * UP.p * 100))
print("B. 滾存")
for r in carry_rows:
    print(f"   carry {r['carry']:.0%}: mean {r['mean']:.2f} (理論 {r['theory_mean']:.2f})  CV {r['cv']:.3f}  P(<¼均) {r['p_below_quarter_mean']*100:.1f}%  P99/均 {r['p99_over_mean']:.2f}")
print("C. 看獎池玩家 total RTP:", dict(zip(ks.tolist(), np.round(total_h, 4).tolist())), " 損益兩平 k≈%.2f" % breakeven_k)
print("   逐注模擬驗證（p=1/500）解析 vs 模擬:", np.round(rtp_ana_sim * 100, 3).tolist(), np.round(rtp_sim * 100, 3).tolist())
print("D. 重複 40 次（單次模擬的雜訊）：", {k: round(v*100,3) for k, v in res["players"]["repeat40"].items()}, "（單位：個百分點 RTP）")
print("D. 玩家視角：", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in pz.items()}, ensure_ascii=False))
