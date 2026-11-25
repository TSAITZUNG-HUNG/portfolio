"""示範三：兩趟產出逐位元比對 ＋ 不變式檢查 ＋ 突變測試。

流程：配平趟（只算賠付）→ 產出趟（建完整 books）→ 逐位元比對 → 其他不變式。
然後故意弄壞 5 種東西，示範每一種都被擋下。

執行：python demos/03_automated_verification/run.py
"""
import copy
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from slotlab import verify as V  # noqa: E402
from slotlab.toy_slot import Game, pay_table_x100  # noqa: E402

OUT = pathlib.Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
GAME, SEED, N = Game(), 20261125, 5_000
TABLE = pay_table_x100(GAME)

bal = V.balance_pass(GAME, SEED, N)
books = V.emit_pass(GAME, SEED, N)
rng = np.random.default_rng(1)
lookup = [(b["id"], int(w), b["payoutMultiplier"]) for b, w in zip(books, rng.integers(1, 10_000, N))]
claimed = V.lookup_rtp(lookup)

report = {"n_books": N, "clean": {}, "mutations": {}}

def run(name, fn, store):
    try:
        fn(); store[name] = "PASS"
    except V.VerificationError as e:
        store[name] = f"FAIL: {e}"

print(f"== 乾淨資料（{N:,} 本）：全部檢查應 PASS ==")
checks = {
    "兩趟逐位元相同": lambda: V.check_two_pass_bitwise(bal, books),
    "事件順序與 finalWin": lambda: V.check_event_consistency(books),
    "可見＝實付（盤面×賠付表）": lambda: V.check_visible_equals_paid(books, TABLE),
    "查找表與 books 逐筆一致": lambda: V.check_lookup_matches_books(lookup, books),
    "RTP 恆等式": lambda: V.check_rtp_identity(lookup, claimed),
}
for k, f in checks.items():
    run(k, f, report["clean"]); print(f"  {k:<26} {report['clean'][k]}")
assert all(v == "PASS" for v in report["clean"].values())

print("\n== 突變：故意弄壞，每一項都必須 FAIL（被擋下） ==")
def mut_leaky():
    V.check_two_pass_bitwise(bal, V.emit_pass(GAME, SEED, N, leaky_emit=True))
def mut_tamper():
    b = copy.deepcopy(books); i = next(k for k, x in enumerate(b) if x["payoutMultiplier"] > 0)
    b[i]["payoutMultiplier"] += 100
    for e in b[i]["events"]:
        if "amount" in e: e["amount"] += 100
    V.check_visible_equals_paid(b, TABLE)
def mut_scale():
    b = copy.deepcopy(books)
    for x in b:
        x["payoutMultiplier"] = int(x["payoutMultiplier"] * 0.96)
        for e in x["events"]:
            if "amount" in e: e["amount"] = x["payoutMultiplier"]
    V.check_visible_equals_paid(b, TABLE)
def mut_lookup():
    l = list(lookup); l[10] = (l[10][0], l[10][1], l[10][2] + 1); V.check_lookup_matches_books(l, books)
def mut_rtp():
    V.check_rtp_identity(lookup, claimed + 0.0005)

for k, f in {
    "事件產生偷耗亂數（兩趟發散）": mut_leaky,
    "只改 book 金額、不改盤面（合成 book）": mut_tamper,
    "輸出端縮放係數 win×0.96": mut_scale,
    "查找表某筆賠付差 1": mut_lookup,
    "宣稱的 RTP 差 0.05 個百分點": mut_rtp,
}.items():
    run(k, f, report["mutations"])
    caught = report["mutations"][k].startswith("FAIL")
    print(f"  [{'擋下' if caught else '漏網!!'}] {k}\n         → {report['mutations'][k][:90]}")
assert all(v.startswith("FAIL") for v in report["mutations"].values()), "有突變漏網"

(OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
# 範例輸出：前 3 本 book（JSONL）與查找表前 3 列
with open(OUT / "sample_books.jsonl", "w") as f:
    for b in books[:3]:
        f.write(json.dumps(b, ensure_ascii=False) + "\n")
print("\n全部突變都被擋下；範例輸出寫入 results/")
