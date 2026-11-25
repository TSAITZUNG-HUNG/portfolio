"""
自動化驗證：兩趟產出（配平趟 vs 產出趟）逐位元比對 ＋ 一組不變式檢查。

為什麼要兩趟：大樣本時，把每一轉的完整事件（盤面、贏分、結算…）全留在記憶體
會跑不動。所以「配平趟」只算賠付數值，「產出趟」才建立完整 books。
危險在於：兩趟若走不同的路，產出的 books 就可能和配平時的數字對不上。
解法是**單一路徑**：兩趟呼叫同一個 `play()`，只差一個旗標，再用逐位元比對把關。

關鍵設計：
1. 每一本 book 用 (seed, book_id) 獨立播種 → 結果與計算順序無關、可重現。
2. 事件產生**不得消耗亂數**（純函式）；否則兩趟的亂數序列會發散。
3. 驗證器只看**輸出**（books、查找表、賠付表），不 import 遊戲引擎 —— 避免
   「引擎寫錯、斷言跟著錯」的同源偏誤。
"""
from __future__ import annotations

import numpy as np

from .toy_slot import (BLANK, N_CELLS, SYMBOLS, Game, MAX_WIN_X100,
                       pay_table_x100, payout_from_counts)

SYMBOL_NAMES = SYMBOLS + ["-"]          # 索引 0..5 = A..F，6 = 空白


def play(game: Game, seed: int, book_id: int, emit_events: bool, *, leaky_emit: bool = False):
    """
    玩一轉。回傳 (payout_x100, events 或 None)。
    leaky_emit=True 是「故意壞掉的版本」：產出趟多消耗一次亂數，用來測驗證器擋得住。
    """
    rng = np.random.default_rng([seed, book_id])
    cells = rng.choice(len(game.probs), size=N_CELLS, p=np.asarray(game.probs))
    counts = np.bincount(cells, minlength=len(game.probs))
    payout = int(payout_from_counts(counts, pay_table_x100(game)))
    if not emit_events:
        return payout, None
    if leaky_emit:
        rng.random()                                    # ← 錯誤：事件產生不該動亂數
        cells = rng.choice(len(game.probs), size=N_CELLS, p=np.asarray(game.probs))
        counts = np.bincount(cells, minlength=len(game.probs))
        payout = int(payout_from_counts(counts, pay_table_x100(game)))
    events = [
        {"type": "reveal", "grid": [SYMBOL_NAMES[c] for c in cells]},
        {"type": "win", "amount": payout},
        {"type": "finalWin", "amount": payout},
    ]
    return payout, events


def balance_pass(game: Game, seed: int, n: int) -> np.ndarray:
    """配平趟：只算賠付數值，不建 books。"""
    return np.array([play(game, seed, i, False)[0] for i in range(n)], dtype=np.int64)


def emit_pass(game: Game, seed: int, n: int, *, leaky_emit: bool = False):
    """產出趟：建立完整 books（每本一個 dict）。"""
    books = []
    for i in range(n):
        payout, events = play(game, seed, i, True, leaky_emit=leaky_emit)
        books.append({"id": i, "payoutMultiplier": payout, "events": events})
    return books


# ------------------------------ 檢查（只讀輸出） ------------------------------

class VerificationError(AssertionError):
    pass


def check_two_pass_bitwise(balance_payouts: np.ndarray, books: list) -> None:
    """配平趟與產出趟的賠付必須逐位元相同。"""
    emitted = np.array([b["payoutMultiplier"] for b in books], dtype=np.int64)
    if emitted.shape != balance_payouts.shape:
        raise VerificationError(f"本數不同：{balance_payouts.shape} vs {emitted.shape}")
    bad = np.flatnonzero(emitted != balance_payouts)
    if bad.size:
        raise VerificationError(f"兩趟賠付不同：{bad.size} 本（例：id={int(bad[0])}）")


def recompute_payout_from_grid(grid: list, pay_table: np.ndarray) -> int:
    """從「玩家畫面上看得到的盤面」依公布的賠付表重算賠付（可見＝實付）。"""
    total = 0
    for k, name in enumerate(SYMBOLS):
        c = sum(1 for g in grid if g == name)
        total += int(pay_table[k][c])
    return min(total, MAX_WIN_X100)


def check_visible_equals_paid(books: list, pay_table: np.ndarray) -> None:
    """賠付表 × 盤面上每個符號 == 入帳金額。"""
    for b in books:
        grid = next(e["grid"] for e in b["events"] if e["type"] == "reveal")
        expect = recompute_payout_from_grid(grid, pay_table)
        if expect != b["payoutMultiplier"]:
            raise VerificationError(f"可見≠實付：book {b['id']} 盤面算出 {expect}，入帳 {b['payoutMultiplier']}")


def check_event_consistency(books: list) -> None:
    """finalWin 金額 == payoutMultiplier；事件順序固定為 reveal → win → finalWin。"""
    for b in books:
        types = [e["type"] for e in b["events"]]
        if types != ["reveal", "win", "finalWin"]:
            raise VerificationError(f"book {b['id']} 事件順序異常：{types}")
        if b["events"][-1]["amount"] != b["payoutMultiplier"]:
            raise VerificationError(f"book {b['id']} finalWin≠payoutMultiplier")


def check_lookup_matches_books(lookup: list[tuple[int, int, int]], books: list) -> None:
    """查找表 (id, weight, payout) 的第 3 欄須與 books 同 id 逐筆一致，權重為正整數。"""
    by_id = {b["id"]: b["payoutMultiplier"] for b in books}
    if len(lookup) != len(books):
        raise VerificationError("查找表與 books 筆數不同")
    for i, w, pay in lookup:
        if not (isinstance(w, int) and w >= 1):
            raise VerificationError(f"id {i} 權重非正整數：{w}")
        if by_id.get(i) != pay:
            raise VerificationError(f"id {i} 查找表賠付 {pay} ≠ books {by_id.get(i)}")


def lookup_rtp(lookup: list[tuple[int, int, int]]) -> float:
    """RTP = Σ(payout × weight) / Σ(weight) / 100。"""
    tw = sum(w for _, w, _ in lookup)
    return sum(w * pay for _, w, pay in lookup) / tw / 100.0


def check_rtp_identity(lookup, claimed_rtp: float, tol: float = 1e-9) -> None:
    got = lookup_rtp(lookup)
    if abs(got - claimed_rtp) > tol:
        raise VerificationError(f"RTP 恆等式不成立：查找表算出 {got:.9f}，宣稱 {claimed_rtp:.9f}")
