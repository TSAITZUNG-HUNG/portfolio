"""自動化驗證的測試 —— 包含「故意把東西弄壞，驗證器必須擋下」的突變測試。"""
import copy
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from slotlab import verify as V  # noqa: E402
from slotlab.toy_slot import Game, pay_table_x100  # noqa: E402

GAME, SEED, N = Game(), 7, 400
TABLE = pay_table_x100(GAME)


@pytest.fixture(scope="module")
def clean():
    bal = V.balance_pass(GAME, SEED, N)
    books = V.emit_pass(GAME, SEED, N)
    rng = np.random.default_rng(1)
    weights = rng.integers(1, 1000, size=N).tolist()
    lookup = [(b["id"], w, b["payoutMultiplier"]) for b, w in zip(books, weights)]
    return bal, books, lookup


def test_clean_run_passes_every_check(clean):
    bal, books, lookup = clean
    V.check_two_pass_bitwise(bal, books)
    V.check_event_consistency(books)
    V.check_visible_equals_paid(books, TABLE)
    V.check_lookup_matches_books(lookup, books)
    V.check_rtp_identity(lookup, V.lookup_rtp(lookup))


def test_per_book_seeding_is_order_independent():
    a = [V.play(GAME, SEED, i, False)[0] for i in range(50)]
    b = [V.play(GAME, SEED, i, False)[0] for i in reversed(range(50))][::-1]
    assert a == b


# ----------------------------- 突變測試：必須擋下 -----------------------------

def test_catches_event_generation_consuming_rng():
    """產出趟多耗一次亂數 → 兩趟發散 → 逐位元比對必須擋下。"""
    bal = V.balance_pass(GAME, SEED, N)
    leaky = V.emit_pass(GAME, SEED, N, leaky_emit=True)
    with pytest.raises(V.VerificationError, match="兩趟賠付不同"):
        V.check_two_pass_bitwise(bal, leaky)


def test_catches_tampered_payout_in_book(clean):
    """只改 book 的賠付、不改盤面（＝合成 book／偷改金額）→ 可見≠實付。"""
    _, books, _ = clean
    bad = copy.deepcopy(books)
    i = next(k for k, b in enumerate(bad) if b["payoutMultiplier"] > 0)
    bad[i]["payoutMultiplier"] += 100
    for e in bad[i]["events"]:
        if "amount" in e:
            e["amount"] += 100
    with pytest.raises(V.VerificationError, match="可見≠實付"):
        V.check_visible_equals_paid(bad, TABLE)


def test_catches_scaled_payout_everywhere(clean):
    """輸出端縮放係數（win×k）→ 全面違反可見＝實付。"""
    _, books, _ = clean
    bad = copy.deepcopy(books)
    for b in bad:
        b["payoutMultiplier"] = int(b["payoutMultiplier"] * 0.96)
        for e in b["events"]:
            if "amount" in e:
                e["amount"] = b["payoutMultiplier"]
    with pytest.raises(V.VerificationError):
        V.check_visible_equals_paid(bad, TABLE)


def test_catches_lookup_table_mismatch(clean):
    _, books, lookup = clean
    bad = list(lookup)
    bad[5] = (bad[5][0], bad[5][1], bad[5][2] + 1)
    with pytest.raises(V.VerificationError, match="查找表賠付"):
        V.check_lookup_matches_books(bad, books)


def test_catches_non_integer_or_zero_weight(clean):
    _, books, lookup = clean
    bad = list(lookup)
    bad[3] = (bad[3][0], 0, bad[3][2])
    with pytest.raises(V.VerificationError, match="正整數"):
        V.check_lookup_matches_books(bad, books)


def test_catches_wrong_rtp_claim(clean):
    _, _, lookup = clean
    with pytest.raises(V.VerificationError, match="RTP 恆等式"):
        V.check_rtp_identity(lookup, V.lookup_rtp(lookup) + 0.0005)


def test_catches_event_order_swap(clean):
    _, books, _ = clean
    bad = copy.deepcopy(books)
    bad[0]["events"][0], bad[0]["events"][1] = bad[0]["events"][1], bad[0]["events"][0]
    with pytest.raises(V.VerificationError, match="事件順序"):
        V.check_event_consistency(bad)
