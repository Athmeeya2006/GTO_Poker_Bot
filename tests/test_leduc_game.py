import sys

sys.path.insert(0, ".")

from core.leduc_poker import (
    get_all_deals,
    is_terminal_leduc,
    get_actions_leduc,
    get_payoff_leduc,
    LeducCFR,
)


def test_leduc_deal_count():
    deals = get_all_deals()
    assert len(deals) == 120


def test_leduc_round_actions():
    assert get_actions_leduc("") == ["c", "b"]
    assert get_actions_leduc("c") == ["c", "b"]
    assert get_actions_leduc("b") == ["c", "f", "r"]
    assert get_actions_leduc("cb") == ["c", "f"]


def test_leduc_terminal_detection():
    assert is_terminal_leduc("bf")
    assert not is_terminal_leduc("")
    assert not is_terminal_leduc("c")


def test_leduc_payoff_zero_sum():
    cards = [1, 3, 2]
    for terminal_history in ("bf", "cc/cc", "bc/cc", "cc/bf"):
        p0 = get_payoff_leduc(terminal_history, cards, 0)
        p1 = get_payoff_leduc(terminal_history, cards, 1)
        assert p0 + p1 == 0


def test_leduc_cfr_smoke():
    solver = LeducCFR()
    solver.train(1)
    strat = solver.get_full_strategy()
    assert len(strat) > 0
