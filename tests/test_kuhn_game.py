# tests/test_kuhn_game.py
"""
Unit tests for Kuhn Poker game mechanics.
Uses pytest - no if __name__ == "__main__" blocks.
"""
import pytest
from core.kuhn_poker import (
    is_terminal, get_actions, current_player,
    get_payoff, get_info_set, CARDS, CARD_NAMES
)


class TestTerminalDetection:
    """Verify all terminal/non-terminal history classifications."""

    @pytest.mark.parametrize("history", ["bc", "bf", "cc", "cbc", "cbf"])
    def test_terminal_states(self, history):
        assert is_terminal(history) is True

    @pytest.mark.parametrize("history", ["", "c", "b", "cb"])
    def test_non_terminal_states(self, history):
        assert is_terminal(history) is False


class TestPayoffs:
    """Verify payoffs for every terminal history."""

    def test_bet_call_higher_card_wins(self):
        """P0 has K(3), P1 has J(1). P0 bets, P1 calls. P0 wins 2."""
        assert get_payoff("bc", [3, 1], 0) == 2
        assert get_payoff("bc", [3, 1], 1) == -2

    def test_bet_call_lower_card_wins(self):
        """P0 has J(1), P1 has K(3). P0 bets, P1 calls. P1 wins 2."""
        assert get_payoff("bc", [1, 3], 0) == -2
        assert get_payoff("bc", [1, 3], 1) == 2

    def test_bet_fold(self):
        """P0 bets, P1 folds. P0 wins 1 regardless of cards."""
        assert get_payoff("bf", [1, 3], 0) == 1
        assert get_payoff("bf", [1, 3], 1) == -1
        assert get_payoff("bf", [3, 1], 0) == 1  # wins even without best card

    def test_check_check_showdown(self):
        """Both check, showdown. Higher card wins 1."""
        assert get_payoff("cc", [1, 2], 0) == -1
        assert get_payoff("cc", [1, 2], 1) == 1
        assert get_payoff("cc", [3, 1], 0) == 1
        assert get_payoff("cc", [3, 1], 1) == -1

    def test_check_bet_call(self):
        """P0 checks, P1 bets, P0 calls. Showdown for pot of 4."""
        assert get_payoff("cbc", [3, 1], 0) == 2
        assert get_payoff("cbc", [3, 1], 1) == -2
        assert get_payoff("cbc", [1, 3], 0) == -2
        assert get_payoff("cbc", [1, 3], 1) == 2

    def test_check_bet_fold(self):
        """P0 checks, P1 bets, P0 folds. P1 wins 1."""
        assert get_payoff("cbf", [3, 1], 0) == -1
        assert get_payoff("cbf", [3, 1], 1) == 1

    def test_zero_sum_all_terminals(self):
        """Every terminal must satisfy P0_payoff + P1_payoff == 0."""
        for hist in ["bc", "bf", "cc", "cbc", "cbf"]:
            for c0 in CARDS:
                for c1 in CARDS:
                    if c0 == c1:
                        continue
                    p0 = get_payoff(hist, [c0, c1], 0)
                    p1 = get_payoff(hist, [c0, c1], 1)
                    assert p0 + p1 == 0, (
                        f"Not zero-sum: history={hist}, cards=[{c0},{c1}], "
                        f"payoffs=({p0}, {p1})"
                    )


class TestInfoSets:
    """Verify information set encoding."""

    def test_p0_opening(self):
        assert get_info_set("", [3, 2], 0) == "K:"
        assert get_info_set("", [1, 2], 0) == "J:"

    def test_p1_facing_bet(self):
        assert get_info_set("b", [3, 1], 1) == "J:b"
        assert get_info_set("b", [3, 2], 1) == "Q:b"

    def test_p0_facing_check_bet(self):
        assert get_info_set("cb", [1, 3], 0) == "J:cb"
        assert get_info_set("cb", [3, 1], 0) == "K:cb"

    def test_info_set_hides_opponent_card(self):
        """Info set should NOT depend on opponent's card."""
        i1 = get_info_set("b", [3, 1], 1)
        i2 = get_info_set("b", [3, 2], 1)
        # P1 has J in first case, Q in second - different info sets
        # But if P1 had the SAME card, info sets should be identical
        i3 = get_info_set("b", [2, 1], 1)
        i4 = get_info_set("b", [3, 1], 1)
        assert i3 == i4  # P1 has J:b regardless of P0's card


class TestCurrentPlayer:
    """Verify player turn assignment."""

    def test_initial_is_p0(self):
        assert current_player("") == 0

    def test_after_check_is_p1(self):
        assert current_player("c") == 1

    def test_after_bet_is_p1(self):
        assert current_player("b") == 1

    def test_after_check_bet_is_p0(self):
        assert current_player("cb") == 0


class TestActions:
    """Verify action availability at each decision point."""

    def test_opening(self):
        assert get_actions("") == ["c", "b"]

    def test_after_check(self):
        assert get_actions("c") == ["c", "b"]

    def test_after_bet(self):
        assert get_actions("b") == ["c", "f"]

    def test_after_check_bet(self):
        assert get_actions("cb") == ["c", "f"]

    def test_no_actions_at_terminal(self):
        for hist in ["bc", "bf", "cc", "cbc", "cbf"]:
            assert get_actions(hist) == []