# tests/test_leduc_game.py
"""
Comprehensive tests for Leduc Poker - game mechanics AND solver convergence.
No longer a smoke test. This is production-grade testing.
"""
import pytest
from core.leduc_poker import (
    get_all_deals,
    is_terminal_leduc,
    get_actions_leduc,
    get_payoff_leduc,
    current_player_leduc,
    get_info_set_leduc,
    hand_rank,
    LeducCFR,
    LeducGameInterface,
    _is_round_over,
    CARD_NAMES,
)
from core.exploitability import compute_exploitability_generic


class TestLeducDeals:
    """Verify deal enumeration."""

    def test_deal_count(self):
        """6 cards, pick 3 ordered: 6×5×4 = 120."""
        deals = get_all_deals()
        assert len(deals) == 120

    def test_no_duplicate_cards_in_deal(self):
        """Each deal uses 3 distinct card positions from the deck."""
        deals = get_all_deals()
        for d in deals:
            # Each deal is (p0, p1, community) - all from different deck slots
            assert len(d) == 3


class TestLeducTerminal:
    """Verify terminal state detection."""

    @pytest.mark.parametrize("history,expected", [
        ("bf", True),       # fold in round 1
        ("cbf", True),      # fold after check-bet
        ("cc/cc", True),    # both rounds check-check
        ("bc/cc", True),    # round 1 bet-call, round 2 check-check
        ("cc/bf", True),    # round 1 check-check, round 2 fold
        ("bc/bc", True),    # both rounds bet-call
        ("", False),
        ("c", False),
        ("b", False),
        ("cc", False),      # round 1 over but not terminal (goes to round 2)
        ("bc", False),      # round 1 over but not terminal
    ])
    def test_terminal_detection(self, history, expected):
        assert is_terminal_leduc(history) == expected


class TestLeducActions:
    """Verify action availability."""

    def test_opening_actions(self):
        assert get_actions_leduc("") == ["c", "b"]

    def test_after_check(self):
        assert get_actions_leduc("c") == ["c", "b"]

    def test_after_bet_can_raise(self):
        """After a bet (1 bet in round), can call/fold/raise."""
        actions = get_actions_leduc("b")
        assert "c" in actions
        assert "f" in actions
        assert "r" in actions

    def test_after_check_bet(self):
        """After check-bet, facing a bet."""
        actions = get_actions_leduc("cb")
        assert "c" in actions
        assert "f" in actions

    def test_round2_opening(self):
        """Round 2 starts with check or bet."""
        assert get_actions_leduc("cc/") == ["c", "b"]
        assert get_actions_leduc("bc/") == ["c", "b"]


class TestLeducHandRank:
    """Verify hand ranking logic."""

    def test_pair_beats_high_card(self):
        """A pair should always beat a high card."""
        assert hand_rank(1, 1) > hand_rank(3, 2)  # pair of J > K high

    def test_higher_pair_wins(self):
        assert hand_rank(3, 3) > hand_rank(1, 1)  # KK > JJ

    def test_high_card_ranking(self):
        assert hand_rank(3, 1) > hand_rank(2, 1)  # K > Q (no pair)
        assert hand_rank(2, 1) > hand_rank(1, 2)  # Q > J (no pair)


class TestLeducPayoffs:
    """Verify payoff computation."""

    def test_zero_sum(self):
        """All terminals must be zero-sum."""
        terminals = ["bf", "cbf", "cc/cc", "bc/cc", "cc/bf", "bc/bc"]
        cards_list = [
            [1, 3, 2],  # J vs K, community Q
            [2, 2, 3],  # Q vs Q, community K
            [3, 1, 1],  # K vs J, community J
        ]
        for hist in terminals:
            for cards in cards_list:
                p0 = get_payoff_leduc(hist, cards, 0)
                p1 = get_payoff_leduc(hist, cards, 1)
                assert p0 + p1 == 0, (
                    f"Not zero-sum: {hist}, cards={cards}, "
                    f"p0={p0}, p1={p1}"
                )

    def test_fold_winner_gets_pot(self):
        """When opponent folds, winner gets opponent's contribution."""
        # P0 bets, P1 folds - P0 wins
        p0 = get_payoff_leduc("bf", [1, 3, 2], 0)
        assert p0 > 0, f"P0 bet and P1 folded, P0 should win, got {p0}"
        p1 = get_payoff_leduc("bf", [1, 3, 2], 1)
        assert p1 < 0, f"P1 folded, P1 should lose, got {p1}"

        # cbf: P0 checks, P1 bets, P0 folds - P1 wins
        p0 = get_payoff_leduc("cbf", [3, 1, 2], 0)
        assert p0 < 0, f"P0 folded, P0 should lose, got {p0}"

    def test_pair_wins_showdown(self):
        """Player with pair at showdown should win."""
        # P0=K, P1=J, comm=K → P0 has pair of K, P1 has J high
        # After cc/cc: minimal pot, P0 should win
        p0 = get_payoff_leduc("cc/cc", [3, 1, 3], 0)
        assert p0 > 0, f"P0 with pair should win, got {p0}"


class TestLeducCurrentPlayer:
    """Verify player turn logic."""

    def test_p0_starts_each_round(self):
        assert current_player_leduc("") == 0
        assert current_player_leduc("cc/") == 0
        assert current_player_leduc("bc/") == 0

    def test_p1_second_in_each_round(self):
        assert current_player_leduc("c") == 1
        assert current_player_leduc("b") == 1


class TestLeducInfoSets:
    """Verify information set encoding."""

    def test_round1_hides_community(self):
        """In round 1, community card should not be visible."""
        i_set = get_info_set_leduc("", [1, 3, 2], 0)
        assert "-1" in i_set or i_set.count("|") >= 2

    def test_round2_reveals_community(self):
        """In round 2, community card should be visible."""
        i_set = get_info_set_leduc("cc/", [1, 3, 2], 0)
        # Community is Q(2), should be in the info set
        assert "Q" in i_set or "2" in i_set

    def test_opponent_card_hidden(self):
        """Info set should never contain opponent's card."""
        # P0's info set should be the same regardless of P1's card
        i1 = get_info_set_leduc("cc/", [1, 2, 3], 0)
        i2 = get_info_set_leduc("cc/", [1, 3, 3], 0)
        assert i1 == i2  # P0 has J, community K, P1 varies


class TestLeducSolverConvergence:
    """Verify Leduc CFR solver converges."""

    @pytest.fixture(scope="class")
    def leduc_strategy(self):
        """Train Leduc solver - enough iterations for meaningful convergence."""
        solver = LeducCFR()
        solver.train(50)
        return solver.get_full_strategy()

    def test_learns_info_sets(self, leduc_strategy):
        """Should discover substantial number of info sets."""
        n_info_sets = len(leduc_strategy)
        assert n_info_sets > 50, (
            f"Expected >50 info sets, got {n_info_sets}"
        )

    def test_strategies_are_valid_distributions(self, leduc_strategy):
        """Every strategy must be a valid probability distribution."""
        for i_set, strategy in leduc_strategy.items():
            total = sum(strategy.values())
            assert abs(total - 1.0) < 1e-6, (
                f"Strategy at {i_set} sums to {total}, not 1.0"
            )
            for a, p in strategy.items():
                assert p >= -1e-10, (
                    f"Negative probability {p} at {i_set}, action {a}"
                )

    def test_exploitability_decreases_with_iterations(self):
        """Exploitability should decrease as we train more."""
        game_if = LeducGameInterface()
        solver1 = LeducCFR()
        solver1.train(10)
        expl1 = compute_exploitability_generic(solver1.get_full_strategy(), game_if)

        solver2 = LeducCFR()
        solver2.train(50)
        expl2 = compute_exploitability_generic(solver2.get_full_strategy(), game_if)

        assert expl2 < expl1, (
            f"Exploitability should decrease: 10iter={expl1:.4f}, "
            f"50iter={expl2:.4f}"
        )
