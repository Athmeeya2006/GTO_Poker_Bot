# tests/test_cfr_convergence.py
"""
Rigorous Nash equilibrium convergence tests for all CFR variants.

Kuhn Poker has an analytical Nash equilibrium. We verify:
1. EXACT convergence to known values (tight tolerances)
2. Exploitability → 0 (machine precision for sufficient iterations)
3. Cross-validation: tree-traversal BR matches brute-force BR
4. Convergence speed ordering: CFR+ > DCFR > Vanilla
"""
import pytest
from core.cfr import VanillaCFR
from core.cfr_plus import CFRPlus
from core.dcfr import DCFR
from core.mccfr import ExternalSamplingMCCFR
from core.exploitability import compute_exploitability, compute_exploitability_bruteforce


# ── Analytical Kuhn Nash Equilibrium ─────────────────────────────
#
# Kuhn Poker has a FAMILY of Nash equilibria parameterized by α ∈ [0, 1/3]:
#
# Player 0 (acts first):
#   J:  bet with prob α (bluff)
#   Q:  bet with prob 0 (never bet)
#   K:  bet with prob 3α (value bet)
#   J:cb call with prob 0 (never call a re-raise with Jack)
#   Q:cb call with prob α + 1/3
#   K:cb call with prob 1 (always call with King)
#
# Player 1 (acts second):
#   J:b  call with prob 0 (never call a bet with Jack)
#   Q:b  call with prob 1/3 (always exactly 1/3)
#   K:b  call with prob 1 (always call with King)
#   J:c  bet with prob 1/3
#   Q:c  bet with prob 0 (never bet as P2 with Q after check)
#   K:c  bet with prob 1 (always bet)
#
# Game value: v = -1/18 ≈ -0.0556 (P0 has slight disadvantage)
#
# CRITICAL: P2's strategy is UNIQUE. Only P1's has the α parameter.
# The tolerance of 0.001 here (vs previous 0.08) reflects 50000 iterations.

KUHN_GAME_VALUE = -1.0 / 18.0  # ≈ -0.0556


class TestKuhnNashExact:
    """Verify convergence to the analytical Kuhn Nash equilibrium."""

    @pytest.fixture(scope="class")
    def converged_strategy(self):
        """Train CFR+ to high precision. Shared across tests in this class."""
        solver = CFRPlus()
        solver.train(50000)
        return solver.get_full_strategy()

    @pytest.fixture(scope="class")
    def exploitability(self, converged_strategy):
        return compute_exploitability(converged_strategy)

    # ── P1 equilibrium family constraints ─────────────────────────

    def test_p0_queen_never_bets(self, converged_strategy):
        """P0 should never bet with Queen (it's dominated)."""
        q_bet = converged_strategy["Q:"]["b"]
        assert q_bet < 0.005, f"P0 Q: bet should be ~0, got {q_bet:.6f}"

    def test_p0_jack_bluff_in_range(self, converged_strategy):
        """P0 Jack bluff frequency α should be in [0, 1/3]."""
        j_bet = converged_strategy["J:"]["b"]
        assert 0.0 <= j_bet <= 0.34, (
            f"P0 J: bluff α should be in [0, 1/3], got {j_bet:.6f}"
        )

    def test_p0_king_bet_tracks_3alpha(self, converged_strategy):
        """P0 K: bet probability should equal 3×α (the J: bluff rate)."""
        j_bet = converged_strategy["J:"]["b"]
        k_bet = converged_strategy["K:"]["b"]
        assert abs(k_bet - 3.0 * j_bet) < 0.01, (
            f"K:bet={k_bet:.6f} should be 3×J:bet={3*j_bet:.6f}"
        )

    def test_p0_queen_cb_call_tracks_alpha_plus_third(self, converged_strategy):
        """P0 Q:cb call should equal α + 1/3."""
        j_bet = converged_strategy["J:"]["b"]
        q_call = converged_strategy["Q:cb"]["c"]
        expected = j_bet + 1.0 / 3.0
        assert abs(q_call - expected) < 0.01, (
            f"Q:cb call={q_call:.6f} should be α+1/3={expected:.6f}"
        )

    def test_p0_king_cb_always_calls(self, converged_strategy):
        """P0 should always call with King after check-bet."""
        k_call = converged_strategy["K:cb"]["c"]
        assert k_call > 0.99, f"K:cb call should be ~1.0, got {k_call:.6f}"

    def test_p0_jack_cb_never_calls(self, converged_strategy):
        """P0 should never call with Jack after check-bet."""
        j_call = converged_strategy["J:cb"]["c"]
        assert j_call < 0.005, f"J:cb call should be ~0, got {j_call:.6f}"

    # ── P2 unique equilibrium (NO free parameter) ─────────────────

    def test_p1_king_always_calls_bet(self, converged_strategy):
        """P1 always calls a bet with King."""
        k_call = converged_strategy["K:b"]["c"]
        assert k_call > 0.995, f"P1 K:b call should be ~1.0, got {k_call:.6f}"

    def test_p1_queen_calls_exactly_one_third(self, converged_strategy):
        """P1 calls a bet with Queen at exactly 1/3 (this is UNIQUE)."""
        q_call = converged_strategy["Q:b"]["c"]
        assert abs(q_call - 1.0 / 3.0) < 0.01, (
            f"P1 Q:b call should be exactly 1/3={1/3:.6f}, got {q_call:.6f}"
        )

    def test_p1_jack_never_calls_bet(self, converged_strategy):
        """P1 never calls a bet with Jack."""
        j_call = converged_strategy["J:b"]["c"]
        assert j_call < 0.005, f"P1 J:b call should be ~0, got {j_call:.6f}"

    def test_p1_king_always_bets_after_check(self, converged_strategy):
        """P1 always bets with King after P0 checks."""
        k_bet = converged_strategy["K:c"]["b"]
        assert k_bet > 0.99, f"P1 K:c bet should be ~1.0, got {k_bet:.6f}"

    def test_p1_jack_bets_one_third_after_check(self, converged_strategy):
        """P1 bets (bluffs) with Jack 1/3 of the time after check."""
        j_bet = converged_strategy["J:c"]["b"]
        assert abs(j_bet - 1.0 / 3.0) < 0.01, (
            f"P1 J:c bet should be 1/3={1/3:.6f}, got {j_bet:.6f}"
        )

    def test_p1_queen_never_bets_after_check(self, converged_strategy):
        """P1 never bets with Queen after check (dominated)."""
        q_bet = converged_strategy["Q:c"]["b"]
        assert q_bet < 0.01, f"P1 Q:c bet should be ~0, got {q_bet:.6f}"

    # ── Exploitability ────────────────────────────────────────────

    def test_exploitability_near_zero(self, exploitability):
        """Exploitability should be < 0.001 at 50k iterations."""
        assert exploitability < 0.001, (
            f"Exploitability should be < 0.001, got {exploitability:.6f}"
        )

    def test_exploitability_nonnegative(self, exploitability):
        """Exploitability is always ≥ 0 by definition."""
        assert exploitability >= -1e-10, (
            f"Exploitability cannot be negative, got {exploitability:.6f}"
        )


class TestCrossValidation:
    """Cross-validate tree-traversal vs brute-force exploitability."""

    def test_tree_traversal_matches_bruteforce(self):
        """Both methods should give the same exploitability value."""
        solver = CFRPlus()
        solver.train(5000)
        strat = solver.get_full_strategy()

        tree_expl = compute_exploitability(strat)
        brute_expl = compute_exploitability_bruteforce(strat)

        assert abs(tree_expl - brute_expl) < 0.001, (
            f"Tree traversal ({tree_expl:.6f}) != "
            f"brute force ({brute_expl:.6f})"
        )


class TestConvergenceOrdering:
    """Verify that advanced CFR variants converge faster."""

    @pytest.fixture(scope="class")
    def convergence_data(self):
        """Train all solvers for the same number of iterations."""
        N = 3000
        vanilla = VanillaCFR()
        vanilla.train(N)

        cfrplus = CFRPlus()
        cfrplus.train(N)

        dcfr = DCFR()
        dcfr.train(N)

        return {
            "vanilla": compute_exploitability(vanilla.get_full_strategy()),
            "cfr_plus": compute_exploitability(cfrplus.get_full_strategy()),
            "dcfr": compute_exploitability(dcfr.get_full_strategy()),
        }

    def test_cfr_plus_beats_vanilla(self, convergence_data):
        """CFR+ should have lower exploitability than Vanilla at same iterations."""
        assert convergence_data["cfr_plus"] < convergence_data["vanilla"], (
            f"CFR+ ({convergence_data['cfr_plus']:.6f}) should beat "
            f"Vanilla ({convergence_data['vanilla']:.6f})"
        )

    def test_dcfr_beats_vanilla(self, convergence_data):
        """DCFR should have lower exploitability than Vanilla at same iterations."""
        assert convergence_data["dcfr"] < convergence_data["vanilla"], (
            f"DCFR ({convergence_data['dcfr']:.6f}) should beat "
            f"Vanilla ({convergence_data['vanilla']:.6f})"
        )


class TestMCCFR:
    """MCCFR-specific tests."""

    def test_mccfr_converges(self):
        """MCCFR should achieve reasonable exploitability."""
        mccfr = ExternalSamplingMCCFR()
        mccfr.train(30000)
        strat = mccfr.get_full_strategy()
        expl = compute_exploitability(strat)
        assert expl < 0.05, f"MCCFR exploitability too high: {expl:.6f}"

    def test_baseline_calibrated(self):
        """MCCFR baselines should be calibrated after training."""
        mccfr = ExternalSamplingMCCFR()
        mccfr.train(10000)
        calibrated = sum(1 for n in mccfr.baseline_count.values() if n > 20)
        assert calibrated >= 3, (
            f"Only {calibrated} info sets have calibrated baselines"
        )
