# tests/test_trading.py
"""
Tests for the Glosten-Milgrom isomorphism module.
"""
import pytest
from trading.glosten_milgrom import GlostenMilgromModel
from core.cfr_plus import CFRPlus
from core.exploitability import compute_exploitability


class TestGMModelMechanics:
    """Verify Glosten-Milgrom model computations."""

    def test_spread_positive(self):
        """Bid-ask spread should always be positive with informed traders."""
        gm = GlostenMilgromModel(V_L=1, V_H=3, mu=0.5)
        assert gm.bid_ask_spread() > 0

    def test_ask_above_bid(self):
        """Ask price should always exceed bid price."""
        gm = GlostenMilgromModel(V_L=1, V_H=3, mu=0.5)
        assert gm.optimal_ask_price() > gm.optimal_bid_price()

    def test_spread_increases_with_mu(self):
        """Higher informed-trader fraction -> wider spread."""
        gm_low = GlostenMilgromModel(V_L=1, V_H=3, mu=0.1)
        gm_high = GlostenMilgromModel(V_L=1, V_H=3, mu=0.9)
        assert gm_high.bid_ask_spread() > gm_low.bid_ask_spread()

    def test_zero_mu_zero_spread(self):
        """With no informed traders, spread should be zero."""
        gm = GlostenMilgromModel(V_L=1, V_H=3, mu=0.0)
        assert abs(gm.bid_ask_spread()) < 0.01

    def test_ask_between_mid_and_high(self):
        """Ask price should be between V_mid and V_H."""
        gm = GlostenMilgromModel(V_L=1, V_H=3, mu=0.5)
        ask = gm.optimal_ask_price()
        assert gm.V_mid < ask < gm.V_H

    def test_bid_between_low_and_mid(self):
        """Bid price should be between V_L and V_mid."""
        gm = GlostenMilgromModel(V_L=1, V_H=3, mu=0.5)
        bid = gm.optimal_bid_price()
        assert gm.V_L < bid < gm.V_mid


class TestGMKuhnMapping:
    """Verify the Kuhn-GM isomorphism."""

    @pytest.fixture(scope="class")
    def cfr_data(self):
        """Train solver and extract equilibrium data."""
        solver = CFRPlus()
        solver.train(30000)
        strat = solver.get_full_strategy()
        bluff_freq = strat.get("J:", {}).get("b", 0.0)
        expl = compute_exploitability(strat)
        return {"strategy": strat, "bluff_freq": bluff_freq, "expl": expl}

    def test_bluff_frequency_in_nash_range(self, cfr_data):
        """Bluff frequency should be in [0, 1/3]."""
        bf = cfr_data["bluff_freq"]
        assert 0.0 <= bf <= 0.34, f"Bluff freq {bf} not in Nash range"

    def test_mu_alpha_roundtrip(self, cfr_data):
        """mu -> alpha -> mu should roundtrip consistently."""
        gm = GlostenMilgromModel(V_L=1, V_H=3, mu=0.6)
        alpha = gm.implied_bluff_frequency_from_mu()
        mu_back = gm.implied_mu_from_bluff_frequency(alpha)
        assert abs(mu_back - gm.mu) < 0.001, (
            f"Roundtrip failed: {gm.mu} -> {alpha} -> {mu_back}"
        )

    def test_zero_profit_at_nash(self, cfr_data):
        """MM breaks even at Nash bluff frequency (zero-profit condition)."""
        bf = cfr_data["bluff_freq"]
        gm = GlostenMilgromModel(V_L=1, V_H=3, mu=0.6)
        profit = gm.expected_profit_per_trade(bf)
        # At Nash alpha in [0, 1/3], the MM should approximately break even.
        # The exact zero-profit condition holds at alpha = 1/3 analytically.
        assert abs(profit) < 0.05, (
            f"MM should approximately break even at Nash, profit={profit:.6f}"
        )

    def test_informed_posterior_at_nash(self, cfr_data):
        """At Nash (alpha in [0, 1/3]), P(informed|bet) in [0.75, 1.0]."""
        bf = cfr_data["bluff_freq"]
        p_informed = 1.0 / (1.0 + bf)
        # Equilibrium family: alpha in [0, 1/3] -> P(informed|bet) in [0.75, 1.0]
        assert 0.74 <= p_informed <= 1.01, (
            f"P(informed|bet) = {p_informed:.4f}, expected in [0.75, 1.0]"
        )


class TestGMLimitations:
    """Tests documenting where the Kuhn-GM analogy breaks down."""

    def test_spread_scales_linearly_with_value_range(self):
        """Binary GM spread scales linearly with (V_H - V_L); real markets use integral expectations."""
        gm_narrow = GlostenMilgromModel(V_L=1, V_H=2, mu=0.5)
        gm_wide = GlostenMilgromModel(V_L=1, V_H=10, mu=0.5)

        ratio = gm_wide.bid_ask_spread() / gm_narrow.bid_ask_spread()
        expected_ratio = (10 - 1) / (2 - 1)
        assert abs(ratio - expected_ratio) < 0.1, (
            f"Binary-value spread should scale linearly with value range. "
            f"Got ratio {ratio:.2f}, expected {expected_ratio:.2f}"
        )

    def test_single_period_vs_dynamic(self):
        """Model is static; real markets have time-varying spreads and inventory management."""
        gm = GlostenMilgromModel(V_L=1, V_H=3, mu=0.5)
        spread = gm.bid_ask_spread()
        assert spread > 0  # just verify the static model works

    def test_no_inventory_risk(self):
        """Model assumes zero inventory risk; spread is constant regardless of order history."""
        gm = GlostenMilgromModel(V_L=1, V_H=3, mu=0.5)
        spread1 = gm.bid_ask_spread()
        spread2 = gm.bid_ask_spread()
        assert spread1 == spread2  # static model - documenting limitation

    def test_no_correlation_across_assets(self):
        """No multi-asset correlation; two independent models share no information."""
        gm1 = GlostenMilgromModel(V_L=1, V_H=3, mu=0.5)
        gm2 = GlostenMilgromModel(V_L=2, V_H=4, mu=0.5)
        # Two independent models - no cross-asset information leakage
        assert gm1.bid_ask_spread() == gm2.bid_ask_spread()  # same mu -> same spread
