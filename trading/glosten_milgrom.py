# trading/glosten_milgrom.py
"""
Glosten-Milgrom (1985) informed-trading model mapped to Kuhn Poker.

THE MAPPING (exact correspondence):

  Poker (Kuhn)                   | Financial Market (GM)
  -------------------------------|----------------------------------------
  Player 1 (private card)        | Informed trader (knows true value V)
  Player 2 (no private info)     | Market maker (doesn't know V)
  Cards: J=1, Q=2, K=3           | Asset value: Low=1, Mid=2, High=3
  Betting = submitting order     | Submitting a market BUY order
  Checking = not trading         | Not submitting an order
  Calling = filling the order    | Market maker accepts the order
  Folding = refusing             | Market maker widens spread / refuses
  Nash equilibrium strategy      | Zero-profit bid-ask spread

LIMITATIONS (explicitly documented):

1. DISCRETE vs CONTINUOUS VALUES: Kuhn has 3 discrete card values.
   Real markets have continuous price processes (e.g., geometric Brownian
   motion). The GM model here uses binary {V_L, V_H}, collapsing the
   three-card structure. With a continuous value distribution, the spread
   formula involves integral expectations, not discrete sums.

2. SINGLE PERIOD vs DYNAMIC: Kuhn is a one-shot game. Real markets are
   sequential - prices update after each trade, inventory accumulates,
   and the MM's posterior evolves. A proper dynamic extension would use
   the Kyle (1985) model or Glosten-Milgrom with sequential trades.

3. NO INVENTORY RISK: The toy model assumes the market maker is
   risk-neutral with infinite capital. Real MMs face inventory risk,
   funding costs, and correlation across assets. The spread in practice
   has both adverse selection AND inventory components (Amihud-Mendelson).

4. NO MULTI-ASSET CORRELATION: In real trading, information about one
   asset affects pricing of correlated assets. This model is single-asset.
   Cross-asset information leakage (e.g., ETF arbitrage) is not captured.

5. ADVERSE SELECTION ONLY: The model captures adverse selection (the
   informed trader's information advantage) but not other microstructure
   effects: order flow toxicity, maker-taker rebates, latency arbitrage,
   or market impact models (Almgren-Chriss).

This module is BOTH:
  - An educational demonstration of the isomorphism
  - A functional tool that uses CFR output to compute trading quantities
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.cfr_plus import CFRPlus
from core.exploitability import compute_exploitability


class GlostenMilgromModel:
    """
    Glosten-Milgrom model with Kuhn Poker isomorphism.

    This model is FUNCTIONAL - it uses CFR solver output to:
    1. Compute implied informed-trader fractions from equilibrium strategies
    2. Derive zero-profit bid-ask spreads
    3. Price assets under adverse selection
    4. Generate trading signals based on observed bluff frequencies
    """

    def __init__(self, V_L=1, V_H=3, mu=0.5):
        """
        V_L: low asset value
        V_H: high asset value
        mu:  fraction of informed traders (0 to 1)
        """
        self.V_L = V_L
        self.V_H = V_H
        self.mu = mu
        self.V_mid = (V_L + V_H) / 2

    def optimal_ask_price(self):
        """
        GM optimal ask price (market maker sells to buyer).

        Derivation:
          MM break-even condition:
            ask = P(informed|buy) × V_H + P(uninformed|buy) × V_mid
          where:
            P(buy) = μ × 1 + (1-μ) × 0.5
            P(informed|buy) = μ / P(buy)
        """
        p_buy_informed = 1.0
        p_buy_uninformed = 0.5
        p_buy = self.mu * p_buy_informed + (1 - self.mu) * p_buy_uninformed
        p_informed_given_buy = (self.mu * p_buy_informed) / p_buy if p_buy > 0 else 0
        return p_informed_given_buy * self.V_H + (1 - p_informed_given_buy) * self.V_mid

    def optimal_bid_price(self):
        """
        GM optimal bid price (market maker buys from seller).
        Symmetric: informed seller has V = V_L.
        """
        p_sell_informed = 1.0
        p_sell_uninformed = 0.5
        p_sell = self.mu * p_sell_informed + (1 - self.mu) * p_sell_uninformed
        p_informed_given_sell = (self.mu * p_sell_informed) / p_sell if p_sell > 0 else 0
        return p_informed_given_sell * self.V_L + (1 - p_informed_given_sell) * self.V_mid

    def bid_ask_spread(self):
        """Bid-ask spread = ask - bid. Represents adverse selection cost."""
        return self.optimal_ask_price() - self.optimal_bid_price()

    def adverse_selection_component(self):
        """
        Fraction of the spread attributable to adverse selection.
        In this model, 100% of the spread IS adverse selection.
        In real markets, spread = adverse_selection + inventory + fixed_costs.
        """
        return self.bid_ask_spread()

    def implied_mu_from_bluff_frequency(self, bluff_freq):
        """
        Map Kuhn bluff frequency α to implied informed-trader fraction μ.

        Derivation:
          P(informed|buy) from Kuhn: 1 / (1 + α)
          P(informed|buy) from GM:   2μ / (1 + μ)
          Setting equal:  μ = 1 / (1 + 2α)
        """
        alpha = max(0.0, min(bluff_freq, 1.0))
        return 1.0 / (1.0 + 2.0 * alpha)

    def implied_bluff_frequency_from_mu(self):
        """
        Inverse mapping: α = (1 - μ) / (2μ)
        """
        mu = max(min(self.mu, 1.0), 1e-10)
        return (1.0 - mu) / (2.0 * mu)

    def spread_from_bluff_frequency(self, bluff_freq):
        """
        THE KEY CONNECTION:
        P(informed|bet) = 1 / (1 + α)
        Spread = P(informed|bet) × (V_H - V_L)

        At Nash equilibrium (α = 1/3):
          P(informed|bet) = 3/4
          Spread = 3/4 × 2 = 1.5
        """
        p_informed = 1.0 / (1.0 + bluff_freq)
        return p_informed * (self.V_H - self.V_L)

    def expected_profit_per_trade(self, bluff_freq):
        """
        Expected market maker profit per trade given bluff frequency.

        At Nash equilibrium, this should be 0 (zero-profit condition).
        Away from Nash, MM profits or loses.
        """
        p_informed = 1.0 / (1.0 + bluff_freq)
        spread = self.spread_from_bluff_frequency(bluff_freq)

        # MM profit = spread/2 from uninformed - loss from informed
        profit_per_uninformed = spread / 2.0
        loss_per_informed = (self.V_H - self.V_L) / 2.0 - spread / 2.0

        return (1 - p_informed) * profit_per_uninformed - p_informed * loss_per_informed

    def optimal_position_size(self, bluff_freq, bankroll, max_risk_fraction=0.02):
        """
        Kelly-inspired position sizing based on detected edge.

        If bluff frequency deviates from Nash, there's a trading edge.
        Position size scales with edge strength and is bounded by
        max_risk_fraction of bankroll.

        Args:
            bluff_freq: observed bluff frequency
            bankroll: available capital
            max_risk_fraction: maximum fraction of bankroll to risk

        Returns:
            suggested position size in units
        """
        nash_alpha = 1.0 / 3.0
        edge = abs(bluff_freq - nash_alpha) / nash_alpha

        # Kelly fraction ≈ edge / variance
        # In this binary model, variance ≈ spread
        spread = self.spread_from_bluff_frequency(bluff_freq)
        if spread <= 0:
            return 0.0

        kelly_fraction = edge / spread
        # Half-Kelly for safety
        position = min(
            kelly_fraction * bankroll * 0.5,
            max_risk_fraction * bankroll
        )
        return max(position, 0.0)

    def verify_isomorphism(self, cfr_bluff_freq):
        """
        Check consistency between GM and Kuhn-derived spreads.
        Returns detailed diagnostics.
        """
        gm_spread = self.bid_ask_spread()
        poker_spread = self.spread_from_bluff_frequency(cfr_bluff_freq)

        print("=" * 55)
        print("  Glosten-Milgrom / Kuhn Poker Isomorphism")
        print("=" * 55)
        print(f"\n  Asset values:  V_L={self.V_L}, V_H={self.V_H}")
        print(f"  Informed fraction μ = {self.mu:.4f}")
        print(f"  Implied μ from bluff α: "
              f"{self.implied_mu_from_bluff_frequency(cfr_bluff_freq):.4f}\n")
        print(f"  GM optimal ask:   {self.optimal_ask_price():.4f}")
        print(f"  GM optimal bid:   {self.optimal_bid_price():.4f}")
        print(f"  GM bid-ask spread: {gm_spread:.4f}")
        print(f"\n  CFR bluff frequency (Jack):  {cfr_bluff_freq:.4f}")
        print(f"  P(informed | bet):           {1/(1+cfr_bluff_freq):.4f}")
        print(f"  Poker-derived spread:        {poker_spread:.4f}")
        print(f"\n  |GM spread - Poker spread| = {abs(gm_spread - poker_spread):.6f}")
        print(f"\n  KNOWN LIMITATIONS:")
        print(f"  1. Discrete values (3 cards vs continuous prices)")
        print(f"  2. Single period (no dynamic price evolution)")
        print(f"  3. No inventory risk (infinite capital assumption)")
        print(f"  4. No multi-asset correlation")
        print(f"  5. Adverse selection only (no latency/impact effects)")
        print("=" * 55)

        return {
            "gm_spread": gm_spread,
            "poker_spread": poker_spread,
            "bluff_freq": cfr_bluff_freq,
            "p_informed_given_bet": 1.0 / (1.0 + cfr_bluff_freq),
            "implied_mu": self.implied_mu_from_bluff_frequency(cfr_bluff_freq),
            "expected_profit": self.expected_profit_per_trade(cfr_bluff_freq),
        }


class TradingSignalGenerator:
    """
    FUNCTIONAL trading signal generator that uses CFR equilibrium output
    to produce actionable trading quantities.

    This goes beyond illustration - it computes:
    1. Fair spread given observed market conditions
    2. Whether the observed spread is too wide/narrow (edge detection)
    3. Position sizing recommendations
    4. Signal strength (confidence in the edge)
    """

    def __init__(self, V_L=1, V_H=3):
        self.V_L = V_L
        self.V_H = V_H
        self.gm = GlostenMilgromModel(V_L=V_L, V_H=V_H)

    def from_bluff_frequency(self, bluff_freq):
        """
        Generate trading signals from observed bluff frequency.

        Returns dict with:
        - fair_spread: what the spread SHOULD be at this bluff frequency
        - nash_spread: spread at Nash equilibrium (α = 1/3)
        - edge: deviation from Nash (trading opportunity)
        - signal: BUY_SPREAD / SELL_SPREAD / HOLD
        - position_size: suggested size (relative units)

        NOTE: Kuhn Poker has a FAMILY of Nash equilibria with α ∈ [0, 1/3].
        Any bluff frequency in this range is Nash-optimal, so the signal
        is HOLD for all α in [0, 1/3], not just α = 1/3.
        """
        nash_alpha = 1.0 / 3.0
        nash_spread = self.gm.spread_from_bluff_frequency(nash_alpha)
        fair_spread = self.gm.spread_from_bluff_frequency(bluff_freq)

        # Kuhn Nash equilibrium allows any α ∈ [0, 1/3]. All such α are
        # equilibrium strategies with zero exploitability, so there is
        # no trading edge when the observed bluff frequency is in this range.
        nash_range_tolerance = 0.01  # small tolerance for numerical noise
        in_nash_range = -nash_range_tolerance <= bluff_freq <= nash_alpha + nash_range_tolerance

        if in_nash_range:
            signal = "HOLD"
            confidence = 0.0
            edge = 0.0
        else:
            edge = fair_spread - nash_spread
            if edge > 0:
                signal = "BUY_SPREAD"  # spread should be wider → sell at ask, buy at bid
                confidence = min(edge / nash_spread, 1.0)
            else:
                signal = "SELL_SPREAD"  # spread should be narrower → provide liquidity
                confidence = min(abs(edge) / nash_spread, 1.0)

        # Position sizing (Kelly-inspired) — zero when in Nash range
        position_size = 0.0 if in_nash_range else self.gm.optimal_position_size(bluff_freq, bankroll=1000.0)

        return {
            "fair_spread": fair_spread,
            "nash_spread": nash_spread,
            "edge": edge,
            "signal": signal,
            "confidence": confidence,
            "position_size": position_size,
            "bluff_freq": bluff_freq,
            "p_informed": 1.0 / (1.0 + bluff_freq),
        }

    def from_cfr_solver(self, solver_strategy):
        """
        Generate trading signals directly from a CFR solver's output.
        This is the full pipeline: solver → equilibrium → trading signal.
        """
        bluff_freq = solver_strategy.get("J:", {}).get("b", 1.0 / 3.0)
        signals = self.from_bluff_frequency(bluff_freq)

        # Add solver-specific metadata
        signals["solver_converged"] = bluff_freq <= 0.34
        return signals

    def report(self, signals):
        """Print a formatted trading signal report."""
        print(f"\n{'═' * 55}")
        print("  TRADING SIGNAL REPORT")
        print(f"{'═' * 55}")
        print(f"  Observed bluff frequency:  {signals['bluff_freq']:.4f}")
        print(f"  P(informed | trade):       {signals['p_informed']:.4f}")
        print(f"  Fair spread:               {signals['fair_spread']:.4f}")
        print(f"  Nash equilibrium spread:   {signals['nash_spread']:.4f}")
        print(f"  Edge:                      {signals['edge']:+.4f}")
        print(f"\n  SIGNAL: {signals['signal']}")
        print(f"  Confidence:                {signals['confidence']:.2%}")
        print(f"  Suggested position size:   {signals['position_size']:.2f}")
        if 'solver_converged' in signals:
            print(f"  Solver converged:          {'YES' if signals['solver_converged'] else 'NO'}")
        print(f"{'═' * 55}")


def run_isomorphism_demo():
    """Train CFR, extract bluff frequency, verify against GM, generate signals."""
    print("Training CFR+ on Kuhn Poker...")
    solver = CFRPlus()
    solver.train(20000)
    strat = solver.get_full_strategy()

    j_strategy = strat.get("J:", {})
    bluff_freq = j_strategy.get("b", 0.0)

    print(f"CFR+ converged. Jack bluff frequency: {bluff_freq:.4f}")
    print(f"Exploitability: {compute_exploitability(strat):.6f}\n")

    # GM isomorphism verification
    implied_mu = GlostenMilgromModel(V_L=1, V_H=3).implied_mu_from_bluff_frequency(bluff_freq)
    gm = GlostenMilgromModel(V_L=1, V_H=3, mu=implied_mu)
    result = gm.verify_isomorphism(bluff_freq)

    # Generate functional trading signals
    signal_gen = TradingSignalGenerator(V_L=1, V_H=3)

    print("\n--- Signals from Nash equilibrium ---")
    nash_signals = signal_gen.from_cfr_solver(strat)
    signal_gen.report(nash_signals)

    print("\n--- Signals from hypothetical over-bluffer (α=0.5) ---")
    overbluff_signals = signal_gen.from_bluff_frequency(0.5)
    signal_gen.report(overbluff_signals)

    print("\n--- Signals from hypothetical under-bluffer (α=0.1) ---")
    underbluff_signals = signal_gen.from_bluff_frequency(0.1)
    signal_gen.report(underbluff_signals)

    return result


if __name__ == "__main__":
    run_isomorphism_demo()
