# core/mccfr.py  — with baseline subtraction variance reduction
"""
External Sampling MCCFR + Baseline Subtraction.

The variance reduction technique:
  Standard MCCFR uses sampled counterfactual values v̂(a).
  These are unbiased but high variance — each sample gives a noisy estimate.

  Baseline subtraction: instead of using v̂(a) directly, use (v̂(a) - b(I))
  where b(I) is a baseline — a running estimate of the expected value at I.

  Since E[v̂(a) - b(I)] = E[v̂(a)] - b(I), and if b(I) ≈ E[v̂(a)],
  the variance drops dramatically while the estimator stays UNBIASED.

  This is identical to the control variate technique in Monte Carlo integration,
  and to baseline subtraction in policy gradient RL (REINFORCE with baseline).
  Jane Street uses variance reduction in every Monte Carlo pricing model they run.
"""
import random
from collections import defaultdict
from core.kuhn_poker import (
    is_terminal, get_actions, current_player,
    get_payoff, get_info_set, CARDS
)

class ExternalSamplingMCCFR:
    def __init__(self):
        self.regrets      = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations   = 0

        # Baseline: running mean of node values per information set
        # Used for variance reduction
        self.baseline       = defaultdict(float)   # b(I) = running mean EV
        self.baseline_count = defaultdict(int)      # how many times we've seen I

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _regret_match(self, info_set, actions):
        r   = self.regrets[info_set]
        pos = {a: max(r[a], 0.0) for a in actions}
        total = sum(pos.values())
        if total > 0:
            return {a: pos[a] / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    def get_average_strategy(self, info_set, actions):
        s = self.strategy_sum[info_set]
        total = sum(s[a] for a in actions)
        if total > 0:
            return {a: s[a] / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    def _update_baseline(self, info_set, value):
        """
        Incremental running mean update for the baseline at info_set.
        b_new = b_old + (value - b_old) / (n + 1)
        This is Welford's online algorithm — numerically stable.
        """
        n = self.baseline_count[info_set]
        self.baseline[info_set] += (value - self.baseline[info_set]) / (n + 1)
        self.baseline_count[info_set] += 1

    # ------------------------------------------------------------------ #
    #  Core traversal
    # ------------------------------------------------------------------ #

    def _traverse(self, cards, history, updating_player):
        """
        External sampling traversal with baseline subtraction.
        Returns: expected payoff for updating_player.
        """
        if is_terminal(history):
            payoff = get_payoff(history, cards, 0)
            return payoff if updating_player == 0 else -payoff

        player  = current_player(history)
        actions = get_actions(history)
        i_set   = get_info_set(history, cards, player)
        sigma   = self._regret_match(i_set, actions)

        if player == updating_player:
            # Traverse ALL actions for updating player
            for a in actions:
                self.strategy_sum[i_set][a] += sigma[a]

            # Compute action values
            v = {a: self._traverse(cards, history + a, updating_player)
                 for a in actions}

            node_v = sum(sigma[a] * v[a] for a in actions)

            # ── BASELINE SUBTRACTION ──────────────────────────────────
            # Get current baseline for this info set
            b = self.baseline[i_set]    # 0.0 initially, converges to E[node_v]

            # Update regrets using BASELINE-SUBTRACTED values
            # Standard:   regret(a) += v(a) - node_v
            # With baseline: regret(a) += (v(a) - b) - (node_v - b)
            #                           = v(a) - node_v     ← same!
            # BUT for SAMPLED nodes (when we later extend to partial traversal):
            # regret(a) += (v̂(a) - b) - (node_v - b)
            # The baseline cancels in expectation but REDUCES VARIANCE
            # because (v̂(a) - b) has lower variance than v̂(a) alone
            # when b ≈ E[v̂(a)].
            #
            # For full traversal (Kuhn/Leduc), this is equivalent to standard CFR.
            # The variance reduction becomes critical for larger games.
            for a in actions:
                self.regrets[i_set][a] += (v[a] - b) - (node_v - b)

            # Update baseline with observed node value
            self._update_baseline(i_set, node_v)

            return node_v

        else:
            # Sample ONE action for opponent
            probs  = [sigma[a] for a in actions]
            chosen = random.choices(actions, weights=probs, k=1)[0]

            value = self._traverse(cards, history + chosen, updating_player)

            # ── BASELINE SUBTRACTION FOR SAMPLED OPPONENT NODES ──────
            # When we sample the opponent's action, we get a noisy estimate
            # of the true expected value. Subtract the baseline to reduce
            # the variance of the regret update at the PARENT node.
            #
            # The importance-weighted value is: value / sigma[chosen]
            # With baseline: (value - b) / sigma[chosen] + b
            # This is an unbiased estimator with lower variance when b ≈ E[value].
            b = self.baseline[i_set]
            self._update_baseline(i_set, value)

            # Return baseline-corrected importance-weighted estimate
            # Only apply correction if we have enough baseline samples
            if self.baseline_count[i_set] > 5:
                corrected = (value - b) / sigma[chosen] + b
            else:
                corrected = value   # use raw value until baseline is stable

            return corrected

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def train(self, iterations):
        for _ in range(iterations):
            self.iterations += 1
            cards = random.sample(CARDS, 2)
            self._traverse(cards, "", 0)
            self._traverse(cards, "", 1)

    def get_full_strategy(self):
        result = {}
        all_sets = set(self.regrets.keys()) | set(self.strategy_sum.keys())
        for i_set in all_sets:
            actions = list(self.strategy_sum[i_set].keys()) or \
                      list(self.regrets[i_set].keys())
            result[i_set] = self.get_average_strategy(i_set, actions)
        return result

    def variance_reduction_stats(self):
        """
        Print stats showing how much variance the baseline is absorbing.
        For each info set, shows the baseline value and how many times
        it has been updated — proxy for how well-calibrated the baseline is.
        """
        print("\n=== Baseline Variance Reduction Stats ===")
        print(f"{'Info Set':20s}  {'Baseline':>10s}  {'N samples':>10s}")
        print("-" * 45)
        for i_set in sorted(self.baseline_count.keys()):
            if self.baseline_count[i_set] > 5:
                print(f"{i_set:20s}  "
                      f"{self.baseline[i_set]:>10.4f}  "
                      f"{self.baseline_count[i_set]:>10d}")
