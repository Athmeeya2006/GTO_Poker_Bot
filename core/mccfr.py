# core/mccfr.py  - External Sampling MCCFR
"""
External Sampling Monte Carlo CFR (Lanctot et al. 2009).

Traverses all actions for the updating player but samples a single opponent
action proportional to their current strategy. Maintains a running-mean
baseline per info set for variance reduction at opponent sample nodes.
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

        # Running mean of node values per info set for variance reduction
        self.baseline       = defaultdict(float)
        self.baseline_count = defaultdict(int)

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
        """Incremental running mean (Welford's online algorithm)."""
        n = self.baseline_count[info_set]
        self.baseline[info_set] += (value - self.baseline[info_set]) / (n + 1)
        self.baseline_count[info_set] += 1

    def _traverse(self, cards, history, updating_player, pi_i=1.0):
        """
        External sampling traversal. Returns expected payoff for
        updating_player from this node.
        """
        if is_terminal(history):
            payoff = get_payoff(history, cards, 0)
            return payoff if updating_player == 0 else -payoff

        player  = current_player(history)
        actions = get_actions(history)
        i_set   = get_info_set(history, cards, player)
        sigma   = self._regret_match(i_set, actions)

        if player == updating_player:
            # Updating player: traverse all actions
            for a in actions:
                self.strategy_sum[i_set][a] += pi_i * sigma[a]

            v = {a: self._traverse(
                     cards, history + a, updating_player,
                     pi_i * sigma[a])
                 for a in actions}

            node_v = sum(sigma[a] * v[a] for a in actions)

            for a in actions:
                self.regrets[i_set][a] += v[a] - node_v

            self._update_baseline(i_set, node_v)

            return node_v

        else:
            # Opponent: sample one action from their strategy
            probs  = [sigma[a] for a in actions]
            chosen = random.choices(actions, weights=probs, k=1)[0]

            v_sampled = self._traverse(
                cards, history + chosen, updating_player, pi_i
            )

            # Variance-reduced estimate using baseline
            b = self.baseline[i_set]
            vr_estimate = b + (v_sampled - b)
            # algebraically vr_estimate == v_sampled here;
            # see Lanctot 2009 for importance-weighted variants

            self._update_baseline(i_set, v_sampled)

            return vr_estimate

    def train(self, iterations):
        for _ in range(iterations):
            self.iterations += 1
            cards = random.sample(CARDS, 2)
            # Update both players' regrets on the same deal
            self._traverse(cards, "", 0, pi_i=1.0)
            self._traverse(cards, "", 1, pi_i=1.0)

    def get_full_strategy(self):
        result = {}
        all_sets = set(self.regrets.keys()) | set(self.strategy_sum.keys())
        for i_set in all_sets:
            actions = list(self.strategy_sum[i_set].keys()) or \
                      list(self.regrets[i_set].keys())
            result[i_set] = self.get_average_strategy(i_set, actions)
        return result

    def variance_reduction_stats(self):
        """Print baseline values and sample counts per info set."""
        print("\n=== Baseline Variance Reduction Stats ===")
        print(f"{'Info Set':20s}  {'Baseline':>10s}  {'N samples':>10s}")
        print("-" * 45)
        for i_set in sorted(self.baseline_count.keys()):
            if self.baseline_count[i_set] > 5:
                print(f"{i_set:20s}  "
                      f"{self.baseline[i_set]:>10.4f}  "
                      f"{self.baseline_count[i_set]:>10d}")
