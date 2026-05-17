# core/mccfr.py  - External Sampling MCCFR
"""
External Sampling Monte Carlo CFR.

External sampling traverses ALL actions for the updating player, but
SAMPLES a single action for the opponent (proportional to the opponent's
current strategy). This gives an unbiased estimate of counterfactual values
with lower per-iteration cost than vanilla CFR.

Key properties:
  - The updating player explores all actions → exact counterfactual values
    for their own info sets (no variance there).
  - The opponent's action is sampled → introduces variance, but the
    estimator is unbiased because sampling from σ_{-i} and observing
    the raw payoff gives E[v] = Σ σ(a) × v(a).
  - No importance weighting is needed at opponent sample nodes in standard
    external sampling MCCFR (Lanctot et al. 2009).

Baseline subtraction (variance reduction):
  At opponent sample nodes, we maintain a running mean of observed values
  as a baseline b(I). The variance-reduced estimator is:
    v̂ = (v_sampled - b) + b = v_sampled  (for raw external sampling)
  For the updating player's regret, the baseline cancels in full traversal.
  We track baselines purely for diagnostics and future partial-traversal use.
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
        # Tracked for diagnostics; in standard external sampling with
        # full traversal for the updating player, the baseline cancels
        # algebraically in the regret update.
        self.baseline       = defaultdict(float)
        self.baseline_count = defaultdict(int)

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
        This is Welford's online algorithm - numerically stable.
        """
        n = self.baseline_count[info_set]
        self.baseline[info_set] += (value - self.baseline[info_set]) / (n + 1)
        self.baseline_count[info_set] += 1

    # ------------------------------------------------------------------ #
    #  Core traversal
    # ------------------------------------------------------------------ #

    def _traverse(self, cards, history, updating_player, pi_i=1.0):
        """
        External sampling traversal.

        Args:
            cards: the dealt cards for this sample
            history: action history string
            updating_player: which player's regrets we're updating (0 or 1)
            pi_i: reach probability of the updating player to this node

        Returns: expected payoff for updating_player from this node.
        """
        if is_terminal(history):
            payoff = get_payoff(history, cards, 0)
            return payoff if updating_player == 0 else -payoff

        player  = current_player(history)
        actions = get_actions(history)
        i_set   = get_info_set(history, cards, player)
        sigma   = self._regret_match(i_set, actions)

        if player == updating_player:
            # ── UPDATING PLAYER: traverse ALL actions ──────────────────
            # Accumulate reach-weighted strategy for average computation.
            # The weight is the updating player's reach probability pi_i,
            # matching vanilla CFR's strategy_sum[i][a] += reach_i * σ(a).
            for a in actions:
                self.strategy_sum[i_set][a] += pi_i * sigma[a]

            # Compute action values by traversing all actions
            v = {a: self._traverse(
                     cards, history + a, updating_player,
                     pi_i * sigma[a])
                 for a in actions}

            node_v = sum(sigma[a] * v[a] for a in actions)

            # Update regrets: standard CFR regret = v(a) - node_v
            for a in actions:
                self.regrets[i_set][a] += v[a] - node_v

            # Track baseline for diagnostics
            self._update_baseline(i_set, node_v)

            return node_v

        else:
            # ── OPPONENT: sample ONE action from their strategy ────────
            # Standard external sampling: sample from σ_{-i}, return raw
            # value. No importance weighting needed because:
            #   E[v_sampled] = Σ σ(a) × v(a) = true expected value
            probs  = [sigma[a] for a in actions]
            chosen = random.choices(actions, weights=probs, k=1)[0]

            value = self._traverse(
                cards, history + chosen, updating_player, pi_i
            )

            # Track baseline for diagnostics
            self._update_baseline(i_set, value)

            # Return raw value — no importance weighting
            return value

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

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
        """
        Print stats showing baseline values and calibration.
        For each info set, shows the baseline value and how many times
        it has been updated - proxy for how well-calibrated the baseline is.
        """
        print("\n=== Baseline Variance Reduction Stats ===")
        print(f"{'Info Set':20s}  {'Baseline':>10s}  {'N samples':>10s}")
        print("-" * 45)
        for i_set in sorted(self.baseline_count.keys()):
            if self.baseline_count[i_set] > 5:
                print(f"{i_set:20s}  "
                      f"{self.baseline[i_set]:>10.4f}  "
                      f"{self.baseline_count[i_set]:>10d}")
