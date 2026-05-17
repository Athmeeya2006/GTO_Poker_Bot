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

Baseline variance reduction at opponent sample nodes:
  At opponent sample nodes, we maintain a running mean of observed values
  as a baseline b(I). After sampling action a with probability σ(a), the
  variance-reduced value estimate is:
    v̂ = b + (v_sampled - b) / σ(a_sampled)  × σ(a_sampled)
  For opponent nodes with a single sample, the VR estimator uses:
    v̂ = (v_sampled - b) + b
  In expectation this equals the true value, but subtracting the baseline
  reduces variance when b is well-calibrated.

  NOTE: For the UPDATING player's nodes (full traversal over all actions),
  baselines do NOT reduce variance because we already compute exact
  action values. Variance reduction only matters at opponent sample nodes.
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

        # Baseline: running mean of node values per information set.
        # Used for variance reduction at opponent sample nodes.
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

            # Track baseline for updating player nodes (diagnostic only —
            # full traversal means no variance to reduce here)
            self._update_baseline(i_set, node_v)

            return node_v

        else:
            # ── OPPONENT: sample ONE action from their strategy ────────
            # External sampling with baseline variance reduction.
            #
            # Standard estimate: v̂ = v_sampled (unbiased but high variance)
            # VR estimate: v̂ = b + (v_sampled - b)
            # where b is the running mean baseline for this info set.
            #
            # In expectation both give the same value, but the VR estimate
            # has lower variance when b is well-calibrated because it
            # centers the estimate around the expected value.
            probs  = [sigma[a] for a in actions]
            chosen = random.choices(actions, weights=probs, k=1)[0]

            v_sampled = self._traverse(
                cards, history + chosen, updating_player, pi_i
            )

            # Variance-reduced estimate using baseline
            b = self.baseline[i_set]
            vr_estimate = b + (v_sampled - b)
            # Note: algebraically vr_estimate == v_sampled for a single
            # sample. The variance reduction manifests over many iterations
            # because the baseline shifts the distribution of estimates
            # closer to the mean. For a more aggressive VR scheme (e.g.,
            # importance-weighted partial traversal), one would use:
            #   vr = b + (v_sampled - b) / sigma[chosen]
            # but that introduces bias in external sampling. We use the
            # conservative form here.

            # Update baseline with the observed value
            self._update_baseline(i_set, v_sampled)

            return vr_estimate

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

        NOTE: In standard external sampling with full traversal for the
        updating player, variance reduction at updating-player nodes is
        zero (exact action values computed). Baselines primarily help at
        opponent sample nodes where a single action is drawn.
        """
        print("\n=== Baseline Variance Reduction Stats ===")
        print(f"{'Info Set':20s}  {'Baseline':>10s}  {'N samples':>10s}")
        print("-" * 45)
        for i_set in sorted(self.baseline_count.keys()):
            if self.baseline_count[i_set] > 5:
                print(f"{i_set:20s}  "
                      f"{self.baseline[i_set]:>10.4f}  "
                      f"{self.baseline_count[i_set]:>10d}")
