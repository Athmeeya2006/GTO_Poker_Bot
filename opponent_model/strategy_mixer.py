# opponent_model/strategy_mixer.py
"""
Blends GTO strategy and exploitative counter-strategy based on model confidence.

Key design:
  - When confidence is low: play GTO (safe, unexploitable)
  - When confidence is high: deviate toward exploit
  - Blending is linear in confidence → smooth transition, no abrupt switches
  - When opponent adapts and KL → 0: revert to GTO automatically

Exploitation strategy:
  Instead of hard-coded magic numbers, the counter-strategy is computed
  as a best response against the opponent's estimated action frequencies.
  This uses the tree-traversal BR machinery from exploitability.py.
"""
from core.kuhn_poker import (
    is_terminal, get_actions, current_player,
    get_payoff, get_info_set, CARDS
)
from itertools import permutations
from collections import defaultdict


def _compute_counter_strategy(opponent_estimated_strategy, our_player):
    """
    Compute a best-response strategy for our_player against the
    opponent's estimated strategy.

    Uses the same tree-traversal BR algorithm as exploitability.py:
    1. Accumulate opponent-reach-weighted action values per info set
    2. Pick the best action at each info set

    Args:
        opponent_estimated_strategy: dict {info_set: {action: prob}}
            estimated from the Dirichlet model
        our_player: 0 or 1 - which player we are

    Returns:
        dict {info_set: {action: prob}} - the best response strategy
    """
    all_deals = list(permutations(CARDS, 2))
    infoset_action_values = defaultdict(lambda: defaultdict(float))

    for cards in all_deals:
        _br_traverse(
            list(cards), "", our_player, opponent_estimated_strategy,
            infoset_action_values, opp_reach=1.0
        )

    br_strategy = {}
    for info_set, action_vals in infoset_action_values.items():
        best_action = max(action_vals, key=action_vals.get)
        actions = list(action_vals.keys())
        # softened BR: 90/10 for robustness against model error
        br_strategy[info_set] = {}
        n_other = max(len(actions) - 1, 1)
        for a in actions:
            if a == best_action:
                br_strategy[info_set][a] = 0.90
            else:
                br_strategy[info_set][a] = 0.10 / n_other

    return br_strategy


def _br_traverse(cards, history, br_player, opponent_strategy,
                 infoset_action_values, opp_reach=1.0):
    """Walk tree accumulating reach-weighted action values for BR player."""
    if is_terminal(history):
        return get_payoff(history, cards, br_player)

    player = current_player(history)
    actions = get_actions(history)
    info_set = get_info_set(history, cards, player)

    if player == br_player:
        action_vals = {}
        for a in actions:
            action_vals[a] = _br_traverse(
                cards, history + a, br_player, opponent_strategy,
                infoset_action_values, opp_reach
            )
            infoset_action_values[info_set][a] += opp_reach * action_vals[a]
        return max(action_vals.values())
    else:
        sigma = opponent_strategy.get(info_set, None)
        if sigma is None:
            sigma = {a: 1.0 / len(actions) for a in actions}

        value = 0.0
        for a in actions:
            child_val = _br_traverse(
                cards, history + a, br_player, opponent_strategy,
                infoset_action_values, opp_reach * sigma.get(a, 0.0)
            )
            value += sigma.get(a, 0.0) * child_val
        return value


class StrategyMixer:
    def __init__(self, gto_strategy, confidence_threshold=0.65):
        """
        gto_strategy: dict {info_set: {action: prob}} - pre-computed Nash solution
        confidence_threshold: minimum confidence before any exploitation
        """
        self.gto      = gto_strategy
        self.threshold = confidence_threshold

    def get_exploitative_strategy(self, info_set, leak_type, actions,
                                  opponent_estimated_strategy=None):
        """
        Returns a counter-strategy for the given info set.

        If an opponent estimated strategy is available, computes the
        actual best response. Otherwise falls back to GTO.

        Args:
            info_set: the current information set
            leak_type: string describing the leak (e.g. "folds_too_much")
            actions: available actions at this info set
            opponent_estimated_strategy: Dirichlet posterior means, if available
        """
        if opponent_estimated_strategy is not None:
            # Determine which player the bot is (opponent acts at
            # the observed info sets; we act at our own info sets).
            history = info_set.split(":", 1)[1] if ":" in info_set else ""
            our_player = current_player(history)

            br = _compute_counter_strategy(
                opponent_estimated_strategy, our_player
            )
            if info_set in br:
                return br[info_set]

        # Fallback: return GTO if BR computation doesn't cover this info set
        return self.gto.get(info_set, {a: 1.0/len(actions) for a in actions})

    def get_mixed_strategy(self, info_set, actions, dirichlet_model,
                           nash_strategy, leak_type=None):
        """
        Confidence-gated strategy blending.

        Uses posterior variance for confidence gating:
        - Low confidence → pure GTO
        - High confidence → blend toward computed best response
        """
        gto_strat = self.gto.get(info_set, {a: 1.0 / len(actions) for a in actions})

        # Use variance-based confidence (not KL proxy)
        confidence = dirichlet_model.confidence_from_variance(
            info_set, actions, nash_strategy
        )

        if confidence < self.threshold or leak_type is None:
            return gto_strat, confidence

        # Build opponent's estimated strategy from Dirichlet posteriors
        opponent_est = {}
        for obs_info_set in dirichlet_model.n_observations:
            if dirichlet_model.n_observations[obs_info_set] >= 3:
                obs_actions = list(dirichlet_model.counts[obs_info_set].keys())
                opponent_est[obs_info_set] = dirichlet_model.posterior_mean(
                    obs_info_set, obs_actions
                )

        # Compute best-response counter-strategy if we have opponent data
        if opponent_est:
            exploit_strat = self.get_exploitative_strategy(
                info_set, leak_type, actions,
                opponent_estimated_strategy=opponent_est
            )
        else:
            exploit_strat = gto_strat

        # Linear interpolation: 0 = pure GTO, 1 = pure exploit
        mixed = {}
        for a in actions:
            g = gto_strat.get(a, 1.0 / len(actions))
            e = exploit_strat.get(a, 1.0 / len(actions))
            mixed[a] = (1 - confidence) * g + confidence * e

        total = sum(mixed.values())
        mixed = {a: v / total for a, v in mixed.items()}

        return mixed, confidence
