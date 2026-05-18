# core/cfr_plus.py
from collections import defaultdict
from itertools import permutations
from core.kuhn_poker import (
    is_terminal, get_actions, current_player,
    get_payoff, get_info_set, CARDS
)

class CFRPlus:
    """
    CFR+ (Tammelin 2014).
    Two changes from vanilla CFR:
    1. Regrets are floored at 0 each update (no negative accumulation).
    2. Average strategy uses LINEAR weighting: iteration t has weight t.
       This dramatically accelerates convergence.

    Linear weight self.t is per iteration (shared across all deal permutations),
    matching the convention in OpenSpiel and Pluribus.
    """
    def __init__(self):
        self.regrets      = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.t = 0   # current iteration number

    def _regret_match(self, info_set, actions):
        r = self.regrets[info_set]
        # CFR+: floor at 0 before matching (positive part only)
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

    def _cfr(self, cards, history, p0, p1):
        if is_terminal(history):
            return get_payoff(history, cards, 0)

        player  = current_player(history)
        actions = get_actions(history)
        i_set   = get_info_set(history, cards, player)
        sigma   = self._regret_match(i_set, actions)

        # LINEAR weighting: weight = self.t
        reach_i = p0 if player == 0 else p1
        for a in actions:
            self.strategy_sum[i_set][a] += self.t * reach_i * sigma[a]

        v = {}
        for a in actions:
            if player == 0:
                v[a] = self._cfr(cards, history + a, p0 * sigma[a], p1)
            else:
                v[a] = self._cfr(cards, history + a, p0, p1 * sigma[a])

        node_v   = sum(sigma[a] * v[a] for a in actions)
        cf_reach = p1 if player == 0 else p0

        for a in actions:
            if player == 0:
                raw = cf_reach * (v[a] - node_v)
            else:
                raw = cf_reach * (-(v[a] - node_v))
            # CFR+: floor regrets at 0 immediately
            self.regrets[i_set][a] = max(self.regrets[i_set][a] + raw, 0.0)

        return node_v

    def train(self, iterations):
        for _ in range(iterations):
            self.t += 1
            for cards in permutations(CARDS, 2):
                self._cfr(list(cards), "", 1.0, 1.0)

    def get_full_strategy(self):
        result = {}
        all_sets = set(self.strategy_sum.keys()) | set(self.regrets.keys())
        for i_set in all_sets:
            actions = list(self.strategy_sum[i_set].keys()) or list(self.regrets[i_set].keys())
            result[i_set] = self.get_average_strategy(i_set, actions)
        return result
