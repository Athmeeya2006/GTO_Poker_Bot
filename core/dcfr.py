# core/dcfr.py
from collections import defaultdict
from itertools import permutations
from core.kuhn_poker import (
    is_terminal,
    get_actions,
    current_player,
    get_payoff,
    get_info_set,
    CARDS,
)


class DCFR:
    """
    Discounted CFR (DCFR) for Kuhn Poker.

    Uses separate temporal discounting for positive and negative regrets and
    discounted averaging for strategy accumulation.
    """

    def __init__(self, alpha: float = 1.5, beta: float = 0.0, gamma: float = 2.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.t = 0
        self.regrets = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

    def _regret_match(self, info_set, actions):
        r = self.regrets[info_set]
        pos = {a: max(r[a], 0.0) for a in actions}
        total = sum(pos.values())
        if total > 0:
            return {a: pos[a] / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    def _discount_regrets(self, info_set, actions):
        if self.t <= 1:
            return
        pos_scale = (self.t ** self.alpha) / (self.t ** self.alpha + 1.0)
        neg_scale = (self.t ** self.beta) / (self.t ** self.beta + 1.0)
        for a in actions:
            r = self.regrets[info_set][a]
            self.regrets[info_set][a] = r * (pos_scale if r > 0 else neg_scale)

    def _discount_strategy_sum(self, info_set, actions):
        if self.t <= 1:
            return
        scale = (self.t / (self.t + 1.0)) ** self.gamma
        for a in actions:
            self.strategy_sum[info_set][a] *= scale

    def _cfr(self, cards, history, p0, p1):
        if is_terminal(history):
            return get_payoff(history, cards, 0)

        player = current_player(history)
        actions = get_actions(history)
        i_set = get_info_set(history, cards, player)

        sigma = self._regret_match(i_set, actions)

        reach_i = p0 if player == 0 else p1
        for a in actions:
            self.strategy_sum[i_set][a] += reach_i * sigma[a]

        v = {}
        for a in actions:
            if player == 0:
                v[a] = self._cfr(cards, history + a, p0 * sigma[a], p1)
            else:
                v[a] = self._cfr(cards, history + a, p0, p1 * sigma[a])

        node_v = sum(sigma[a] * v[a] for a in actions)
        cf_reach = p1 if player == 0 else p0

        for a in actions:
            delta = (v[a] - node_v) if player == 0 else (-(v[a] - node_v))
            self.regrets[i_set][a] += cf_reach * delta

        return node_v

    def train(self, iterations: int):
        for _ in range(iterations):
            self.t += 1

            # Apply discounting ONCE per iteration (before tree traversals)
            for i_set in list(self.regrets.keys()):
                actions = list(self.regrets[i_set].keys())
                self._discount_regrets(i_set, actions)
            for i_set in list(self.strategy_sum.keys()):
                actions = list(self.strategy_sum[i_set].keys())
                self._discount_strategy_sum(i_set, actions)

            for cards in permutations(CARDS, 2):
                self._cfr(list(cards), "", 1.0, 1.0)

    def get_average_strategy(self, info_set, actions):
        s = self.strategy_sum[info_set]
        total = sum(s[a] for a in actions)
        if total > 0:
            return {a: s[a] / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    def get_full_strategy(self):
        result = {}
        all_sets = set(self.strategy_sum.keys()) | set(self.regrets.keys())
        for i_set in all_sets:
            actions = list(self.strategy_sum[i_set].keys()) or list(self.regrets[i_set].keys())
            result[i_set] = self.get_average_strategy(i_set, actions)
        return result
