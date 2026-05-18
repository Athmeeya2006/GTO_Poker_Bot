# core/cfr.py  - alternating-updates vanilla CFR
"""
Vanilla CFR with alternating updates (Zinkevich et al. 2007).
Each iteration does two traversals, updating one player's regrets per pass
to keep strategies consistent within each traversal.
"""
from collections import defaultdict
from itertools import permutations
from core.kuhn_poker import (
    is_terminal, get_actions, current_player,
    get_payoff, get_info_set, CARDS
)

class VanillaCFR:
    def __init__(self):
        self.regrets  = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

    def _regret_match(self, info_set, actions):
        r = self.regrets[info_set]
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

    def _cfr_player(self, cards, history, p0, p1, updating_player):
        """
        Traverse the game tree, updating regrets for only the updating_player.
        Returns the expected value from Player 0's perspective.
        """
        if is_terminal(history):
            return get_payoff(history, cards, 0)

        player  = current_player(history)
        actions = get_actions(history)
        i_set   = get_info_set(history, cards, player)
        sigma   = self._regret_match(i_set, actions)

        # Accumulate weighted strategy (for average strategy computation)
        reach_i = p0 if player == 0 else p1
        for a in actions:
            self.strategy_sum[i_set][a] += reach_i * sigma[a]

        # Recurse to get action utilities (always from P0 perspective)
        v = {}
        for a in actions:
            next_hist = history + a
            if player == 0:
                v[a] = self._cfr_player(cards, next_hist, p0 * sigma[a], p1, updating_player)
            else:
                v[a] = self._cfr_player(cards, next_hist, p0, p1 * sigma[a], updating_player)

        # Node value from P0 perspective
        node_v = sum(sigma[a] * v[a] for a in actions)

        # Only update regrets for the updating_player
        if player == updating_player:
            # Counterfactual reach = opponent's reach probability
            cf_reach = p1 if player == 0 else p0

            for a in actions:
                # For P0: regret = v(a) - node_v
                # For P1: regret = -(v(a) - node_v)  because v is from P0's perspective
                if player == 0:
                    self.regrets[i_set][a] += cf_reach * (v[a] - node_v)
                else:
                    self.regrets[i_set][a] += cf_reach * (-(v[a] - node_v))

        return node_v

    def train(self, iterations):
        for _ in range(iterations):
            # Alternating updates: fix P1's strategy, update P0;
            # then fix P0's strategy, update P1.
            for cards in permutations(CARDS, 2):
                self._cfr_player(list(cards), "", 1.0, 1.0, updating_player=0)
            for cards in permutations(CARDS, 2):
                self._cfr_player(list(cards), "", 1.0, 1.0, updating_player=1)
            self.iterations += 1

    def get_full_strategy(self):
        result = {}
        all_info_sets = set(self.regrets.keys()) | set(self.strategy_sum.keys())
        for i_set in all_info_sets:
            actions = list(self.strategy_sum[i_set].keys()) or list(self.regrets[i_set].keys())
            result[i_set] = self.get_average_strategy(i_set, actions)
        return result
