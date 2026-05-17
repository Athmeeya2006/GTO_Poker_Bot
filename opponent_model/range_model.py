# opponent_model/range_model.py
"""
Range-based Bayesian opponent model.

Instead of discrete opponent 'types', we track a probability distribution
over every hand the opponent could hold, updated after each observed action.

This is what real poker solvers do.

Core idea:
  P(opponent holds hand h | actions observed) 
    ∝ P(actions | opponent holds h) × P(opponent holds h)

After each action, hands the opponent would NEVER play get downweighted.
After a few streets, the range becomes very precise.
"""
import numpy as np
from collections import defaultdict

class RangeModel:
    def __init__(self, possible_hands, prior=None):
        """
        possible_hands: list of hand identifiers (e.g. [1, 2, 3] for J/Q/K)
        prior: initial probability distribution (uniform if None)
        """
        self.hands = possible_hands
        n = len(possible_hands)

        if prior is None:
            self.weights = {h: 1.0 / n for h in possible_hands}
        else:
            self.weights = dict(zip(possible_hands, prior))

        self.observations = []   # log of (action, info_set) tuples

    def update(self, action, strategy_profile, info_set_fn):
        """
        Bayesian update: observe opponent taking 'action'.

        strategy_profile: the GTO strategy dict {info_set: {action: prob}}
        info_set_fn: callable(hand) -> info_set_key for the opponent's hand

        For each possible hand h:
          new_weight(h) ∝ old_weight(h) × P(action | opponent has hand h)
          where P(action | h) comes from the GTO strategy as the BASELINE.
          Deviations from GTO show up as weight imbalances.
        """
        unnormalized = {}
        for h in self.hands:
            i_set = info_set_fn(h)
            if i_set in strategy_profile and action in strategy_profile[i_set]:
                likelihood = strategy_profile[i_set][action]
            else:
                likelihood = 0.5   # uniform if not in strategy

            # Cromwell's rule: never 0
            likelihood = max(likelihood, 0.01)
            unnormalized[h] = self.weights[h] * likelihood

        total = sum(unnormalized.values())
        if total > 0:
            self.weights = {h: w / total for h, w in unnormalized.items()}
        else:
            # Reset to uniform if total collapses (shouldn't happen with Cromwell)
            n = len(self.hands)
            self.weights = {h: 1.0 / n for h in self.hands}

        self.observations.append((action, info_set_fn))

    def most_likely_hand(self):
        return max(self.weights, key=self.weights.get)

    def entropy(self):
        """Shannon entropy of the range. Low = confident. High = uncertain."""
        e = 0.0
        for p in self.weights.values():
            if p > 0:
                e -= p * np.log2(p)
        return e

    def max_entropy(self):
        return np.log2(len(self.hands))

    def confidence(self):
        """0 = totally uncertain, 1 = certain."""
        return 1.0 - self.entropy() / self.max_entropy()

    def get_weights(self):
        return dict(self.weights)