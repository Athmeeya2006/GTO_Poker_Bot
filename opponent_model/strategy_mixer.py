# opponent_model/strategy_mixer.py
"""
Blends GTO strategy and exploitative counter-strategy based on model confidence.

Key design:
  - When confidence is low: play GTO (safe, unexploitable)
  - When confidence is high: deviate toward exploit
  - Blending is linear in confidence → smooth transition, no abrupt switches
  - When opponent adapts and KL → 0: revert to GTO automatically
"""
class StrategyMixer:
    def __init__(self, gto_strategy, confidence_threshold=0.65):
        """
        gto_strategy: dict {info_set: {action: prob}} - pre-computed Nash solution
        confidence_threshold: minimum confidence before any exploitation
        """
        self.gto      = gto_strategy
        self.threshold = confidence_threshold

    def get_exploitative_strategy(self, info_set, leak_type, actions):
        """
        Returns a best-response strategy for a given leak type.
        These are hard-coded counter-strategies to known leak patterns.
        """
        if "folds_too_much" in leak_type:
            # Opponent over-folds → bluff more
            if "b" in actions:
                return {"b": 0.90, "c": 0.10} if len(actions) == 2 else \
                       {a: (0.85 if a == "b" else 0.075) for a in actions}

        if "calls_too_much" in leak_type:
            # Opponent calls too much (calling station) → never bluff, always value bet
            # Check if this looks like a strong-hand position from the info_set
            parts = info_set.split("|")
            looks_strong = ("K" in info_set) or (len(parts) > 1 and "Q" in parts[1])
            if looks_strong:
                return {"b": 0.95, "c": 0.05} if len(actions) == 2 else \
                       {a: (0.90 if a == "b" else 0.10 / max(len(actions)-1,1))
                        for a in actions}
            else:
                return {"b": 0.02, "c": 0.98} if len(actions) == 2 else \
                       {a: (0.01 if a == "b" else 0.99 / max(len(actions)-1,1))
                        for a in actions}

        # Default: return GTO
        return self.gto.get(info_set, {a: 1.0/len(actions) for a in actions})

    def get_mixed_strategy(self, info_set, actions, dirichlet_model,
                           nash_strategy, leak_type=None):
        """
        Revised version - uses posterior variance for confidence gating.
        """
        gto_strat = self.gto.get(info_set, {a: 1.0 / len(actions) for a in actions})

        # Use variance-based confidence (not KL proxy)
        confidence = dirichlet_model.confidence_from_variance(
            info_set, actions, nash_strategy
        )

        if confidence < self.threshold or leak_type is None:
            return gto_strat, confidence

        exploit_strat = self.get_exploitative_strategy(info_set, leak_type, actions)

        # Linear interpolation: 0 = pure GTO, 1 = pure exploit
        mixed = {}
        for a in actions:
            g = gto_strat.get(a, 1.0 / len(actions))
            e = exploit_strat.get(a, 1.0 / len(actions))
            mixed[a] = (1 - confidence) * g + confidence * e

        total = sum(mixed.values())
        mixed = {a: v / total for a, v in mixed.items()}

        return mixed, confidence
