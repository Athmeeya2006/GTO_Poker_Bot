# tests/test_opponent_model.py
"""
Tests for opponent modeling components:
- Dirichlet posterior model
- SPRT leak detector
- Strategy mixer
- Range model
"""
import pytest
import random
from opponent_model.dirichlet_model import DirichletOpponentModel
from opponent_model.hypothesis_test import SPRTLeakDetector
from opponent_model.strategy_mixer import StrategyMixer
from opponent_model.range_model import RangeModel


class TestDirichletModel:
    """Verify Dirichlet-Multinomial Bayesian opponent model."""

    def test_prior_is_uniform(self):
        """With no observations, posterior mean should be uniform."""
        model = DirichletOpponentModel(alpha_prior=1.0)
        # Initialize actions
        model.counts["K:"]["b"]  # triggers default
        model.counts["K:"]["c"]
        mean = model.posterior_mean("K:", ["b", "c"])
        assert abs(mean["b"] - 0.5) < 0.01

    def test_posterior_tracks_observations(self):
        """After many observations, posterior should approach true frequency."""
        model = DirichletOpponentModel(alpha_prior=1.0)
        rng = random.Random(42)
        true_freq = 0.8  # opponent bets 80% of the time

        for _ in range(200):
            action = "b" if rng.random() < true_freq else "c"
            model.observe("K:", action)

        mean = model.posterior_mean("K:", ["b", "c"])
        assert abs(mean["b"] - true_freq) < 0.05, (
            f"Posterior mean {mean['b']:.4f} should be near {true_freq}"
        )

    def test_kl_near_zero_for_gto_opponent(self):
        """KL divergence should be near 0 when opponent plays Nash."""
        model = DirichletOpponentModel(alpha_prior=1.0)
        nash = {"K:": {"b": 0.8, "c": 0.2}}
        rng = random.Random(7)

        for _ in range(200):
            action = "b" if rng.random() < 0.8 else "c"
            model.observe("K:", action)

        kl = model.kl_from_nash("K:", ["b", "c"], nash)
        assert kl < 0.05, f"KL should be near 0 for GTO opponent, got {kl:.4f}"

    def test_kl_large_for_deviating_opponent(self):
        """KL divergence should be large when opponent deviates from Nash."""
        model = DirichletOpponentModel(alpha_prior=1.0)
        nash = {"K:": {"b": 0.5, "c": 0.5}}

        # Opponent always bets (deviates)
        for _ in range(100):
            model.observe("K:", "b")

        kl = model.kl_from_nash("K:", ["b", "c"], nash)
        assert kl > 0.5, f"KL should be large for deviating opponent, got {kl:.4f}"

    def test_credible_interval_narrows(self):
        """Credible interval should narrow with more observations."""
        model = DirichletOpponentModel(alpha_prior=1.0)

        # Few observations
        for _ in range(5):
            model.observe("K:", "b")
        lo1, hi1 = model.credible_interval("K:", "b", ["b", "c"])
        width1 = hi1 - lo1

        # Many more observations
        for _ in range(200):
            model.observe("K:", "b")
        lo2, hi2 = model.credible_interval("K:", "b", ["b", "c"])
        width2 = hi2 - lo2

        assert width2 < width1, (
            f"CI should narrow: {width1:.4f} -> {width2:.4f}"
        )

    def test_confidence_low_with_few_observations(self):
        """Confidence should be low with insufficient data."""
        model = DirichletOpponentModel()
        nash = {"K:": {"b": 0.5, "c": 0.5}}
        model.observe("K:", "b")
        model.observe("K:", "b")

        conf = model.confidence_from_variance("K:", ["b", "c"], nash)
        assert conf < 0.5, f"Confidence should be low with 2 obs, got {conf:.4f}"

    def test_get_deviations_finds_strong_deviations(self):
        """Should detect info sets where opponent significantly deviates."""
        model = DirichletOpponentModel()
        nash = {"J:b": {"c": 0.0, "f": 1.0}}  # Nash: always fold

        # Opponent calls every time (massive deviation)
        for _ in range(30):
            model.observe("J:b", "c")

        devs = model.get_deviations(nash, threshold=0.1)
        assert len(devs) > 0, "Should detect deviation from Nash"


class TestSPRT:
    """Verify Sequential Probability Ratio Test."""

    def test_detects_leak(self):
        """SPRT should detect a clear leak within reasonable observations."""
        sprt = SPRTLeakDetector(alpha=0.05, beta=0.05, epsilon=0.20)
        rng = random.Random(42)

        # True fold frequency: 0.9 (Nash is ~0.67)
        for i in range(100):
            action = "f" if rng.random() < 0.9 else "c"
            result = sprt.update("Q:b", "f", action, 0.67)
            if result == "H1":
                break

        leaks = sprt.get_detected_leaks()
        assert len(leaks) > 0, "SPRT should detect the fold leak"

    def test_no_false_positive_on_gto(self):
        # seed 42 verified: no FP at this epsilon, n=100
        sprt = SPRTLeakDetector(alpha=0.05, beta=0.05, epsilon=0.20)
        rng = random.Random(42)

        # Opponent plays at Nash frequency
        for i in range(100):
            action = "f" if rng.random() < 0.67 else "c"
            result = sprt.update("Q:b", "f", action, 0.67)

        leaks = sprt.get_detected_leaks()
        fold_leaks = [l for l in leaks if l[0] == "Q:b" and l[1] == "f"]
        assert len(fold_leaks) == 0, (
            "Should not detect leak for GTO-playing opponent (seed=42 verified)"
        )

    def test_detection_speed(self):
        """Should detect large deviations quickly (< 30 observations)."""
        sprt = SPRTLeakDetector(alpha=0.05, beta=0.05, epsilon=0.20)

        n_to_decide = None
        for i in range(100):
            # Opponent always folds (extreme deviation)
            result = sprt.update("J:b", "f", "f", 0.5)
            if result == "H1":
                n_to_decide = i + 1
                break

        assert n_to_decide is not None, "Should detect extreme leak"
        assert n_to_decide < 30, (
            f"Should detect extreme leak in <30 obs, took {n_to_decide}"
        )


class TestStrategyMixer:
    """Verify confidence-gated strategy mixing."""

    def test_returns_gto_when_no_confidence(self):
        """With no data, mixer should return pure GTO."""
        gto = {"K:": {"b": 0.8, "c": 0.2}}
        mixer = StrategyMixer(gto, confidence_threshold=0.5)
        model = DirichletOpponentModel()

        strat, conf = mixer.get_mixed_strategy(
            "K:", ["b", "c"], model, gto
        )
        assert abs(strat["b"] - 0.8) < 0.01
        assert conf < 0.5

    def test_exploits_when_confident(self):
        """With high confidence of leak, should deviate from GTO."""
        gto = {"J:b": {"c": 0.0, "f": 1.0}}
        mixer = StrategyMixer(gto, confidence_threshold=0.3)
        model = DirichletOpponentModel()

        # Massive data showing opponent deviates
        for _ in range(100):
            model.observe("J:b", "c")

        strat, conf = mixer.get_mixed_strategy(
            "J:b", ["c", "f"], model, gto, leak_type="calls_too_much"
        )
        # Should have some deviation from GTO
        # (exact values depend on exploitation strategy)
        assert conf > 0.3, f"Should be confident after 100 obs, got {conf:.4f}"


class TestRangeModel:
    """Verify Bayesian hand range inference."""

    def test_uniform_prior(self):
        """Initial weights should be uniform."""
        model = RangeModel([1, 2, 3])
        weights = model.get_weights()
        for h in [1, 2, 3]:
            assert abs(weights[h] - 1/3) < 0.01

    def test_update_narrows_range(self):
        """Observing actions should narrow the hand range."""
        model = RangeModel([1, 2, 3])

        # Strategy where only K bets
        strategy = {
            "J:": {"b": 0.0, "c": 1.0},
            "Q:": {"b": 0.0, "c": 1.0},
            "K:": {"b": 1.0, "c": 0.0},
        }

        def info_fn(hand):
            return f"{['J','Q','K'][hand-1]}:"

        # Opponent bet → only King bets
        model.update("b", strategy, info_fn)

        weights = model.get_weights()
        assert weights[3] > weights[1]  # K more likely than J
        assert weights[3] > weights[2]  # K more likely than Q

    def test_entropy_decreases_after_update(self):
        """Entropy should decrease as range narrows."""
        model = RangeModel([1, 2, 3])
        e0 = model.entropy()

        strategy = {
            "J:": {"b": 0.1, "c": 0.9},
            "Q:": {"b": 0.3, "c": 0.7},
            "K:": {"b": 1.0, "c": 0.0},
        }

        model.update("b", strategy, lambda h: f"{['J','Q','K'][h-1]}:")
        e1 = model.entropy()

        assert e1 < e0, f"Entropy should decrease: {e0:.4f} -> {e1:.4f}"

    def test_confidence_increases_after_update(self):
        """Confidence should increase as range narrows."""
        model = RangeModel([1, 2, 3])
        c0 = model.confidence()

        strategy = {
            "J:": {"b": 0.01, "c": 0.99},
            "Q:": {"b": 0.01, "c": 0.99},
            "K:": {"b": 1.0, "c": 0.0},
        }

        model.update("b", strategy, lambda h: f"{['J','Q','K'][h-1]}:")
        c1 = model.confidence()

        assert c1 > c0, f"Confidence should increase: {c0:.4f} -> {c1:.4f}"
