# opponent_model/dirichlet_model.py
"""
Dirichlet-Multinomial Bayesian model for opponent ACTION tendencies.

For each information set I, the opponent has a true (unknown) action 
frequency vector θ_I. We maintain a Dirichlet posterior over θ_I.

Dirichlet is the conjugate prior for multinomial distributions, so the
posterior update is just: alpha[a] += 1 when action a is observed.

This gives us:
- Posterior mean of opponent's action frequencies
- 95% credible intervals on each action
- KL divergence from Nash equilibrium
"""
import numpy as np
from scipy.stats import beta as beta_dist
from collections import defaultdict

class DirichletOpponentModel:
    def __init__(self, alpha_prior=1.0):
        """
        alpha_prior: Dirichlet concentration parameter.
          1.0 = Laplace smoothing (uniform prior)
          <1  = sparse prior (assume opponent rarely uses some actions)
          >1  = smoothed toward uniform
        """
        self.alpha_prior = alpha_prior
        # counts[info_set][action] = alpha_prior + observed_count
        self.counts = defaultdict(lambda: defaultdict(lambda: alpha_prior))
        self.n_observations = defaultdict(int)

    def observe(self, info_set, action):
        """Record that opponent took 'action' at 'info_set'."""
        self.counts[info_set][action] += 1.0
        self.n_observations[info_set] += 1

    def posterior_mean(self, info_set, actions):
        """
        Expected action frequencies under the Dirichlet posterior.
        This is your best estimate of opponent's strategy.
        """
        counts = {a: self.counts[info_set][a] for a in actions}
        total  = sum(counts.values())
        return {a: counts[a] / total for a in actions}

    def credible_interval(self, info_set, action, actions, ci=0.95):
        """
        95% credible interval on P(opponent plays 'action' at info_set).
        Uses Beta distribution (marginal of Dirichlet).
        """
        alpha_a     = self.counts[info_set][action]
        alpha_rest  = sum(self.counts[info_set][a] for a in actions if a != action)
        lo, hi      = beta_dist.ppf([(1-ci)/2, 1-(1-ci)/2], alpha_a, alpha_rest)
        return lo, hi

    def posterior_variance(self, info_set, action, actions):
        """
        Variance of the posterior estimate of P(opponent plays action).

        For Dirichlet-Multinomial, the marginal for each action is Beta.
        Beta(alpha_a, alpha_0 - alpha_a) has variance:
          Var = (alpha_a * (alpha_0 - alpha_a)) / (alpha_0^2 * (alpha_0 + 1))

        High variance = uncertain = stay close to GTO.
        Low variance  = confident = can exploit.
        """
        alpha_a = self.counts[info_set][action]
        alpha_rest = sum(self.counts[info_set][a] for a in actions if a != action)
        alpha_0 = alpha_a + alpha_rest

        if alpha_0 < 2:
            return 1.0   # maximum uncertainty if almost no data

        variance = (alpha_a * alpha_rest) / (alpha_0 ** 2 * (alpha_0 + 1))
        return variance

    def confidence_from_variance(self, info_set, actions, nash_strategy):
        """
        Composite confidence score:
          1. Low posterior variance on the most exploitable action -> high confidence
          2. KL divergence from Nash is large -> there's actually something to exploit

        Both conditions must hold to deviate from GTO.
        This is the exact trading analogy: signal must exist AND be precise.
        """
        if self.n_observations.get(info_set, 0) < 3:
            return 0.0

        # Find the action with highest deviation from Nash
        P = self.posterior_mean(info_set, actions)
        Q = nash_strategy.get(info_set, {a: 1.0 / len(actions) for a in actions})

        max_dev_action = max(actions, key=lambda a: abs(P.get(a, 0) - Q.get(a, 0)))

        # Variance on that action
        var = self.posterior_variance(info_set, max_dev_action, actions)

        # KL signal strength
        kl = self.kl_from_nash(info_set, actions, nash_strategy)

        # Confidence = signal strength x precision
        # var is small when confident, so use 1/(1 + var*scale)
        precision = 1.0 / (1.0 + var * 20.0)
        kl_signal = min(kl / 0.3, 1.0)     # saturates at KL = 0.3

        return precision * kl_signal

    def kl_from_nash(self, info_set, actions, nash_strategy):
        """
        KL divergence between opponent's estimated strategy and Nash.
        KL(P || Q) = Σ P(a) * log(P(a) / Q(a))
        High KL = opponent deviates from GTO = exploitable.
        """
        P = self.posterior_mean(info_set, actions)
        Q = nash_strategy.get(info_set, {a: 1.0/len(actions) for a in actions})

        kl = 0.0
        for a in actions:
            p = P.get(a, 1e-10)
            q = max(Q.get(a, 1e-10), 1e-10)
            if p > 0:
                kl += p * np.log(p / q)
        return kl

    def total_kl_divergence(self, nash_strategy):
        """
        Sum of KL divergences across all observed information sets.
        This is your single-number 'how exploitable is this opponent' score.
        """
        total_kl = 0.0
        for i_set in self.n_observations:
            if self.n_observations[i_set] < 2:
                continue   # Need at least some data
            actions = list(self.counts[i_set].keys())
            if i_set in nash_strategy:
                total_kl += self.kl_from_nash(i_set, actions, nash_strategy)
        return total_kl

    def get_deviations(self, nash_strategy, threshold=0.15):
        """
        Find information sets where opponent significantly deviates from Nash.
        Returns list of (info_set, action, opponent_freq, nash_freq, deviation).
        Sorted by deviation size.
        """
        deviations = []
        for i_set in self.n_observations:
            if self.n_observations[i_set] < 5:
                continue
            actions = list(self.counts[i_set].keys())
            P = self.posterior_mean(i_set, actions)
            Q = nash_strategy.get(i_set, {a: 1.0/len(actions) for a in actions})

            for a in actions:
                dev = abs(P.get(a, 0) - Q.get(a, 0))
                if dev > threshold:
                    deviations.append((i_set, a, P.get(a, 0), Q.get(a, 0), dev))

        deviations.sort(key=lambda x: -x[4])
        return deviations

    def summary(self, nash_strategy):
        """Print a readable summary of opponent tendencies."""
        print("=== Opponent Model Summary ===")
        total_kl = self.total_kl_divergence(nash_strategy)
        print(f"Total KL from Nash: {total_kl:.4f}  (0=GTO, higher=more exploitable)\n")

        devs = self.get_deviations(nash_strategy)
        if devs:
            print("Significant deviations from Nash:")
            for i_set, action, opp_f, nash_f, dev in devs[:5]:
                print(f"  {i_set:20s}  action={action}  "
                      f"opponent={opp_f:.2f}  nash={nash_f:.2f}  "
                      f"deviation={dev:.2f}")
        else:
            print("No significant deviations detected yet.")
