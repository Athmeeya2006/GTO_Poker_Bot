# opponent_model/hypothesis_test.py
"""
Sequential Probability Ratio Test (SPRT) for exploit detection.
Wald (1947) — optimal sequential test.

H0: opponent plays action a with probability p0 (GTO frequency)
H1: opponent plays action a with probability p1 = p0 + epsilon (leaked frequency)

SPRT accumulates a log-likelihood ratio. When it crosses upper threshold A,
we reject H0 (leak detected). When it crosses lower threshold B, we accept H0.

This reaches a decision in the MINIMUM number of observations on average,
among all tests with the same false positive and false negative rates.
"""
import numpy as np

class SPRTLeakDetector:
    def __init__(self, alpha=0.05, beta=0.05, epsilon=0.20):
        """
        alpha:   false positive rate (wrongly conclude there's a leak)
        beta:    false negative rate (miss a real leak)
        epsilon: minimum deviation from Nash we want to detect
        """
        self.alpha   = alpha
        self.beta    = beta
        self.epsilon = epsilon

        # SPRT thresholds
        self.A = np.log((1 - beta) / alpha)          # upper: reject H0
        self.B = np.log(beta / (1 - alpha))          # lower: accept H0

        # Per info_set per action: log-likelihood ratio
        self.log_LR = {}
        self.status = {}   # "H0", "H1", or "continue"
        self.n_obs  = {}

    def _key(self, info_set, action):
        return f"{info_set}|{action}"

    def update(self, info_set, action, observed_action, nash_freq):
        """
        Update the SPRT for 'action' at 'info_set'.

        observed_action: what opponent actually did
        nash_freq: P(action | Nash) = p0
        """
        key = self._key(info_set, action)
        if key not in self.log_LR:
            self.log_LR[key] = 0.0
            self.n_obs[key]  = 0
            self.status[key] = "continue"

        if self.status[key] != "continue":
            return self.status[key]   # test already decided

        p0 = nash_freq
        p1 = min(p0 + self.epsilon, 0.99)   # alternative hypothesis

        self.n_obs[key] += 1

        if observed_action == action:
            # Observed the action we're testing
            if p0 > 0 and p1 > 0:
                self.log_LR[key] += np.log(p1 / p0)
        else:
            # Did NOT observe the action
            if (1-p0) > 0 and (1-p1) > 0:
                self.log_LR[key] += np.log((1 - p1) / (1 - p0))

        # Check thresholds
        lr = self.log_LR[key]
        if lr >= self.A:
            self.status[key] = "H1"    # Leak detected
        elif lr <= self.B:
            self.status[key] = "H0"    # No leak, play GTO
        else:
            self.status[key] = "continue"   # Need more data

        return self.status[key]

    def get_detected_leaks(self):
        """Returns list of (info_set, action) where H1 was accepted."""
        leaks = []
        for key, status in self.status.items():
            if status == "H1":
                parts     = key.rsplit("|", 1)
                info_set  = parts[0]
                action    = parts[1]
                leaks.append((info_set, action, self.n_obs[key]))
        return leaks

    def observations_to_decide(self, info_set, action):
        """How many observations have we accumulated for this test?"""
        key = self._key(info_set, action)
        return self.n_obs.get(key, 0)

    def summary(self):
        leaks = self.get_detected_leaks()
        continuing = sum(1 for s in self.status.values() if s == "continue")
        H0_accepted = sum(1 for s in self.status.values() if s == "H0")
        print(f"SPRT Summary: {len(leaks)} leaks detected, "
              f"{H0_accepted} tests accepted H0, "
              f"{continuing} tests ongoing")
        if leaks:
            print("Detected leaks:")
            for i_set, action, n in leaks:
                print(f"  {i_set} | action={action} | after {n} obs")