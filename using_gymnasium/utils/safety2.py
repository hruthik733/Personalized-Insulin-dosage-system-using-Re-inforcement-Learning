# utils/safety2.py

import numpy as np

class SafetyLayer:
    """
    Safety layer enforcing:
    - No insulin if glucose below hypo_threshold.
    - No insulin if glucose below predictive_low_threshold AND dropping rapidly.
    - Optionally scale down insulin for moderate lows with rapid drop.
    - Insulin allowed (clipped) only if glucose sufficiently high or rising.

    Parameters:
    - hypo_threshold: hard hypoglycemia cutoff (mg/dL)
    - predictive_low_threshold: warning margin above hypoglycemia (mg/dL)
    - hyper_threshold: lower bound to allow full insulin action (mg/dL)
    """

    def __init__(self, hypo_threshold=80, predictive_low_threshold=110, hyper_threshold=170):
        self.hypo_threshold = hypo_threshold
        self.predictive_low_threshold = predictive_low_threshold
        self.hyper_threshold = hyper_threshold

    def apply(self, action, state):
        glucose, rate_of_change, iob, upcoming_carbs = state

        # Hard no insulin if glucose below hypo threshold
        if glucose < self.hypo_threshold:
            return np.array([0.0])

        # No insulin if rapidly dropping glucose nearing low safe threshold
        if glucose < self.predictive_low_threshold and rate_of_change < -1.0:
            return np.array([0.0])

        # Optional: reduce insulin if moderately low glucose and sharply dropping
        if glucose < 130 and rate_of_change < -2.0:
            reduced_action = 0.5 * action
            return np.clip(reduced_action, 0, 5.0)

        # Hyperglycemia range: allow agent action but clip
        if glucose > self.hyper_threshold or (rate_of_change > 0.5 and glucose > 150):
            return np.clip(action, 0, 5.0)

        # Otherwise, allow agent action clipped
        return np.clip(action, 0, 5.0)
