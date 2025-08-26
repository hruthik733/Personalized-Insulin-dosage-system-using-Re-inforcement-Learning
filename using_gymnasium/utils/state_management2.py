# utils/state_management2.py

import collections
import numpy as np
from scipy.stats import gamma

def get_pkpd_discount_factors(t_peak, t_end, n_steps):
    """
    Pharmacokinetics/Pharmacodynamics insulin absorption curve
    """
    shape_k = 2
    scale_theta = t_peak / (shape_k - 1)
    time_points = np.linspace(0, t_end, n_steps)
    pdf_values = gamma.pdf(time_points, a=shape_k, scale=scale_theta)
    f_k = pdf_values / np.max(pdf_values)
    cdf_values = gamma.cdf(time_points, a=shape_k, scale=scale_theta)
    F_k = cdf_values
    return f_k, F_k

class StateRewardManager:
    def __init__(self, state_dim):
        self.glucose_history = collections.deque(maxlen=2)
        self.insulin_history = collections.deque(maxlen=160)
        self.reset()
        _, self.F_k = get_pkpd_discount_factors(
            t_peak=55, t_end=480, n_steps=160
        )
        self.running_state_mean = np.zeros(state_dim)
        self.running_state_std = np.ones(state_dim)
        self.n_observations = 0

    def update_normalization_stats(self, state):
        self.n_observations += 1
        old_mean = self.running_state_mean.copy()
        self.running_state_mean += (state - self.running_state_mean) / self.n_observations
        self.running_state_std += (state - old_mean) * (state - self.running_state_mean)

    def get_normalized_state(self, state):
        self.update_normalization_stats(state)
        std = np.sqrt(self.running_state_std / (self.n_observations if self.n_observations > 1 else 1))
        return (state - self.running_state_mean) / (std + 1e-8)

    def calculate_iob(self):
        """
        Insulin On Board from history and PK/PD curve
        """
        return np.sum(np.array(list(self.insulin_history)[::-1]) * (1 - self.F_k))

    def get_full_state(self, observation, upcoming_carbs=0):
        """
        observation: glucose level (mg/dL)
        """
        self.glucose_history.append(observation)
        rate = (
            (self.glucose_history[1] - self.glucose_history[0]) / 3.0
            if len(self.glucose_history) == 2
            else 0.0
        )
        iob = self.calculate_iob()
        return np.array([observation, rate, iob, upcoming_carbs])

    def get_reward(self, state):
        """
        Balanced and safe glucose control reward:
        1. Strong target at ~100 mg/dL using Gaussian shaping.
        2. Heavy penalty for hypoglycemia.
        3. Moderate penalty for hyperglycemia.
        4. Penalize excessive IOB and glucose volatility.
        """
        g, rate, iob, _ = state

        # --- 1. Gaussian-shaped target reward ---
        target_glucose = 100
        sigma = 20.0   # smaller sigma → sharper peak around target
        # Gaussian reward in [0, 1], then scale to, say, 0–10
        proximity_reward = 10.0 * np.exp(-0.5 * ((g - target_glucose) / sigma) ** 2)

        reward = proximity_reward

        # --- 2. Strong hypoglycemia penalty ---
        if g < 70:
            reward -= 200 * (1 + (70 - g) / 10)  # harsher if extremely low

        # --- 3. Hyperglycemia penalty ---
        if g > 180:
            reward -= min((g - 180) * 0.2, 50)  # cap extreme penalty

        # --- 4. Penalize high IOB to avoid stacking ---
        reward -= 0.5 * (iob ** 2)

        # --- 5. Penalize large rate of change to encourage stability ---
        reward -= 0.1 * (abs(rate) ** 1.5)

        return reward

    def reset(self):
        """
        Initialize history with safe baseline glucose and zero insulin
        """
        self.glucose_history.clear()
        for _ in range(2):
            self.glucose_history.append(140)  # start in safe mid-range
        self.insulin_history.clear()
        for _ in range(160):
            self.insulin_history.append(0)
