# utils/state_management.py

import collections
import numpy as np
from scipy.stats import gamma

def get_pkpd_discount_factors(t_peak, t_end, n_steps):
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
        _, self.F_k = get_pkpd_discount_factors(t_peak=55, t_end=480, n_steps=160)
        self.running_state_mean, self.running_state_std, self.n_observations = np.zeros(state_dim), np.ones(state_dim), 0

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
        return np.sum(np.array(list(self.insulin_history)[::-1]) * (1 - self.F_k))
        
    def get_full_state(self, observation, upcoming_carbs=0):
        self.glucose_history.append(observation)
        rate = (self.glucose_history[1] - self.glucose_history[0]) / 3.0 if len(self.glucose_history) == 2 else 0.0
        iob = self.calculate_iob()
        return np.array([observation, rate, iob, upcoming_carbs])

    def get_reward(self, state):
        # Unpack the state; note that upcoming_carbs is now part of it
        g, _, iob, _ = state 
        reward = 0
        if 70 <= g <= 180: reward += 1
        if g < 70: reward -= 200
        if g > 180: reward -= (g - 180) * 0.1
        reward -= 0.5 * (iob**2)
        return reward

    def reset(self):
        self.glucose_history.clear()
        for _ in range(2): self.glucose_history.append(140)
        self.insulin_history.clear()
        for _ in range(160): self.insulin_history.append(0)