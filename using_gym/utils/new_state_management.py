import collections
import numpy as np
from scipy.stats import gamma
import warnings
from datetime import time, datetime

def get_pkpd_discount_factors(t_peak, t_end, n_steps):
    shape_k = 2
    scale_theta = t_peak / (shape_k - 1)
    time_points = np.linspace(0, t_end, n_steps)
    pdf_values = gamma.pdf(time_points, a=shape_k, scale=scale_theta)
    f_k = pdf_values / np.max(pdf_values)
    cdf_values = gamma.cdf(time_points, a=shape_k, scale=scale_theta)
    F_k = cdf_values
    return f_k, F_k

def risk_index(bg_values, horizon):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        bg_values = np.array(bg_values[-horizon:])
        bg_values[bg_values < 1] = 1
        fBG = 1.509 * (np.log(bg_values)**1.084 - 5.381)
        rl = 10 * fBG[fBG < 0]**2
        rh = 10 * fBG[fBG > 0]**2
        lbgi = np.nan_to_num(np.mean(rl))
        hbgi = np.nan_to_num(np.mean(rh))
        ri = lbgi + hbgi
    return ri

class StateRewardManager:
    def __init__(self, state_dim, cohort_name='adult'):
        self.cohort_name = cohort_name
        self.glucose_history = collections.deque(maxlen=6)
        self.insulin_history = collections.deque(maxlen=160)
        self.reset()
        _, self.F_k = get_pkpd_discount_factors(t_peak=55, t_end=480, n_steps=160)

        # Updated normalization stats for the 12-dimensional state
        adult_mean = np.array([140.0, 0.0, 2.0, 10.0, 0.0, 0.0] + [140.0] * 6)
        adult_std = np.array([40.0, 1.5, 2.0, 20.0, 1.0, 1.0] + [40.0] * 6)
        
        adolescent_mean = np.array([150.0, 0.0, 2.5, 15.0, 0.0, 0.0] + [150.0] * 6)
        adolescent_std = np.array([50.0, 2.0, 2.5, 25.0, 1.0, 1.0] + [50.0] * 6)

        if self.cohort_name == 'adolescent':
            self.state_mean = adolescent_mean
            self.state_std = adolescent_std
        else:
            self.state_mean = adult_mean
            self.state_std = adult_std

    def get_normalized_state(self, state):
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def calculate_iob(self):
        return np.sum(np.array(list(self.insulin_history)[::-1]) * (1 - self.F_k))

    def get_full_state(self, observation, current_sim_time, upcoming_carbs=0):
        glucose_value = observation[0]
        self.glucose_history.append(glucose_value)
        
        padded_history = list(self.glucose_history)
        while len(padded_history) < 6:
            padded_history.insert(0, padded_history[0])

        rate = (padded_history[-1] - padded_history[-2]) / 5.0
        iob = self.calculate_iob()

        time_in_seconds = current_sim_time.hour * 3600 + current_sim_time.minute * 60 + current_sim_time.second
        sin_time = np.sin(2 * np.pi * time_in_seconds / 86400)
        cos_time = np.cos(2 * np.pi * time_in_seconds / 86400)

        state_vector = np.concatenate([
            [glucose_value, rate, iob, upcoming_carbs, sin_time, cos_time],
            padded_history
        ])
        return state_vector

    def get_reward(self, state):
        if self.cohort_name == 'adolescent':
            return self.get_adolescent_reward(state)
        else:
            return self.get_adult_reward(state)

    def get_adult_reward(self, state):
        glucose = state[0]
        iob = state[2]
        reward = 0
        if 70 <= glucose <= 180: reward += 1
        if glucose < 70: reward -= 0.2 * ((70 - glucose)**2)
        if glucose > 180: reward -= 0.1 * ((glucose - 180)**2)
        reward -= 0.001 * (iob**2)
        return reward

    def get_adolescent_reward(self, state):
        glucose = state[0]
        rate_of_change = state[1]
        iob = state[2]
        reward = -risk_index([glucose], 1)
        reward -= 0.1 * (rate_of_change**2)
        reward -= 0.02 * (iob**2)
        if glucose <= 40:
            reward -= 200
        return reward

    def reset(self):
        self.glucose_history.clear()
        for _ in range(6): self.glucose_history.append(140)
        self.insulin_history.clear()
        for _ in range(160): self.insulin_history.append(0)