import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import collections
from scipy.stats import gamma

# --- Helper Functions for Bio-Inspired Components ---

def get_pkpd_discount_factors(t_peak, t_end, n_steps):
    """
    Generates the PK/PD-driven discount factors based on a gamma distribution,
    as described in the paper.

    Args:
        t_peak (int): The time to peak insulin action in minutes.
        t_end (int): The total duration of insulin action in minutes.
        n_steps (int): The number of time steps corresponding to t_end.

    Returns:
        tuple: A tuple containing (f_k, F_k), the PDF-like and CDF-like
               discount factors.
    """
    # The paper uses a gamma distribution with shape k=2
    shape_k = 2
    # The scale is derived from the peak time: t_peak = (k-1) * scale
    scale_theta = t_peak / (shape_k - 1)
    
    # Create time points for the distribution
    time_points = np.linspace(0, t_end, n_steps)
    
    # Gamma PDF for f_k (sequential influence)
    pdf_values = gamma.pdf(time_points, a=shape_k, scale=scale_theta)
    f_k = pdf_values / np.max(pdf_values) # Min-max normalization
    
    # Gamma CDF for F_k (cumulative influence)
    cdf_values = gamma.cdf(time_points, a=shape_k, scale=scale_theta)
    F_k = cdf_values # CDF is already scaled between 0 and 1
    
    return f_k, F_k


# --- Agent Definition ---

class BioInspiredAgentWithState(object):
    """
    An agent that now calculates the full state representation, including IOB.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        
        # Bio-inspired reward parameters
        self.g_target = 120.0
        self.m_target = -1/15.0
        
        # State calculation components
        self.glucose_history = collections.deque(maxlen=2)
        self.insulin_history = collections.deque(maxlen=160) # Store last 8 hours (160 * 3 min)
        for _ in range(160): self.insulin_history.append(0)

        # Get the discount factors for IOB calculation
        # Assuming peak insulin at 55 mins, duration 8 hours (480 mins)
        # Number of steps = 480 mins / 3 mins/step = 160
        self.f_k, self.F_k = get_pkpd_discount_factors(t_peak=55, t_end=480, n_steps=160)

    def select_action(self, state):
        """Selects a random action for now."""
        return self.action_space.sample()

    def calculate_iob(self):
        """
        Calculates Insulin on Board (IOB) using the formula from the paper.
        IOB_t = sum(i_{t-k} * (1 - F_k))
        """
        # Reverse the insulin history to align with F_k (most recent insulin first)
        recent_insulin = np.array(list(self.insulin_history)[::-1])
        iob = np.sum(recent_insulin * (1 - self.F_k))
        return iob

    def get_full_state(self, observation):
        """
        Constructs the full state tuple (g, v, iob).

        Args:
            observation (float): The current glucose reading.

        Returns:
            tuple: The full state (glucose, glucose_rate, iob).
        """
        self.glucose_history.append(observation)
        
        if len(self.glucose_history) < 2:
            glucose_rate = 0.0
        else:
            glucose_rate = (self.glucose_history[1] - self.glucose_history[0]) / 3.0
            
        iob = self.calculate_iob()
        
        return (observation, glucose_rate, iob)

    def reset(self):
        """Resets the agent's internal state."""
        self.glucose_history.clear()
        self.insulin_history.clear()
        for _ in range(160): self.insulin_history.append(0)
        print("Agent state and histories have been reset.")


# --- Main Simulation Script ---

def main():
    PATIENT_NAME = 'adolescent#001'
    CLEAN_PATIENT_NAME = PATIENT_NAME.replace('#', '-')
    ENV_ID = f'simglucose/{CLEAN_PATIENT_NAME}-v0'

    register(
        id=ENV_ID,
        entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        max_episode_steps=288,
        kwargs={"patient_name": PATIENT_NAME},
    )

    env = gymnasium.make(ENV_ID)
    agent = BioInspiredAgentWithState(env)
    
    observation_array, info = env.reset()
    observation = observation_array[0]
    agent.reset()
    agent.glucose_history.append(observation)

    print(f"\n--- Starting Simulation with {PATIENT_NAME} ---")

    for t in range(200):
        # Get the full state
        current_state = agent.get_full_state(observation)
        g, v, iob = current_state

        # Agent selects action based on state
        action = agent.select_action(current_state)
        
        # Store the action taken for future IOB calculations
        agent.insulin_history.append(action[0])
        
        # Environment steps forward
        next_obs_array, reward, terminated, truncated, info = env.step(action)
        observation = next_obs_array[0]
        
        print(f"\n-> Timestep {t + 1}")
        print(f"  - State (g, v, IOB): ({g:.2f} mg/dL, {v:.2f} mg/dL/min, {iob:.2f} U)")
        print(f"  - Action Taken (Insulin): {action[0]:.4f} U/min")
        
        if terminated or truncated:
            print(f"\nEpisode finished after {t + 1} timesteps.")
            observation_array, info = env.reset()
            observation = observation_array[0]
            agent.reset()
            agent.glucose_history.append(observation)

    env.close()
    print("\n--- Simulation Finished ---")

if __name__ == "__main__":
    main()
