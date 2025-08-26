import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import collections
from scipy.stats import gamma
import torch
import torch.nn as nn
from torch.distributions import Categorical

# --- Helper Functions for Bio-Inspired Components (from previous step) ---

def get_pkpd_discount_factors(t_peak, t_end, n_steps):
    shape_k = 2
    scale_theta = t_peak / (shape_k - 1)
    time_points = np.linspace(0, t_end, n_steps)
    pdf_values = gamma.pdf(time_points, a=shape_k, scale=scale_theta)
    f_k = pdf_values / np.max(pdf_values)
    cdf_values = gamma.cdf(time_points, a=shape_k, scale=scale_theta)
    F_k = cdf_values
    return f_k, F_k

# --- PPO Neural Network Definitions ---

class Actor(nn.Module):
    """
    The Actor network learns the policy, i.e., maps state to action.
    """
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.relu(self.layer_1(state))
        x = self.relu(self.layer_2(x))
        # Output is a probability distribution over actions
        action_probs = self.softmax(self.layer_3(x))
        return action_probs

class Critic(nn.Module):
    """
    The Critic network learns the value function, i.e., estimates the
    expected return from a given state.
    """
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, 1) # Outputs a single value
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.layer_1(state))
        x = self.relu(self.layer_2(x))
        value = self.layer_3(x)
        return value

# --- PPO Agent Definition ---

class PPOAgent(object):
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        # Discretize the action space as in the paper
        self.action_dim = 72
        self.action_bins = self._create_action_bins()

        # Initialize Actor and Critic networks
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)
        
        # State calculation components
        self.glucose_history = collections.deque(maxlen=2)
        self.insulin_history = collections.deque(maxlen=160)
        for _ in range(160): self.insulin_history.append(0)
        _, self.F_k = get_pkpd_discount_factors(t_peak=55, t_end=480, n_steps=160)

    def _create_action_bins(self):
        # As per paper: 72 categories, including 0 pmol and 70 doses
        # from 10 pmol to 1,800 pmol.
        # This is a simplified version. A more precise implementation would
        # match the paper's 5 pmol and 50 pmol increments.
        # For now, we use linear spacing.
        # Note: The environment expects U/min, not pmol. We will need to convert.
        # For now, we use the environment's continuous space and discretize it.
        low = self.env.action_space.low[0]
        high = self.env.action_space.high[0]
        return np.linspace(low, high, self.action_dim)

    def select_action(self, state):
        """
        Selects an action using the Actor network.
        """
        # Convert state to a PyTorch tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action probabilities from the actor
        action_probs = self.actor(state_tensor)
        
        # Create a categorical distribution and sample an action index
        dist = Categorical(action_probs)
        action_index = dist.sample()
        
        # Get the continuous action value from our bins
        action_val = self.action_bins[action_index.item()]
        
        # Return the action in the format the environment expects
        return np.array([action_val])

    def calculate_iob(self):
        recent_insulin = np.array(list(self.insulin_history)[::-1])
        iob = np.sum(recent_insulin * (1 - self.F_k))
        return iob

    def get_full_state(self, observation):
        self.glucose_history.append(observation)
        glucose_rate = (self.glucose_history[1] - self.glucose_history[0]) / 3.0 if len(self.glucose_history) == 2 else 0.0
        iob = self.calculate_iob()
        return np.array([observation, glucose_rate, iob])

    def reset(self):
        self.glucose_history.clear()
        self.insulin_history.clear()
        for _ in range(160): self.insulin_history.append(0)
        print("Agent state and histories have been reset.")


# --- Main Simulation Script ---

def main():
    PATIENT_NAME = 'adolescent#001'
    CLEAN_PATIENT_NAME = PATIENT_NAME.replace('#', '-')
    ENV_ID = f'simglucose/{CLEAN_PATIENT_NAME}-v0'

    register(id=ENV_ID, entry_point="simglucose.envs:T1DSimGymnaisumEnv", max_episode_steps=288, kwargs={"patient_name": PATIENT_NAME})

    env = gymnasium.make(ENV_ID)
    # The state from the environment is just glucose, but our agent needs (g, v, iob)
    # We will manually construct the state for the agent.
    # To do this correctly, we need to know the state dimension.
    # Let's redefine the env observation space to match our agent's needs.
    env.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)

    agent = PPOAgent(env)
    
    observation_array, info = env.reset()
    agent.reset()
    
    # Manually create the first full state
    observation = observation_array[0]
    agent.glucose_history.append(observation)
    current_state = agent.get_full_state(observation)

    print(f"\n--- Starting Simulation with PPO Agent (Untrained) ---")

    for t in range(200):
        action = agent.select_action(current_state)
        agent.insulin_history.append(action[0])
        
        next_obs_array, reward, terminated, truncated, info = env.step(action)
        
        # Construct the next full state
        next_observation = next_obs_array[0]
        next_state = agent.get_full_state(next_observation)
        
        print(f"\n-> Timestep {t + 1}")
        print(f"  - State (g, v, IOB): ({current_state[0]:.2f}, {current_state[1]:.2f}, {current_state[2]:.2f})")
        print(f"  - Action Taken (Insulin): {action[0]:.4f} U/min")
        
        current_state = next_state
        
        if terminated or truncated:
            print(f"\nEpisode finished after {t + 1} timesteps.")
            observation_array, info = env.reset()
            agent.reset()
            observation = observation_array[0]
            agent.glucose_history.append(observation)
            current_state = agent.get_full_state(observation)

    env.close()
    print("\n--- Simulation Finished ---")

if __name__ == "__main__":
    main()
