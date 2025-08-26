import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import collections
from scipy.stats import gamma
import torch
import torch.nn as nn
from torch.distributions import Categorical
import simglucose.simulation.scenario_gen as scgen
from simglucose.simulation.scenario import CustomScenario
from datetime import datetime
import matplotlib.pyplot as plt
import os

# --- Helper Functions for Bio-Inspired Components ---
def get_pkpd_discount_factors(t_peak, t_end, n_steps):
    shape_k = 2
    scale_theta = t_peak / (shape_k - 1)
    time_points = np.linspace(0, t_end, n_steps)
    pdf_values = gamma.pdf(time_points, a=shape_k, scale=scale_theta)
    f_k = pdf_values / np.max(pdf_values)
    cdf_values = gamma.cdf(time_points, a=shape_k, scale=scale_theta)
    F_k = cdf_values
    return f_k, F_k

# --- NEW: Rule-Based Safety Layer ---
class SafetyLayer:
    def __init__(self):
        # Define safety thresholds
        self.hypo_threshold = 80  # Override if BG is below 80 mg/dL
        self.predictive_low_threshold = 110 # Check for rapid drops if BG is below 110 mg/dL

    def apply(self, action, state):
        """
        Overrides the agent's action if it is unsafe.

        Args:
            action: The action chosen by the RL agent.
            state: The current unnormalized state (g, v, iob).

        Returns:
            The corrected, safe action.
        """
        glucose, rate_of_change, _ = state
        
        # Rule 1: Absolute Hypoglycemia Override
        # If glucose is already low, deliver no insulin.
        if glucose < self.hypo_threshold:
            return np.array([0.0]) # Force action to zero

        # Rule 2: Predictive Low Glucose Suspend
        # If glucose is trending down and approaching the low threshold, suspend insulin.
        if glucose < self.predictive_low_threshold and rate_of_change < -1.0:
            return np.array([0.0]) # Force action to zero
            
        # If no safety rules are triggered, allow the agent's chosen action.
        return action

# --- Memory Buffer ---
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# --- PPO LSTM Neural Network Definitions ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(state_dim, n_latent_var, batch_first=True)
        self.layer_1 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state, hidden):
        state = state.unsqueeze(1)
        lstm_out, hidden = self.lstm(state, hidden)
        x = self.relu(self.layer_1(lstm_out))
        action_probs = self.softmax(self.layer_2(x))
        return action_probs.squeeze(1), hidden

class Critic(nn.Module):
    def __init__(self, state_dim, n_latent_var):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(state_dim, n_latent_var, batch_first=True)
        self.layer_1 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, 1)
        self.relu = nn.ReLU()

    def forward(self, state, hidden):
        state = state.unsqueeze(1)
        lstm_out, hidden = self.lstm(state, hidden)
        x = self.relu(self.layer_1(lstm_out))
        value = self.layer_2(x)
        return value.squeeze(1), hidden

# --- PPO Agent with Bio-Inspired Learning ---
class BioInspiredPPOAgent:
    def __init__(self, env, state_dim, action_dim, n_latent_var, lr, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.actor = Actor(state_dim, action_dim, n_latent_var)
        self.actor_old = Actor(state_dim, action_dim, n_latent_var)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, n_latent_var)
        
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + 
            list(self.critic.parameters()), lr=lr
        )
        self.MseLoss = nn.MSELoss()

        self.action_bins = np.linspace(env.action_space.low[0], env.action_space.high[0], action_dim)

    def select_action(self, state, hidden, memory):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, hidden = self.actor_old(state_tensor, hidden)
        dist = Categorical(action_probs)
        action_index = dist.sample()
        action_val = self.action_bins[action_index.item()]
        
        if memory is not None:
            memory.states.append(state_tensor)
            memory.actions.append(action_index)
            memory.logprobs.append(dist.log_prob(action_index))
        
        return np.array([action_val]), hidden

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.cat(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()

        for _ in range(self.K_epochs):
            action_probs, _ = self.actor(old_states, None)
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            state_values, _ = self.critic(old_states, None)
            
            advantages = rewards - state_values.squeeze().detach()
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + \
                   0.5 * self.MseLoss(state_values.squeeze(), rewards) - \
                   0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.actor_old.load_state_dict(self.actor.state_dict())

# --- State Normalization and Reward Calculation ---
class StateRewardManager:
    def __init__(self, state_dim):
        self.glucose_history = collections.deque(maxlen=2)
        self.insulin_history = collections.deque(maxlen=160)
        self.reset()
        self.f_k, self.F_k = get_pkpd_discount_factors(t_peak=55, t_end=480, n_steps=160)

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
        recent_insulin = np.array(list(self.insulin_history)[::-1])
        return np.sum(recent_insulin * (1 - self.F_k))
        
    def get_full_state(self, observation):
        self.glucose_history.append(observation)
        rate = (self.glucose_history[1] - self.glucose_history[0]) / 3.0 if len(self.glucose_history) == 2 else 0.0
        iob = self.calculate_iob()
        return np.array([observation, rate, iob])

    def get_reward(self, state):
        g, v, iob = state
        reward = 0
        
        if 70 <= g <= 180:
            reward += 1
        
        if g < 70:
            reward -= 200
            
        if g > 180:
            reward -= (g - 180) * 0.1
            
        reward -= iob * 0.05
        
        return reward

    def reset(self):
        self.glucose_history.clear()
        for _ in range(2): self.glucose_history.append(140)
        self.insulin_history.clear()
        for _ in range(160): self.insulin_history.append(0)

# --- Main Training and Evaluation Script ---
def main():
    PATIENT_NAME = 'adolescent#001'
    max_episodes = 2000
    update_timestep = 5000
    lr = 5e-5
    eps_clip = 0.1
    gamma = 0.99
    K_epochs = 4
    max_timesteps = 288
    n_latent_var = 64
    
    if not os.path.exists('./models'):
        os.makedirs('./models')
    actor_path = f'./models/ppo_actor_safe_lstm_{PATIENT_NAME.replace("#", "-")}.pth'
    critic_path = f'./models/ppo_critic_safe_lstm_{PATIENT_NAME.replace("#", "-")}.pth'

    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())
    meal_scenario = scgen.RandomScenario(start_time=start_time, seed=1)
    CLEAN_PATIENT_NAME = PATIENT_NAME.replace('#', '-')
    ENV_ID = f'simglucose/{CLEAN_PATIENT_NAME}-v0'
    
    register(id=ENV_ID, entry_point="simglucose.envs:T1DSimGymnaisumEnv", max_episode_steps=max_timesteps, kwargs={"patient_name": PATIENT_NAME, "custom_scenario": meal_scenario})
    env = gymnasium.make(ENV_ID)
    
    state_dim = 3
    action_dim = 72
    
    memory = Memory()
    agent = BioInspiredPPOAgent(env, state_dim, action_dim, n_latent_var, lr, gamma, K_epochs, eps_clip)
    manager = StateRewardManager(state_dim)
    safety_layer = SafetyLayer() # Initialize the safety layer
    
    timestep = 0
    
    episode_lengths = []
    final_glucose_levels = []
    
    print("--- Starting Training with Safety Layer & LSTM Agent ---")
    for i_episode in range(1, max_episodes + 1):
        obs_array, info = env.reset()
        manager.reset()
        unnormalized_state = manager.get_full_state(obs_array[0])
        current_state = manager.get_normalized_state(unnormalized_state)
        hidden = None
        
        for t in range(max_timesteps):
            timestep += 1
            
            # Agent proposes an action
            proposed_action, hidden = agent.select_action(current_state, hidden, memory)
            # Safety layer corrects the action if necessary
            safe_action = safety_layer.apply(proposed_action, unnormalized_state)
            
            manager.insulin_history.append(safe_action[0])
            obs_array, _, terminated, truncated, _ = env.step(safe_action)
            done = terminated or truncated
            
            unnormalized_state = manager.get_full_state(obs_array[0])
            current_state = manager.get_normalized_state(unnormalized_state)
            
            reward = manager.get_reward(unnormalized_state)
            
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                timestep = 0
            
            if done:
                break
        
        episode_lengths.append(t + 1)
        final_glucose_levels.append(unnormalized_state[0])
        
        if i_episode % 100 == 0:
            print(f"Episode {i_episode}/{max_episodes} finished. Avg Length: {np.mean(episode_lengths[-100:]):.2f}")
    
    print("--- Training Finished ---")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_lengths)
    plt.title('Episode Length over Time')
    plt.xlabel('Episode')
    plt.ylabel('Length (Timesteps)')
    
    plt.subplot(1, 2, 2)
    plt.plot(final_glucose_levels)
    plt.axhline(y=70, color='r', linestyle='--', label='Hypo Threshold')
    plt.title('Final Glucose Level per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Glucose (mg/dL)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Saving trained model...")
    torch.save(agent.actor_old.state_dict(), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)
    
    print("\n--- Starting Evaluation ---")
    eval_scenario = CustomScenario(start_time=start_time, scenario=[(7, 45), (12, 70), (18, 80)])
    env.close()
    
    register(id=ENV_ID + "-eval", entry_point="simglucose.envs:T1DSimGymnaisumEnv", max_episode_steps=max_timesteps, kwargs={"patient_name": PATIENT_NAME, "custom_scenario": eval_scenario})
    eval_env = gymnasium.make(ENV_ID + "-eval")
    
    eval_agent = BioInspiredPPOAgent(eval_env, state_dim, action_dim, n_latent_var, lr, gamma, K_epochs, eps_clip)
    eval_agent.actor_old.load_state_dict(torch.load(actor_path))
    
    manager.reset()
    obs_array, info = eval_env.reset()
    unnormalized_state = manager.get_full_state(obs_array[0])
    current_state = manager.get_normalized_state(unnormalized_state)
    glucose_history = [obs_array[0]]
    hidden_eval = None
    
    for t in range(max_timesteps):
        # During evaluation, the safety layer is still active for a fair comparison
        proposed_action, hidden_eval = eval_agent.select_action(current_state, hidden_eval, None)
        safe_action = safety_layer.apply(proposed_action, unnormalized_state)
        
        manager.insulin_history.append(safe_action[0])
        obs_array, _, terminated, truncated, _ = eval_env.step(safe_action)
        
        unnormalized_state = manager.get_full_state(obs_array[0])
        current_state = manager.get_normalized_state(unnormalized_state)
        glucose_history.append(obs_array[0])
        if terminated or truncated:
            break
            
    eval_env.close()
    
    glucose_history = np.array(glucose_history)
    time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
    time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
    time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
    mean_glucose = np.mean(glucose_history)

    print("\n--- Evaluation Results ---")
    print(f"Mean Glucose: {mean_glucose:.2f} mg/dL")
    print(f"Time in Range (70-180 mg/dL): {time_in_range:.2f}%")
    print(f"Time in Hypoglycemia (<70 mg/dL): {time_hypo:.2f}%")
    print(f"Time in Hyperglycemia (>180 mg/dL): {time_hyper:.2f}%")

    plt.figure(figsize=(15, 6))
    plt.plot(glucose_history)
    plt.axhline(y=180, color='r', linestyle='--', label='Hyperglycemia Threshold')
    plt.axhline(y=70, color='orange', linestyle='--', label='Hypoglycemia Threshold')
    plt.axhline(y=140, color='g', linestyle='-', label='Target')
    plt.title(f'Trained Agent Performance for {PATIENT_NAME}')
    plt.xlabel('Time (steps)')
    plt.ylabel('Blood Glucose (mg/dL)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
