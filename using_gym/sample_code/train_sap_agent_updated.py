import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import collections
from scipy.stats import gamma
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import simglucose.simulation.scenario_gen as scgen
from simglucose.simulation.scenario import CustomScenario
from datetime import datetime
import matplotlib.pyplot as plt
import os
import random

# --- Helper Functions & Safety Layer ---
def get_pkpd_discount_factors(t_peak, t_end, n_steps):
    shape_k = 2
    scale_theta = t_peak / (shape_k - 1)
    time_points = np.linspace(0, t_end, n_steps)
    pdf_values = gamma.pdf(time_points, a=shape_k, scale=scale_theta)
    f_k = pdf_values / np.max(pdf_values)
    cdf_values = gamma.cdf(time_points, a=shape_k, scale=scale_theta)
    F_k = cdf_values
    return f_k, F_k

class SafetyLayer:
    def __init__(self):
        self.hypo_threshold = 80
        self.predictive_low_threshold = 110
        self.max_insulin_rate = 2.0 # Stricter max insulin in U/min

    def apply(self, action, state):
        glucose, rate_of_change, _ = state
        # Rule 1 & 2: Suspend insulin if low or dropping fast
        if glucose < self.hypo_threshold or (glucose < self.predictive_low_threshold and rate_of_change < -1.0):
            return np.array([0.0])
        # Rule 3: Clip the action to a safe maximum
        return np.clip(action, 0.0, self.max_insulin_rate)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in np.random.choice(len(self.buffer), batch_size)])
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

# --- Soft Actor-Critic (SAC) Network Definitions ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_3 = nn.Linear(n_latent_var, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer_1(x))
        return self.layer_3(F.relu(self.layer_2(x)))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.mean_layer = nn.Linear(n_latent_var, action_dim)
        self.log_std_layer = nn.Linear(n_latent_var, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t) - torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# --- SAC Agent Implementation ---
class SACAgent:
    def __init__(self, env, state_dim, action_dim, n_latent_var, lr, gamma, tau, alpha):
        self.gamma, self.tau, self.alpha = gamma, tau, alpha
        self.max_action = float(env.action_space.high[0])
        self.actor = Actor(state_dim, action_dim, n_latent_var, self.max_action)
        self.critic_1 = Critic(state_dim, action_dim, n_latent_var)
        self.critic_1_target = Critic(state_dim, action_dim, n_latent_var)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2 = Critic(state_dim, action_dim, n_latent_var)
        self.critic_2_target = Critic(state_dim, action_dim, n_latent_var)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy().flatten()

    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state, action, reward, next_state, done = torch.FloatTensor(state), torch.FloatTensor(action), torch.FloatTensor(reward).unsqueeze(1), torch.FloatTensor(next_state), torch.FloatTensor(done).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q = torch.min(self.critic_1_target(next_state, next_action), self.critic_2_target(next_state, next_action)) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        critic_1_loss = F.mse_loss(self.critic_1(state, action), target_q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        critic_2_loss = F.mse_loss(self.critic_2(state, action), target_q)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_action, log_prob = self.actor.sample(state)
        actor_loss = (self.alpha * log_prob - torch.min(self.critic_1(state, new_action), self.critic_2(state, new_action))).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# --- State Manager ---
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
        
    def get_full_state(self, observation):
        self.glucose_history.append(observation)
        rate = (self.glucose_history[1] - self.glucose_history[0]) / 3.0 if len(self.glucose_history) == 2 else 0.0
        return np.array([observation, rate, self.calculate_iob()])

    def get_reward(self, state):
        g, v, iob = state
        reward = 0
        
        # Graduated penalty and recovery bonus
        previous_g = self.glucose_history[0]
        if g < 40:
            reward -= 500  # Harsh penalty
        elif g < 70:
            reward -= (70 - g) * 2  # Scaled penalty
        elif g <= 180:
            reward += 1  # In-range bonus
            if previous_g < 70: # Recovery bonus
                reward += 100
        else:
            reward -= (g - 180) * 0.05
        
        reward -= iob * 0.01
        return reward

    def reset(self):
        self.glucose_history.clear()
        for _ in range(2): self.glucose_history.append(140)
        self.insulin_history.clear()
        for _ in range(160): self.insulin_history.append(0)

# --- Live Plotting and Evaluation ---
def evaluate_and_plot(agent, manager, safety_layer, env_id, episode_num):
    eval_env = gymnasium.make(env_id)
    obs_array, _ = eval_env.reset()
    manager.reset()
    unnormalized_state = manager.get_full_state(obs_array[0])
    current_state = manager.get_normalized_state(unnormalized_state)
    glucose_history = [obs_array[0]]
    
    for _ in range(288): # Full day simulation
        action = agent.select_action(current_state)
        safe_action = safety_layer.apply(action, unnormalized_state)
        manager.insulin_history.append(safe_action[0])
        obs_array, _, terminated, truncated, _ = eval_env.step(safe_action)
        unnormalized_state = manager.get_full_state(obs_array[0])
        current_state = manager.get_normalized_state(unnormalized_state)
        glucose_history.append(obs_array[0])
        if terminated or truncated: break
    eval_env.close()
    
    plt.clf()
    plt.plot(glucose_history, label='SAC Agent')
    plt.axhline(y=180, color='r', linestyle=':', label='Hyperglycemia')
    plt.axhline(y=70, color='orange', linestyle=':', label='Hypoglycemia')
    plt.axhline(y=140, color='g', linestyle='-', label='Target')
    plt.title(f'Live Performance after Episode {episode_num}')
    plt.xlabel('Time (steps)'); plt.ylabel('Blood Glucose (mg/dL)'); plt.legend(); plt.grid(True)
    plt.pause(0.1)

# --- Main Training Script ---
def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    PATIENT_NAME = 'adolescent#001'
    max_episodes = 2000
    lr, gamma, tau, alpha = 3e-4, 0.99, 0.005, 0.2
    batch_size, n_latent_var, replay_buffer_size = 256, 256, 1000000
    max_timesteps = 288
    learning_starts = 1000
    
    if not os.path.exists('./models'): os.makedirs('./models')
    actor_path = f'./models/sac_actor_{PATIENT_NAME.replace("#", "-")}.pth'
    
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())
    meal_scenario = scgen.RandomScenario(start_time=start_time, seed=SEED)
    CLEAN_PATIENT_NAME = PATIENT_NAME.replace('#', '-')
    ENV_ID = f'simglucose/{CLEAN_PATIENT_NAME}-v0'
    
    try:
        register(id=ENV_ID, entry_point="simglucose.envs:T1DSimGymnaisumEnv", max_episode_steps=max_timesteps, kwargs={"patient_name": PATIENT_NAME, "custom_scenario": meal_scenario})
    except gymnasium.error.Error:
        print(f"Environment {ENV_ID} already registered. Skipping.")
        
    env = gymnasium.make(ENV_ID)
    env.action_space.seed(SEED)
    
    state_dim, action_dim = 3, 1
    agent = SACAgent(env, state_dim, action_dim, n_latent_var, lr, gamma, tau, alpha)
    manager = StateRewardManager(state_dim)
    safety_layer = SafetyLayer()
    replay_buffer = ReplayBuffer(replay_buffer_size)
    
    episode_lengths, final_glucose_levels = [], []
    total_timesteps_taken = 0
    
    plt.ion()
    plt.figure(figsize=(10, 5))

    print("--- Starting Training with SAC Agent (Stable) ---")
    for i_episode in range(1, max_episodes + 1):
        obs_array, _ = env.reset(seed=SEED + i_episode)
        manager.reset()
        unnormalized_state = manager.get_full_state(obs_array[0])
        current_state = manager.get_normalized_state(unnormalized_state)
        episode_reward = 0
        
        for t in range(max_timesteps):
            if total_timesteps_taken < learning_starts:
                proposed_action = env.action_space.sample()
            else:
                proposed_action = agent.select_action(current_state)

            safe_action = safety_layer.apply(proposed_action, unnormalized_state)
            
            manager.insulin_history.append(safe_action[0])
            next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
            done = terminated or truncated
            
            next_unnormalized_state = manager.get_full_state(next_obs_array[0])
            next_state = manager.get_normalized_state(next_unnormalized_state)
            
            reward = manager.get_reward(unnormalized_state)
            replay_buffer.push(current_state, safe_action, reward, next_state, done)
            
            current_state, unnormalized_state = next_state, next_unnormalized_state
            episode_reward += reward
            total_timesteps_taken += 1
            
            if total_timesteps_taken > learning_starts and len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)
            
            if done: break
        
        episode_lengths.append(t + 1)
        final_glucose_levels.append(unnormalized_state[0])
        
        if i_episode % 50 == 0:
            print(f"Episode {i_episode}/{max_episodes} | Length: {t+1} | Reward: {episode_reward:.2f}")
            evaluate_and_plot(agent, manager, safety_layer, ENV_ID, i_episode)
    
    plt.ioff()
    print("--- Training Finished ---")
    
    print(f"Saving trained model..."); torch.save(agent.actor.state_dict(), actor_path)
    
    print("\nFinal evaluation plot is displayed.")
    plt.show()

if __name__ == '__main__':
    main()
