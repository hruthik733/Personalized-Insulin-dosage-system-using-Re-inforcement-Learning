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
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# --- Helper Functions & Safety Layer (from previous steps) ---
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

    def apply(self, action, state):
        glucose, rate_of_change, _ = state
        if glucose < self.hypo_threshold or (glucose < self.predictive_low_threshold and rate_of_change < -1.0):
            return np.array([0.0])
        return action

# --- NEW: Replay Buffer for Off-Policy Learning (SAC) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in np.random.choice(len(self.buffer), batch_size)])
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

# --- NEW: Soft Actor-Critic (SAC) Network Definitions ---
class Critic(nn.Module): # Q-Network
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_3 = nn.Linear(n_latent_var, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

class Actor(nn.Module): # Policy Network
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
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2) # Clamp for stability
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# --- NEW: SAC Agent Implementation ---
class SACAgent:
    def __init__(self, env, state_dim, action_dim, n_latent_var, lr, gamma, tau, alpha):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
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
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# --- State Manager (from previous steps) ---
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
        g, _, iob = state
        reward = 0
        if 70 <= g <= 180: reward += 1
        if g < 70: reward -= 200
        if g > 180: reward -= (g - 180) * 0.1
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
    lr = 3e-4
    gamma = 0.99
    tau = 0.005 # Target network update rate
    alpha = 0.2 # Entropy regularization coefficient
    batch_size = 256
    n_latent_var = 256
    replay_buffer_size = 1000000
    max_timesteps = 288
    
    if not os.path.exists('./models'): os.makedirs('./models')
    actor_path = f'./models/sac_actor_{PATIENT_NAME.replace("#", "-")}.pth'
    
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())
    meal_scenario = scgen.RandomScenario(start_time=start_time, seed=1)
    CLEAN_PATIENT_NAME = PATIENT_NAME.replace('#', '-')
    ENV_ID = f'simglucose/{CLEAN_PATIENT_NAME}-v0'
    
    register(id=ENV_ID, entry_point="simglucose.envs:T1DSimGymnaisumEnv", max_episode_steps=max_timesteps, kwargs={"patient_name": PATIENT_NAME, "custom_scenario": meal_scenario})
    env = gymnasium.make(ENV_ID)
    
    state_dim = 3
    action_dim = 1 # SAC works with continuous actions
    
    agent = SACAgent(env, state_dim, action_dim, n_latent_var, lr, gamma, tau, alpha)
    manager = StateRewardManager(state_dim)
    safety_layer = SafetyLayer()
    replay_buffer = ReplayBuffer(replay_buffer_size)
    
    episode_lengths, final_glucose_levels = [], []
    
    print("--- Starting Training with SAC Agent ---")
    for i_episode in range(1, max_episodes + 1):
        obs_array, _ = env.reset()
        manager.reset()
        unnormalized_state = manager.get_full_state(obs_array[0])
        current_state = manager.get_normalized_state(unnormalized_state)
        episode_reward = 0
        
        for t in range(max_timesteps):
            proposed_action = agent.select_action(current_state)
            safe_action = safety_layer.apply(proposed_action, unnormalized_state)
            
            manager.insulin_history.append(safe_action[0])
            next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
            done = terminated or truncated
            
            next_unnormalized_state = manager.get_full_state(next_obs_array[0])
            next_state = manager.get_normalized_state(next_unnormalized_state)
            
            reward = manager.get_reward(unnormalized_state)
            replay_buffer.push(current_state, safe_action, reward, next_state, done)
            
            current_state = next_state
            unnormalized_state = next_unnormalized_state
            episode_reward += reward
            
            if len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)
            
            if done: break
        
        episode_lengths.append(t + 1)
        final_glucose_levels.append(unnormalized_state[0])
        
        if i_episode % 50 == 0:
            print(f"Episode {i_episode}/{max_episodes} | Length: {t+1} | Reward: {episode_reward:.2f}")
    
    print("--- Training Finished ---")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(episode_lengths); plt.title('Episode Length over Time')
    plt.subplot(1, 2, 2); plt.plot(final_glucose_levels); plt.axhline(y=70, color='r', linestyle='--'); plt.title('Final Glucose Level')
    plt.tight_layout(); plt.show()

    print(f"Saving trained model..."); torch.save(agent.actor.state_dict(), actor_path)
    
    print("\n--- Starting Evaluation ---")
    eval_scenario = CustomScenario(start_time=start_time, scenario=[(7, 45), (12, 70), (18, 80)])
    eval_env = gymnasium.make(ENV_ID, custom_scenario=eval_scenario) # Re-use the ID with a new scenario
    
    eval_agent = SACAgent(eval_env, state_dim, action_dim, n_latent_var, lr, gamma, tau, alpha)
    eval_agent.actor.load_state_dict(torch.load(actor_path))
    
    manager.reset()
    obs_array, _ = eval_env.reset()
    unnormalized_state = manager.get_full_state(obs_array[0])
    current_state = manager.get_normalized_state(unnormalized_state)
    glucose_history = [obs_array[0]]
    
    for t in range(max_timesteps):
        action = eval_agent.select_action(current_state)
        safe_action = safety_layer.apply(action, unnormalized_state)
        manager.insulin_history.append(safe_action[0])
        obs_array, _, terminated, truncated, _ = eval_env.step(safe_action)
        unnormalized_state = manager.get_full_state(obs_array[0])
        current_state = manager.get_normalized_state(unnormalized_state)
        glucose_history.append(obs_array[0])
        if terminated or truncated: break
            
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
    plt.plot(glucose_history, label='SAC Agent')
    plt.axhline(y=180, color='r', linestyle=':', label='Hyperglycemia Threshold')
    plt.axhline(y=70, color='orange', linestyle=':', label='Hypoglycemia Threshold')
    plt.axhline(y=140, color='g', linestyle='-', label='Target')
    plt.title(f'SAC Agent Performance for {PATIENT_NAME}')
    plt.xlabel('Time (steps)'); plt.ylabel('Blood Glucose (mg/dL)'); plt.legend(); plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
