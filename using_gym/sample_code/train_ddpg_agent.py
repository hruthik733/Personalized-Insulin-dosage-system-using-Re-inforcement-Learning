import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import collections
from scipy.stats import gamma
import torch
import torch.nn as nn
import torch.nn.functional as F
import simglucose.simulation.scenario_gen as scgen
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.basal_bolus_ctrller import BBController
from datetime import datetime
import matplotlib.pyplot as plt
import os
import random

# --- Helper Functions ---
def get_pkpd_discount_factors(t_peak, t_end, n_steps):
    shape_k = 2
    scale_theta = t_peak / (shape_k - 1)
    time_points = np.linspace(0, t_end, n_steps)
    pdf_values = gamma.pdf(time_points, a=shape_k, scale=scale_theta)
    f_k = pdf_values / np.max(pdf_values)
    cdf_values = gamma.cdf(time_points, a=shape_k, scale=scale_theta)
    F_k = cdf_values
    return f_k, F_k

# --- OU Noise for Exploration ---
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

# --- Safety Layer ---
class SafetyLayer:
    def __init__(self):
        self.hypo_threshold = 55  # Only block insulin at dangerous hypoglycemia
        self.severe_hyper_threshold = 350  # You could also block if BG explodes

    def apply(self, action, state):
        glucose, rate_of_change, iob, meal = state  # Expecting 4 values now
        if glucose < self.hypo_threshold:
            return np.array([0.0])
        return np.clip(action, 0.0, 2.0)


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in indices])
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    def __len__(self):
        return len(self.buffer)
    def clear(self):
        self.buffer.clear()

# --- DDPG Networks ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, action_dim)
        self.max_action = max_action
    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        return self.max_action * torch.tanh(self.layer_3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 1)
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

# --- DDPG Agent ---
class DDPGAgent:
    def __init__(self, env, state_dim, action_dim, lr, gamma, tau):
        self.gamma, self.tau = gamma, tau
        self.max_action = float(env.action_space.high[0])
        self.actor = Actor(state_dim, action_dim, self.max_action)
        self.actor_target = Actor(state_dim, action_dim, self.max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()
    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + ((1 - done) * self.gamma * target_q).detach()
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# --- State & Reward Manager ---
class StateRewardManager:
    def __init__(self, state_dim):
        self.glucose_history = collections.deque(maxlen=5)
        self.insulin_history = collections.deque(maxlen=160)
        self.meal_history = collections.deque(maxlen=1)  # Only most recent meal info
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
        std = np.sqrt(self.running_state_std / max(self.n_observations,1))
        return (state - self.running_state_mean) / (std + 1e-8)

    def calculate_iob(self):
        return np.sum(np.array(list(self.insulin_history)[::-1]) * (1 - self.F_k))

    def get_full_state(self, observation, info=None):
        # Attempt to extract meal amount from env info, fallback to zero
        meal = 0.0
        if info and "meal" in info:
            meal = info["meal"]
        self.glucose_history.append(observation)
        self.meal_history.append(meal)
        rate = 0.0
        if len(self.glucose_history) > 1:
            time_span = (len(self.glucose_history) - 1) * 3.0
            rate = (self.glucose_history[-1] - self.glucose_history[0]) / time_span if time_span > 0 else 0
        # State: [BG, dBG/dt, IOB, recent_meal]
        return np.array([observation, rate, self.calculate_iob(), self.meal_history[-1]])

    def get_reward(self, state, action):
        g, _, iob, meal = state
        reward = 1.0 if 90 <= g <= 140 else 0.0  # bonus for tight range
        reward -= abs(g - 110) * 0.04
        if g < 70:
            reward -= (70 - g) * 1.5
        if g < 40:
            reward -= 50
        if g > 180:
            reward -= (g - 180) * 0.40
        reward -= iob * 0.008
        return reward

    def reset(self):
        self.glucose_history.clear()
        self.insulin_history.clear()
        self.meal_history.clear()
        for _ in range(5): self.glucose_history.append(140)
        for _ in range(160): self.insulin_history.append(0)
        self.meal_history.append(0)

# --- Evaluation & Plotting ---
def evaluate_and_plot(agent, manager, safety_layer, env_id, episode_num):
    eval_env = gymnasium.make(env_id)
    obs_array, info = eval_env.reset()
    manager.reset()
    unnormalized_state = manager.get_full_state(obs_array[0], info)
    current_state = manager.get_normalized_state(unnormalized_state)
    glucose_history, action_history = [obs_array[0]], []
    for t in range(288):
        action = agent.select_action(current_state)
        safe_action = safety_layer.apply(action, unnormalized_state)
        manager.insulin_history.append(safe_action[0])
        obs_array, _, terminated, truncated, info = eval_env.step(safe_action)
        unnormalized_state = manager.get_full_state(obs_array[0], info)
        current_state = manager.get_normalized_state(unnormalized_state)
        glucose_history.append(obs_array[0])
        action_history.append(safe_action[0])
        if terminated or truncated:
            break
    eval_env.close()
    plt.clf()
    plt.plot(glucose_history, label='DDPG BG')
    plt.plot(action_history, label='Insulin', alpha=0.5)
    plt.axhline(y=180, color='r', linestyle=':', label='Hyperglycemia')
    plt.axhline(y=70, color='orange', linestyle=':', label='Hypoglycemia')
    plt.title(f'Live Performance after Episode {episode_num}')
    plt.xlabel('Time (steps)'); plt.ylabel('Glucose (mg/dL) / Insulin')
    plt.legend(); plt.grid(True); plt.pause(0.1)

# --- Main Training Loop ---
def main():
    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    PATIENT_NAME = 'adolescent#001'
    max_episodes, lr, gamma, tau = 2000, 3e-5, 0.985, 0.005
    batch_size, replay_buffer_size, max_timesteps = 128, 100_000, 288
    warmup_episodes = 16
    state_dim, action_dim = 4, 1  # state now includes meal
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())
    meal_scenario = scgen.RandomScenario(start_time=start_time, seed=SEED)
    CLEAN_PATIENT_NAME = PATIENT_NAME.replace("#", "-")
    ENV_ID = f'simglucose/{CLEAN_PATIENT_NAME}-v0'
    if not os.path.exists('./models'): os.makedirs('./models')
    actor_path = f'./models/ddpg_actor_{CLEAN_PATIENT_NAME}.pth'
    try:
        register(id=ENV_ID, entry_point="simglucose.envs:T1DSimGymnaisumEnv", max_episode_steps=max_timesteps, kwargs={"patient_name": PATIENT_NAME, "custom_scenario": meal_scenario})
    except gymnasium.error.Error:
        pass
    env = gymnasium.make(ENV_ID)
    env.action_space.seed(SEED)
    agent = DDPGAgent(env, state_dim, action_dim, lr, gamma, tau)
    manager = StateRewardManager(state_dim)
    safety_layer = SafetyLayer()
    replay_buffer = ReplayBuffer(replay_buffer_size)
    ou_noise = OUNoise(action_dim)
    print("--- Expert Warm-up ---")
    expert_controller = BBController()
    for i_episode in range(warmup_episodes):
        obs_array, info = env.reset(seed=SEED + i_episode)
        manager.reset()
        for t in range(max_timesteps):
            from simglucose.simulation.env import Observation
            expert_obs = Observation(CGM=obs_array[0])
            expert_action_val = expert_controller.policy(expert_obs, 0, False, **info).basal
            expert_action = np.array([expert_action_val])
            manager.insulin_history.append(expert_action[0])
            next_obs_array, _, terminated, truncated, info = env.step(expert_action)
            unnormalized_state = manager.get_full_state(next_obs_array[0], info)
            next_state = manager.get_normalized_state(unnormalized_state)
            reward = manager.get_reward(unnormalized_state, expert_action)
            replay_buffer.push(manager.get_normalized_state(manager.get_full_state(obs_array[0], info)), expert_action, reward, next_state, terminated or truncated)
            obs_array = next_obs_array
            if terminated or truncated: break
        print(f"Warm-up episode {i_episode+1}/{warmup_episodes} done.")

    plt.ion(); plt.figure(figsize=(10, 5))
    print("--- DDPG Training ---")
    for i_episode in range(1, max_episodes + 1):
        obs_array, info = env.reset(seed=SEED + i_episode + warmup_episodes)
        manager.reset()
        ou_noise.reset()
        unnormalized_state = manager.get_full_state(obs_array[0], info)
        current_state = manager.get_normalized_state(unnormalized_state)
        episode_reward = 0
        for t in range(max_timesteps):
            action = agent.select_action(current_state)
            action += ou_noise.sample()
            safe_action = safety_layer.apply(action, unnormalized_state)
            manager.insulin_history.append(safe_action[0])
            next_obs_array, _, terminated, truncated, info = env.step(safe_action)
            next_unnormalized_state = manager.get_full_state(next_obs_array[0], info)
            next_state = manager.get_normalized_state(next_unnormalized_state)
            reward = manager.get_reward(next_unnormalized_state, safe_action)
            replay_buffer.push(current_state, safe_action, reward, next_state, terminated or truncated)
            current_state = next_state
            unnormalized_state = next_unnormalized_state
            episode_reward += reward
            agent.update(replay_buffer, batch_size)
            if terminated or truncated: break
            if t < 8 or t % 36 == 0:
                # Example debug log
                print(f"Ep{ i_episode }|Step { t }|BG { unnormalized_state[0]:.1f} |A { safe_action[0]:.2f} |R { reward:.2f}")
        if i_episode % 30 == 0:
            print(f"Ep {i_episode}/{max_episodes} | Steps: {t+1} | Reward: {episode_reward:.2f}")
            evaluate_and_plot(agent, manager, safety_layer, ENV_ID, i_episode)
        # Optional: clear buffer on major reward change
        # if i_episode % 500 == 0: replay_buffer.clear()

    plt.ioff(); print("--- Training Finished ---")
    torch.save(agent.actor.state_dict(), actor_path)
    print("\nModel saved and final plot shown."); plt.show()

if __name__ == '__main__':
    main()
