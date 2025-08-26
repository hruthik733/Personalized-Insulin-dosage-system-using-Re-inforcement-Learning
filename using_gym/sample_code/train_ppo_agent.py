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

# --- PPO Neural Network Definitions ---
class Actor(nn.Module):
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
        action_probs = self.softmax(self.layer_3(x))
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.layer_1(state))
        x = self.relu(self.layer_2(x))
        return self.layer_3(x)

# --- PPO Agent ---
class PPOAgent:
    def __init__(self, env, state_dim, action_dim, lr, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        self.actor_old = Actor(state_dim, action_dim)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.MseLoss = nn.MSELoss()

        self.action_bins = np.linspace(env.action_space.low[0], env.action_space.high[0], action_dim)

    def select_action(self, state, memory):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor_old(state_tensor)
        dist = Categorical(action_probs)
        action_index = dist.sample()
        action_val = self.action_bins[action_index.item()]
        
        if memory is not None:
            memory.states.append(state_tensor)
            memory.actions.append(action_index)
            memory.logprobs.append(dist.log_prob(action_index))
        
        return np.array([action_val])

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
            action_probs = self.actor(old_states)
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            state_values = self.critic(old_states).squeeze()
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards)
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.actor_old.load_state_dict(self.actor.state_dict())

# --- State Calculation Component ---
class StateManager:
    def __init__(self):
        self.glucose_history = collections.deque(maxlen=2)
        self.insulin_history = collections.deque(maxlen=160)
        self.reset()
        _, self.F_k = get_pkpd_discount_factors(t_peak=55, t_end=480, n_steps=160)
        
    def calculate_iob(self):
        recent_insulin = np.array(list(self.insulin_history)[::-1])
        return np.sum(recent_insulin * (1 - self.F_k))
        
    def get_full_state(self, observation):
        self.glucose_history.append(observation)
        rate = (self.glucose_history[1] - self.glucose_history[0]) / 3.0 if len(self.glucose_history) == 2 else 0.0
        iob = self.calculate_iob()
        return np.array([observation, rate, iob])

    def reset(self):
        self.glucose_history.clear()
        for _ in range(2): self.glucose_history.append(140) # Start with a neutral value
        self.insulin_history.clear()
        for _ in range(160): self.insulin_history.append(0)

# --- Main Training and Evaluation Script ---
def main():
    # --- Hyperparameters ---
    PATIENT_NAME = 'adolescent#001'
    max_episodes = 500
    max_timesteps = 288
    update_timestep = 2000
    lr = 0.002
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2
    
    # --- File Paths for Saved Model ---
    if not os.path.exists('./models'):
        os.makedirs('./models')
    actor_path = f'./models/ppo_actor_{PATIENT_NAME.replace("#", "-")}.pth'
    critic_path = f'./models/ppo_critic_{PATIENT_NAME.replace("#", "-")}.pth'

    # --- Environment Setup ---
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
    agent = PPOAgent(env, state_dim, action_dim, lr, gamma, K_epochs, eps_clip)
    state_manager = StateManager()
    
    timestep = 0
    
    # --- TRAINING PHASE ---
    print("--- Starting Training ---")
    for i_episode in range(1, max_episodes + 1):
        obs_array, info = env.reset()
        state_manager.reset()
        current_state = state_manager.get_full_state(obs_array[0])
        
        for t in range(max_timesteps):
            timestep += 1
            action = agent.select_action(current_state, memory)
            state_manager.insulin_history.append(action[0])
            obs_array, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                timestep = 0
            
            current_state = state_manager.get_full_state(obs_array[0])
            if done:
                break
        
        if i_episode % 50 == 0:
            print(f"Episode {i_episode}/{max_episodes} finished.")
    
    print("--- Training Finished ---")
    
    # --- SAVING MODEL ---
    print(f"Saving trained model to {actor_path} and {critic_path}")
    torch.save(agent.actor_old.state_dict(), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)
    
    # --- EVALUATION PHASE ---
    print("\n--- Starting Evaluation ---")
    # Create a fixed scenario for fair evaluation
    eval_scenario = CustomScenario(start_time=start_time, scenario=[(7, 45), (12, 70), (18, 80)])
    env.close() # Close training env
    register(id=ENV_ID + "-eval", entry_point="simglucose.envs:T1DSimGymnaisumEnv", max_episode_steps=max_timesteps, kwargs={"patient_name": PATIENT_NAME, "custom_scenario": eval_scenario})
    eval_env = gymnasium.make(ENV_ID + "-eval")
    
    # Load the trained agent
    eval_agent = PPOAgent(eval_env, state_dim, action_dim, lr, gamma, K_epochs, eps_clip)
    eval_agent.actor_old.load_state_dict(torch.load(actor_path))
    
    state_manager.reset()
    obs_array, info = eval_env.reset()
    current_state = state_manager.get_full_state(obs_array[0])
    
    glucose_history = [obs_array[0]]
    
    for t in range(max_timesteps):
        action = eval_agent.select_action(current_state, None) # No memory needed for eval
        state_manager.insulin_history.append(action[0])
        obs_array, reward, terminated, truncated, info = eval_env.step(action)
        current_state = state_manager.get_full_state(obs_array[0])
        glucose_history.append(obs_array[0])
        if terminated or truncated:
            break
            
    eval_env.close()
    
    # --- CALCULATE AND DISPLAY RESULTS ---
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

    # --- PLOT RESULTS ---
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
