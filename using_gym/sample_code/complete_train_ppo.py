# import gymnasium
# from gymnasium.envs.registration import register
# import numpy as np
# import collections
# from scipy.stats import gamma
# import torch
# import torch.nn as nn
# from torch.distributions import Categorical
# import simglucose.simulation.scenario_gen as scgen
# from simglucose.simulation.scenario import CustomScenario
# from datetime import datetime
# import matplotlib.pyplot as plt
# import os

# # --- Helper Functions for Bio-Inspired Components ---
# def get_pkpd_discount_factors(t_peak, t_end, n_steps):
#     shape_k = 2
#     scale_theta = t_peak / (shape_k - 1)
#     time_points = np.linspace(0, t_end, n_steps)
#     pdf_values = gamma.pdf(time_points, a=shape_k, scale=scale_theta)
#     f_k = pdf_values / np.max(pdf_values)
#     cdf_values = gamma.cdf(time_points, a=shape_k, scale=scale_theta)
#     F_k = cdf_values
#     return f_k, F_k

# # --- Memory Buffer ---
# class Memory:
#     def __init__(self):
#         self.actions = []
#         self.states = []
#         self.logprobs = []
#         self.rewards_long = []
#         self.rewards_short = []
#         self.is_terminals = []
    
#     def clear_memory(self):
#         del self.actions[:]
#         del self.states[:]
#         del self.logprobs[:]
#         del self.rewards_long[:]
#         del self.rewards_short[:]
#         del self.is_terminals[:]

# # --- PPO Neural Network Definitions ---
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Actor, self).__init__()
#         self.layer_1 = nn.Linear(state_dim, 64)
#         self.layer_2 = nn.Linear(64, 64)
#         self.layer_3 = nn.Linear(64, action_dim)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, state):
#         x = self.relu(self.layer_1(state))
#         x = self.relu(self.layer_2(x))
#         action_probs = self.softmax(self.layer_3(x))
#         return action_probs

# class Critic(nn.Module):
#     def __init__(self, state_dim):
#         super(Critic, self).__init__()
#         self.layer_1 = nn.Linear(state_dim, 64)
#         self.layer_2 = nn.Linear(64, 64)
#         self.layer_3 = nn.Linear(64, 1)
#         self.relu = nn.ReLU()

#     def forward(self, state):
#         x = self.relu(self.layer_1(state))
#         x = self.relu(self.layer_2(x))
#         return self.layer_3(x)

# # --- PPO Agent with Bio-Inspired Learning ---
# class BioInspiredPPOAgent:
#     def __init__(self, env, state_dim, action_dim, lr, gamma, K_epochs, eps_clip):
#         self.gamma = gamma
#         self.eps_clip = eps_clip
#         self.K_epochs = K_epochs
        
#         self.actor = Actor(state_dim, action_dim)
#         self.actor_old = Actor(state_dim, action_dim)
#         self.actor_old.load_state_dict(self.actor.state_dict())
        
#         # Two critic networks as per the paper
#         self.critic_long = Critic(state_dim)
#         self.critic_short = Critic(state_dim)
        
#         self.optimizer = torch.optim.Adam(
#             list(self.actor.parameters()) + 
#             list(self.critic_long.parameters()) + 
#             list(self.critic_short.parameters()), lr=lr
#         )
#         self.MseLoss = nn.MSELoss()

#         self.action_bins = np.linspace(env.action_space.low[0], env.action_space.high[0], action_dim)

#     def select_action(self, state, memory):
#         state_tensor = torch.FloatTensor(state).unsqueeze(0)
#         with torch.no_grad():
#             action_probs = self.actor_old(state_tensor)
#         dist = Categorical(action_probs)
#         action_index = dist.sample()
#         action_val = self.action_bins[action_index.item()]
        
#         if memory is not None:
#             # The state is now only added to memory here
#             memory.states.append(state_tensor)
#             memory.actions.append(action_index)
#             memory.logprobs.append(dist.log_prob(action_index))
        
#         return np.array([action_val])

#     def update(self, memory):
#         # Calculate discounted rewards for both long and short term
#         rewards_long = []
#         discounted_reward_long = 0
#         for reward, is_terminal in zip(reversed(memory.rewards_long), reversed(memory.is_terminals)):
#             if is_terminal:
#                 discounted_reward_long = 0
#             discounted_reward_long = reward + (self.gamma * discounted_reward_long)
#             rewards_long.insert(0, discounted_reward_long)

#         rewards_short = []
#         discounted_reward_short = 0
#         for reward, is_terminal in zip(reversed(memory.rewards_short), reversed(memory.is_terminals)):
#             if is_terminal:
#                 discounted_reward_short = 0
#             discounted_reward_short = reward + (self.gamma * discounted_reward_short)
#             rewards_short.insert(0, discounted_reward_short)
        
#         rewards_long = torch.tensor(rewards_long, dtype=torch.float32)
#         rewards_short = torch.tensor(rewards_short, dtype=torch.float32)

#         old_states = torch.cat(memory.states).detach()
#         old_actions = torch.stack(memory.actions).detach()
#         old_logprobs = torch.stack(memory.logprobs).detach()

#         for _ in range(self.K_epochs):
#             action_probs = self.actor(old_states)
#             dist = Categorical(action_probs)
#             logprobs = dist.log_prob(old_actions)
            
#             # Get values from both critics
#             state_values_long = self.critic_long(old_states).squeeze()
#             state_values_short = self.critic_short(old_states).squeeze()
            
#             # Calculate advantages for both streams
#             advantages_long = rewards_long - state_values_long.detach()
#             advantages_short = rewards_short - state_values_short.detach()
            
#             # Combine advantages as per paper's equation (10)
#             # A_pk/pd = (G_short - V_short) + alpha * (G_long - V_long)
#             alpha = 0.09 # Scaling factor from the paper
#             advantages = advantages_short + alpha * advantages_long

#             # Normalize the final advantage
#             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            
#             ratios = torch.exp(logprobs - old_logprobs.detach())
#             surr1 = ratios * advantages
#             surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
#             # Loss is Actor loss + Critic_long loss + Critic_short loss
#             loss = -torch.min(surr1, surr2) + \
#                    0.5 * self.MseLoss(state_values_long, rewards_long) + \
#                    0.5 * self.MseLoss(state_values_short, rewards_short)
            
#             self.optimizer.zero_grad()
#             loss.mean().backward()
#             self.optimizer.step()
            
#         self.actor_old.load_state_dict(self.actor.state_dict())

# # --- State and Reward Calculation Component ---
# class StateRewardManager:
#     def __init__(self):
#         self.glucose_history = collections.deque(maxlen=2)
#         self.insulin_history = collections.deque(maxlen=160)
#         self.reset()
#         self.f_k, self.F_k = get_pkpd_discount_factors(t_peak=55, t_end=480, n_steps=160)
#         self.g_target = 120.0
#         self.m_target = -1/15.0

#     def calculate_iob(self):
#         recent_insulin = np.array(list(self.insulin_history)[::-1])
#         return np.sum(recent_insulin * (1 - self.F_k))
        
#     def get_full_state(self, observation):
#         self.glucose_history.append(observation)
#         rate = (self.glucose_history[1] - self.glucose_history[0]) / 3.0 if len(self.glucose_history) == 2 else 0.0
#         iob = self.calculate_iob()
#         return np.array([observation, rate, iob])

#     def get_bio_rewards(self, state):
#         g, v, _ = state
#         # R_long
#         dl = abs(g - self.g_target)
#         r_long = -dl if 70 <= g <= 180 else -3 * dl
#         # R_short
#         dr = abs(self.m_target * (g - self.g_target) - v)
#         if g < 100:
#             r_short = -5 * dr if v < -0.6 else (-3 * dr if v < 3 else 0)
#         elif 100 <= g < 160:
#             r_short = 0 if v >= 3 else -dr
#         elif 160 <= g < 180:
#             r_short = -5 * dr if v >= 3 else -dr
#         else:
#             r_short = -5 * dr if v >= 1.5 else -3 * dr
#         return r_long, r_short

#     def reset(self):
#         self.glucose_history.clear()
#         for _ in range(2): self.glucose_history.append(140)
#         self.insulin_history.clear()
#         for _ in range(160): self.insulin_history.append(0)

# # --- Main Training and Evaluation Script ---
# def main():
#     PATIENT_NAME = 'adolescent#001'
    
#     # --- HYPERPARAMETER TUNING ---
#     # The agent needs more time to learn and smaller, more stable updates.
#     max_episodes = 1000     # Increased from 500 to give more learning time
#     update_timestep = 4000   # Increased from 2000 to learn from more diverse experiences
#     lr = 0.0001              # Reduced from 0.0003 for more stable policy updates
#     # ---------------------------

#     gamma = 0.99
#     K_epochs = 4
#     eps_clip = 0.2
#     max_timesteps = 288
    
#     if not os.path.exists('./models'):
#         os.makedirs('./models')
#     actor_path = f'./models/ppo_actor_bio_{PATIENT_NAME.replace("#", "-")}.pth'
#     critic_long_path = f'./models/ppo_critic_long_{PATIENT_NAME.replace("#", "-")}.pth'
#     critic_short_path = f'./models/ppo_critic_short_{PATIENT_NAME.replace("#", "-")}.pth'

#     now = datetime.now()
#     start_time = datetime.combine(now.date(), datetime.min.time())
#     meal_scenario = scgen.RandomScenario(start_time=start_time, seed=1)
#     CLEAN_PATIENT_NAME = PATIENT_NAME.replace('#', '-')
#     ENV_ID = f'simglucose/{CLEAN_PATIENT_NAME}-v0'
    
#     register(id=ENV_ID, entry_point="simglucose.envs:T1DSimGymnaisumEnv", max_episode_steps=max_timesteps, kwargs={"patient_name": PATIENT_NAME, "custom_scenario": meal_scenario})
#     env = gymnasium.make(ENV_ID)
    
#     state_dim = 3
#     action_dim = 72
    
#     memory = Memory()
#     agent = BioInspiredPPOAgent(env, state_dim, action_dim, lr, gamma, K_epochs, eps_clip)
#     manager = StateRewardManager()
    
#     timestep = 0
    
#     print("--- Starting Training with Bio-Inspired Rewards ---")
#     for i_episode in range(1, max_episodes + 1):
#         obs_array, info = env.reset()
#         manager.reset()
#         current_state = manager.get_full_state(obs_array[0])
        
#         for t in range(max_timesteps):
#             timestep += 1
#             action = agent.select_action(current_state, memory)
#             manager.insulin_history.append(action[0])
#             obs_array, _, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
            
#             current_state = manager.get_full_state(obs_array[0])
#             r_long, r_short = manager.get_bio_rewards(current_state)
            
#             # This line was the bug, it has been removed.
#             # memory.states.append(torch.FloatTensor(current_state).unsqueeze(0)) 
            
#             memory.rewards_long.append(r_long)
#             memory.rewards_short.append(r_short)
#             memory.is_terminals.append(done)
            
#             if timestep % update_timestep == 0:
#                 agent.update(memory)
#                 memory.clear_memory()
#                 timestep = 0
            
#             if done:
#                 break
        
#         if i_episode % 50 == 0:
#             print(f"Episode {i_episode}/{max_episodes} finished.")
    
#     print("--- Training Finished ---")
    
#     print(f"Saving trained model...")
#     torch.save(agent.actor_old.state_dict(), actor_path)
#     torch.save(agent.critic_long.state_dict(), critic_long_path)
#     torch.save(agent.critic_short.state_dict(), critic_short_path)
    
#     print("\n--- Starting Evaluation ---")
#     eval_scenario = CustomScenario(start_time=start_time, scenario=[(7, 45), (12, 70), (18, 80)])
#     env.close()
    
#     # The 'force=True' argument is removed from this line.
#     register(id=ENV_ID + "-eval", entry_point="simglucose.envs:T1DSimGymnaisumEnv", max_episode_steps=max_timesteps, kwargs={"patient_name": PATIENT_NAME, "custom_scenario": eval_scenario})
#     eval_env = gymnasium.make(ENV_ID + "-eval")
    
#     eval_agent = BioInspiredPPOAgent(eval_env, state_dim, action_dim, lr, gamma, K_epochs, eps_clip)
#     eval_agent.actor_old.load_state_dict(torch.load(actor_path))
    
#     manager.reset()
#     obs_array, info = eval_env.reset()
#     current_state = manager.get_full_state(obs_array[0])
#     glucose_history = [obs_array[0]]
    
#     for t in range(max_timesteps):
#         action = eval_agent.select_action(current_state, None)
#         manager.insulin_history.append(action[0])
#         obs_array, _, terminated, truncated, _ = eval_env.step(action)
#         current_state = manager.get_full_state(obs_array[0])
#         glucose_history.append(obs_array[0])
#         if terminated or truncated:
#             break
            
#     eval_env.close()
    
#     glucose_history = np.array(glucose_history)
#     time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
#     time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
#     time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
#     mean_glucose = np.mean(glucose_history)

#     print("\n--- Evaluation Results ---")
#     print(f"Mean Glucose: {mean_glucose:.2f} mg/dL")
#     print(f"Time in Range (70-180 mg/dL): {time_in_range:.2f}%")
#     print(f"Time in Hypoglycemia (<70 mg/dL): {time_hypo:.2f}%")
#     print(f"Time in Hyperglycemia (>180 mg/dL): {time_hyper:.2f}%")

#     plt.figure(figsize=(15, 6))
#     plt.plot(glucose_history)
#     plt.axhline(y=180, color='r', linestyle='--', label='Hyperglycemia Threshold')
#     plt.axhline(y=70, color='orange', linestyle='--', label='Hypoglycemia Threshold')
#     plt.axhline(y=140, color='g', linestyle='-', label='Target')
#     plt.title(f'Trained Agent Performance for {PATIENT_NAME}')
#     plt.xlabel('Time (steps)')
#     plt.ylabel('Blood Glucose (mg/dL)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# if __name__ == '__main__':
#     main()









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
        self.rewards_long = []
        self.rewards_short = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards_long[:]
        del self.rewards_short[:]
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

# --- PPO Agent with Bio-Inspired Learning ---
class BioInspiredPPOAgent:
    def __init__(self, env, state_dim, action_dim, lr, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.actor = Actor(state_dim, action_dim)
        self.actor_old = Actor(state_dim, action_dim)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        self.critic_long = Critic(state_dim)
        self.critic_short = Critic(state_dim)
        
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + 
            list(self.critic_long.parameters()) + 
            list(self.critic_short.parameters()), lr=lr
        )
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
        rewards_long = []
        discounted_reward_long = 0
        for reward, is_terminal in zip(reversed(memory.rewards_long), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward_long = 0
            discounted_reward_long = reward + (self.gamma * discounted_reward_long)
            rewards_long.insert(0, discounted_reward_long)

        rewards_short = []
        discounted_reward_short = 0
        for reward, is_terminal in zip(reversed(memory.rewards_short), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward_short = 0
            discounted_reward_short = reward + (self.gamma * discounted_reward_short)
            rewards_short.insert(0, discounted_reward_short)
        
        rewards_long = torch.tensor(rewards_long, dtype=torch.float32)
        rewards_short = torch.tensor(rewards_short, dtype=torch.float32)

        old_states = torch.cat(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()

        for _ in range(self.K_epochs):
            action_probs = self.actor(old_states)
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            state_values_long = self.critic_long(old_states).squeeze()
            state_values_short = self.critic_short(old_states).squeeze()
            
            advantages_long = rewards_long - state_values_long.detach()
            advantages_short = rewards_short - state_values_short.detach()
            
            alpha = 0.09
            advantages = advantages_short + alpha * advantages_long
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + \
                   0.5 * self.MseLoss(state_values_long, rewards_long) + \
                   0.5 * self.MseLoss(state_values_short, rewards_short) - \
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
        self.g_target = 120.0
        self.m_target = -1/15.0

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

    def get_bio_rewards(self, state):
        g, v, _ = state
        dl = abs(g - self.g_target)
        
        # R_long with asymmetric penalty for hypoglycemia
        if g < 70:
            r_long = -10 * dl # Greatly penalize hypoglycemia
        elif g > 180:
            r_long = -3 * dl
        else:
            r_long = -dl
        
        dr = abs(self.m_target * (g - self.g_target) - v)
        if g < 100:
            r_short = -5 * dr if v < -0.6 else (-3 * dr if v < 3 else 0)
        elif 100 <= g < 160:
            r_short = 0 if v >= 3 else -dr
        elif 160 <= g < 180:
            r_short = -5 * dr if v >= 3 else -dr
        else:
            r_short = -5 * dr if v >= 1.5 else -3 * dr
        return r_long, r_short

    def reset(self):
        self.glucose_history.clear()
        for _ in range(2): self.glucose_history.append(140)
        self.insulin_history.clear()
        for _ in range(160): self.insulin_history.append(0)

# --- Main Training and Evaluation Script ---
def main():
    PATIENT_NAME = 'adolescent#001'
    
    # --- HYPERPARAMETER TUNING FOR STABILITY ---
    max_episodes = 1500      # More training time
    update_timestep = 5000   # Learn from larger batches of experience
    lr = 5e-5                # Slower, more stable learning rate (0.00005)
    eps_clip = 0.1           # Smaller policy updates
    # -------------------------------------------

    gamma = 0.99
    K_epochs = 4
    max_timesteps = 288
    
    if not os.path.exists('./models'):
        os.makedirs('./models')
    actor_path = f'./models/ppo_actor_bio_{PATIENT_NAME.replace("#", "-")}.pth'
    critic_long_path = f'./models/ppo_critic_long_{PATIENT_NAME.replace("#", "-")}.pth'
    critic_short_path = f'./models/ppo_critic_short_{PATIENT_NAME.replace("#", "-")}.pth'

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
    agent = BioInspiredPPOAgent(env, state_dim, action_dim, lr, gamma, K_epochs, eps_clip)
    manager = StateRewardManager(state_dim)
    
    timestep = 0
    
    print("--- Starting Training with Bio-Inspired Rewards & State Normalization ---")
    for i_episode in range(1, max_episodes + 1):
        obs_array, info = env.reset()
        manager.reset()
        unnormalized_state = manager.get_full_state(obs_array[0])
        current_state = manager.get_normalized_state(unnormalized_state)
        
        for t in range(max_timesteps):
            timestep += 1
            action = agent.select_action(current_state, memory)
            manager.insulin_history.append(action[0])
            obs_array, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            unnormalized_state = manager.get_full_state(obs_array[0])
            current_state = manager.get_normalized_state(unnormalized_state)
            
            r_long, r_short = manager.get_bio_rewards(unnormalized_state)
            
            memory.rewards_long.append(r_long)
            memory.rewards_short.append(r_short)
            memory.is_terminals.append(done)
            
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                timestep = 0
            
            if done:
                break
        
        if i_episode % 50 == 0:
            print(f"Episode {i_episode}/{max_episodes} finished.")
    
    print("--- Training Finished ---")
    
    print(f"Saving trained model...")
    torch.save(agent.actor_old.state_dict(), actor_path)
    torch.save(agent.critic_long.state_dict(), critic_long_path)
    torch.save(agent.critic_short.state_dict(), critic_short_path)
    
    print("\n--- Starting Evaluation ---")
    eval_scenario = CustomScenario(start_time=start_time, scenario=[(7, 45), (12, 70), (18, 80)])
    env.close()
    
    register(id=ENV_ID + "-eval", entry_point="simglucose.envs:T1DSimGymnaisumEnv", max_episode_steps=max_timesteps, kwargs={"patient_name": PATIENT_NAME, "custom_scenario": eval_scenario})
    eval_env = gymnasium.make(ENV_ID + "-eval")
    
    eval_agent = BioInspiredPPOAgent(eval_env, state_dim, action_dim, lr, gamma, K_epochs, eps_clip)
    eval_agent.actor_old.load_state_dict(torch.load(actor_path))
    
    manager.reset()
    obs_array, info = eval_env.reset()
    unnormalized_state = manager.get_full_state(obs_array[0])
    current_state = manager.get_normalized_state(unnormalized_state)
    glucose_history = [obs_array[0]]
    
    for t in range(max_timesteps):
        action = eval_agent.select_action(current_state, None)
        manager.insulin_history.append(action[0])
        obs_array, _, terminated, truncated, _ = eval_env.step(action)
        
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
