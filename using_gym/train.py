import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
import torch.nn as nn
import os
import yaml
from tqdm import tqdm
import random
from datetime import datetime

# --- Local Imports ---
from agents.hybrid_agent import HybridAgent
from utils.new_state_management import StateRewardManager
from utils.safety import SafetyLayer
from simglucose.simulation.scenario import CustomScenario
import simglucose.simulation.scenario_gen as scgen

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.next_states = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.next_states[:]

def train():
    # --- Load Config ---
    with open('./configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- Setup ---
    device = torch.device("cpu")
    config['project']['device'] = 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(config['project']['seed'])
    np.random.seed(config['project']['seed'])
    random.seed(config['project']['seed'])

    # --- Environment ---
    ENV_ID = config['environment']['env_id']
    try:
        register(id=ENV_ID, entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv", max_episode_steps=config['environment']['max_episode_steps'], kwargs={"patient_name": config['environment']['patient_cohort']})
    except gymnasium.error.Error:
        print(f"Environment {ENV_ID} already registered.")
    
    env = gymnasium.make(ENV_ID)
    
    # --- Agent and Utilities ---
    agent = HybridAgent(config)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config['training']['lr'])
    mse_loss = nn.MSELoss()
    
    manager = StateRewardManager(state_dim=config['agent']['state_dim'], cohort_name='adult')
    safety_layer = SafetyLayer()

    # --- Training ---
    memory = Memory()
    time_step = 0
    
    print("--- Starting Training ---")
    pbar = tqdm(total=config['training']['total_timesteps'])
    
    while time_step < config['training']['total_timesteps']:
        obs_array, info = env.reset()
        manager.reset()

        core_env = env.unwrapped.env.env
        current_sim_time = core_env.time
        episode_scenario = env.unwrapped.env.custom_scenario
        upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
        unnormalized_state = manager.get_full_state(obs_array, current_sim_time, upcoming_carbs)
        state = manager.get_normalized_state(unnormalized_state)
        
        for t in range(config['environment']['max_episode_steps']):
            time_step += 1
            pbar.update(1)

            state_tensor = torch.FloatTensor(state).to(device)
            action, log_prob = agent.act(state_tensor)
            
            unscaled_action = action.cpu().numpy()

            safe_action = safety_layer.apply(unscaled_action, unnormalized_state)
            clipped_action = np.clip(safe_action, 0.0, config['agent']['max_action'])

            manager.insulin_history.append(clipped_action.item())
            next_obs_array, _, terminated, truncated, _ = env.step(clipped_action)
            done = terminated or truncated

            core_env = env.unwrapped.env.env
            current_sim_time = core_env.time
            episode_scenario = env.unwrapped.env.custom_scenario
            upcoming_carbs = episode_scenario.get_action(current_sim_time).meal if episode_scenario else 0
            next_unnormalized_state = manager.get_full_state(next_obs_array, current_sim_time, upcoming_carbs)
            next_state = manager.get_normalized_state(next_unnormalized_state)
            
            reward = manager.get_reward(unnormalized_state)

            memory.states.append(state_tensor)
            memory.actions.append(action)
            memory.log_probs.append(log_prob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            memory.next_states.append(torch.FloatTensor(next_state).to(device))
            
            if time_step > 0 and time_step % config['training']['update_timestep'] == 0:
                # This is a simplified update step for demonstration
                # A full PPO implementation would be more complex
                if len(memory.states) > 1:
                    old_states = torch.stack(memory.states).detach().squeeze(1)
                    old_actions = torch.stack(memory.actions).detach().squeeze(1)
                    actual_next_states = torch.stack(memory.next_states).detach().squeeze(1)
                    
                    predicted_next_states = agent.predict_next_state(old_states, old_actions)
                    prediction_loss = mse_loss(predicted_next_states, actual_next_states)
                    
                    ppo_loss = -torch.stack(memory.log_probs).mean()
                    
                    composite_loss = ppo_loss + config['training']['prediction_loss_weight'] * prediction_loss
                    
                    optimizer.zero_grad()
                    composite_loss.backward()
                    optimizer.step()
                    
                    memory.clear()
                    print(f"\n--- Update performed at step {time_step} ---")


            state = next_state
            unnormalized_state = next_unnormalized_state
            if done or truncated:
                break
    
    pbar.close()
    print("--- Training Finished ---")
    # Use the corrected key for BOTH the save and print commands
    torch.save(agent.state_dict(), config['agent']['model_path'])
    print(f"Model saved to {config['agent']['model_path']}")

if __name__ == '__main__':
    train()