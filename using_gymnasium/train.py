# train.py

import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import os

# --- Local Imports from our modules ---
from agents.sac_agent import SACAgent
from utils.replay_buffer import ReplayBuffer
from utils.safety import SafetyLayer
from utils.state_management import StateRewardManager
import simglucose.simulation.scenario_gen as scgen
from simglucose.simulation.scenario import CustomScenario

def main():
    # --- Hyperparameters & Configuration ---
    PATIENT_NAME = 'adolescent#001'
    max_episodes = 2000
    lr = 3e-4
    gamma_val = 0.99
    tau = 0.005
    alpha = 0.2
    batch_size = 256
    n_latent_var = 256
    replay_buffer_size = 1000000
    max_timesteps_per_episode = 288

    # --- Setup ---
    if not os.path.exists('./models'):
        os.makedirs('./models')
    actor_path = f'./models/sac_actor_{PATIENT_NAME.replace("#", "-")}.pth'

    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    # --- Environment Registration ---
    meal_scenario = scgen.RandomScenario(start_time=start_time, seed=1)
    CLEAN_PATIENT_NAME = PATIENT_NAME.replace('#', '-')
    ENV_ID = f'simglucose/{CLEAN_PATIENT_NAME}-v0'

    # Using the fully explicit entry point based on the source file
    register(
        id=ENV_ID,
        entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
        max_episode_steps=max_timesteps_per_episode,
        kwargs={"patient_name": PATIENT_NAME, "custom_scenario": meal_scenario}
    )
    env = gymnasium.make(ENV_ID)

    state_dim = 4
    action_dim = 1

    # --- Initialization ---
    agent = SACAgent(env, state_dim, action_dim, n_latent_var, lr, gamma_val, tau, alpha)
    manager = StateRewardManager(state_dim)
    safety_layer = SafetyLayer()
    replay_buffer = ReplayBuffer(replay_buffer_size)

    episode_lengths, final_glucose_levels = [], []

    # --- Training Loop ---
    print("--- Starting Training with SAC Agent ---")
    for i_episode in range(1, max_episodes + 1):
        obs_array, info = env.reset()
        episode_scenario = info.get('scenario')
        manager.reset()

        # ! THE CORRECT PATH
        current_sim_time = env.unwrapped.env.env.time
        upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0

        unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
        current_state = manager.get_normalized_state(unnormalized_state)
        episode_reward = 0

        for t in range(max_timesteps_per_episode):
            proposed_action = agent.select_action(current_state)
            safe_action = safety_layer.apply(proposed_action, unnormalized_state)

            manager.insulin_history.append(safe_action[0])
            next_obs_array, _, terminated, truncated, info = env.step(safe_action)
            done = terminated or truncated

            # ! THE CORRECT PATH
            current_sim_time = env.unwrapped.env.env.time
            upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0

            next_unnormalized_state = manager.get_full_state(next_obs_array[0], upcoming_carbs)
            next_state = manager.get_normalized_state(next_unnormalized_state)

            reward = manager.get_reward(unnormalized_state)
            replay_buffer.push(current_state, safe_action, reward, next_state, done)

            current_state = next_state
            unnormalized_state = next_unnormalized_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)

            if done:
                break

        episode_lengths.append(t + 1)
        final_glucose_levels.append(unnormalized_state[0])

        if i_episode % 50 == 0:
            print(f"Episode {i_episode}/{max_episodes} | Length: {t+1} | Reward: {episode_reward:.2f}")

    print("--- Training Finished ---")
    torch.save(agent.actor.state_dict(), actor_path)
    print(f"Saved trained model to {actor_path}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(episode_lengths); plt.title('Episode Length over Time')
    plt.subplot(1, 2, 2); plt.plot(final_glucose_levels); plt.axhline(y=70, color='r', linestyle='--'); plt.title('Final Glucose Level')
    plt.tight_layout(); plt.show()

    # --- Evaluation ---
    print("\n--- Starting Evaluation ---")
    eval_scenario = CustomScenario(start_time=start_time, scenario=[(7 * 60, 45), (12 * 60, 70), (18 * 60, 80)])
    eval_env = gymnasium.make(ENV_ID, custom_scenario=eval_scenario)

    eval_agent = SACAgent(eval_env, state_dim, action_dim, n_latent_var, lr, gamma_val, tau, alpha)
    eval_agent.actor.load_state_dict(torch.load(actor_path))

    manager.reset()
    obs_array, info = eval_env.reset()
    episode_scenario = info.get('scenario')

    # ! THE CORRECT PATH
    current_sim_time = eval_env.unwrapped.env.env.time
    upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0
    unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
    current_state = manager.get_normalized_state(unnormalized_state)
    glucose_history = [obs_array[0]]

    for t in range(max_timesteps_per_episode):
        action = eval_agent.select_action(current_state)
        safe_action = safety_layer.apply(action, unnormalized_state)
        manager.insulin_history.append(safe_action[0])
        obs_array, _, terminated, truncated, _ = eval_env.step(safe_action)

        # ! THE CORRECT PATH
        current_sim_time = eval_env.unwrapped.env.env.time
        upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0
        unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)

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
    plt.plot(glucose_history, label='SAC Agent')
    plt.axhline(y=180, color='r', linestyle=':', label='Hyperglycemia Threshold')
    plt.axhline(y=70, color='orange', linestyle=':', label='Hypoglycemia Threshold')
    plt.axhline(y=140, color='g', linestyle='-', label='Target')
    plt.title(f'SAC Agent Performance for {PATIENT_NAME}')
    plt.xlabel('Time (minutes)'); plt.ylabel('Blood Glucose (mg/dL)'); plt.legend(); plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()