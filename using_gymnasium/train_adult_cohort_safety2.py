import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import os
import random
import pandas as pd

# --- Local Imports ---
from agents.sac_agent import SACAgent
from utils.replay_buffer import ReplayBuffer
from utils.state_management2 import StateRewardManager
from utils.safety2 import SafetyLayer
import simglucose.simulation.scenario_gen as scgen
from simglucose.simulation.scenario import CustomScenario

def main():
    # Reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Hyperparameters
    # max_episodes = 2000
    max_episodes = 100
    lr = 3e-4
    gamma_val = 0.99
    tau = 0.005
    alpha = 0.2
    batch_size = 256
    n_latent_var = 256
    replay_buffer_size = 1000000
    max_timesteps_per_episode = 288
    learning_starts = 1000

    # Define the cohort of patients for training
    adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]

    # --- Setup a SINGLE COHORT MODEL ---
    AGENT_NAME = 'sac_cohort'
    model_dir = f'./models/{AGENT_NAME}'
    results_dir = f'./results/{AGENT_NAME}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    actor_path = f'{model_dir}/actor_adult_cohort.pth'

    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    # Register the environment with the LIST of patients
    meal_scenario = scgen.RandomScenario(start_time=start_time, seed=SEED)
    env_id = 'simglucose/adult-cohort-v0'
    try:
        register(
            id=env_id,
            entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
            max_episode_steps=max_timesteps_per_episode,
            kwargs={"patient_name": adult_patients, "custom_scenario": meal_scenario}
        )
    except gymnasium.error.Error:
        pass

    # Env dims
    state_dim = 4
    action_dim = 1

    # Initialize agent and buffer only ONCE
    # We create a dummy env just to get action_dim
    temp_env = gymnasium.make(env_id)
    agent = SACAgent(temp_env, state_dim, action_dim, n_latent_var, lr, gamma_val, tau, alpha)
    temp_env.close()
    
    manager = StateRewardManager(state_dim)
    safety_layer = SafetyLayer()
    replay_buffer = ReplayBuffer(replay_buffer_size)

    total_timesteps_taken = 0

    # --- SINGLE TRAINING LOOP for the whole cohort ---
    print("\n" + "="*50)
    print("--- Starting Training for Adult Cohort Model ---")
    print("="*50)
    for i_episode in range(1, max_episodes + 1):
        # ! FIX: Re-create the environment at the start of each episode
        # This forces the wrapper to call its __init__ and select a new random patient
        env = gymnasium.make(env_id)
        env.action_space.seed(SEED + i_episode) # Re-seed the action space
        
        # Now, reset the newly created environment
        obs_array, info = env.reset(seed=SEED + i_episode)
        
        patient_name_for_episode = info.get('patient_name', 'Unknown')
        episode_scenario = info.get('scenario')
        manager.reset()

        current_sim_time = env.unwrapped.env.env.time
        upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0
        unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
        current_state = manager.get_normalized_state(unnormalized_state)
        episode_reward = 0

        for t in range(max_timesteps_per_episode):
            if total_timesteps_taken < learning_starts:
                action = np.array([np.random.uniform(low=0, high=0.5)])
            else:
                action = agent.select_action(current_state)

            safe_action = safety_layer.apply(action, unnormalized_state)
            manager.insulin_history.append(safe_action[0])
            next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
            done = terminated or truncated

            current_sim_time = env.unwrapped.env.env.time
            upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0
            next_unnormalized_state = manager.get_full_state(next_obs_array[0], upcoming_carbs)
            next_state = manager.get_normalized_state(next_unnormalized_state)

            reward = manager.get_reward(unnormalized_state)
            replay_buffer.push(current_state, safe_action, reward, next_state, done)

            current_state = next_state
            unnormalized_state = next_unnormalized_state
            episode_reward += reward
            total_timesteps_taken += 1

            if total_timesteps_taken > learning_starts and len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)

            if done:
                break
        
        env.close() # Close the environment at the end of the episode

        if i_episode % 50 == 0:
            print(f"Episode {i_episode}/{max_episodes} | Patient: {patient_name_for_episode} | Reward: {episode_reward:.2f}")

    print("--- Training Finished ---")
    torch.save(agent.actor.state_dict(), actor_path)
    print(f"Saved trained cohort model to {actor_path}")

    # --- SYSTEMATIC EVALUATION on each patient ---
    print("\n" + "="*50)
    print("--- Starting Systematic Evaluation ---")
    print("="*50)
    
    all_patient_results = []
    # Load the single trained agent
    eval_agent = SACAgent(env, state_dim, action_dim, n_latent_var, lr, gamma_val, tau, alpha)
    eval_agent.actor.load_state_dict(torch.load(actor_path))
    eval_agent.actor.eval() # Set to evaluation mode

    for patient_name in adult_patients:
        print(f"\n--- Evaluating on Patient: {patient_name} ---")
        meal_times = [7 * 60, 12 * 60, 18 * 60]
        meal_carbs = [45, 70, 80]
        eval_scenario = CustomScenario(start_time=start_time, scenario=list(zip(meal_times, meal_carbs)))
        
        # Create a specific environment for this patient
        eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario, patient_name=patient_name)
        
        manager.reset()
        obs_array, info = eval_env.reset()
        
        # ! FIX: Use .meal instead of .CHO
        current_sim_time = eval_env.unwrapped.env.env.time
        upcoming_carbs = eval_scenario.get_action(current_sim_time).meal if eval_scenario else 0
        unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
        current_state = manager.get_normalized_state(unnormalized_state)
        glucose_history = [obs_array[0]]

        for t in range(max_timesteps_per_episode):
            with torch.no_grad():
                action = eval_agent.select_action(current_state)

            safe_action = safety_layer.apply(action, unnormalized_state)
            manager.insulin_history.append(safe_action[0])
            obs_array, _, terminated, truncated, _ = eval_env.step(safe_action)
            
            # ! FIX: Use .meal instead of .CHO
            current_sim_time = eval_env.unwrapped.env.env.time
            upcoming_carbs = eval_scenario.get_action(current_sim_time).meal if eval_scenario else 0
            unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
            current_state = manager.get_normalized_state(unnormalized_state)
            glucose_history.append(obs_array[0])

            if terminated or truncated:
                break
        
        eval_env.close()

        # Metrics
        glucose_history = np.array(glucose_history)
        time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
        time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
        time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
        mean_glucose = np.mean(glucose_history)
        
        patient_summary = { "Patient": patient_name, "Mean Glucose (mg/dL)": mean_glucose, "Time in Range (%)": time_in_range, "Time Hypo (%)": time_hypo, "Time Hyper (%)": time_hyper, }
        all_patient_results.append(patient_summary)

        # Plotting
        plt.figure(figsize=(15, 6))
        time_axis_minutes = np.arange(len(glucose_history)) * 5
        plt.plot(time_axis_minutes, glucose_history, label='SAC Agent Glucose')
        plt.axhline(y=180, color='r', linestyle=':', label='Hyper Threshold')
        plt.axhline(y=70, color='orange', linestyle=':', label='Hypo Threshold')
        for meal_time in meal_times:
            plt.axvline(x=meal_time, color='black', linestyle='--')
        plt.title(f'Performance for {patient_name}')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Blood Glucose (mg/dL)')
        plt.legend()
        plt.grid(True)
        plot_path = f'{results_dir}/evaluation_plot_{patient_name.replace("#", "-")}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved evaluation plot to {plot_path}")

    # SUMMARY
    print("\n" + "="*56)
    print("---           OVERALL PERFORMANCE SUMMARY           ---")
    print("="*56)
    results_df = pd.DataFrame(all_patient_results).set_index('Patient')
    print(results_df)
    print("\n--- Average Performance Across All Patients ---")
    print(results_df.mean().to_string())

if __name__ == '__main__':
    main()