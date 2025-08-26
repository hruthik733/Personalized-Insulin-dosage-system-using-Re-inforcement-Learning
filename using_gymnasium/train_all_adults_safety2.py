# import gymnasium
# from gymnasium.envs.registration import register
# import numpy as np
# import torch
# from datetime import datetime
# import matplotlib.pyplot as plt
# import os
# import random

# # --- Local Imports ---
# from agents.sac_agent import SACAgent
# from utils.replay_buffer import ReplayBuffer
# from utils.state_management2 import StateRewardManager   # balanced reward version
# from utils.safety2 import SafetyLayer                    # improved Safety Layer

# import simglucose.simulation.scenario_gen as scgen
# from simglucose.simulation.scenario import CustomScenario

# def main():
#     # Reproducibility
#     SEED = 42
#     random.seed(SEED)
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)

#     # Hyperparameters
#     max_episodes = 200
#     lr = 3e-4
#     gamma_val = 0.99
#     tau = 0.005
#     alpha = 0.2
#     batch_size = 256
#     n_latent_var = 256
#     replay_buffer_size = 1000000
#     max_timesteps_per_episode = 288  # 24 hours @ 5-min steps
#     learning_starts = 1000

#     # Patient list
#     adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]
#     all_patient_results = []

#     for patient_name in adult_patients:
#         print("\n" + "="*50)
#         print(f"--- Starting Training for Patient: {patient_name} ---")
#         print("="*50)

#         # Setup output dirs
#         AGENT_NAME = 'sac'
#         model_dir = f'./models/{AGENT_NAME}'
#         results_dir = f'./results/{AGENT_NAME}'
#         os.makedirs(model_dir, exist_ok=True)
#         os.makedirs(results_dir, exist_ok=True)
#         actor_path = f'{model_dir}/actor_{patient_name.replace("#", "-")}.pth'

#         now = datetime.now()
#         start_time = datetime.combine(now.date(), datetime.min.time())

#         # Register simglucose env for patient
#         meal_scenario = scgen.RandomScenario(start_time=start_time, seed=SEED)
#         clean_patient_name = patient_name.replace('#', '-')
#         env_id = f'simglucose/{clean_patient_name}-v0'

#         try:
#             register(
#                 id=env_id,
#                 entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
#                 max_episode_steps=max_timesteps_per_episode,
#                 kwargs={"patient_name": patient_name, "custom_scenario": meal_scenario}
#             )
#         except gymnasium.error.Error:
#             pass

#         env = gymnasium.make(env_id)
#         env.action_space.seed(SEED)

#         # Env dims
#         state_dim = 4
#         action_dim = 1

#         # Init agent, safety, buffer
#         agent = SACAgent(env, state_dim, action_dim, n_latent_var, lr, gamma_val, tau, alpha)
#         manager = StateRewardManager(state_dim)
#         safety_layer = SafetyLayer()
#         replay_buffer = ReplayBuffer(replay_buffer_size)

#         total_timesteps_taken = 0

#         # TRAINING LOOP
#         for i_episode in range(1, max_episodes + 1):
#             obs_array, info = env.reset(seed=SEED + i_episode)
#             episode_scenario = info.get('scenario')
#             manager.reset()

#             current_sim_time = env.unwrapped.env.env.time
#             upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0
#             unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
#             current_state = manager.get_normalized_state(unnormalized_state)
#             episode_reward = 0

#             for t in range(max_timesteps_per_episode):
#                 if total_timesteps_taken < learning_starts:
#                     action = np.array([np.random.uniform(low=0, high=0.5)])
#                 else:
#                     action = agent.select_action(current_state)

#                 # Apply safety
#                 safe_action = safety_layer.apply(action, unnormalized_state)

#                 # Step env
#                 manager.insulin_history.append(safe_action[0])
#                 next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
#                 done = terminated or truncated

#                 # Next state
#                 current_sim_time = env.unwrapped.env.env.time
#                 upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0
#                 next_unnormalized_state = manager.get_full_state(next_obs_array[0], upcoming_carbs)
#                 next_state = manager.get_normalized_state(next_unnormalized_state)

#                 # Reward + buffer
#                 reward = manager.get_reward(unnormalized_state)
#                 replay_buffer.push(current_state, safe_action, reward, next_state, done)

#                 current_state = next_state
#                 unnormalized_state = next_unnormalized_state
#                 episode_reward += reward
#                 total_timesteps_taken += 1

#                 if total_timesteps_taken > learning_starts and len(replay_buffer) > batch_size:
#                     agent.update(replay_buffer, batch_size)

#                 if done:
#                     break

#             if i_episode % 50 == 0:
#                 print(f"Episode {i_episode}/{max_episodes} | Reward: {episode_reward:.2f}")

#         print(f"--- Training Finished for {patient_name} ---")
#         torch.save(agent.actor.state_dict(), actor_path)
#         print(f"Saved trained model to {actor_path}")

#         # EVALUATION LOOP
#         print(f"\n--- Starting Evaluation for {patient_name} ---")
#         eval_scenario = CustomScenario(start_time=start_time,
#                                        scenario=[(7 * 60, 45), (12 * 60, 70), (18 * 60, 80)])
#         eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)

#         eval_agent = SACAgent(eval_env, state_dim, action_dim, n_latent_var, lr, gamma_val, tau, alpha)
#         eval_agent.actor.load_state_dict(torch.load(actor_path))

#         manager.reset()
#         obs_array, info = eval_env.reset()
#         episode_scenario = info.get('scenario')

#         current_sim_time = eval_env.unwrapped.env.env.time
#         upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0
#         unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
#         current_state = manager.get_normalized_state(unnormalized_state)
#         glucose_history = [obs_array[0]]

#         for t in range(max_timesteps_per_episode):
#             action = eval_agent.select_action(current_state)
#             safe_action = safety_layer.apply(action, unnormalized_state)
#             manager.insulin_history.append(safe_action[0])
#             obs_array, _, terminated, truncated, _ = eval_env.step(safe_action)

#             current_sim_time = eval_env.unwrapped.env.env.time
#             upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0
#             unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
#             current_state = manager.get_normalized_state(unnormalized_state)
#             glucose_history.append(obs_array[0])

#             if terminated or truncated:
#                 break

#         eval_env.close()

#         # Metrics
#         glucose_history = np.array(glucose_history)
#         time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
#         time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
#         time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
#         mean_glucose = np.mean(glucose_history)

#         print("\n--- Evaluation Results ---")
#         print(f"Mean Glucose: {mean_glucose:.2f} mg/dL")
#         print(f"Time in Range (70-180 mg/dL): {time_in_range:.2f}%")
#         print(f"Time Hypo (<70 mg/dL): {time_hypo:.2f}%")
#         print(f"Time Hyper (>180 mg/dL): {time_hyper:.2f}%")

#         patient_summary = {
#             "Patient": patient_name,
#             "Mean Glucose (mg/dL)": mean_glucose,
#             "Time in Range (%)": time_in_range,
#             "Time Hypo (%)": time_hypo,
#             "Time Hyper (%)": time_hyper,
#         }
#         all_patient_results.append(patient_summary)

#         # Plot
#         plt.figure(figsize=(15, 6))
#         plt.plot(glucose_history, label='SAC Agent')
#         plt.axhline(y=180, color='r', linestyle=':', label='Hyper Threshold')
#         plt.axhline(y=70, color='orange', linestyle=':', label='Hypo Threshold')
#         plt.axhline(y=100, color='g', linestyle='--', label='Target 100 mg/dL')
#         plt.title(f'Performance for {patient_name}')
#         plt.xlabel('Time step (5 min)')
#         plt.ylabel('Blood Glucose (mg/dL)')
#         plt.legend()
#         plt.grid(True)

#         plot_path = f'{results_dir}/evaluation_plot_{clean_patient_name}.png'
#         plt.savefig(plot_path)
#         plt.close()
#         print(f"Saved evaluation plot to {plot_path}")

#     # SUMMARY
#     print("\n" + "="*56)
#     print("---           OVERALL PERFORMANCE SUMMARY          ---")
#     print("="*56)
#     import pandas as pd
#     results_df = pd.DataFrame(all_patient_results).set_index('Patient')
#     print(results_df)
#     print("\n--- Average Performance Across All Patients ---")
#     print(results_df.mean().to_string())

# if __name__ == '__main__':
#     main()





















import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import os
import random

# --- Local Imports ---
from agents.sac_agent import SACAgent
from utils.replay_buffer import ReplayBuffer
from utils.state_management2 import StateRewardManager   # balanced reward version
from utils.safety2 import SafetyLayer                    # improved Safety Layer

import simglucose.simulation.scenario_gen as scgen
from simglucose.simulation.scenario import CustomScenario

def main():
    # Reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Hyperparameters
    max_episodes = 200
    lr = 3e-4
    gamma_val = 0.99
    tau = 0.005
    alpha = 0.2
    batch_size = 256
    n_latent_var = 256
    replay_buffer_size = 1000000
    max_timesteps_per_episode = 288  # 24 hours @ 5-min steps
    learning_starts = 1000

    # Patient list
    adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]
    all_patient_results = []

    for patient_name in adult_patients:
        print("\n" + "="*50)
        print(f"--- Starting Training for Patient: {patient_name} ---")
        print("="*50)

        # Setup output dirs
        AGENT_NAME = 'sac'
        model_dir = f'./models/{AGENT_NAME}'
        results_dir = f'./results/{AGENT_NAME}'
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        actor_path = f'{model_dir}/actor_{patient_name.replace("#", "-")}.pth'

        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())

        # Register simglucose env for patient
        meal_scenario = scgen.RandomScenario(start_time=start_time, seed=SEED)
        clean_patient_name = patient_name.replace('#', '-')
        env_id = f'simglucose/{clean_patient_name}-v0'

        try:
            register(
                id=env_id,
                entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
                max_episode_steps=max_timesteps_per_episode,
                kwargs={"patient_name": patient_name, "custom_scenario": meal_scenario}
            )
        except gymnasium.error.Error:
            pass

        env = gymnasium.make(env_id)
        env.action_space.seed(SEED)

        # Env dims
        state_dim = 4
        action_dim = 1

        # Init agent, safety, buffer
        agent = SACAgent(env, state_dim, action_dim, n_latent_var, lr, gamma_val, tau, alpha)
        manager = StateRewardManager(state_dim)
        safety_layer = SafetyLayer()
        replay_buffer = ReplayBuffer(replay_buffer_size)

        total_timesteps_taken = 0

        # TRAINING LOOP
        for i_episode in range(1, max_episodes + 1):
            obs_array, info = env.reset(seed=SEED + i_episode)
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

                # Apply safety
                safe_action = safety_layer.apply(action, unnormalized_state)

                # Step env
                manager.insulin_history.append(safe_action[0])
                next_obs_array, _, terminated, truncated, _ = env.step(safe_action)
                done = terminated or truncated

                # Next state
                current_sim_time = env.unwrapped.env.env.time
                upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0
                next_unnormalized_state = manager.get_full_state(next_obs_array[0], upcoming_carbs)
                next_state = manager.get_normalized_state(next_unnormalized_state)

                # Reward + buffer
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

            if i_episode % 50 == 0:
                print(f"Episode {i_episode}/{max_episodes} | Reward: {episode_reward:.2f}")

        print(f"--- Training Finished for {patient_name} ---")
        torch.save(agent.actor.state_dict(), actor_path)
        print(f"Saved trained model to {actor_path}")

        # EVALUATION LOOP
        print(f"\n--- Starting Evaluation for {patient_name} ---")
        # Define the fixed meal times and amounts for evaluation
        meal_times = [7 * 60, 12 * 60, 18 * 60] # In minutes from start of day
        meal_carbs = [45, 70, 80]
        eval_scenario = CustomScenario(start_time=start_time,
                                       scenario=list(zip(meal_times, meal_carbs)))
        
        eval_env = gymnasium.make(env_id, custom_scenario=eval_scenario)

        eval_agent = SACAgent(eval_env, state_dim, action_dim, n_latent_var, lr, gamma_val, tau, alpha)
        eval_agent.actor.load_state_dict(torch.load(actor_path))
        eval_agent.actor.eval() # Set to evaluation mode

        manager.reset()
        obs_array, info = eval_env.reset()
        episode_scenario = info.get('scenario')

        current_sim_time = eval_env.unwrapped.env.env.time
        upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0
        unnormalized_state = manager.get_full_state(obs_array[0], upcoming_carbs)
        current_state = manager.get_normalized_state(unnormalized_state)
        glucose_history = [obs_array[0]]

        for t in range(max_timesteps_per_episode):
            # In evaluation, use the deterministic action without exploration noise
            with torch.no_grad():
                action = eval_agent.select_action(current_state)

            safe_action = safety_layer.apply(action, unnormalized_state)
            manager.insulin_history.append(safe_action[0])
            obs_array, _, terminated, truncated, _ = eval_env.step(safe_action)

            current_sim_time = eval_env.unwrapped.env.env.time
            upcoming_carbs = episode_scenario.get_action(current_sim_time).CHO if episode_scenario else 0
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

        print("\n--- Evaluation Results ---")
        print(f"Mean Glucose: {mean_glucose:.2f} mg/dL")
        print(f"Time in Range (70-180 mg/dL): {time_in_range:.2f}%")
        print(f"Time Hypo (<70 mg/dL): {time_hypo:.2f}%")
        print(f"Time Hyper (>180 mg/dL): {time_hyper:.2f}%")

        patient_summary = {
            "Patient": patient_name,
            "Mean Glucose (mg/dL)": mean_glucose,
            "Time in Range (%)": time_in_range,
            "Time Hypo (%)": time_hypo,
            "Time Hyper (%)": time_hyper,
        }
        all_patient_results.append(patient_summary)

        # Plot
        plt.figure(figsize=(15, 6))
        # The x-axis is in 5-minute steps, so we create a time axis in minutes
        time_axis_minutes = np.arange(len(glucose_history)) * 5
        
        plt.plot(time_axis_minutes, glucose_history, label='SAC Agent Glucose')
        plt.axhline(y=180, color='r', linestyle=':', label='Hyper Threshold')
        plt.axhline(y=70, color='orange', linestyle=':', label='Hypo Threshold')
        plt.axhline(y=100, color='g', linestyle='--', label='Target 100 mg/dL')
        
        # ! NEW: Add vertical lines for each meal
        for meal_time, meal_amount in zip(meal_times, meal_carbs):
            plt.axvline(x=meal_time, color='black', linestyle='--', 
                        label=f'Meal ({meal_amount}g) at {meal_time//60}:00')
        
        plt.title(f'Performance for {patient_name}')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Blood Glucose (mg/dL)')
        
        # Improve legend handling to avoid duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.grid(True)

        plot_path = f'{results_dir}/evaluation_plot_{clean_patient_name}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved evaluation plot to {plot_path}")

    # SUMMARY
    print("\n" + "="*56)
    print("---           OVERALL PERFORMANCE SUMMARY          ---")
    print("="*56)
    import pandas as pd
    results_df = pd.DataFrame(all_patient_results).set_index('Patient')
    print(results_df)
    print("\n--- Average Performance Across All Patients ---")
    print(results_df.mean().to_string())

if __name__ == '__main__':
    main()
