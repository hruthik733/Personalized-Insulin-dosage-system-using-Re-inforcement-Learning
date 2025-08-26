import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import os
import random
import pandas as pd

# --- Local Imports from our modules ---
from agents.hybrid_agent import HybridAgent
from utils.safety import SafetyLayer
from utils.new_state_management import StateRewardManager
from simglucose.simulation.scenario import CustomScenario

def evaluate():
    # --- 1. Load Config ---
    with open('./configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- 2. Setup ---
    SEED = config['project']['seed']
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    config['project']['device'] = 'cpu' # Ensure evaluation runs on CPU
    print(f"Using device: {device}")

    # --- 3. Define Patients and Model Path ---
    PATIENT_COHORT = config['environment']['patient_cohort']
    actor_path = config['agent']['model_path']
    results_dir = './results/hybrid_agent_evaluation'
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    
    # --- 4. Environment Setup ---
    ENV_ID = config['environment']['env_id']
    try:
        register(id=ENV_ID, entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv")
    except gymnasium.error.Error:
        print(f"Environment {ENV_ID} already registered.")
        
    # --- 5. Load the Trained Agent ---
    eval_agent = HybridAgent(config)
    eval_agent.load_state_dict(torch.load(actor_path, map_location=device))
    eval_agent.eval() # Set model to evaluation mode
    print(f"Successfully loaded trained model from {actor_path}")

    # --- 6. Systematic Evaluation Loop ---
    print("\n--- Starting Systematic Evaluation on Adult Cohort ---")
    all_patient_results = []
    start_time = datetime.combine(datetime.now().date(), datetime.min.time())
    meal_times = [7 * 60, 12 * 60, 18 * 60]
    meal_carbs = [45, 70, 80]
    eval_scenario = CustomScenario(start_time=start_time, scenario=list(zip(meal_times, meal_carbs)))
    
    for patient_name in PATIENT_COHORT:
        print(f"--- Evaluating on Patient: {patient_name} ---")
        eval_env = gymnasium.make(ENV_ID, custom_scenario=eval_scenario, patient_name=patient_name)
        
        manager = StateRewardManager(state_dim=config['agent']['state_dim'], cohort_name='adult')
        safety_layer = SafetyLayer()
        obs_array, info = eval_env.reset()
        
        glucose_history = []
        insulin_history = []

        unnormalized_state = manager.get_full_state(obs_array, eval_env.unwrapped.env.env.time, 0)
        current_state = manager.get_normalized_state(unnormalized_state)
        glucose_history.append(unnormalized_state[0])
        insulin_history.append(0.0)

        for t in range(config['environment']['max_episode_steps']):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(current_state).to(device)
                scaled_action, _ = eval_agent.act(state_tensor)
                unscaled_action = scaled_action.cpu().numpy()

            safe_action = safety_layer.apply(unscaled_action, unnormalized_state)
            clipped_action = np.clip(safe_action, 0.0, config['agent']['max_action'])
            
            manager.insulin_history.append(clipped_action.item())
            obs_array, _, terminated, truncated, _ = eval_env.step(clipped_action)
            
            core_env = eval_env.unwrapped.env.env
            current_sim_time = core_env.time
            upcoming_carbs = eval_env.unwrapped.env.custom_scenario.get_action(current_sim_time).meal
            unnormalized_state = manager.get_full_state(obs_array, current_sim_time, upcoming_carbs)
            current_state = manager.get_normalized_state(unnormalized_state)
            
            glucose_history.append(unnormalized_state[0])
            insulin_history.append(clipped_action.item())
            if terminated or truncated:
                break
        
        eval_env.close()

        # ! NEW: Generate and Save Plot for this patient
        fig, ax1 = plt.subplots(figsize=(15, 7))
        time_axis = np.arange(len(glucose_history)) * 5 # Time in minutes

        # Plot Glucose
        ax1.plot(time_axis, glucose_history, 'b-', label='Blood Glucose (mg/dL)')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Blood Glucose (mg/dL)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, linestyle='--')
        
        # Add TIR lines and fill
        ax1.axhline(y=180, color='r', linestyle=':', label='Hyperglycemia (>180)')
        ax1.axhline(y=70, color='orange', linestyle=':', label='Hypoglycemia (<70)')
        ax1.fill_between(time_axis, 70, 180, color='green', alpha=0.15, label='Time in Range (70-180)')
        
        # Plot Meals as vertical lines
        for meal_time, meal_carb in zip(meal_times, meal_carbs):
            ax1.axvline(x=meal_time, color='black', linestyle='--', label=f'{meal_carb}g Meal')

        # Plot Insulin on a second y-axis
        ax2 = ax1.twinx()
        ax2.bar(time_axis, insulin_history, width=5, color='gray', alpha=0.6, label='Insulin Dose (U/hr)')
        ax2.set_ylabel('Insulin Dose (U/hr)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        fig.suptitle(f'Hybrid Agent Performance for {patient_name}', fontsize=16)
        # Create a single legend for both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_path = f'{results_dir}/evaluation_plot_{patient_name.replace("#", "-")}.png'
        plt.savefig(plot_path)
        plt.close(fig) # Close the figure to free up memory
        print(f"Saved evaluation plot to {plot_path}")

        # --- Calculate and Store Metrics ---
        glucose_history = np.array(glucose_history)
        time_in_range = np.sum((glucose_history >= 70) & (glucose_history <= 180)) / len(glucose_history) * 100
        time_hypo = np.sum(glucose_history < 70) / len(glucose_history) * 100
        time_hyper = np.sum(glucose_history > 180) / len(glucose_history) * 100
        mean_glucose = np.mean(glucose_history)
        all_patient_results.append({
            "Patient": patient_name, "Mean Glucose (mg/dL)": mean_glucose,
            "Time in Range (%)": time_in_range, "Time Hypo (%)": time_hypo,
            "Time Hyper (%)": time_hyper
        })

    # --- 7. Final Summary ---
    print("\n\n========================================================")
    print("---      COHORT OVERALL PERFORMANCE SUMMARY      ---")
    print("========================================================")
    results_df = pd.DataFrame(all_patient_results)
    results_df.set_index('Patient', inplace=True)
    print("\n--- Detailed Results Per Patient ---")
    print(results_df.round(2))
    average_performance = results_df.mean()
    print("\n--- Average Performance ---")
    print(average_performance.round(2).to_string())
    results_df.to_csv(f'{results_dir}/cohort_evaluation_summary.csv')
    print(f"\nSaved detailed summary results to {results_dir}/cohort_evaluation_summary.csv")

if __name__ == '__main__':
    evaluate()