import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import collections

# --- Agent Definition ---
# This is the BioInspiredRLAgent from our previous step.
# It calculates the custom rewards as defined in the base paper.

class BioInspiredRLAgent:
    """
    An agent that uses the bio-inspired reward functions from the paper.
    It takes random actions to test the reward calculation within the
    correct Gymnasium environment setup.
    """
    def __init__(self, env, g_target=120.0, m_target=-1/15.0):
        """
        Initializes the agent.
        """
        self.env = env
        self.action_space = env.action_space
        self.g_target = g_target
        self.m_target = m_target
        self.glucose_history = collections.deque(maxlen=2)

    def select_action(self, state):
        """
        Selects an action using a random policy for now.
        """
        return self.action_space.sample()

    def _calculate_long_term_reward(self, g):
        """
        Calculates the long-term reward (R_long) based on the paper.
        """
        dl = abs(g - self.g_target)
        if 70 <= g <= 180:
            return -dl
        else:
            return -3 * dl

    def _calculate_short_term_reward(self, g, v):
        """
        Calculates the short-term reward (R_short) based on the paper.
        """
        dr = abs(self.m_target * (g - self.g_target) - v)
        if g < 100:
            return -5 * dr if v < -0.6 else (-3 * dr if v < 3 else 0)
        elif 100 <= g < 160:
            return 0 if v >= 3 else -dr
        elif 160 <= g < 180:
            return -5 * dr if v >= 3 else -dr
        else: # g >= 180
            return -5 * dr if v >= 1.5 else -3 * dr

    def calculate_rewards(self, observation):
        """
        Calculates both long-term and short-term rewards.
        """
        current_glucose = observation
        self.glucose_history.append(current_glucose)
        
        if len(self.glucose_history) < 2:
            glucose_rate = 0.0
        else:
            # Rate is (current_glucose - previous_glucose) / time_interval
            # Simglucose sample time is 3 minutes by default
            glucose_rate = (self.glucose_history[1] - self.glucose_history[0]) / 3.0

        r_long = self._calculate_long_term_reward(current_glucose)
        r_short = self._calculate_short_term_reward(current_glucose, glucose_rate)

        return r_long, r_short

    def reset(self):
        """
        Resets the agent's internal state.
        """
        self.glucose_history.clear()
        print("Agent state has been reset.")


# --- Main Simulation Script ---

def main():
    """
    The main function to register the environment, create it, and
    run the simulation with our agent.
    """
    PATIENT_NAME = 'adolescent#001'
    
    # Create a "clean" name for the environment ID by replacing '#' with '-'
    CLEAN_PATIENT_NAME = PATIENT_NAME.replace('#', '-')
    ENV_ID = f'simglucose/{CLEAN_PATIENT_NAME}-v0'

    print(f"Registering new environment: {ENV_ID}")

    # This is the new, correct way to register the environment for gymnasium
    register(
        id=ENV_ID,
        entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        max_episode_steps=288,  # 288 steps for a 24-hour simulation (24*60/5)
        # We pass the ORIGINAL patient name with the '#' into the environment
        kwargs={"patient_name": PATIENT_NAME},
    )

    print("Attempting to create the simglucose environment...")
    try:
        # Use gymnasium.make with the newly registered ID
        env = gymnasium.make(ENV_ID)
        print("✅ Environment created successfully!")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return

    # Create an instance of our agent
    agent = BioInspiredRLAgent(env)
    
    # --- Simulation Loop ---
    print(f"\n--- Starting Simulation with {PATIENT_NAME} ---")
    observation_array, info = env.reset()
    # --- THIS IS THE FIX ---
    # Extract the scalar value from the observation array
    observation = observation_array[0]
    
    agent.reset()
    agent.glucose_history.append(observation) # Prime the history with the scalar value

    for t in range(20): # Run for 200 steps
        # The agent selects an action
        action = agent.select_action(observation)
        
        # The environment steps forward
        observation_array, reward, terminated, truncated, info = env.step(action)
        # --- THIS IS THE FIX ---
        # Extract the scalar value from the new observation array
        observation = observation_array[0]
        
        # Our agent calculates its own bio-inspired rewards
        r_long, r_short = agent.calculate_rewards(observation)
        
        print(f"\n-> Timestep {t + 1}")
        # Now this print statement will work correctly
        print(f"  - Observation (CGM): {observation:.2f} mg/dL")
        print(f"  - Action Taken (Insulin): {action[0]:.4f} U/min")
        print(f"  - Default Env Reward: {reward:.4f}")
        print(f"  - Bio-Inspired R_long: {r_long:.4f}")
        print(f"  - Bio-Inspired R_short: {r_short:.4f}")
        
        if terminated or truncated:
            print(f"\nEpisode finished after {t + 1} timesteps.")
            observation_array, info = env.reset()
            observation = observation_array[0]
            agent.reset()
            agent.glucose_history.append(observation)

    env.close()
    print("\n--- Simulation Finished ---")

if __name__ == "__main__":
    main()
