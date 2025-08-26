# import gym
# import simglucose
# from pprint import pprint

# def test_simglucose_env():
#     """
#     Initializes the simglucose environment and runs a few steps with random actions.
#     """
#     print("Attempting to create the simglucose environment...")
    
#     # The standard ID for the simglucose environment is 'simglucose-v0'
#     try:
#         env = gym.make('simglucose-v0')
#         print("✅ Environment created successfully!")
#     except Exception as e:
#         print(f"❌ Error creating environment: {e}")
#         return

#     # Reset the environment to get the first observation
#     observation = env.reset()
#     print("\n--- Starting Simulation ---")
#     print(f"Initial Observation (Blood Glucose): {observation}")

#     # Run the simulation for a few steps
#     num_steps = 5
#     for i in range(num_steps):
#         # Get a random action from the environment's action space
#         # In simglucose, this is a random insulin dose
#         action = env.action_space.sample()
        
#         # Take the action and get the results
#         observation, reward, done, info = env.step(action)
        
#         print(f"\n-> Step {i + 1}")
#         print(f"  Action (Insulin Dose): {action}")
#         print(f"  Next Observation (Blood Glucose): {observation}")
#         print(f"  Reward: {reward}")
#         print(f"  Episode Done: {done}")
#         print( "  Info:")
#         pprint(info)
        
#         # If the episode is finished, reset the environment
#         if done:
#             print("\nEpisode finished. Resetting environment.")
#             observation = env.reset()

#     # Clean up the environment
#     env.close()
#     print("\n--- Simulation Finished ---")
#     print("✅ Test script completed.")

# if __name__ == "__main__":
#     test_simglucose_env()






import gymnasium as gym
import simglucose
import numpy as np
from pprint import pprint
import collections

class BioInspiredRLAgent:
    """
    An agent that uses the bio-inspired reward functions from the paper.
    For now, it takes random actions to test the reward calculation.
    """
    def __init__(self, env, g_target=120.0, m_target=-1/15.0):
        """
        Initializes the agent.

        Args:
            env: The Simglucose environment.
            g_target (float): The target blood glucose level for long-term reward.
            m_target (float): The target slope for the short-term reward.
        """
        self.env = env
        self.g_target = g_target
        self.m_target = m_target
        # Use a deque to store the last two glucose readings to calculate the rate
        self.glucose_history = collections.deque(maxlen=2)

    def select_action(self, state):
        """
        Selects an action. For now, this is random.
        In the future, this will use a trained policy network.
        """
        # For this initial version, we take a random action
        return self.env.action_space.sample()

    def _calculate_long_term_reward(self, g):
        """
        Calculates the long-term reward (R_long) based on the paper.
        This reward encourages maintaining a basal glucose level.

        Args:
            g (float): The current blood glucose level.

        Returns:
            float: The calculated long-term reward.
        """
        dl = abs(g - self.g_target)
        if 70 <= g <= 180:
            return -dl
        else:
            # Penalize out-of-range values more heavily
            return -3 * dl

    def _calculate_short_term_reward(self, g, v):
        """
        Calculates the short-term reward (R_short) based on the paper.
        This reward encourages rapid response to glucose changes.

        Args:
            g (float): The current blood glucose level.
            v (float): The current rate of change of blood glucose.

        Returns:
            float: The calculated short-term reward.
        """
        dr = abs(self.m_target * (g - self.g_target) - v)
        
        # Penalize based on clinical risk zones defined in the paper
        if g < 100:
            if v < -0.6:
                return -5 * dr
            elif v >= 3:
                return 0
            else:
                return -3 * dr
        elif 100 <= g < 160:
            if v >= 3:
                # Natural postprandial increase, no penalty
                return 0
            else:
                return -dr
        elif 160 <= g < 180:
            if v >= 3:
                return -5 * dr
            else:
                return -dr
        else: # g >= 180
            if v >= 1.5:
                return -5 * dr
            else:
                return -3 * dr

    def calculate_rewards(self, observation):
        """
        Calculates both long-term and short-term rewards.

        Args:
            observation: The observation object from the environment.

        Returns:
            tuple: A tuple containing (long_term_reward, short_term_reward).
        """
        current_glucose = observation.CGM
        self.glucose_history.append(current_glucose)
        
        # Calculate glucose rate of change (v)
        if len(self.glucose_history) < 2:
            glucose_rate = 0.0
        else:
            # Rate is (current_glucose - previous_glucose) / time_interval
            # Simglucose sample time is 3 minutes by default
            glucose_rate = (self.glucose_history[1] - self.glucose_history[0]) / 3.0

        r_long = self._calculate_long_term_reward(current_glucose)
        r_short = self._calculate_short_term_reward(current_glucose, glucose_rate)

        return r_long, r_short


def run_simulation():
    """
    Initializes the environment and agent, then runs a simulation loop.
    """
    print("Attempting to create the simglucose environment...")
    try:
        # Use gymnasium.make for compatibility
        env = gym.make('simglucose-v0')
        print("✅ Environment created successfully!")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return

    # Initialize our agent
    agent = BioInspiredRLAgent(env)
    
    # Reset the environment to get the first observation
    observation, info = env.reset()
    agent.glucose_history.append(observation.CGM) # Prime the history

    print("\n--- Starting Simulation with Bio-Inspired Agent ---")
    print(f"Initial Observation (CGM): {observation.CGM:.2f} mg/dL")

    # Run the simulation for a few steps
    num_steps = 10
    for i in range(num_steps):
        # The agent selects an action based on the current state
        action = agent.select_action(observation)
        
        # The environment steps forward based on the action
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        # Our agent calculates its own bio-inspired rewards
        r_long, r_short = agent.calculate_rewards(next_observation)
        
        print(f"\n-> Step {i + 1}")
        print(f"  Action (Insulin Dose): {action[0]:.4f} U")
        print(f"  Next Observation (CGM): {next_observation.CGM:.2f} mg/dL")
        print(f"  Default Env Reward: {reward:.4f}")
        print(f"  Bio-Inspired R_long: {r_long:.4f}")
        print(f"  Bio-Inspired R_short: {r_short:.4f}")
        
        observation = next_observation
        
        if terminated or truncated:
            print("\nEpisode finished. Resetting environment.")
            observation, info = env.reset()
            agent.glucose_history.clear()
            agent.glucose_history.append(observation.CGM)


    # Clean up the environment
    env.close()
    print("\n--- Simulation Finished ---")
    print("✅ Agent script completed.")

if __name__ == "__main__":
    run_simulation()
