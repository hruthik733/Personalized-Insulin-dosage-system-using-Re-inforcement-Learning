import gym
from gym.envs.registration import register
from simglucose.simulation.scenario import CustomScenario
from datetime import datetime

# Register SimGlucose env with custom meal scenario
start_time = datetime(2018, 1, 1, 0, 0, 0)
meal_scenario = CustomScenario(start_time=start_time, scenario=[(1, 20)])

register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002', 'custom_scenario': meal_scenario}
)

# Import OpenAI Baselines PPO2 related modules
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy

# Create vectorized environment (required by Baselines)
env = DummyVecEnv([lambda: gym.make('simglucose-adolescent2-v0')])

# Train PPO agent for 100,000 timesteps
model = ppo2.learn(policy=MlpPolicy, env=env, total_timesteps=100000)

# Test trained agent
obs = env.reset()
total_reward = 0.0
for t in range(100):
    action, _states = model.step(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]
    env.render()
    if done:
        print(f"Episode finished after {t + 1} timesteps")
        break

print("Total Reward:", total_reward)
env.close()
