import gym

# Register gym environment. By specifying kwargs,
# you are able to choose which patient or patients to simulate.
# patient_name must be 'adolescent#001' to 'adolescent#010',
# or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
# It can also be a list of patient names
# You can also specify a custom scenario or a list of custom scenarios
# If you chose a list of patient names or a list of custom scenarios,
# every time the environment is reset, a random patient and scenario will be
# chosen from the list

from gym.envs.registration import register
from simglucose.simulation.scenario import CustomScenario
from datetime import datetime

start_time = datetime(2018, 1, 1, 0, 0, 0)
meal_scenario = CustomScenario(start_time=start_time, scenario=[(5,20)])


register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002',
            'custom_scenario': meal_scenario}
)

env = gym.make('simglucose-adolescent2-v0')

observation = env.reset()
for t in range(500):
    env.render(mode='human')
    print(observation)
    # Action in the gym environment is a scalar
    # representing the basal insulin, which differs from
    # the regular controller action outside the gym
    # environment (a tuple (basal, bolus)).
    # In the perfect situation, the agent should be able
    # to control the glucose only through basal instead
    # of asking patient to take bolus
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break