import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import copy

# --- 1. Register the simglucose Environment ---
try:
    register(
        id='simglucose-adult1-v0',
        entry_point='simglucose.envs:T1DSimGymnaisumEnv',
        kwargs={'patient_name': 'adult#001'}
    )
except gymnasium.error.Error:
    print("Environment 'simglucose-adult1-v0' is already registered.")

# --- 2. Define a Balanced Reward Function ---
def calculate_reward(glucose):
    if 70 <= glucose <= 180:
        return 1.0
    elif glucose < 70:
        return -0.1 * ((70 - glucose)**2)
    else:
        return -0.1 * ((glucose - 180)**2)

# --- 3. Define a Simple RL Agent (DQN) ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_space):
        self.state_dim = state_dim
        self.action_space = action_space
        self.action_dim = len(action_space)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(state_dim, self.action_dim).to(self.device)
        # ! NEW: Create a Target Network for stable learning
        self.target_net = DQN(state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        # ! NEW: Increase buffer size for more diverse experiences
        self.buffer = ReplayBuffer(50000)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        # ! NEW: Slower epsilon decay to encourage more exploration
        self.epsilon_decay = 0.9995

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def learn(self, batch_size):
        if len(self.buffer) < batch_size:
            return
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        
        # ! NEW: Use the stable target_net to calculate next Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        target_q_values = rewards + (1 - dones) * 0.99 * next_q_values
        
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# --- 4. Main Training Loop ---
if __name__ == '__main__':
    env = gymnasium.make('simglucose-adult1-v0')
    
    DISCRETE_ACTIONS = [0.0, 1.0, 2.5, 5.0]
    
    state_dimension = env.observation_space.shape[0]
    agent = DQNAgent(state_dim=state_dimension, action_space=DISCRETE_ACTIONS)
    
    # ! NEW: Increased episodes for more training time
    num_episodes = 500
    # ! NEW: Start learning only after collecting some random data
    learning_starts = 1000
    target_update_frequency = 10 # Update target network every 10 episodes
    
    episode_rewards = []
    total_steps = 0

    for i_episode in range(1, num_episodes + 1):
        state, info = env.reset()
        total_reward = 0
        
        for t in range(288):
            total_steps += 1
            action_index = agent.select_action(state)
            insulin_dose = DISCRETE_ACTIONS[action_index]
            
            next_state, reward, done, truncated, info = env.step(np.array([insulin_dose]))
            custom_reward = calculate_reward(state[0])
            total_reward += custom_reward

            agent.buffer.push(state, action_index, custom_reward, next_state, done)
            
            # Start learning only after the buffer has enough samples
            if total_steps > learning_starts:
                agent.learn(batch_size=64)

            state = next_state
            
            if done or truncated:
                break
        
        # ! NEW: Periodically update the target network
        if i_episode % target_update_frequency == 0 and total_steps > learning_starts:
            agent.update_target_net()
            print(f"--- Target Network Updated at Episode {i_episode} ---")
        
        episode_rewards.append(total_reward)
        print(f"Episode {i_episode}/{num_episodes} | Length: {t+1} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()