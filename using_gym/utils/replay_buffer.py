import torch
import numpy as np

class VectorizedReplayBuffer:
    def __init__(self, num_envs, capacity, state_dim, action_dim, device):
        self.num_envs = num_envs
        self.capacity = capacity
        self.device = device
        
        self.states = np.zeros((capacity, num_envs, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, num_envs), dtype=np.float32)
        self.next_states = np.zeros((capacity, num_envs, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, num_envs), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        num_transitions = self.size * self.num_envs
        if num_transitions < batch_size:
             # Not enough samples to form a full batch
             return None

        # Sample time steps and environments
        time_indices = np.random.randint(0, self.size, batch_size)
        env_indices = np.random.randint(0, self.num_envs, batch_size)
        
        states_batch = torch.FloatTensor(self.states[time_indices, env_indices]).to(self.device)
        actions_batch = torch.FloatTensor(self.actions[time_indices, env_indices]).to(self.device)
        rewards_batch = torch.FloatTensor(self.rewards[time_indices, env_indices]).to(self.device)
        next_states_batch = torch.FloatTensor(self.next_states[time_indices, env_indices]).to(self.device)
        dones_batch = torch.FloatTensor(self.dones[time_indices, env_indices]).to(self.device)
        
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch

    def __len__(self):
        return self.size * self.num_envs