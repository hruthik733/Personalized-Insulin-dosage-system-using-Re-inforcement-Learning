# agents/sac_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# --- Soft Actor-Critic (SAC) Network Definitions ---
class Critic(nn.Module): # Q-Network
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_3 = nn.Linear(n_latent_var, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

class Actor(nn.Module): # Policy Network
    def __init__(self, state_dim, action_dim, n_latent_var, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.mean_layer = nn.Linear(n_latent_var, action_dim)
        self.log_std_layer = nn.Linear(n_latent_var, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2) # Clamp for stability
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# --- SAC Agent Implementation ---
class SACAgent:
    def __init__(self, env, state_dim, action_dim, n_latent_var, lr, gamma, tau, alpha):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.max_action = float(env.action_space.high[0])

        self.actor = Actor(state_dim, action_dim, n_latent_var, self.max_action)
        self.critic_1 = Critic(state_dim, action_dim, n_latent_var)
        self.critic_1_target = Critic(state_dim, action_dim, n_latent_var)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = Critic(state_dim, action_dim, n_latent_var)
        self.critic_2_target = Critic(state_dim, action_dim, n_latent_var)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy().flatten()

    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)