import torch
import torch.nn as nn
from torch.distributions import Normal

class HybridAgent(nn.Module):
    def __init__(self, config):
        super(HybridAgent, self).__init__()
        
        state_dim = config['agent']['state_dim']
        action_dim = config['agent']['action_dim']
        feature_size = config['agent']['feature_extractor_size']
        ac_hidden_size = config['agent']['actor_critic_hidden_size']
        pred_hidden_size = config['agent']['prediction_hidden_size']
        self.max_action = config['agent']['max_action']
        self.device = torch.device(config['project']['device'])

        # 1. Shared Body
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size),
            nn.ReLU()
        ).to(self.device)

        # 2. Control Head
        self.actor = nn.Sequential(
            nn.Linear(feature_size, ac_hidden_size),
            nn.Tanh(),
            nn.Linear(ac_hidden_size, action_dim),
            nn.Tanh()
        ).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Linear(feature_size, ac_hidden_size),
            nn.Tanh(),
            nn.Linear(ac_hidden_size, 1)
        ).to(self.device)
        
        # 3. Prediction Head
        self.prediction_head = nn.Sequential(
            nn.Linear(feature_size + action_dim, pred_hidden_size),
            nn.ReLU(),
            nn.Linear(pred_hidden_size, state_dim)
        ).to(self.device)
        
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim)).to(self.device)

    def act(self, state):
        # ! FIX: Ensure state has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        features = self.feature_extractor(state)
        action_mean = self.actor(features) * self.max_action
        
        action_std = self.action_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(-1)
        
        return action.detach(), action_log_prob.detach()

    def evaluate(self, state, action):
        features = self.feature_extractor(state)
        action_mean = self.actor(features) * self.max_action
        
        # ! FIX: Ensure action_log_std is expanded correctly
        action_std = self.action_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        
        action_log_probs = dist.log_prob(action).sum(-1)
        dist_entropy = dist.entropy().sum(-1)
        state_values = self.critic(features)
        
        return action_log_probs, state_values, dist_entropy

    def predict_next_state(self, state, action):
        features = self.feature_extractor(state)
        prediction_input = torch.cat([features, action], dim=-1)
        predicted_next_state_delta = self.prediction_head(prediction_input)
        # The model predicts the CHANGE in state
        predicted_next_state = state + predicted_next_state_delta
        return predicted_next_state