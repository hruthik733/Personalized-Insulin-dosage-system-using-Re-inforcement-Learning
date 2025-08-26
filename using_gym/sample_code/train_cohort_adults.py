#!/usr/bin/env python3
"""
Reinforcement Learning Training Script for Adult Cohort Insulin Prediction
Using simglucose with gymnasium and comprehensive state management

Features:
- Multi-patient adult cohort training
- Enhanced state space with glucose history, IOB, meals, time features
- Asymmetric reward function favoring hypoglycemia avoidance
- PPO algorithm with safety constraints
- Comprehensive logging and monitoring
"""

import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from collections import deque
from datetime import datetime, timedelta
import pickle
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add simglucose to path if needed
sys.path.append('simglucose')

from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedT1DEnv(gym.Env):
    """Enhanced T1D Environment with comprehensive state management"""
    
    def __init__(self, patient_names: List[str], max_episode_steps: int = 1000):
        super().__init__()
        
        # Patient pool - all adults
        self.patient_names = patient_names
        self.current_patient_idx = 0
        self.max_episode_steps = max_episode_steps
        
        # Initialize history buffers
        self.glucose_history_size = 12  # 1 hour history (5-min intervals)
        self.hypo_hyper_history_size = 24  # 2 hour history for hypo/hyper events
        
        # State components
        self.glucose_history = deque(maxlen=self.glucose_history_size)
        self.hypo_history = deque(maxlen=self.hypo_hyper_history_size)
        self.hyper_history = deque(maxlen=self.hypo_hyper_history_size)
        
        # Insulin tracking
        self.insulin_history = deque(maxlen=24)  # Track insulin for IOB calculation
        self.last_insulin_time = 0
        
        # Meal tracking
        self.last_meal_time = 0
        self.last_meal_carbs = 0
        self.meal_absorbed = 0
        
        # Time tracking
        self.episode_start_time = None
        self.current_step = 0
        
        # Action space: basal insulin rate (0-5 U/hr)
        self.action_space = spaces.Box(low=0.0, high=5.0, shape=(1,), dtype=np.float32)
        
        # Observation space: comprehensive state vector
        obs_dim = (1 +  # current glucose
                  self.glucose_history_size +  # glucose history
                  1 +  # insulin on board
                  1 +  # time since last insulin
                  3 +  # meal info (time, carbs, absorption)
                  2 +  # time features (hour of day, normalized)
                  2 +  # hypo/hyper history counts
                  1)   # glucose rate of change
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self._create_environment()
        
    def _create_environment(self):
        """Create simglucose environment for current patient"""
        patient_name = self.patient_names[self.current_patient_idx]
        
        # Create patient, sensor, and pump
        self.patient = T1DPatient.withName(patient_name)
        self.sensor = CGMSensor.withName('Dexcom', seed=np.random.randint(1000))
        self.pump = InsulinPump.withName('Insulet')
        
        # Create random scenario
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.scenario = RandomScenario(start_time=start_time, seed=np.random.randint(1000))
        
        # Create T1D simulation environment
        self.sim_env = T1DSimEnv(self.patient, self.sensor, self.pump, self.scenario)
        
    def _calculate_iob(self) -> float:
        """Calculate Insulin On Board using exponential decay model"""
        if not self.insulin_history:
            return 0.0
            
        iob = 0.0
        current_time = self.current_step * 5  # 5-minute intervals
        
        for insulin_time, insulin_amount in self.insulin_history:
            time_diff = current_time - insulin_time  # minutes
            if time_diff > 0:
                # Exponential decay with 4-hour half-life
                decay_factor = np.exp(-time_diff / (4 * 60 / np.log(2)))
                iob += insulin_amount * decay_factor
                
        return iob
    
    def _calculate_glucose_rate_of_change(self) -> float:
        """Calculate glucose rate of change (mg/dL per 5-min)"""
        if len(self.glucose_history) < 2:
            return 0.0
        return list(self.glucose_history)[-1] - list(self.glucose_history)[-2]
    
    def _update_meal_absorption(self):
        """Update meal absorption model"""
        if self.last_meal_time > 0:
            time_since_meal = (self.current_step * 5 - self.last_meal_time) / 60.0  # hours
            # Simple absorption model: peak at 1 hour, complete by 4 hours
            if time_since_meal <= 4:
                absorption_rate = np.exp(-((time_since_meal - 1) ** 2) / 2)
                self.meal_absorbed = min(1.0, self.meal_absorbed + absorption_rate * 0.1)
    
    def _get_observation(self, cgm_reading: float) -> np.array:
        """Construct comprehensive observation vector"""
        obs = []
        
        # Current glucose (normalized)
        obs.append(cgm_reading / 400.0)
        
        # Glucose history (normalized and padded)
        glucose_hist = list(self.glucose_history)
        while len(glucose_hist) < self.glucose_history_size:
            glucose_hist.append(cgm_reading)  # Pad with current reading
        obs.extend([g / 400.0 for g in glucose_hist])
        
        # Insulin on board (normalized)
        iob = self._calculate_iob()
        obs.append(iob / 20.0)  # Normalize by reasonable max IOB
        
        # Time since last insulin (normalized by 6 hours)
        time_since_insulin = min((self.current_step * 5 - self.last_insulin_time) / (6 * 60), 1.0)
        obs.append(time_since_insulin)
        
        # Meal information
        time_since_meal = min((self.current_step * 5 - self.last_meal_time) / (6 * 60), 1.0)
        obs.append(time_since_meal)
        obs.append(self.last_meal_carbs / 100.0)  # Normalize by 100g carbs
        obs.append(self.meal_absorbed)
        
        # Time features
        hour_of_day = (self.current_step * 5 / 60) % 24  # Hour of day
        obs.append(np.sin(2 * np.pi * hour_of_day / 24))  # Circular encoding
        obs.append(np.cos(2 * np.pi * hour_of_day / 24))
        
        # Hypo/hyper history (count in last 2 hours)
        hypo_count = sum(1 for h in self.hypo_history if h > 0) / len(self.hypo_history)
        hyper_count = sum(1 for h in self.hyper_history if h > 0) / len(self.hyper_history)
        obs.append(hypo_count)
        obs.append(hyper_count)
        
        # Glucose rate of change (normalized)
        glucose_roc = self._calculate_glucose_rate_of_change()
        obs.append(glucose_roc / 50.0)  # Normalize by 50 mg/dL per 5-min
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self, glucose: float, action: float) -> float:
        """Enhanced reward function with safety emphasis"""
        
        # Base reward based on glucose zones
        if glucose < 54:  # Severe hypoglycemia
            reward = -100
        elif glucose < 70:  # Mild hypoglycemia  
            reward = -50
        elif 70 <= glucose <= 180:  # Target range
            reward = 10
        elif 180 < glucose <= 250:  # Mild hyperglycemia
            reward = -5
        elif 250 < glucose <= 300:  # Moderate hyperglycemia
            reward = -15
        else:  # Severe hyperglycemia
            reward = -30
            
        # Additional penalties and bonuses
        
        # Glucose stability bonus
        if len(self.glucose_history) >= 2:
            glucose_change = abs(self._calculate_glucose_rate_of_change())
            if glucose_change < 10:  # Stable glucose
                reward += 2
            elif glucose_change > 30:  # Rapid change penalty
                reward -= 5
        
        # IOB safety check
        iob = self._calculate_iob()
        if iob > 15 and glucose < 100:  # High IOB with lowish glucose
            reward -= 10
        
        # Time-in-range bonus
        if 80 <= glucose <= 160:  # Tight control bonus
            reward += 5
            
        # Excessive insulin penalty
        if action > 3.0 and glucose < 150:
            reward -= 5
            
        return reward
    
    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # Convert action to insulin dose
        basal_insulin = float(action[0])
        
        # Take step in simglucose environment - handle the Step structure
        step_result = self.sim_env.step(basal_insulin)
        
        # simglucose returns a Step named tuple with (observation, reward, done, info)
        if hasattr(step_result, 'observation'):
            # Extract components from Step structure
            obs = step_result
            done = step_result.done
            info = step_result.info
        else:
            # Handle other possible return formats
            if len(step_result) == 5:
                # New gymnasium API: obs, reward, terminated, truncated, info
                obs, sim_reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                # Old gym API: obs, reward, done, info
                obs, sim_reward, done, info = step_result
            else:
                raise ValueError(f"Unexpected step result format: {type(step_result)}")
        
        # Extract CGM reading - handle different observation formats
        if hasattr(obs, 'observation') and hasattr(obs.observation, 'CGM'):
            # Handle simglucose Step structure: Step(observation=Observation(CGM=value), ...)
            cgm_reading = float(obs.observation.CGM)
        elif isinstance(obs, np.ndarray):
            cgm_reading = float(obs[0])
        elif hasattr(obs, 'CGM'):
            # Handle named tuple format with CGM attribute
            cgm_reading = float(obs.CGM)
        elif hasattr(obs, '_fields') and 'CGM' in obs._fields:
            # Handle named tuple by field name
            cgm_reading = float(getattr(obs, 'CGM'))
        elif isinstance(obs, (tuple, list)) and len(obs) > 0:
            # Handle tuple/list format
            cgm_reading = float(obs[0])
        else:
            # Try to find glucose value in the observation
            logger.warning(f"Unknown observation format: {type(obs)}, {obs}")
            if hasattr(obs, '__dict__'):
                # Look for common glucose field names
                for attr in ['glucose', 'bg', 'cgm', 'CGM']:
                    if hasattr(obs, attr):
                        cgm_reading = float(getattr(obs, attr))
                        break
                else:
                    raise ValueError(f"Cannot extract glucose reading from observation: {obs}")
            else:
                raise ValueError(f"Cannot extract glucose reading from observation type: {type(obs)}")
        
        # Update tracking variables
        self.current_step += 1
        
        # Update histories
        self.glucose_history.append(cgm_reading)
        self.hypo_history.append(1 if cgm_reading < 70 else 0)
        self.hyper_history.append(1 if cgm_reading > 250 else 0)
        
        # Track insulin administration
        if basal_insulin > 0:
            self.insulin_history.append((self.current_step * 5, basal_insulin / 12))  # Convert hourly to 5-min dose
            self.last_insulin_time = self.current_step * 5
        
        # Update meal absorption
        self._update_meal_absorption()
        
        # Check for meal events (from scenario)
        meal_amount = 0
        if isinstance(info, dict):
            meal_amount = info.get('meal', 0)
        elif hasattr(info, 'meal'):
            meal_amount = info.meal
        
        if meal_amount > 0:
            self.last_meal_time = self.current_step * 5
            self.last_meal_carbs = meal_amount
            self.meal_absorbed = 0
        
        # Calculate reward
        reward = self._calculate_reward(cgm_reading, basal_insulin)
        
        # Check termination conditions
        terminated = done or (cgm_reading < 40) or (cgm_reading > 500)
        truncated = self.current_step >= self.max_episode_steps
        
        # Create enhanced observation
        observation = self._get_observation(cgm_reading)
        
        # Enhanced info dictionary
        enhanced_info = {
            'glucose': cgm_reading,
            'iob': self._calculate_iob(),
            'patient': self.patient_names[self.current_patient_idx],
            'step': self.current_step,
            'meal_carbs': self.last_meal_carbs,
            'glucose_roc': self._calculate_glucose_rate_of_change(),
        }
        
        # Add original info if it exists and is a dict
        if isinstance(info, dict):
            enhanced_info.update(info)
        
        return observation, reward, terminated, truncated, enhanced_info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.array, Dict]:
        """Reset environment, optionally switching to different patient"""
        
        super().reset(seed=seed)
        
        # Randomly select patient from cohort
        self.current_patient_idx = np.random.randint(len(self.patient_names))
        
        # Create new environment for selected patient
        self._create_environment()
        
        # Reset tracking variables
        self.current_step = 0
        self.last_insulin_time = 0
        self.last_meal_time = 0
        self.last_meal_carbs = 0
        self.meal_absorbed = 0
        
        # Reset histories
        self.glucose_history.clear()
        self.hypo_history.clear()
        self.hyper_history.clear()
        self.insulin_history.clear()
        
        # Reset simglucose environment - handle the Step structure
        reset_result = self.sim_env.reset()
        
        # simglucose returns a Step named tuple even on reset
        if hasattr(reset_result, 'observation'):
            # Extract observation from Step structure
            obs = reset_result
            info = reset_result.info if hasattr(reset_result, 'info') else {}
        elif len(reset_result) == 2:
            # Standard gymnasium API
            obs, info = reset_result
        else:
            # Old gym API (returns only obs)
            obs = reset_result
            info = {}
        
        # Debug: Print observation structure on first reset
        if self.current_step == 0:
            logger.info(f"Observation type: {type(obs)}")
            logger.info(f"Observation content: {obs}")
            if hasattr(obs, '_fields'):
                logger.info(f"Named tuple fields: {obs._fields}")
            elif hasattr(obs, '__dict__'):
                logger.info(f"Object attributes: {list(obs.__dict__.keys())}")
        
        # Extract initial CGM reading - handle different observation formats
        if hasattr(obs, 'observation') and hasattr(obs.observation, 'CGM'):
            # Handle simglucose Step structure: Step(observation=Observation(CGM=value), ...)
            cgm_reading = float(obs.observation.CGM)
        elif isinstance(obs, np.ndarray):
            cgm_reading = float(obs[0])
        elif hasattr(obs, 'CGM'):
            # Handle named tuple format with CGM attribute
            cgm_reading = float(obs.CGM)
        elif hasattr(obs, '_fields') and 'CGM' in obs._fields:
            # Handle named tuple by field name
            cgm_reading = float(getattr(obs, 'CGM'))
        elif isinstance(obs, (tuple, list)) and len(obs) > 0:
            # Handle tuple/list format
            cgm_reading = float(obs[0])
        else:
            # Try to find glucose value in the observation
            logger.warning(f"Unknown observation format: {type(obs)}, {obs}")
            if hasattr(obs, '__dict__'):
                # Look for common glucose field names
                for attr in ['glucose', 'bg', 'cgm', 'CGM']:
                    if hasattr(obs, attr):
                        cgm_reading = float(getattr(obs, attr))
                        break
                else:
                    raise ValueError(f"Cannot extract glucose reading from observation: {obs}")
            else:
                raise ValueError(f"Cannot extract glucose reading from observation type: {type(obs)}")
        
        # Initialize histories with current reading
        self.glucose_history.append(cgm_reading)
        self.hypo_history.append(0)
        self.hyper_history.append(0)
        
        # Create initial observation
        observation = self._get_observation(cgm_reading)
        
        enhanced_info = {
            'glucose': cgm_reading,
            'patient': self.patient_names[self.current_patient_idx],
        }
        
        # Add original info if it exists and is a dict
        if isinstance(info, dict):
            enhanced_info.update(info)
        
        return observation, enhanced_info

# PPO Implementation
class PPONetwork(nn.Module):
    """PPO Actor-Critic Network"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Parameter(torch.ones(action_dim) * 0.5)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs):
        features = self.shared(obs)
        
        # Actor outputs
        action_mean = torch.clamp(self.actor_mean(features), 0, 5)  # Clamp to valid insulin range
        action_std = torch.clamp(self.actor_std.expand_as(action_mean), 0.1, 2.0)
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_std, value

class PPOTrainer:
    """PPO Training Algorithm"""
    
    def __init__(self, env, lr: float = 3e-4, gamma: float = 0.99, 
                 clip_eps: float = 0.2, epochs: int = 10):
        
        self.env = env
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.network = PPONetwork(obs_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_glucose_stats = []
        
    def collect_rollout(self, max_steps: int = 2000):
        """Collect rollout data"""
        
        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        obs, _ = self.env.reset()
        
        for step in range(max_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action_mean, action_std, value = self.network(obs_tensor)
                
            # Sample action
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            
            # Clamp action to valid range
            action_clamped = torch.clamp(action, 0, 5)
            
            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(action_clamped.numpy())
            done = terminated or truncated
            
            # Store transition
            observations.append(obs)
            actions.append(action_clamped.squeeze().numpy())
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            dones.append(done)
            
            obs = next_obs
            
            if done:
                # Log episode statistics
                episode_reward = sum(rewards[-step:] if step < len(rewards) else rewards)
                self.episode_rewards.append(episode_reward)
                
                obs, _ = self.env.reset()
        
        return {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'values': np.array(values),
            'log_probs': np.array(log_probs),
            'dones': np.array(dones)
        }
    
    def compute_returns_and_advantages(self, rollout):
        """Compute returns and advantages using GAE"""
        
        rewards = rollout['rewards']
        values = rollout['values']
        dones = rollout['dones']
        
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                gae = 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * 0.95 * gae  # GAE with lambda=0.95
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update_policy(self, rollout, returns, advantages):
        """Update policy using PPO"""
        
        observations = torch.FloatTensor(rollout['observations'])
        actions = torch.FloatTensor(rollout['actions'])
        old_log_probs = torch.FloatTensor(rollout['log_probs'])
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        for _ in range(self.epochs):
            # Forward pass
            action_mean, action_std, values = self.network(observations)
            
            # Calculate new log probabilities
            dist = torch.distributions.Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions.unsqueeze(-1)).sum(-1)
            
            # PPO policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Entropy bonus for exploration
            entropy = dist.entropy().mean()
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
    
    def train(self, total_steps: int = 1000000, log_interval: int = 10000):
        """Main training loop"""
        
        logger.info("Starting PPO training...")
        
        steps_collected = 0
        
        while steps_collected < total_steps:
            # Collect rollout
            rollout = self.collect_rollout(max_steps=2000)
            steps_collected += len(rollout['rewards'])
            
            # Compute returns and advantages
            returns, advantages = self.compute_returns_and_advantages(rollout)
            
            # Update policy
            self.update_policy(rollout, returns, advantages)
            
            # Logging
            if steps_collected % log_interval == 0:
                recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 100 else self.episode_rewards
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                
                logger.info(f"Steps: {steps_collected}/{total_steps}, Avg Reward: {avg_reward:.2f}, "
                          f"Episodes: {len(self.episode_rewards)}")
        
        logger.info("Training completed!")
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        logger.info(f"Model loaded from {path}")

def main():
    """Main training function"""
    
    # Adult patient names (10 adults in simglucose)
    adult_patients = [f'adult#{i:03d}' for i in range(1, 11)]
    
    logger.info(f"Training on adult cohort: {adult_patients}")
    
    # Create enhanced environment
    env = EnhancedT1DEnv(patient_names=adult_patients, max_episode_steps=1000)
    
    # Create trainer
    trainer = PPOTrainer(env, lr=3e-4, gamma=0.99)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    try:
        # Train the model
        trainer.train(total_steps=1000000, log_interval=10000)
        
        # Save the trained model
        model_path = f"results/ppo_insulin_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        trainer.save_model(model_path)
        
        # Save training statistics
        stats = {
            'episode_rewards': trainer.episode_rewards,
            'adult_patients': adult_patients
        }
        
        with open(f"results/training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl", 'wb') as f:
            pickle.dump(stats, f)
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save partial results
        model_path = f"results/ppo_insulin_model_interrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        trainer.save_model(model_path)
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()