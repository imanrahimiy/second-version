"""
Multi-agent Reinforcement Learning Framework for Mining Optimization
Implements three specialized agents as described in the revised manuscript:
- Parameter Agent: Adaptive SA/LNS/GA parameter tuning
- Scheduling Agent: Operational decision making
- Resource Agent: Dynamic capacity allocation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

@dataclass
class RLConfig:
    """Configuration for RL agents"""
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    epsilon: float = 0.1  # Exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    buffer_size: int = 10000
    batch_size: int = 64
    update_frequency: int = 4
    hidden_size: int = 256

class ReplayBuffer:
    """Experience replay buffer for RL agents"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)


class PolicyNetwork(nn.Module):
    """Neural network for policy approximation"""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ParameterAgent:
    """
    Agent for adaptive parameter tuning of SA/LNS/GA algorithms
    Learns optimal parameter settings based on convergence patterns
    """
    
    def __init__(self, config: RLConfig):
        self.config = config
        
        # State: [convergence_rate, diversity, iteration_progress, current_temp, destruction_size]
        self.state_size = 5
        
        # Action: parameter adjustments
        # [temp_adjustment, cooling_adjustment, destruction_adjustment, mutation_adjustment]
        self.action_size = 4
        
        self.policy_net = PolicyNetwork(self.state_size, self.action_size, config.hidden_size)
        self.target_net = PolicyNetwork(self.state_size, self.action_size, config.hidden_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.epsilon = config.epsilon
        self.steps = 0
        
        # Current parameters
        self.current_params = {
            'temperature': 300.0,
            'cooling_rate': 0.95,
            'destruction_size': 0.2,
            'mutation_rate': 0.1
        }
    
    def get_state(self, metrics: Dict) -> np.ndarray:
        """Convert optimization metrics to state vector"""
        state = np.array([
            metrics.get('convergence_rate', 0.0),
            metrics.get('diversity', 0.5),
            metrics.get('iteration_progress', 0.0),
            self.current_params['temperature'] / 500,  # Normalize
            self.current_params['destruction_size']
        ], dtype=np.float32)
        return state
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select parameter adjustment action using epsilon-greedy"""
        if random.random() < self.epsilon:
            # Exploration: random adjustments
            return np.random.uniform(-0.1, 0.1, self.action_size)
        
        # Exploitation: use policy network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            action = torch.tanh(q_values).numpy()[0] * 0.1  # Scale to [-0.1, 0.1]
        
        return action
    
    def adjust_parameters(self, iteration: int) -> Dict:
        """Adjust algorithm parameters based on current state"""
        metrics = {
            'iteration_progress': iteration / 100,  # Assume 100 max iterations
            'convergence_rate': 0.01,  # Placeholder
            'diversity': 0.7  # Placeholder
        }
        
        state = self.get_state(metrics)
        action = self.select_action(state)
        
        # Apply adjustments
        self.current_params['temperature'] *= (1 + action[0])
        self.current_params['cooling_rate'] = np.clip(
            self.current_params['cooling_rate'] + action[1] * 0.05, 0.85, 0.99
        )
        self.current_params['destruction_size'] = np.clip(
            self.current_params['destruction_size'] + action[2], 0.1, 0.4
        )
        self.current_params['mutation_rate'] = np.clip(
            self.current_params['mutation_rate'] + action[3] * 0.05, 0.05, 0.3
        )
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_min, 
                          self.epsilon * self.config.epsilon_decay)
        
        return self.current_params
    
    def update_policy(self, reward: float):
        """Update policy based on reward signal"""
        # Store experience and train if buffer is sufficient
        if len(self.replay_buffer) >= self.config.batch_size:
            self._train_step()
        
        # Update target network periodically
        if self.steps % self.config.update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.steps += 1
    
    def _train_step(self):
        """Single training step using experience replay"""
        batch = self.replay_buffer.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Compute current Q values
        current_q = self.policy_net(states)
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states)
            target_q = rewards + self.config.gamma * torch.max(next_q, dim=1, keepdim=True)[0] * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class SchedulingAgent:
    """
    Agent for real-time operational decisions and strategy selection
    Learns optimal scheduling strategies based on geological scenarios
    """
    
    def __init__(self, config: RLConfig):
        self.config = config
        
        # State: dual values, geological features, capacity utilization
        self.state_size = 20  # Simplified state representation
        
        # Action: strategy selection (highgrade, balanced, spatial)
        self.action_size = 3
        
        self.policy_net = PolicyNetwork(self.state_size, self.action_size, config.hidden_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        
        self.current_state = None
        self.epsilon = config.epsilon
        
    def update_state(self, dual_values: Dict, geological_features: Optional[Dict] = None):
        """Update agent state based on current optimization status"""
        state_vector = []
        
        # Add dual values (normalized)
        for key in sorted(dual_values.keys())[:10]:
            state_vector.append(dual_values[key] / 1000)  # Normalize
        
        # Add geological features if available
        if geological_features:
            state_vector.extend([
                geological_features.get('mean_grade', 0),
                geological_features.get('grade_variance', 0),
                geological_features.get('spatial_correlation', 0)
            ])
        
        # Pad to state size
        while len(state_vector) < self.state_size:
            state_vector.append(0.0)
        
        self.current_state = np.array(state_vector[:self.state_size], dtype=np.float32)
    
    def select_strategy(self, dual_values: Dict, equipment_type: str) -> str:
        """Select scheduling strategy based on current state"""
        self.update_state(dual_values)
        
        if random.random() < self.epsilon:
            # Exploration
            strategies = ['highgrade', 'risk_balanced', 'spatial']
            return random.choice(strategies)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            action_idx = torch.argmax(q_values).item()
        
        strategies = ['highgrade', 'risk_balanced', 'spatial']
        return strategies[action_idx]
    
    def update_policy(self, reward: float):
        """Update policy based on reward"""
        # Simple policy gradient update
        if self.current_state is not None:
            state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            
            # Compute loss (negative reward for gradient ascent)
            loss = -reward * torch.max(q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_min, 
                          self.epsilon * self.config.epsilon_decay)


class ResourceAgent:
    """
    Agent for dynamic capacity allocation and resource management
    Learns optimal resource distribution across periods and equipment
    """
    
    def __init__(self, config: RLConfig):
        self.config = config
        
        # State: capacity utilization per period, equipment efficiency
        self.state_size = 12  # 6 periods + 6 efficiency metrics
        
        # Action: capacity allocation adjustments
        self.action_size = 6  # One per period
        
        self.policy_net = PolicyNetwork(self.state_size, self.action_size, config.hidden_size)
        self.value_net = PolicyNetwork(self.state_size, 1, config.hidden_size)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=config.learning_rate)
        
        # Feature weights for geological integration
        self.feature_weights = np.array([0.4, 0.3, 0.3])  # alteration, structure, distance
        
    def get_feature_weights(self) -> np.ndarray:
        """Return current feature weights for geological integration"""
        return self.feature_weights
    
    def allocate_capacity(self, current_utilization: np.ndarray, 
                         efficiency_metrics: np.ndarray) -> np.ndarray:
        """
        Allocate capacity across periods based on current state
        Returns capacity adjustment factors
        """
        state = np.concatenate([current_utilization, efficiency_metrics])
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            # Get capacity adjustments from policy network
            adjustments = torch.tanh(self.policy_net(state_tensor))
            adjustments = adjustments.numpy()[0] * 0.2  # Scale to [-0.2, 0.2]
        
        # Apply adjustments to base capacity
        base_capacity = 6.5e6
        adjusted_capacity = base_capacity * (1 + adjustments)
        
        # Ensure total capacity remains constant
        total_adjustment = np.sum(adjusted_capacity) - (base_capacity * 6)
        adjusted_capacity -= total_adjustment / 6
        
        return adjusted_capacity
    
    def update_feature_weights(self, geological_performance: Dict):
        """
        Update feature weights based on geological performance metrics
        """
        # Simple adaptive weighting based on feature importance
        if 'feature_importance' in geological_performance:
            importance = geological_performance['feature_importance']
            # Normalize to sum to 1
            self.feature_weights = importance / np.sum(importance)
    
    def update_policy(self, reward: float):
        """
        Update policy using actor-critic method
        """
        # Placeholder for actor-critic update
        # In practice, would need state, action, next_state for proper update
        pass


class MultiAgentCoordinator:
    """
    Coordinates multiple RL agents for integrated optimization
    Implements reward function from manuscript:
    R(t) = α·NPV_improvement + β·Constraint_satisfaction + γ·Efficiency - δ·Risk
    """
    
    def __init__(self, config: RLConfig):
        self.config = config
        
        # Initialize agents
        self.parameter_agent = ParameterAgent(config)
        self.scheduling_agent = SchedulingAgent(config)
        self.resource_agent = ResourceAgent(config)
        
        # Reward weights
        self.alpha = 0.4  # NPV improvement weight
        self.beta = 0.3   # Constraint satisfaction weight
        self.gamma = 0.2  # Efficiency weight
        self.delta = 0.1  # Risk penalty weight
        
        # Performance tracking
        self.previous_npv = 0
        self.performance_history = []
    
    def calculate_reward(self, current_npv: float, 
                        constraint_violations: float,
                        computation_time: float,
                        scenario_variance: float) -> float:
        """
        Calculate unified reward signal for all agents
        R(t) = α·NPV_improvement + β·Constraint_satisfaction + γ·Efficiency - δ·Risk
        """
        # NPV improvement (normalized to millions)
        npv_improvement = (current_npv - self.previous_npv) / 1e6
        self.previous_npv = current_npv
        
        # Constraint satisfaction (1 - normalized violations)
        constraint_satisfaction = 1.0 / (1.0 + constraint_violations)
        
        # Computational efficiency (inverse of time)
        efficiency = 1.0 / (1.0 + computation_time / 100)
        
        # Risk penalty (coefficient of variation)
        risk_penalty = scenario_variance / (current_npv + 1e-6)
        
        # Combined reward
        reward = (self.alpha * npv_improvement +
                 self.beta * constraint_satisfaction +
                 self.gamma * efficiency -
                 self.delta * risk_penalty)
        
        return reward
    
    def update_all_agents(self, performance_metrics: Dict):
        """Update all agents based on performance metrics"""
        reward = self.calculate_reward(
            performance_metrics.get('npv', 0),
            performance_metrics.get('violations', 0),
            performance_metrics.get('time', 100),
            performance_metrics.get('variance', 0)
        )
        
        # Update each agent
        self.parameter_agent.update_policy(reward)
        self.scheduling_agent.update_policy(reward)
        self.resource_agent.update_policy(reward)
        
        # Track performance
        self.performance_history.append({
            'reward': reward,
            'npv': performance_metrics.get('npv', 0),
            'violations': performance_metrics.get('violations', 0)
        })
    
    def get_agents(self) -> Dict:
        """Return dictionary of all agents"""
        return {
            'parameter': self.parameter_agent,
            'schedule': self.scheduling_agent,
            'resource': self.resource_agent
        }
    
    def save_models(self, path: str):
        """Save all agent models"""
        torch.save({
            'parameter_policy': self.parameter_agent.policy_net.state_dict(),
            'schedule_policy': self.scheduling_agent.policy_net.state_dict(),
            'resource_policy': self.resource_agent.policy_net.state_dict(),
            'resource_value': self.resource_agent.value_net.state_dict()
        }, path)
    
    def load_models(self, path: str):
        """Load all agent models"""
        checkpoint = torch.load(path)
        self.parameter_agent.policy_net.load_state_dict(checkpoint['parameter_policy'])
        self.scheduling_agent.policy_net.load_state_dict(checkpoint['schedule_policy'])
        self.resource_agent.policy_net.load_state_dict(checkpoint['resource_policy'])
        self.resource_agent.value_net.load_state_dict(checkpoint['resource_value'])