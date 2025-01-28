import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import csv
import pandas as pd
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch.nn.functional as F 

##############################################
# 1. Base Agent Class
##############################################

class Agent:
    """
    Base Agent class. It should be inherited by specific algorithm
    implementations.
    """
    def select_action(self, state):
        raise NotImplementedError

    def update(self, transitions):
        """
        transitions can be a list of (state, action, reward, next_state, done,
        log_prob, etc.)
        """
        raise NotImplementedError

    def save_checkpoint(self, filepath):
        raise NotImplementedError
        
    def load_checkpoint(self, filepath):
        raise NotImplementedError

###############################################
# 2. ActorCriticAgent - Baseline Implementation
###############################################

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, activation="relu"):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Map activation string to actual function - optimize later
        activation_functions = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid
        }
        self.activation = activation_functions.get(activation.lower(), F.relu)
        
        in_dim = state_dim
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
            
        self.mean = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = self.activation(layer(x))
        mean = self.mean(x)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_layers, activation="relu"):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Map activation string to actual function
        activation_functions = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid
        }
        self.activation = activation_functions.get(activation.lower(), F.relu)
        
        in_dim = state_dim
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
            
        self.value = nn.Linear(in_dim, 1)

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.value(x)

class ActorCriticAgent(Agent):
    def __init__(
        self,
        state_dim,
        action_dim,
        policy_lr=1e-3,
        value_lr=1e-3,
        gamma=0.99,
        policy_network=None,
        value_network=None
    ):
        super().__init__()
        self.gamma = gamma
        
        # Use configuration or defaults for policy network
        policy_config = policy_network or {"hidden_layers": [64, 64], "activation": "relu"}
        self.policy_net = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=policy_config["hidden_layers"],
            activation=policy_config["activation"]
        )
        
        # Use configuration or defaults for value network
        value_config = value_network or {"hidden_layers": [64, 64], "activation": "relu"}
        self.value_net = ValueNetwork(
            state_dim=state_dim,
            hidden_layers=value_config["hidden_layers"],
            activation=value_config["activation"]
        )
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.episode_transitions = []

    def select_action(self, state):
        # Convert state to torch if needed
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        dist = self.policy_net(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Detach for numpy conversion, but keep original for training
        action_numpy = torch.clip(action.detach(), -2, 2).numpy()
        return action_numpy, action, log_prob

    def store_transition(self, transition):
        # transition: (state, action_tensor, reward, next_state, done, log_prob)
        self.episode_transitions.append(transition)

    def _compute_returns_and_advantages(self, transitions):
        rewards = [tr[2] for tr in transitions]
        states = [tr[0] for tr in transitions]
        values = []

        with torch.no_grad():
            for s in states:
                s_tensor = torch.tensor(s, dtype=torch.float32)
                values.append(self.value_net(s_tensor).item())

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        advantages = returns - values
        return returns, advantages

    def update(self, transitions):
        """
        Using the transitions from a full episode to update policy and value.
        transitions: list of (state, action, reward, next_state, done, log_prob)
        """
        returns, advantages = self._compute_returns_and_advantages(transitions)
        returns = returns.unsqueeze(1)
        advantages = advantages.unsqueeze(1)

        # Update Value Network
        states = torch.tensor([t[0] for t in transitions], dtype=torch.float32)
        value_preds = self.value_net(states)
        value_loss = nn.MSELoss()(value_preds, returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update Policy Network
        total_policy_loss = 0
        for i, (state, action, reward, next_state, done, _) in enumerate(transitions):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            dist = self.policy_net(state_tensor)
            log_prob = dist.log_prob(action).sum(dim=-1)
            total_policy_loss = total_policy_loss - log_prob * advantages[i].item()

        # Average the policy loss
        policy_loss = total_policy_loss / len(transitions)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Clear stored transitions
        self.episode_transitions = []

        return value_loss.item(), policy_loss.item()

    def save_checkpoint(self, filepath):
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])

##############################################
# 3. Environment Wrapper Base
##############################################

class EnvironmentWrapper:
    def reset(self):
        """Returns the initial observation"""
        raise NotImplementedError

    def step(self, action):
        """
        Performs action, returns next_state, reward, done, info
        """
        raise NotImplementedError

##############################################
# 4. PendulumEnvWrapper
##############################################

class PendulumEnvWrapper(EnvironmentWrapper):
    def __init__(self, env_name="Pendulum-v1", max_episode_steps=200, reward_threshold=-250):
        # Create environment with max_episode_steps
        self.env = gym.make(
            env_name,
            render_mode="rgb_array",
            max_episode_steps=max_episode_steps
        )
        self.env_name = env_name
        self.reward_threshold = reward_threshold
        self.reset()

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

##############################################
# 5. Training Loop (Environment & Agent independent)
##############################################

class Logger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_file = os.path.join(log_dir, f'run_{self.current_time}.csv')
        self.data = []
        
    def log(self, episode, metrics):
        """
        Log metrics for an episode
        metrics: dictionary of metric_name: value pairs
        """
        metrics['episode'] = episode
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.data.append(metrics)
        
        # Write to CSV after each episode
        if len(self.data) == 1:  # First entry, create CSV with headers
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
                writer.writeheader()
                writer.writerow(self.data[0])
        else:  # Append to existing CSV
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
                writer.writerow(self.data[-1])
    
    def get_data(self):
        """Return all logged data as a pandas DataFrame"""
        return pd.DataFrame(self.data)

def train(agent, env, num_episodes=10, 
          checkpoint_freq=100,
          video_freq=500,
          video_dir='videos',
          log_dir='logs',
          checkpoint_dir='checkpoints'
          ):
    
    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize logger
    logger = Logger(log_dir)
    
    # Wrap environment with video recorder
    video_env = RecordVideo(
        env.env,
        video_folder=f"{video_dir}/{logger.current_time}",
        episode_trigger=lambda ep_id: ep_id % video_freq == 0,
        name_prefix="rl-video"
    )
    
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        
        state, _ = video_env.reset()
        done = False
        episode_reward = 0.0
        transitions = []
        value_losses = []
        policy_losses = []
        
        while not done:
            action_numpy, action_tensor, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, info = video_env.step(action_numpy)
            done = terminated or truncated
            transitions.append((state, action_tensor.detach(), reward, next_state, done, log_prob))
            state = next_state
            episode_reward += reward
        
        value_loss, policy_loss = agent.update(transitions)
        value_losses.append(value_loss)
        policy_losses.append(policy_loss)
        
        logger.log(episode, {
            'reward': episode_reward,
            'value_loss': np.mean(value_losses),
            'policy_loss': np.mean(policy_losses),
        })
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_checkpoint(f'{checkpoint_dir}/best_model.pt')
        
        if episode % checkpoint_freq == 0:
            agent.save_checkpoint(f'{checkpoint_dir}/checkpoint_{episode}.pt')
        
        print(f"Episode {episode + 1} - Reward: {episode_reward:.2f}")
    
    video_env.close()
    return logger.get_data()

##############################################
# Main
##############################################

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Set random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Print resolved config
    print(OmegaConf.to_yaml(cfg))
    
    # Create environment wrapper
    env = instantiate(cfg.environment)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Create agent
    agent = instantiate(cfg.model, 
                       state_dim=obs_dim,
                       action_dim=act_dim)

    # Train
    train(
        agent=agent,
        env=env,
        num_episodes=cfg.training.num_episodes,
        checkpoint_freq=cfg.training.checkpoint_freq,
        video_freq=cfg.training.video_freq,
        video_dir=cfg.training.video_dir,
        log_dir=cfg.training.log_dir,
        checkpoint_dir=cfg.training.checkpoint_dir
    )

if __name__ == "__main__":
    main()