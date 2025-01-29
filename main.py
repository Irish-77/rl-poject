import os
import torch
import numpy as np
from gymnasium.wrappers import RecordVideo

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.utils.logger import Logger
from src.agents import ActorCriticAgent
from src.environments import PendulumEnvWrapper

def train(agent, env, exploration_strategy=None, num_episodes=10, 
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
            # Add exploration logic
            if exploration_strategy and exploration_strategy.should_explore():
                action_numpy = env.action_space.sample()
                action_tensor = torch.FloatTensor(action_numpy)
                log_prob = torch.tensor(0.0)  # dummy log prob for random actions
            else:
                action_numpy, action_tensor, log_prob = agent.select_action(state)
                
            next_state, reward, terminated, truncated, info = video_env.step(action_numpy)
            done = terminated or truncated
            transitions.append((state, action_tensor.detach(), reward, next_state, done, log_prob))
            state = next_state
            episode_reward += reward
            
        # Update exploration strategy and get logging info
        exploration_info = {}
        if exploration_strategy:
            exploration_strategy.update()
            exploration_info = exploration_strategy.get_logging_info()
        
        value_loss, policy_loss = agent.update(transitions)
        value_losses.append(value_loss)
        policy_losses.append(policy_loss)
        
        logger.log(episode, {
            'reward': episode_reward,
            'value_loss': np.mean(value_losses),
            'policy_loss': np.mean(policy_losses),
            **exploration_info  # Add exploration metrics to logging
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

    # Create exploration strategy
    exploration_strategy = instantiate(cfg.exploration) if "exploration" in cfg else None
    
    # Train with exploration
    train(
        agent=agent,
        env=env,
        exploration_strategy=exploration_strategy,
        num_episodes=cfg.training.num_episodes,
        checkpoint_freq=cfg.training.checkpoint_freq,
        video_freq=cfg.training.video_freq,
        video_dir=cfg.training.video_dir,
        log_dir=cfg.training.log_dir,
        checkpoint_dir=cfg.training.checkpoint_dir
    )

if __name__ == "__main__":
    main()