import gymnasium as gym

from src.environments import EnvironmentWrapper

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