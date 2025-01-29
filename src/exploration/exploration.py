from collections import defaultdict
from dataclasses import dataclass
import numpy as np

@dataclass
class ExplorationConfig:
    initial_epsilon: float = 1.0
    min_epsilon: float = 0.01
    decay_rate: float = 0.995

class ExplorationStrategy:
    def should_explore(self) -> bool:
        raise NotImplementedError
    
    def update(self):
        pass
    
    def get_logging_info(self) -> dict:
        return {}

class EpsilonGreedy(ExplorationStrategy):
    def __init__(self, config: ExplorationConfig):
        self.epsilon = config.initial_epsilon
        self.min_epsilon = config.min_epsilon
        self.decay_rate = config.decay_rate
    
    def should_explore(self) -> bool:
        return np.random.random() < self.epsilon
    
    def update(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
    
    def get_logging_info(self) -> dict:
        return {
            'epsilon': self.epsilon,
            'exploration_rate': self.epsilon * 100  # as percentage
        }