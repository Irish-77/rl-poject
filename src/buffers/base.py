from abc import ABC, abstractmethod
import torch
import os

class Buffer(ABC):
    def __init__(self, capacity, save_dir='buffer_checkpoints'):
        self.capacity = capacity
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    @abstractmethod
    def add(self, state, action, reward, next_state, done):
        pass
    
    @abstractmethod
    def sample(self, batch_size):
        pass
    
    def save(self, episode):
        path = os.path.join(self.save_dir, f'buffer_{episode}.pt')
        torch.save(self.get_state_dict(), path)
    
    def load(self, episode):
        path = os.path.join(self.save_dir, f'buffer_{episode}.pt')
        self.load_state_dict(torch.load(path))
    
    @abstractmethod
    def get_state_dict(self):
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict):
        pass
    
    @abstractmethod
    def get_training_data(self, batch_size):
        """Returns processed data ready for agent training"""
        pass
    
    def process_training_feedback(self, indices=None, errors=None):
        """Optional method to process training feedback (e.g., for prioritized replay)"""
        pass
