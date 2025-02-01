import numpy as np
import torch
from .base import Buffer

class ReplayBuffer(Buffer):
    def __init__(self, capacity, save_dir='buffer_checkpoints'):
        super().__init__(capacity, save_dir)
        self.buffer = []
        self.position = 0
        
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return (torch.FloatTensor(states), 
                torch.FloatTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones))
    
    def get_state_dict(self):
        return {
            'buffer': self.buffer,
            'position': self.position
        }
    
    def load_state_dict(self, state_dict):
        self.buffer = state_dict['buffer']
        self.position = state_dict['position']
    
    def get_training_data(self, batch_size):
        batch = self.sample(batch_size)
        return {
            'states': batch[0],
            'actions': batch[1],
            'rewards': batch[2],
            'next_states': batch[3],
            'dones': batch[4],
            'weights': torch.ones_like(batch[2]),
            'indices': None
        }
