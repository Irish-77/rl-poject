import numpy as np
import torch
from .base import Buffer

class PrioritizedReplayBuffer(Buffer):
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, 
                 save_dir='buffer_checkpoints'):
        super().__init__(capacity, save_dir)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.buffer = []
        self.position = 0
        self.eps = 1e-5
        
    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            probs = self.priorities
        else:
            probs = self.priorities[:len(self.buffer)]
            
        probs = probs ** self.alpha
        probs = probs / probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        weights = torch.FloatTensor(weights)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return (torch.FloatTensor(states),
                torch.FloatTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones),
                weights,
                indices)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.eps
    
    def get_state_dict(self):
        return {
            'buffer': self.buffer,
            'position': self.position,
            'priorities': self.priorities,
            'beta': self.beta
        }
    
    def load_state_dict(self, state_dict):
        self.buffer = state_dict['buffer']
        self.position = state_dict['position']
        self.priorities = state_dict['priorities']
        self.beta = state_dict['beta']
    
    def get_training_data(self, batch_size):
        states, actions, rewards, next_states, dones, weights, indices = self.sample(batch_size)
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'weights': weights,
            'indices': indices
        }
    
    def process_training_feedback(self, indices, errors):
        self.update_priorities(indices, errors)
