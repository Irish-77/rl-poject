import torch
import torch.nn as nn
from torch.distributions import Normal

ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'leakyrelu': nn.LeakyReLU,
    'elu': nn.ELU,
    'gelu': nn.GELU
}

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[64, 64], activation="relu"):
        super().__init__()
        
        # Get activation function from map (case-insensitive)
        activation = activation.lower()
        if activation not in ACTIVATION_MAP:
            print(f"Warning: Activation {activation} not found, using relu")
            activation = 'relu'
        self.activation = ACTIVATION_MAP[activation]()
        
        # Build layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
            
        # Mean network
        self.mean_net = nn.Sequential(
            *layers,
            nn.Linear(prev_dim, action_dim)
        )
        
        # Log std parameter (learnable)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        mean = self.mean_net(state)
        std = torch.exp(self.log_std)
        
        # Ensure proper broadcasting for batched inputs
        if mean.dim() > std.dim():
            std = std.expand_as(mean)
            
        return Normal(mean, std)