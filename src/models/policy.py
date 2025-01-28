import torch
import torch.nn as nn
import torch.nn.functional as F

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