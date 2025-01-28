import torch
import torch.nn as nn
import torch.nn.functional as F

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