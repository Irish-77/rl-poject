import torch
import torch.nn as nn
import torch.optim as optim

from src.agents import Agent
from src.models import PolicyNetwork, ValueNetwork

class ActorCriticAgent(Agent):
    def __init__(
        self,
        state_dim,
        action_dim,
        policy_lr=1e-3,
        value_lr=1e-3,
        gamma=0.99,
        policy_network=None,
        value_network=None,
        device='auto'
    ):
        super().__init__()
        self.gamma = gamma
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                     'mps' if torch.backends.mps.is_available() else 
                                     'cpu')
        else:
            self.device = torch.device(device)
        
        # Use configuration or defaults for policy network
        policy_config = policy_network or {"hidden_layers": [64, 64], "activation": "relu"}
        self.policy_net = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=policy_config["hidden_layers"],
            activation=policy_config["activation"]
        ).to(self.device)
        
        # Use configuration or defaults for value network
        value_config = value_network or {"hidden_layers": [64, 64], "activation": "relu"}
        self.value_net = ValueNetwork(
            state_dim=state_dim,
            hidden_layers=value_config["hidden_layers"],
            activation=value_config["activation"]
        ).to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.episode_transitions = []

    def select_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        else:
            state = state.to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        dist = self.policy_net(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        action_tensor = action.squeeze(0)  # Remove batch dimension for storage
        action_numpy = torch.clip(action_tensor.cpu().detach(), -2, 2).numpy()
        return action_numpy, action_tensor, log_prob

    def store_transition(self, transition):
        # transition: (state, action_tensor, reward, next_state, done, log_prob)
        self.episode_transitions.append(transition)

    def _compute_returns_and_advantages(self, transitions):
        rewards = [tr[2] for tr in transitions]
        states = [tr[0] for tr in transitions]
        values = []

        with torch.no_grad():
            for s in states:
                s_tensor = torch.tensor(s, dtype=torch.float32).to(self.device)
                values.append(self.value_net(s_tensor).cpu().item())

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
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
        states = torch.tensor([t[0] for t in transitions], dtype=torch.float32).to(self.device)
        value_preds = self.value_net(states)
        value_loss = nn.MSELoss()(value_preds, returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update Policy Network
        total_policy_loss = 0
        for i, (state, action, reward, next_state, done, _) in enumerate(transitions):
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            action_tensor = action.to(self.device)  # Move action tensor to correct device
            dist = self.policy_net(state_tensor)
            log_prob = dist.log_prob(action_tensor).sum(dim=-1)
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
            'device': self.device,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])