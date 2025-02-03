#main.py model=tdmpc training.use_buffer=true
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.utils.helper import get_auto_device

class TDMPC2Agent:
    def __init__(self, state_dim, action_dim, planning_horizon=10, num_candidates=100, num_iterations=5,
                 hidden_size=256, lr=3e-4, gamma=0.99, device='auto'):
        """
        TDMPC2 Agent using learned dynamics, value, and policy networks.
        
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            planning_horizon (int): MPC planning horizon.
            num_candidates (int): Number of candidate action sequences to sample.
            num_iterations (int): Number of CEM iterations.
            hidden_size (int): Size of hidden layers.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            device (str): Device to run the agent on ('auto', 'cuda', 'cpu').
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon
        self.num_candidates = num_candidates
        self.num_iterations = num_iterations
        self.gamma = gamma
        
        # Set device
        if device == 'auto':
            self.device = get_auto_device()
        else:
            self.device = torch.device(device)
        
        # Dynamics model: predicts (delta_state, reward) given (state, action)
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim + 1)  # first state_dim outputs: state delta; last output: reward
        ).to(self.device)
        
        # Value network: estimates the value of a state
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self.device)
        
        # Policy network: outputs an action given a state (used as a prior in planning)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  # assuming actions are scaled between -1 and 1
        ).to(self.device)
        
        # Joint optimizer for all networks
        self.optimizer = optim.Adam(
            list(self.dynamics.parameters()) + 
            list(self.value_net.parameters()) + 
            list(self.policy_net.parameters()),
            lr=lr
        )
    
    def select_action(self, state):
        """
        Select an action by planning with the learned dynamics using CEM.
        
        Args:
            state (np.ndarray): Current state (1D numpy array).
            
        Returns:
            action_numpy (np.ndarray): Selected action as a NumPy array.
            action_tensor (torch.Tensor): Selected action as a torch tensor.
            log_prob (torch.Tensor): Dummy log probability (zero), for compatibility.
        """
        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # shape: (1, state_dim)
        # Use CEM-based planning to obtain a sequence of actions
        action_seq = self.cem_planning(state_tensor)
        # Select the first action in the planned sequence
        selected_action = action_seq[0]  # shape: (action_dim,)
        # For compatibility with the training loop, we return a dummy log probability.
        log_prob = torch.tensor(0.0)
        action_tensor = torch.FloatTensor(selected_action).to(self.device)
        action_numpy = action_tensor.cpu().detach().numpy()
        return action_numpy, action_tensor, log_prob

    def cem_planning(self, state):
        """
        Use the Cross-Entropy Method (CEM) to plan an action sequence that maximizes predicted return.
        
        Args:
            state (torch.Tensor): Current state tensor of shape (1, state_dim).
        
        Returns:
            best_action_seq (np.ndarray): Best action sequence (planning_horizon x action_dim).
        """
        H = self.planning_horizon
        N = self.num_candidates
        action_dim = self.action_dim
        
        # Initialize distribution parameters for the action sequence (each action in the horizon)
        mu = torch.zeros(H, action_dim, device=self.device)
        std = torch.ones(H, action_dim, device=self.device)
        
        for _ in range(self.num_iterations):
            # Sample candidate action sequences: shape (N, H, action_dim)
            noise = torch.randn(N, H, action_dim, device=self.device)
            actions = mu.unsqueeze(0) + std.unsqueeze(0) * noise
            # Ensure actions remain in [-1, 1]
            actions = torch.tanh(actions)
            
            # Evaluate the return of each candidate sequence
            returns = self.evaluate_action_sequences(state, actions)
            # Select the top 10% sequences as elites
            K = max(1, int(0.1 * N))
            elite_indices = torch.topk(returns, K, largest=True).indices
            elite_actions = actions[elite_indices]  # shape: (K, H, action_dim)
            # Update distribution parameters
            mu = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-6  # add small constant for numerical stability
        
        best_action_seq = mu.detach().cpu().numpy()
        return best_action_seq

    def evaluate_action_sequences(self, state, action_seqs):
        """
        Evaluate a batch of action sequences by simulating the dynamics model.
        
        Args:
            state (torch.Tensor): Starting state of shape (1, state_dim).
            action_seqs (torch.Tensor): Candidate action sequences of shape (N, H, action_dim).
        
        Returns:
            cumulative_rewards (torch.Tensor): Estimated cumulative rewards for each sequence (shape: (N,)).
        """
        N, H, _ = action_seqs.shape
        # Repeat the state for each candidate sequence
        state_rep = state.repeat(N, 1)  # shape: (N, state_dim)
        cumulative_reward = torch.zeros(N, device=self.device)
        discount = 1.0
        
        for t in range(H):
            actions_t = action_seqs[:, t, :]  # shape: (N, action_dim)
            # Concatenate current state and action
            sa = torch.cat([state_rep, actions_t], dim=-1)  # shape: (N, state_dim + action_dim)
            # Predict state change and reward
            pred = self.dynamics(sa)  # shape: (N, state_dim+1)
            delta_state = pred[:, :self.state_dim]
            reward = pred[:, -1]
            # Update state: assume the dynamics model predicts a state delta
            state_rep = state_rep + delta_state
            cumulative_reward += discount * reward
            discount *= self.gamma
        
        # Add terminal value estimate from the value network
        terminal_value = self.value_net(state_rep).squeeze(-1)
        cumulative_reward += discount * terminal_value
        return cumulative_reward

    def update(self, buffer, batch_size=64):
        """
        Update the dynamics, value, and policy networks using a batch sampled from the replay buffer.
        
        Args:
            buffer: Replay buffer that returns (states, actions, rewards, next_states, dones).
            batch_size (int): Batch size.
        
        Returns:
            value_loss (float): Loss from the value network update.
            policy_loss (float): Loss from the policy network update.
        """
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        # Move to device and add dimensions where needed
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)
        
        # --- Dynamics loss ---
        # Predict state delta and reward from (state, action)
        sa = torch.cat([states, actions], dim=-1)
        pred = self.dynamics(sa)
        pred_delta = pred[:, :self.state_dim]
        pred_reward = pred[:, -1].unsqueeze(1)
        true_delta = next_states - states
        dynamics_loss = nn.MSELoss()(pred_delta, true_delta) + nn.MSELoss()(pred_reward, rewards)
        
        # --- Value loss ---
        # Compute TD target: one-step prediction using next_states
        with torch.no_grad():
            target_value = rewards + self.gamma * self.value_net(next_states) * (1 - dones)
        current_value = self.value_net(states)
        value_loss = nn.MSELoss()(current_value, target_value)
        
        # --- Policy loss ---
        # For simplicity, we use the policy network’s suggested actions and
        # penalize them based on the estimated value (i.e. we try to choose actions
        # that lead to higher state values after one predicted step)
        policy_actions = self.policy_net(states)
        sa_policy = torch.cat([states, policy_actions], dim=-1)
        pred_policy = self.dynamics(sa_policy)
        next_state_est = states + pred_policy[:, :self.state_dim]
        policy_value = self.value_net(next_state_est)
        # We minimize the negative value to encourage better actions
        policy_loss = -policy_value.mean()
        
        total_loss = dynamics_loss + value_loss + policy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return value_loss.item(), policy_loss.item()

    def save_checkpoint(self, file_path):
        """
        Save the agent’s model parameters and optimizer state.
        """
        checkpoint = {
            'dynamics_state_dict': self.dynamics.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, file_path)
