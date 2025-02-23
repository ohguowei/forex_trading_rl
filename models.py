# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_size=12, decision_dim=16, hidden_size=128, num_actions=3):
        super(ActorCritic, self).__init__()
        # Actor branch
        self.actor_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.actor_fc1 = nn.Linear(hidden_size, 32)
        self.actor_fc2 = nn.Linear(32 + decision_dim, 64)
        self.actor_fc3 = nn.Linear(64, 64)
        self.actor_output = nn.Linear(64, num_actions)
        
        # Critic branch
        self.critic_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.critic_fc1 = nn.Linear(hidden_size, 32)
        self.critic_fc2 = nn.Linear(32 + decision_dim, 64)
        self.critic_fc3 = nn.Linear(64, 64)
        self.critic_output = nn.Linear(64, 1)

    def forward(self, state, decisions):
        """
        Args:
            state: Shape (batch_size, sequence_length, input_size).
            decisions: Shape (batch_size, decision_dim).
        Returns:
            policy_logits (batch_size, num_actions),
            value (batch_size, 1).
        """
        # Actor forward pass
        actor_out, _ = self.actor_lstm(state)
        actor_last = actor_out[:, -1, :]  # LSTM output of the last timestep
        x = F.relu(self.actor_fc1(actor_last))
        x = torch.cat([x, decisions], dim=1)  # Add decision history
        x = F.relu(self.actor_fc2(x))
        x = F.relu(self.actor_fc3(x))
        policy_logits = self.actor_output(x)
        
        # Critic forward pass
        critic_out, _ = self.critic_lstm(state)
        critic_last = critic_out[:, -1, :]
        y = F.relu(self.critic_fc1(critic_last))
        y = torch.cat([y, decisions], dim=1)
        y = F.relu(self.critic_fc2(y))
        y = F.relu(self.critic_fc3(y))
        value = self.critic_output(y)
        
        return policy_logits, value
