import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import threading

# Import our OANDA API functions and constants.
from oanda_api import fetch_candle_data, open_position, close_position, ACCOUNT_ID
from feature_extractor import compute_features

##############################################
# 1. Actor-Critic Network Architecture
##############################################
class ActorCritic(nn.Module):
    def __init__(self, input_size=5, decision_dim=16, hidden_size=128, num_actions=3):
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
        # Actor forward pass
        actor_out, _ = self.actor_lstm(state)
        actor_last = actor_out[:, -1, :]  # use last time step
        x = F.relu(self.actor_fc1(actor_last))
        x = torch.cat([x, decisions], dim=1)
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

##############################################
# 2. OANDA Forex Environment Using Real Data
##############################################
class OandaForexEnv:
    def __init__(self, instrument="EUR_USD", units=100, granularity="H1", candle_count=500):
        self.instrument = instrument
        self.units = units
        self.granularity = granularity
        self.candle_count = candle_count
        self.position_open = False
        self.position_side = None
        # Fetch historical data using the oanda_api module.
        self.data = np.array(fetch_candle_data(self.instrument, self.granularity, self.candle_count))
        # Use the imported compute_features function to get features.
        self.features = compute_features(self.data)
        # Set the sliding window index.
        self.current_index = 16


    def compute_features(self, data):
        """
        Compute 5 features for each time step:
         x1 = (c_t - c_{t-1}) / c_{t-1}
         x2 = (h_t - h_{t-1}) / h_{t-1}
         x3 = (l_t - l_{t-1}) / l_{t-1}
         x4 = (h_t - c_t) / c_t
         x5 = (c_t - l_t) / c_t
        """
        features = []
        for i in range(1, len(data)):
            _, h_prev, l_prev, c_prev = data[i-1]
            o, h, l, c = data[i]
            x1 = (c - c_prev) / c_prev
            x2 = (h - h_prev) / h_prev
            x3 = (l - l_prev) / l_prev
            x4 = (h - c) / c if c != 0 else 0
            x5 = (c - l) / c if c != 0 else 0
            features.append([x1, x2, x3, x4, x5])
        return np.array(features)

    def reset(self):
        self.current_index = 16
        self.position_open = False
        self.position_side = None
        return self.features[self.current_index-16:self.current_index]

    def compute_reward(self, action):
        """
        Reward is computed based on the x1 feature (percentage change in close price).
        Action mapping:
          0: long → reward = x1
          1: short → reward = -x1
          2: neutral → reward = 0
        """
        z_t = self.features[self.current_index-1, 0]
        if action == 0:
            delta = 1
        elif action == 1:
            delta = -1
        else:
            delta = 0
        return delta * z_t

    def step(self, action):
        # Manage trade execution using our OANDA API functions.
        if action == 0:  # long
            if not self.position_open or self.position_side != "long":
                open_position(ACCOUNT_ID, self.instrument, self.units, "long")
                self.position_open = True
                self.position_side = "long"
        elif action == 1:  # short
            if not self.position_open or self.position_side != "short":
                open_position(ACCOUNT_ID, self.instrument, self.units, "short")
                self.position_open = True
                self.position_side = "short"
        elif action == 2:  # neutral
            if self.position_open:
                close_position(ACCOUNT_ID, self.instrument)
                self.position_open = False
                self.position_side = None

        reward = self.compute_reward(action)
        self.current_index += 1
        done = False
        if self.current_index >= len(self.features):
            done = True
            next_state = None
        else:
            next_state = self.features[self.current_index-16:self.current_index]
        return next_state, reward, done, {}

##############################################
# 3. Worker Function for Asynchronous Training
##############################################
def worker(global_model, optimizer, worker_id, max_steps=100):
    local_model = ActorCritic()
    local_model.load_state_dict(global_model.state_dict())
    
    env = OandaForexEnv(instrument="EUR_USD", units=100)
    state = env.reset()  # state shape: [16, 5]
    
    # Initialize decisions vector (e.g., previous trading decisions).
    decisions = np.zeros((1, 16))
    
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1, 16, 5]
    decisions = torch.tensor(decisions, dtype=torch.float32)
    
    for step in range(max_steps):
        policy_logits, value = local_model(state, decisions)
        probs = F.softmax(policy_logits, dim=1)
        action = torch.multinomial(probs, num_samples=1)
        
        next_state, reward, done, _ = env.step(action.item())
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32)
        
        advantage = reward_tensor - value
        policy_loss = -torch.log(probs[0, action]) * advantage.detach()
        value_loss = advantage.pow(2)
        loss = policy_loss + value_loss
        
        optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            if local_param.grad is not None:
                global_param._grad = local_param.grad
        optimizer.step()
        local_model.load_state_dict(global_model.state_dict())
        
        if done:
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        else:
            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        
        print(f"Worker {worker_id}, Step {step}, Loss: {loss.item()}, Reward: {reward}")
    print(f"Worker {worker_id} finished training.")

##############################################
# 4. Main Function to Launch Workers
##############################################
def main():
    global_model = ActorCritic()
    global_model.share_memory()  # Share parameters across threads
    optimizer = optim.Adam(global_model.parameters(), lr=0.00004)
    
    num_workers = 2  # Adjust as needed
    threads = []
    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(global_model, optimizer, i))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print("Training finished.")

if __name__ == "__main__":
    main()
