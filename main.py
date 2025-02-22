import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import threading
import time
import argparse

# Import your live trading environment.
from live_env import LiveOandaForexEnv

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
        # Actor forward pass:
        actor_out, _ = self.actor_lstm(state)
        actor_last = actor_out[:, -1, :]
        x = F.relu(self.actor_fc1(actor_last))
        x = torch.cat([x, decisions], dim=1)
        x = F.relu(self.actor_fc2(x))
        x = F.relu(self.actor_fc3(x))
        policy_logits = self.actor_output(x)
        
        # Critic forward pass:
        critic_out, _ = self.critic_lstm(state)
        critic_last = critic_out[:, -1, :]
        y = F.relu(self.critic_fc1(critic_last))
        y = torch.cat([y, decisions], dim=1)
        y = F.relu(self.critic_fc2(y))
        y = F.relu(self.critic_fc3(y))
        value = self.critic_output(y)
        
        return policy_logits, value

##############################################
# 2. Worker Function for Asynchronous Training
##############################################
def worker(global_model, optimizer, optimizer_lock, training_env, max_steps=20):
    """
    A worker that runs for a fixed number of steps on the given environment.
    """
    local_model = ActorCritic()
    local_model.load_state_dict(global_model.state_dict())
    
    state = training_env.reset()  # e.g. sliding window [16, 5]
    decisions = np.zeros((1, 16))
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)   # [1,16,5]
    decisions = torch.tensor(decisions, dtype=torch.float32)
    
    for step in range(max_steps):
        policy_logits, value = local_model(state, decisions)
        probs = F.softmax(policy_logits, dim=1)
        action = torch.multinomial(probs, num_samples=1)
        
        next_state, reward, done, _ = training_env.step(action.item())
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32)
        
        advantage = reward_tensor - value
        policy_loss = -torch.log(probs[0, action]) * advantage.detach()
        value_loss = advantage.pow(2)
        loss = policy_loss + value_loss
        
        optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            if local_param.grad is not None:
                global_param.grad = local_param.grad.clone()
        
        with optimizer_lock:
            optimizer.step()
        
        local_model.load_state_dict(global_model.state_dict())
        
        if done:
            state = training_env.reset()
        else:
            state = next_state
        
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        print(f"[Training Worker] Step: {step}, Loss: {loss.item()}, Reward: {reward}")
        
    print("[Training Worker] Finished training cycle.")

def continuous_training(global_model, optimizer, optimizer_lock):
    """
    Continuously runs training cycles using the live environment (or a dedicated training environment).
    In a real scenario, you might periodically refresh the training environment with new data.
    """
    # Here, for continuous training, we create an environment instance.
    training_env = LiveOandaForexEnv(instrument="EUR_USD", units=100, candle_count=500)
    while True:
        # Run one training cycle (asynchronously via a worker)
        worker(global_model, optimizer, optimizer_lock, training_env, max_steps=20)
        # Optionally, save model checkpoints or log performance.
        print("[Continuous Training] Training cycle complete. Model updated.")
        # Wait a short period before the next training cycle.
        time.sleep(10)  # Adjust as needed

##############################################
# 3. Trading Function (Using Latest Global Model)
##############################################
def trade_live(global_model, live_env, num_steps=10):
    """
    Runs the trading loop. For each 1-hour candle, the system uses the current global model to decide an action.
    """
    global_model.eval()
    state = live_env.reset()  # Get the latest 16 candles as state.
    decisions = np.zeros((1, 16))
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    decisions = torch.tensor(decisions, dtype=torch.float32)
    
    for step in range(num_steps):
        with torch.no_grad():
            policy_logits, _ = global_model(state, decisions)
            probs = torch.softmax(policy_logits, dim=1)
            action = torch.multinomial(probs, num_samples=1).item()
        
        print(f"[Trading] Step {step}, Action: {action}")
        next_state, reward, done, _ = live_env.step(action)
        print(f"[Trading] Reward: {reward}")
        
        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        
        # In live trading, the environment itself waits until the next candle (e.g., 1 hour).
        
    print("[Trading] Finished trading cycle.")
    print("Trade Log:", live_env.trade_log)

##############################################
# 4. Main Function: Hybrid Training & Trading
##############################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='hybrid', choices=['hybrid', 'train', 'trade'],
                        help="Choose 'hybrid' for continuous training & trading; 'train' for training only; 'trade' for trading only.")
    parser.add_argument('--train_steps', type=int, default=20, help="Steps per training cycle.")
    parser.add_argument('--trade_steps', type=int, default=10, help="Trading steps (candles) to process.")
    args = parser.parse_args()
    
    # Create global model and optimizer.
    global_model = ActorCritic()
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=0.00004)
    optimizer_lock = threading.Lock()
    
    if args.mode == 'hybrid':
        # Create live environment for trading.
        live_env = LiveOandaForexEnv(instrument="EUR_USD", units=100, candle_count=500)
        
        # Start continuous training in a background thread.
        training_thread = threading.Thread(target=continuous_training, args=(global_model, optimizer, optimizer_lock), daemon=True)
        training_thread.start()
        
        # In the main thread, run trading loop repeatedly.
        while True:
            trade_live(global_model, live_env, num_steps=args.trade_steps)
            # Optionally, decide on an interval between trading cycles.
            print("[Main] Waiting for next trading cycle...")
            time.sleep(3600)  # Wait one hour before starting the next trading cycle.
            
    elif args.mode == 'train':
        # For pure training mode, run one cycle using a dedicated training environment.
        training_env = LiveOandaForexEnv(instrument="EUR_USD", units=100, candle_count=500)
        worker(global_model, optimizer, optimizer_lock, training_env, max_steps=args.train_steps)
        torch.save(global_model.state_dict(), "trained_global_model.pth")
        print("Training finished. Model saved as 'trained_global_model.pth'.")
        
    elif args.mode == 'trade':
        # For pure trading mode, load the trained model and run trading loop.
        trained_model = ActorCritic()
        trained_model.load_state_dict(torch.load("trained_global_model.pth"))
        live_env = LiveOandaForexEnv(instrument="EUR_USD", units=100, candle_count=500)
        trade_live(trained_model, live_env, num_steps=args.trade_steps)
    
if __name__ == "__main__":
    main()
