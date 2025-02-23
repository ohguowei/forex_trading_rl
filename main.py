# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import threading
import time

from models import ActorCritic
from worker import worker
from live_env import LiveOandaForexEnv
from simulated_env import SimulatedOandaForexEnv

##############################################
# Model Aggregation Function
##############################################
def aggregate_models(models):
    """
    Aggregates multiple models by averaging their weights.
    """
    aggregated_model = ActorCritic()
    aggregated_state_dict = aggregated_model.state_dict()
    
    # Average the parameters from all models
    for key in aggregated_state_dict:
        aggregated_state_dict[key] = torch.mean(
            torch.stack([model.state_dict()[key] for model in models]), dim=0
        )
    
    aggregated_model.load_state_dict(aggregated_state_dict)
    return aggregated_model

##############################################
# Trading Function (Using Latest Aggregated Model)
##############################################
def trade_live(global_model, live_env, num_steps=10):
    """
    Runs the trading loop. For each 1-hour candle, the system
    uses the current global model to decide an action.
    """
    global_model.eval()
    state = live_env.reset()
    decisions = np.zeros((1, 16))  # same shape as in training
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
        
        # Example update of state if not done
        if not done and next_state is not None:
            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        else:
            # If an episode finished, you could reset the environment or break
            break
    
    print("[Trading] Finished trading cycle.")
    print("Trade Log:", live_env.trade_log)

##############################################
# Main Function: Hybrid Training & Trading
##############################################
def main():
    # Hyperparameters
    num_workers = 4  # Number of training workers
    train_steps = 20  # Steps per worker per episode
    trade_steps = 10  # Trading steps (candles) to process
    aggregation_interval = 60  # Aggregate models every 60 seconds

    # Create the global model and optimizer
    global_model = ActorCritic()
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=0.00004)
    optimizer_lock = threading.Lock()

    # Create a live environment (or pass the real API credentials)
    live_env = LiveOandaForexEnv(
        instrument="EUR_USD", 
        units=100, 
        candle_count=500
    )

    # Start multiple training workers (using simulated environment)
    workers = []
    for i in range(num_workers):
        worker_thread = threading.Thread(
            target=worker,
            args=(i, global_model, optimizer, optimizer_lock, train_steps),
            daemon=True
        )
        workers.append(worker_thread)
        worker_thread.start()

    # Periodically aggregate models and do a trading step
    while True:
        #time.sleep(aggregation_interval)  # Let the workers run for a bit

        # Collect local copies from the global model
        local_models = []
        for i in range(num_workers):
            local_model = ActorCritic()
            local_model.load_state_dict(global_model.state_dict())
            local_models.append(local_model)
        
        # Aggregate the local models
        aggregated_model = aggregate_models(local_models)

        # Optionally run a live trading cycle with the aggregated model
        trade_live(aggregated_model, live_env, num_steps=trade_steps)

        print("[Main] Trading cycle completed. You can continue or break the loop.")
        # For demonstration, let’s break after one iteration:
        break

    print("[Main] Done!")

if __name__ == "__main__":
    main()
