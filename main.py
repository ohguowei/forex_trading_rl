# main.py
import datetime
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

#import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
import numpy as np
import threading
import time

from models import ActorCritic
from worker import worker
from live_env import LiveOandaForexEnv

def wait_for_trading_window():
    # Loop until local time is Mon ≥6 AM and Sat <6 AM
    while True:
        now = datetime.datetime.now()
        wd, hr = now.weekday(), now.hour
        # Sunday=6 -> skip; Mon(0)<6 -> skip; Sat(5)≥6 -> skip
        if not (wd == 6 or (wd == 0 and hr < 6) or (wd == 5 and hr >= 6)):
            return
        print("Outside Mon 6AM – Sat 6AM window. Sleeping 60s...")
        time.sleep(60)

def calculate_next_trigger_time():
    """
    Calculate the next trigger time at the next hour and 1 minute mark.
    For example:
    - If the current time is 11:01 AM, the next trigger time is 12:01 PM.
    - If the current time is 9:13 AM, the next trigger time is 10:01 AM.
    """
    now = datetime.datetime.now()
    next_hour = now.replace(minute=1, second=0, microsecond=0) + datetime.timedelta(hours=1)
    return next_hour

def wait_until_next_trigger():
    """
    Wait until the next trigger time (next hour and 1 minute mark).
    """

    next_trigger_time = calculate_next_trigger_time()
    now = datetime.datetime.now()
    time_to_wait = (next_trigger_time - now).total_seconds()
    
    if time_to_wait > 0:
        print(f"Waiting {time_to_wait:.0f} seconds until the next trigger at {next_trigger_time.strftime('%H:%M:%S')}...")
        time.sleep(time_to_wait)
    else:
        print("Next trigger time is in the past. Triggering immediately.")


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
    num_workers = 10 # Number of training workers
    train_steps = 100  # Steps per worker per episode
    trade_steps = 1  # Trading steps (candles) to process
    aggregation_interval = 60  # Aggregate models every 60 seconds

    # Create the global model and optimizer
    global_model = ActorCritic()
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=0.00004)
    optimizer_lock = threading.Lock()

    # Create a live environment (or pass the real API credentials)
    live_env = LiveOandaForexEnv(
        instrument="EUR_USD", 
        units=1000, 
        candle_count=5000
    )
    # Wait here until we’re within Mon 6 AM – Sat 6 AM


    # Periodically aggregate models and do a trading step
    while True:
        #time.sleep(aggregation_interval)  # Let the workers run for a bit
     #   wait_for_trading_window()  # Wait here until we’re within Mon 6 AM – Sat 6 AM
     #   wait_until_next_trigger()
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

        # Collect local copies from the global model
        local_models = []
        for i in range(num_workers):
            local_model = ActorCritic()
            local_model.load_state_dict(global_model.state_dict())
            local_models.append(local_model)
        time.sleep(aggregation_interval)  # Let the workers run for a bit
        # Aggregate the local models
        aggregated_model = aggregate_models(local_models)

        # Optionally run a live trading cycle with the aggregated model
        wait_until_next_trigger()
        trade_live(aggregated_model, live_env, num_steps=trade_steps)

        print("[Main] Trading cycle completed. You can continue or break the loop.")
        # For demonstration, let’s break after one iteration:
        #break

    print("[Main] Done!")

if __name__ == "__main__":
    main()
