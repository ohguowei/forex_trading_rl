import os
import datetime
import time
import threading
import numpy as np
import torch
import torch.optim as optim

from models import ActorCritic
from worker import worker
from live_env import LiveOandaForexEnv
from config import TradingConfig, CURRENCY_CONFIGS

# Directory to save models per currency.
MODEL_DIR = "./models/"

def wait_for_trading_window():
    """
    Loop until the current local time is within the trading window:
    Monday ≥6 AM to Saturday <6 AM.
    """
    while True:
        now = datetime.datetime.now()
        wd, hr = now.weekday(), now.hour
        # Out-of-window conditions: Sunday (6), Monday before 6 AM (0), or Saturday at/after 6 AM (5)
        if not (wd == 6 or (wd == 0 and hr < 6) or (wd == 5 and hr >= 6)):
            return
        print("Outside Mon 6AM – Sat 6AM window. Sleeping 60 seconds...")
        time.sleep(60)

def calculate_next_trigger_time():
    """
    Calculate the next trigger time at the next hour and 1 minute mark.
    """
    now = datetime.datetime.now()
    next_hour = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
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

def trade_live(currency_model, live_env, num_steps=10):
    """
    Run a live trading cycle using the given currency model and live environment.
    """
    currency_model.eval()
    state = live_env.reset()
    decisions = np.zeros((1, 16))
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    decisions = torch.tensor(decisions, dtype=torch.float32)
    
    for step in range(num_steps):
        with torch.no_grad():
            policy_logits, _ = currency_model(state, decisions)
            probs = torch.softmax(policy_logits, dim=1)
            action = torch.multinomial(probs, num_samples=1).item()
        print(f"[Trading] Step {step}, Action: {action}")
        next_state, reward, done, _ = live_env.step(action)
        print(f"[Trading] Reward: {reward}")
        if not done and next_state is not None:
            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        else:
            break
    print("[Trading] Finished trading cycle.")
    print("Trade Log:", live_env.trade_log)

def main():
    #sunday
    num_workers = 20       # Number of workers per currency
    train_steps = 5000     # Training steps for each worker
    
    #weekday
    #num_workers = 5       # Number of workers per currency
    #train_steps = 1000     # Training steps for each worker
    trade_steps = 1       # Number of live trading steps after training
    os.makedirs(MODEL_DIR, exist_ok=True)

    for currency, currency_config in CURRENCY_CONFIGS.items():
        print(f"\n=== Starting training cycle for {currency} ===")
        
        # Ensure we are in the trading window and wait until the next trigger.
        #time.sleep(10)
        #wait_for_trading_window()
        #wait_until_next_trigger()
        
        # Determine the model file path for this currency.
        model_path = os.path.join(MODEL_DIR, f"{currency}.pt")
        
        # Load an existing model if available; otherwise, initialize a new one.
        currency_model = ActorCritic()
        if os.path.exists(model_path):
            currency_model.load_state_dict(torch.load(model_path))
            print(f"Loaded existing model for {currency}.")
        else:
            print(f"Initializing new model for {currency}.")
        
        # Share the model's memory if using threads.
        currency_model.share_memory()
        optimizer = optim.Adam(currency_model.parameters(), lr=0.00004)
        optimizer_lock = threading.Lock()
        
        # Create the live trading environment.
        live_env = LiveOandaForexEnv(
            currency_config,
            candle_count=TradingConfig.CANDLE_COUNT,
            granularity=TradingConfig.GRANULARITY
        )
        
        # Create a barrier for all workers plus the main thread.
        barrier = threading.Barrier(num_workers + 1)
        
        # Start worker threads dedicated to this currency.
        workers = []
        for i in range(num_workers):
            t = threading.Thread(
                target=worker,
                args=(i, currency_model, optimizer, optimizer_lock, train_steps, currency_config, barrier),
                daemon=True
            )
            workers.append(t)
            t.start()
        
        # Main thread waits at the barrier until all workers have finished training.
        barrier.wait()
        
        # Save the updated model for this currency.
        torch.save(currency_model.state_dict(), model_path)
        #print(f"Saved updated model for {currency} to {model_path}.")
        
        # Run a live trading cycle using the updated model.
        trade_live(currency_model, live_env, num_steps=trade_steps)
        
        # Join the worker threads.
        for t in workers:
            t.join()
        
        print(f"=== Finished training cycle for {currency} ===\n")
        

if __name__ == "__main__":
    main()
