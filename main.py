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

import tg_bot            # Contains Telegram bot logic and global "last_trade_status"

# Directory to save models per currency.
MODEL_DIR = "./models/"

def wait_for_trading_window():
    """
    Block until the current local time is within the trading window:
    Monday ≥6 AM to Saturday <6 AM.
    """
    while True:
        now = datetime.datetime.now()
        wd, hr = now.weekday(), now.hour
        # Trading allowed if NOT Sunday (weekday 6), NOT Monday before 6AM, NOT Saturday at/after 6AM.
        if not (wd == 6 or (wd == 0 and hr < 6) or (wd == 5 and hr >= 6)):
            return
        print("Outside Mon 6AM – Sat 6AM trading window. Sleeping 60 seconds...")
        time.sleep(60)

def calculate_next_trigger_time():
    now = datetime.datetime.now()
    next_minute = ((now.minute // 10) + 1) * 10
    if next_minute >= 60:
        next_trigger = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    else:
        next_trigger = now.replace(minute=next_minute, second=0, microsecond=0)
    return next_trigger

def wait_until_next_trigger():
    next_trigger = calculate_next_trigger_time()
    now = datetime.datetime.now()
    wait_time = (next_trigger - now).total_seconds()
    if wait_time > 0:
        print(f"Waiting {wait_time:.0f} seconds until next trigger at {next_trigger.strftime('%H:%M:%S')}...")
        time.sleep(wait_time)
    else:
        print("Next trigger time is in the past. Triggering immediately.")
    return next_trigger

def trade_live(currency_model, live_env, num_steps=10):
    """
    Runs a live trading cycle.
    Updates the Telegram bot's global last_trade_status variable with the most recent trade.
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
    
    # Update the global trade status.
    if live_env.trade_log:
        last_trade = live_env.trade_log[-1]
        tg_bot.last_trade_status = str(last_trade)
        print("Updated last trade status:", tg_bot.last_trade_status)
    else:
        tg_bot.last_trade_status = "No trades executed."

def trading_loop():
    """
    The main trading/training loop. It first waits for the trading window,
    then waits until the next trigger (every 10 minutes), and then executes either
    a training or trading cycle depending on the trigger minute.
    
    It also checks the trading control flag and blocks if trading is paused.
    """
    num_workers = 100      # Number of worker threads for training.
    train_steps = 10       # Training steps per worker.
    trade_steps = 1        # Number of live trading steps per trading cycle.

    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Initialize (or load) models for each currency.
    models = {}
    for currency, currency_config in CURRENCY_CONFIGS.items():
        model_path = os.path.join(MODEL_DIR, f"{currency}.pt")
        currency_model = ActorCritic()
        if os.path.exists(model_path):
            currency_model.load_state_dict(torch.load(model_path))
            print(f"Loaded existing model for {currency}.")
        else:
            print(f"Initializing new model for {currency}.")
        currency_model.share_memory()
        models[currency] = currency_model

    while True:
        # Block until trading is active (tg_bot.trading_event is set).
        tg_bot.trading_event.wait()
        
        wait_for_trading_window()
        next_trigger = wait_until_next_trigger()
        
        # Decide whether to run a training cycle or a trading cycle.
        if next_trigger.minute == 0:
            print(f"\n=== Trigger at {next_trigger.strftime('%H:%M:%S')}: Running TRAINING cycle ===")
            for currency, currency_config in CURRENCY_CONFIGS.items():
                print(f"\n--- Training cycle for {currency} ---")
                model = models[currency]
                optimizer = optim.Adam(model.parameters(), lr=0.00004)
                optimizer_lock = threading.Lock()
                barrier = threading.Barrier(num_workers + 1)
                
                workers = []
                for i in range(num_workers):
                    t = threading.Thread(
                        target=worker,
                        args=(i, model, optimizer, optimizer_lock, train_steps, currency_config, barrier),
                        daemon=True
                    )
                    workers.append(t)
                    t.start()
                
                barrier.wait()
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{currency}.pt"))
                for t in workers:
                    t.join()
                print(f"--- Finished training cycle for {currency} ---")
        else:
            print(f"\n=== Trigger at {next_trigger.strftime('%H:%M:%S')}: Running TRADING cycle ===")
            for currency, currency_config in CURRENCY_CONFIGS.items():
                print(f"\n--- Trading cycle for {currency} ---")
                model = models[currency]
                model.eval()
                live_env = LiveOandaForexEnv(
                    currency_config,
                    candle_count=TradingConfig.CANDLE_COUNT,
                    granularity=TradingConfig.GRANULARITY
                )
                trade_live(model, live_env, num_steps=trade_steps)
                print(f"--- Finished trading cycle for {currency} ---")
        
        print("\nCycle complete. Waiting for the next trigger...\n")

if __name__ == "__main__":
    # Start the trading loop in a background thread.
    trading_thread = threading.Thread(target=trading_loop, daemon=True)
    trading_thread.start()
    
    # Run the Telegram bot in the main thread (to allow proper signal handling).
    tg_bot.run_telegram_bot()
