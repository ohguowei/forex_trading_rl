# live_env.py

import time
import numpy as np
from oanda_api import fetch_candle_data, open_position, close_position, ACCOUNT_ID, ACCESS_TOKEN
from feature_extractor import compute_features

class LiveOandaForexEnv:
    def __init__(self, instrument="EUR_USD", units=100, granularity="H1", candle_count=500):
        self.instrument = instrument
        self.units = units
        self.granularity = granularity
        self.candle_count = candle_count
        # Initially, fetch historical data using your OANDA credentials.
        self.data = np.array(fetch_candle_data(self.instrument, self.granularity, self.candle_count))
        self.features = compute_features(self.data)
        self.current_index = 16

        # Live trade state variables:
        self.position_open = False
        self.position_side = None
        self.entry_price = None
        self.trade_log = []

    def reset(self):
        self.current_index = 16
        self.position_open = False
        self.position_side = None
        self.entry_price = None
        self.trade_log = []
        return self.features[self.current_index-16:self.current_index]

    def update_live_data(self):
        """
        Fetches the most recent candle from OANDA and appends it to the data.
        """
        # Fetch the latest candle using your OANDA API function (credentials are handled in oanda_api.py)
        new_candle = fetch_candle_data(self.instrument, self.granularity, candle_count=1)[0]
        self.data = np.vstack((self.data, new_candle))
        # Recompute features for the new candle (for simplicity, update the entire feature matrix)
        new_features = compute_features(np.vstack((self.data[-2:],)))
        self.features = np.vstack((self.features, new_features))
        self.current_index += 1

    def compute_reward(self, action):
        z_t = self.features[self.current_index-1, 0]
        if action == 0:
            delta = 1
        elif action == 1:
            delta = -1
        else:
            delta = 0
        return delta * z_t

    def live_open_position(self, side):
        # Place a live order using open_position with the imported credentials.
        response = open_position(instrument=self.instrument, account_id=ACCOUNT_ID, units=self.units, side=side)
        if response is not None:
            self.position_open = True
            self.position_side = side
            # Use the current candle's close price as the entry.
            self.entry_price = self.data[self.current_index][3]
            print(f"Live: Opened {side} position on {self.instrument} at {self.entry_price}.")
        else:
            print("Live: Order failed.")

    def live_close_position(self):
        # Close the position using close_position with your ACCOUNT_ID.
        close_position(instrument=self.instrument, account_id=ACCOUNT_ID)
        exit_price = self.data[self.current_index][3]
        profit = 0
        if self.position_side == "long":
            profit = (exit_price - self.entry_price) / self.entry_price
        elif self.position_side == "short":
            profit = (self.entry_price - exit_price) / self.entry_price
        trade_info = {
            "side": self.position_side,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "profit": profit,
            "timestamp": time.time()
        }
        self.trade_log.append(trade_info)
        print(f"Live: Closed {self.position_side} position on {self.instrument} at {exit_price}, Profit: {profit:.4f}")
        self.position_open = False
        self.position_side = None
        self.entry_price = None

    def step(self, action):
        # Execute real trading actions.
        if action == 0:  # long
            if not self.position_open or self.position_side != "long":
                if self.position_open:
                    self.live_close_position()
                self.live_open_position("long")
        elif action == 1:  # short
            if not self.position_open or self.position_side != "short":
                if self.position_open:
                    self.live_close_position()
                self.live_open_position("short")
        elif action == 2:  # neutral: close any open trade
            if self.position_open:
                self.live_close_position()
        
        reward = self.compute_reward(action)
        print("Waiting for next 1-hour candle...")
        time.sleep(3600)  # Wait 1 hour for the next candle.
        self.update_live_data()
        next_state = self.features[self.current_index-16:self.current_index]
        done = False  # In live trading, the loop continues indefinitely.
        return next_state, reward, done, {}
