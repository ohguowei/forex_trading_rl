import numpy as np
from oanda_api import fetch_candle_data, ACCOUNT_ID  # Using our refactored API module
from feature_extractor import compute_features  # Our module for computing fixed features

class SimulatedOandaForexEnv:
    def __init__(self, instrument="EUR_USD", units=100, granularity="H1", candle_count=500):
        self.instrument = instrument
        self.units = units
        self.granularity = granularity
        self.candle_count = candle_count
        # Fetch raw OHLC data using OANDA API function
        self.data = np.array(fetch_candle_data(self.instrument, self.granularity, self.candle_count))
        # Compute normalized features (x1...x5) from OHLC data
        self.features = compute_features(self.data)
        # Use a sliding window of 16 time steps as the state
        self.current_index = 16

        # Trade state variables for simulation:
        self.position_open = False    # True if a trade is open
        self.position_side = None     # "long" or "short"
        self.entry_price = None       # The price at which the current trade was opened
        self.trade_log = []           # Log of closed trades (for evaluation)

    def reset(self):
        self.current_index = 16
        self.position_open = False
        self.position_side = None
        self.entry_price = None
        self.trade_log = []
        # Return initial state: a window of the last 16 feature rows
        return self.features[self.current_index-16:self.current_index]

    def compute_reward(self, action):
        """
        Compute the per-candle reward as:
          r_t = δ_t * z_t
        where z_t = (P_c_t - P_c_{t-1})/P_c_{t-1} is captured by feature x1.
        Action mapping: 0 -> long (δ=1), 1 -> short (δ=-1), 2 -> neutral (δ=0)
        """
        z_t = self.features[self.current_index-1, 0]
        if action == 0:
            delta = 1
        elif action == 1:
            delta = -1
        else:
            delta = 0
        return delta * z_t

    def simulated_open_position(self, side):
        if not self.position_open:
            self.position_open = True
            self.position_side = side
            # Set entry price as the close price from the corresponding candle.
            # Note: since features[0] is derived from data[1], we use data[self.current_index][3] (index 3 = close)
            self.entry_price = self.data[self.current_index][3]
            print(f"Simulated: Opened {side} position on {self.instrument} at entry price {self.entry_price}.")
        else:
            print(f"Simulated: Position already open ({self.position_side}).")

    def simulated_close_position(self):
        if self.position_open:
            # Use the current candle's close price as the exit price.
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
                "duration": self.current_index  # Simplified measure; can be improved
            }
            self.trade_log.append(trade_info)
            print(f"Simulated: Closed {self.position_side} position on {self.instrument} at exit price {exit_price}, Profit: {profit:.4f}")
            self.position_open = False
            self.position_side = None
            self.entry_price = None
        else:
            print("Simulated: No open position to close.")

    def step(self, action):
        """
        Executes one time step:
          - If action is long (0) or short (1), and if no matching position is open, then open that trade.
          - If action is neutral (2), close any open position.
          - When a trade remains open, it is considered continuous over consecutive steps.
        Then the reward for the current candle is computed,
        the environment advances the sliding window, and the next state is returned.
        """
        # Execute trade action:
        if action == 0:  # long
            if not self.position_open or self.position_side != "long":
                if self.position_open:
                    self.simulated_close_position()
                self.simulated_open_position("long")
        elif action == 1:  # short
            if not self.position_open or self.position_side != "short":
                if self.position_open:
                    self.simulated_close_position()
                self.simulated_open_position("short")
        elif action == 2:  # neutral: close any open trade
            if self.position_open:
                self.simulated_close_position()
        
        # Compute reward based on the percentage change in the close price (x1)
        reward = self.compute_reward(action)
        self.current_index += 1
        
        done = False
        if self.current_index >= len(self.features):
            done = True
            next_state = None
        else:
            next_state = self.features[self.current_index-16:self.current_index]
        return next_state, reward, done, {}
