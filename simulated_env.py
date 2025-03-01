import time
import numpy as np
from oanda_api import fetch_candle_data
from feature_extractor import compute_features
from config import TradingConfig

class Trade:
    """
    A class to represent a trade with structured information.
    """
    def __init__(self, side, entry_price, exit_price, profit, timestamp):
        self.side = side
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.profit = profit
        self.timestamp = timestamp

    def __repr__(self):
        return f"Trade(side={self.side}, entry={self.entry_price}, exit={self.exit_price}, profit={self.profit:.4f})"

class SimulatedOandaForexEnv:
    def __init__(self, 
                 instrument=TradingConfig.INSTRUMENT, 
                 units=TradingConfig.SIMULATED_UNITS, 
                 granularity=TradingConfig.GRANULARITY, 
                 candle_count=TradingConfig.CANDLE_COUNT, 
                 spread=TradingConfig.SPREAD):
        self.instrument = instrument
        self.units = units
        self.granularity = granularity
        self.candle_count = candle_count
        self.spread = spread

        # Fetch initial historical data
        self.data = self._fetch_initial_data()
        # Compute your base features (shape: (len(data) - 1, 12))
        self.features = compute_features(self.data)

        # We start after the first 16 candles
        self.current_index = 16

        # Trade state variables
        self.position_open = False
        self.position_side = None
        self.entry_price = None
        self.trade_log = []

    def _fetch_initial_data(self):
        """
        Fetch initial historical data from OANDA API with error handling.
        """
        try:
            data = np.array(fetch_candle_data(self.instrument, self.granularity, self.candle_count))
            if len(data) == 0:
                raise ValueError("No data returned from OANDA API.")
            return data
        except Exception as e:
            print(f"Error fetching initial data: {e}")
            raise

    def reset(self):
        """
        Reset the environment to its initial state.
        Returns a (16, 13) array: 16 timesteps × (12 base features + 1 P/L).
        """
        # Reset index and trade state
        self.current_index = 16
        self.position_open = False
        self.position_side = None
        self.entry_price = None
        self.trade_log = []

        # Re-fetch or reuse the features if you wish, but typically once is enough
        # self.data = self._fetch_initial_data()
        # self.features = compute_features(self.data)

        # Get the last 16 rows of your 12-dim features
        base_features = self.features[self.current_index - 16 : self.current_index]
        # At reset, we have no open trade -> P/L = 0.0
        current_pl = 0.0
        pl_column = np.full((base_features.shape[0], 1), current_pl)
        # Now shape is (16, 13)
        state_with_pl = np.hstack((base_features, pl_column))

        return state_with_pl
    

    def update_live_data(self):
        """
        Fetch the most recent candle from OANDA and append it to the data.
        """
        try:
            new_candle = fetch_candle_data(self.instrument, self.granularity, candle_count=1)[0]
            # For your data shape, you might have [open, high, low, close, volume]
            if len(new_candle) != 4 and len(new_candle) != 5:
                raise ValueError("Invalid candle data returned from OANDA API.")

            self.data = np.vstack((self.data, new_candle))
            # Recompute features for just the new candle
            new_features = compute_features(np.vstack((self.data[-2:],)))
            self.features = np.vstack((self.features, new_features))
            self.current_index += 1
        except Exception as e:
            print(f"Error updating live data: {e}")

    def compute_reward(self, action):
        """
        Compute the reward based on the action and the latest price change in the features.
        This was your original logic: the first feature (index 0) is the % change in close.
        """
        z_t = self.features[self.current_index - 1, 0]  # percentage change
        if action == 0:      # long
            delta = 1
        elif action == 1:    # short
            delta = -1
        else:                # neutral
            delta = 0
        return delta * z_t

    def _apply_spread(self, price, side):
        """
        Apply the spread to the price based on the trade side.
        """
        if side == "long":
            return price + (self.spread / 2)
        elif side == "short":
            return price - (self.spread / 2)
        return price

    def simulated_open_position(self, side):
        """
        Simulate opening a position with spread applied.
        """
        if not self.position_open:
            self.position_open = True
            self.position_side = side
            # Mark the entry price after spread
            self.entry_price = self._apply_spread(self.data[self.current_index][3], side)

    def simulated_close_position(self):
        """
        Simulate closing a position with spread applied.
        """
        if self.position_open:
            exit_price = self._apply_spread(self.data[self.current_index][3], self.position_side)
            if self.position_side == "long":
                profit = (exit_price - self.entry_price) / self.entry_price
            else:  # short
                profit = (self.entry_price - exit_price) / self.entry_price

            # Log the trade
            trade = Trade(
                side=self.position_side,
                entry_price=self.entry_price,
                exit_price=exit_price,
                profit=profit,
                timestamp=time.time()
            )
            self.trade_log.append(trade)

            # Reset position state
            self.position_open = False
            self.position_side = None
            self.entry_price = None

    def step(self, action):
        """
        Execute one step in the simulated trading environment.
        Returns:
          next_state: (16, 13) array (unless done is True),
                      with the P/L as the 13th feature.
          reward: float
          done: bool
          info: dict
        """
        # 1) Execute the trade action
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
        elif action == 2:  # neutral
            if self.position_open:
                self.simulated_close_position()

        # 2) Compute reward
        reward = self.compute_reward(action)

        # 3) Move to the next time step
        self.current_index += 1

        # 4) Check if we are out of data
        done = (self.current_index >= len(self.features))
        if done:
            return None, reward, done, {}

        # 5) Build the next state
        #    a) Take the last 16 rows of base features
        next_features = self.features[self.current_index - 16 : self.current_index]

        #    b) Calculate the current P/L
        if self.position_open:
            # Mark-to-market using the current candle close
            current_price = self.data[self.current_index][3]
            if self.position_side == "long":
                current_pl = (current_price - self.entry_price) / self.entry_price
            else:  # short
                current_pl = (self.entry_price - current_price) / self.entry_price
        else:
            # If no open position, use the last realized P/L or 0
            current_pl = self.trade_log[-1].profit if self.trade_log else 0.0

        #    c) Append the P/L as a new column -> shape (16, 13)
        pl_column = np.full((next_features.shape[0], 1), current_pl)
        next_state = np.hstack((next_features, pl_column))

        return next_state, reward, done, {}
