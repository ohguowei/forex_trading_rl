import time
import numpy as np
from oanda_api import fetch_candle_data
from feature_extractor import compute_features

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
    def __init__(self, instrument="EUR_USD", units=1000, granularity="H1", candle_count=5000, spread=0.0002):
        self.instrument = instrument
        self.units = units
        self.granularity = granularity
        self.candle_count = candle_count
        self.spread = spread  # Spread in pips (e.g., 0.0002 for 2 pips)

        # Fetch initial historical data
        self.data = self._fetch_initial_data()
        self.features = compute_features(self.data)
        self.current_index = 16  # Start after the first 16 candles for the sliding window

        # Trade state variables
        self.position_open = False
        self.position_side = None
        self.entry_price = None
        self.trade_log = []  # List of Trade objects

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
        """
        self.current_index = 16
        self.position_open = False
        self.position_side = None
        self.entry_price = None
        self.trade_log = []
        return self.features[self.current_index-16:self.current_index]

    def update_live_data(self):
        """
        Fetch the most recent candle from OANDA and append it to the data.
        """
        try:
            new_candle = fetch_candle_data(self.instrument, self.granularity, candle_count=1)[0]
            if len(new_candle) != 4:  # Ensure the candle has OHLC data
                raise ValueError("Invalid candle data returned from OANDA API.")
            
            self.data = np.vstack((self.data, new_candle))
            new_features = compute_features(np.vstack((self.data[-2:],)))
            self.features = np.vstack((self.features, new_features))
            self.current_index += 1
        except Exception as e:
            print(f"Error updating live data: {e}")

    def compute_reward(self, action):
        """
        Compute the reward based on the action and the latest price change.
        """
        z_t = self.features[self.current_index-1, 0]  # Percentage change in close price
        if action == 0:  # long
            delta = 1
        elif action == 1:  # short
            delta = -1
        else:  # neutral
            delta = 0
        return delta * z_t

    def _apply_spread(self, price, side):
        """
        Apply the spread to the price based on the trade side.
        - For long positions, the entry price is the ask price (price + spread/2).
        - For short positions, the entry price is the bid price (price - spread/2).
        """
        if side == "long":
            return price + (self.spread / 2)
        elif side == "short":
            return price - (self.spread / 2)
        else:
            return price

    def simulated_open_position(self, side):
        """
        Simulate opening a position with spread applied.
        """
        if not self.position_open:
            self.position_open = True
            self.position_side = side
            # Apply spread to the entry price
            self.entry_price = self._apply_spread(self.data[self.current_index][3], side)
     #       print(f"Simulated: Opened {side} position on {self.instrument} at entry price {self.entry_price}.")
        #else:            
            #print(f"Simulated: Position already open ({self.position_side}).")

    def simulated_close_position(self):
        """
        Simulate closing a position with spread applied.
        """
        if self.position_open:
            # Apply spread to the exit price
            exit_price = self._apply_spread(self.data[self.current_index][3], self.position_side)
            profit = 0
            if self.position_side == "long":
                profit = (exit_price - self.entry_price) / self.entry_price
            elif self.position_side == "short":
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
        #    print(f"Simulated: Closed {self.position_side} position on {self.instrument} at exit price {exit_price}, Profit: {profit:.4f}")

            # Reset position state
            self.position_open = False
            self.position_side = None
            self.entry_price = None
       # else:
       #     print("Simulated: No open position to close.")

    def step(self, action):
        """
        Execute one step in the simulated trading environment.
        """
        # Execute trade action
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
        
        # Compute reward
        reward = self.compute_reward(action)
        self.current_index += 1
        
        # Check if the episode is done
        done = False
        if self.current_index >= len(self.features):
            done = True
            next_state = None
        else:
            next_state = self.features[self.current_index-16:self.current_index]
        
        return next_state, reward, done, {}