import time
import numpy as np
from feature_extractor import compute_features

from oanda_api import (
    fetch_candle_data,
    open_position,
    close_position,
    get_open_positions,
    ACCOUNT_ID, 
    ACCESS_TOKEN
)

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

class LiveOandaForexEnv:
    def __init__(self, instrument="EUR_USD", units=100, granularity="H1", candle_count=5000):
        self.instrument = instrument
        self.units = units
        self.granularity = granularity
        self.candle_count = candle_count

        # Fetch initial historical data
        self.data = self._fetch_initial_data()
        self.features = compute_features(self.data)
        self.current_index = 16

        # Initialize local state about open position
        self.position_open = False
        self.position_side = None  # "long" or "short"
        self.entry_price = None
        self.trade_log = []

        # Now, check if there's already an open position in OANDA
        self._sync_oanda_position_state()

    def _sync_oanda_position_state(self):
        """
        Check OANDA for any open positions in our 'instrument'.
        If found, set self.position_open, self.position_side, and self.entry_price.
        """
        positions_response = get_open_positions(ACCOUNT_ID)
        if not positions_response or "positions" not in positions_response:
            print("No positions found or error in position check.")
            return

        # positions_response["positions"] is typically a list of dicts
        for pos in positions_response["positions"]:
            if pos["instrument"] == self.instrument:
                # OANDA splits positions into 'long' and 'short' subfields
                long_units = float(pos["long"]["units"])
                short_units = float(pos["short"]["units"])

                if long_units > 0:
                    self.position_open = True
                    self.position_side = "long"
                    self.entry_price = float(pos["long"]["averagePrice"])
                    print(f"Detected existing LONG position at price {self.entry_price}.")
                    return
                elif short_units > 0:
                    self.position_open = True
                    self.position_side = "short"
                    self.entry_price = float(pos["short"]["averagePrice"])
                    print(f"Detected existing SHORT position at price {self.entry_price}.")
                    return

        # If no position in that instrument is found
        print(f"No existing position found for {self.instrument} in OANDA.")

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
        Fetches the most recent candle from OANDA and appends it to the data.
        Handles API errors gracefully.
        """
        try:
            new_candle = fetch_candle_data(self.instrument, self.granularity, candle_count=1)[0]
            
            # Ensure the candle has all required fields
            if len(new_candle) != 5:  # [open, high, low, close, volume]
                raise ValueError("Invalid candle data returned from OANDA API.")
            
            # Append the new candle to the data
            self.data = np.vstack((self.data, new_candle))
            
            # Recompute features for the new candle
            new_features = compute_features(np.vstack((self.data[-2:],)))
            self.features = np.vstack((self.features, new_features))
            
            # Increment the current index
            self.current_index += 1
        except Exception as e:
            print(f"Error updating live data: {e}")
            # Optionally, log the error or take corrective action

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

    def live_open_position(self, side):
        """
        Open a live position with error handling.
        """
        if self.position_open:
            print(f"Live: Position already open ({self.position_side}). Cannot open a new position.")
            return

        try:
            response = open_position(instrument=self.instrument, account_id=ACCOUNT_ID, units=self.units, side=side)
            if response is not None:
                self.position_open = True
                self.position_side = side
                self.entry_price = self.data[self.current_index][3]  # Use the current candle's close price
                print(f"Live: Opened {side} position on {self.instrument} at {self.entry_price}.")
            else:
                print("Live: Order failed.")
        except Exception as e:
            print(f"Error opening position: {e}")

    def live_close_position(self):
        """
        Close the current live position with error handling.
        """
        if not self.position_open:
            print("Live: No open position to close.")
            return

        try:
            close_position(instrument=self.instrument, account_id=ACCOUNT_ID)
            exit_price = self.data[self.current_index][3]
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
            print(f"Live: Closed {self.position_side} position on {self.instrument} at {exit_price}, Profit: {profit:.4f}")

            # Reset position state
            self.position_open = False
            self.position_side = None
            self.entry_price = None
        except Exception as e:
            print(f"Error closing position: {e}")

    def step(self, action):
        """
        Execute one step in the live trading environment.
        """
        # Execute trade action
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
        
        # Compute reward
        reward = self.compute_reward(action)
        
        # Update live data
        self.update_live_data()
        next_state = self.features[self.current_index-16:self.current_index]
        done = False  # In live trading, the loop continues indefinitely
        return next_state, reward, done, {}
    
