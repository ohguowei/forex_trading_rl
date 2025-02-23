import numpy as np

def compute_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period + 1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    # Protect against zero-down to avoid division by zero:
    rs = up / down if down != 0 else np.inf

    rsi = np.zeros_like(prices)
    rsi[:period] = 100.0 - (100.0 / (1.0 + rs)) if rs != np.inf else 100.0

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else np.inf
        rsi[i] = 100.0 - (100.0 / (1.0 + rs)) if rs != np.inf else 100.0

    return rsi


def compute_features(data):
    """
    Compute features for each time step from the OHLC data, including volume and RSI.
    
    For each time step i (starting at 1 since we need a previous candle):
      x1 = (c_i - c_{i-1}) / c_{i-1}      : Percentage change in close price
      x2 = (h_i - h_{i-1}) / h_{i-1}      : Percentage change in high price
      x3 = (l_i - l_{i-1}) / l_{i-1}      : Percentage change in low price
      x4 = (h_i - c_i) / c_i              : Relative difference between high and close
      x5 = (c_i - l_i) / c_i              : Relative difference between close and low
      x6 = v_i                            : Volume (normalized)
      x7 = RSI_14                         : RSI with a 14-period window
      x8 = RSI_28                         : RSI with a 28-period window
      x9 = RSI_36                         : RSI with a 36-period window
      x10 = RSI_72                        : RSI with a 72-period window
      x11 = RSI_148                       : RSI with a 148-period window
      x12 = RSI_200                       : RSI with a 200-period window
    
    Returns:
      A NumPy array of shape (len(data)-1, 12) where each row contains [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12].
    """
    features = []
    close_prices = data[:, 3]  # Extract closing prices for RSI calculation
    volumes = data[:, 4]       # Extract volumes

    # Normalize volume (min-max scaling)
    min_volume = np.min(volumes)
    max_volume = np.max(volumes)
    volume_range = max_volume - min_volume
    if volume_range == 0:
        # E.g., set normalized_volume to zeros (or 1.0, or any constant)
        normalized_volume = np.zeros_like(volumes)
    else:
        normalized_volume = (volumes - min_volume) / volume_range
    
    # Compute RSI for different periods
    rsi_14 = compute_rsi(close_prices, 14)
    rsi_28 = compute_rsi(close_prices, 28)
    rsi_36 = compute_rsi(close_prices, 36)
    rsi_72 = compute_rsi(close_prices, 72)
    rsi_148 = compute_rsi(close_prices, 148)
    rsi_200 = compute_rsi(close_prices, 200)

    for i in range(1, len(data)):
        _, h_prev, l_prev, c_prev, _ = data[i-1]
        o, h, l, c, v = data[i]
        
        # Compute percentage changes and relative differences
        x1 = (c - c_prev) / c_prev
        x2 = (h - h_prev) / h_prev
        x3 = (l - l_prev) / l_prev
        x4 = (h - c) / c if c != 0 else 0
        x5 = (c - l) / c if c != 0 else 0
        
        # Normalized volume
        x6 = normalized_volume[i]
        
        # RSI values
        x7 = rsi_14[i]
        x8 = rsi_28[i]
        x9 = rsi_36[i]
        x10 = rsi_72[i]
        x11 = rsi_148[i]
        x12 = rsi_200[i]
        
        features.append([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12])
    
    return np.array(features)