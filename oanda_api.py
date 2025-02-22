# oanda_api.py

import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders

# Set up your credentials and environment.
ACCOUNT_ID = "101-001-26348919-001"
ACCESS_TOKEN = "68ff286dfb6bc058031e66ddcdc72d64-138d97e64d2976820a19a4b179cdcf09"
ENVIRONMENT = "practice"  # Use "live" for real trading


# Create a shared API client.
client = API(access_token=ACCESS_TOKEN, environment=ENVIRONMENT)

def fetch_candle_data(instrument, granularity="H1", candle_count=500):
    """
    Fetch historical candlestick data for a given instrument.
    Returns a list of [open, high, low, close] values.
    """
    params = {
        "granularity": granularity,
        "count": candle_count,
        "price": "M"  # using mid prices
    }
    r = instruments.InstrumentsCandles(instrument, params=params)
    response = client.request(r)
    candles = response["candles"]
    data = []
    for candle in candles:
        mid = candle["mid"]
        o = float(mid["o"])
        h = float(mid["h"])
        l = float(mid["l"])
        c = float(mid["c"])
        data.append([o, h, l, c])
    return data

def open_position(account_id, instrument, units, side):
    """
    Open a market order position.
    `side` should be "long" or "short".
    """
    order_data = {
        "order": {
            "units": str(units if side == "long" else -units),
            "instrument": instrument,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }
    r = orders.OrderCreate(account_id, data=order_data)
    try:
        response = client.request(r)
        print(f"Opened {side} position on {instrument}: {response}")
        return response
    except Exception as e:
        print(f"Order failed: {e}")
        return None

def close_position(account_id, instrument):
    """
    Close the current position.
    (In production, replace this with an appropriate API call.)
    """
    print(f"Closing current position on {instrument}.")
    # For demonstration, this function only prints a message.
    return None
