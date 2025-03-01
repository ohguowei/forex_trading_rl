# config.py

class TradingConfig:
    # You can have separate values for live and simulated environments if needed.
    INSTRUMENT = "EUR_USD"
    LIVE_UNITS = 2250
    SIMULATED_UNITS = 2500
    GRANULARITY = "H1"
    CANDLE_COUNT = 5000
    SPREAD = 0.0002

class OandaConfig:
    ACCOUNT_ID = "001-003-255162-005"
    ACCESS_TOKEN = "c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540"
    ENVIRONMENT = "live"
