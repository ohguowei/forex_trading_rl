# config.py
# Base trading configuration
class TradingConfig:
    GRANULARITY = "H1"
    CANDLE_COUNT = 5000

class CurrencyConfig:
    def __init__(self, instrument, live_units, simulated_units, spread, account_id, access_token, environment):
        self.instrument = instrument
        self.live_units = live_units
        self.simulated_units = simulated_units
        self.spread = spread
        self.account_id = account_id
        self.access_token = access_token
        self.environment = environment  # e.g., "live" or "practice"

CURRENCY_CONFIGS = {
    "EUR_USD": CurrencyConfig(
        instrument="EUR_USD",
        live_units=2250,
        simulated_units=2250,
        spread=0.0002,
        account_id="001-003-255162-005",
        access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
        environment="live"
    ),
    "AUD_USD": CurrencyConfig(
        instrument="AUD_USD",
        live_units=3600,
        simulated_units=3600,
        spread=0.0003,
        account_id="001-003-255162-004",
        access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
        environment="live"
    ),
    "USD_CHF": CurrencyConfig(
        instrument="USD_CHF",
        live_units=2250,
        simulated_units=2250,
        spread=0.0003,
        account_id="101-001-26348919-001",
        access_token="68ff286dfb6bc058031e66ddcdc72d64-138d97e64d2976820a19a4b179cdcf09",
        environment="practice"
        
       #account_id="001-003-255162-002",
       #access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
       #environment="live"
    ),
    "EUR_AUD": CurrencyConfig(
        instrument="EUR_AUD",
        live_units=1000,
        simulated_units=1000,
        spread=0.0004,
        account_id="101-001-26348919-001",
        access_token="68ff286dfb6bc058031e66ddcdc72d64-138d97e64d2976820a19a4b179cdcf09",
        environment="practice"
        
       #account_id="001-003-255162-002",
       #access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
       #environment="live"
    ),
    "EUR_CHF": CurrencyConfig(
        instrument="EUR_CHF",
        live_units=1000,
        simulated_units=1000,
        spread=0.0003,
        account_id="101-001-26348919-001",
        access_token="68ff286dfb6bc058031e66ddcdc72d64-138d97e64d2976820a19a4b179cdcf09",
        environment="practice"
        
       #account_id="001-003-255162-002",
       #access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
       #environment="live"
    ),
    "AUD_CHF": CurrencyConfig(
        instrument="AUD_CHF",
        live_units=1000,
        simulated_units=1000,
        spread=0.0002,
        account_id="101-001-26348919-001",
        access_token="68ff286dfb6bc058031e66ddcdc72d64-138d97e64d2976820a19a4b179cdcf09",
        environment="practice"
        
       #account_id="001-003-255162-002",
       #access_token="c33734921cd0b7b68c721fc18e2019c2-8cfd11c75b7df0c81301e2cf58846540",
       #environment="live"
    ),    
    # Add more currencies as needed...
}
