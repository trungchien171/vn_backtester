# backend.py
import pandas as pd
from vnstock3 import Vnstock

class DataRetrieval:
    def __init__(self, symbols, start_date=None, end_date=None, provider="vnstock3", truncation=None):
        self.symbols = [symbol.upper() for symbol in symbols]
        self.start_date = start_date
        self.end_date = end_date
        self.provider = provider
        self.truncation = truncation

    def load_data(self, interval) -> dict:
        data = {}
        if self.provider == "vnstock3":
            for symbol in self.symbols:
                stock = Vnstock().stock(symbol=symbol, source='VCI')
                df = stock.quote.history(start=self.start_date, end=self.end_date, interval=interval)
                data[symbol] = df
            return data
        else:
            raise ValueError("Provider not supported")

    def load_data_from_csv(self, filepath) -> pd.DataFrame:
        return pd.read_csv(filepath, index_col='date', parse_dates=True)