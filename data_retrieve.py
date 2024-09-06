from typing import Optional
import pandas as pd
from vnstock3 import Vnstock

class DataRetrieval:
    def __init__(
            self,
            symbol: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            provider: str = "vnstock3"

    ):
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.provider = provider

    def load_data(self, interval) -> pd.DataFrame:
        if self.provider == "vnstock3":
            stock = Vnstock().stock(symbol = self.symbol, source='VCI')
            df = stock.quote.history(start = self.start_date, end = self.end_date, interval = interval)
            return df
        else:
            raise ValueError("Provider not supported")
        
    def load_data_from_csv(self, filepath) -> pd.DataFrame:
        return pd.read_csv(filepath, index_col = 'date', parse_dates = True)
        