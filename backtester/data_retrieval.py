from typing import Optional, List, Dict
import pandas as pd
from vnstock3 import Vnstock

class DataRetrieval:
    def __init__(
            self,
            symbols: List[str],
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            provider: str = "vnstock3"

    ):
        self.symbols = [symbol.upper() for symbol in symbols]
        self.start_date = start_date
        self.end_date = end_date
        self.provider = provider

    def load_data(self, interval) -> Dict[str, pd.DataFrame]:
        data = {}
        if self.provider == "vnstock3":
            for symbol in self.symbols:
                stock = Vnstock().stock(symbol = symbol, source='VCI')
                df = stock.quote.history(start = self.start_date, end = self.end_date, interval = interval)
                data[symbol] = df
            return data
        else:
            raise ValueError("Provider not supported")
        
    def load_data_from_csv(self, filepath) -> pd.DataFrame:
        return pd.read_csv(filepath, index_col = 'date', parse_dates = True)
        