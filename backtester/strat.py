from typing import Any, Union, Callable, Dict
import pandas as pd

class Strategy:
    def __init__(
        self,
        signal_logic: Callable[[pd.Series], int] = None,                     
        position_logic: Callable[[pd.DataFrame, pd.Series], pd.Series] = None,
    ):
        self.signal_logic = signal_logic or self.default_signal_logic
        self.position_logic = position_logic or self.default_position_logic
    
    def generate_signals(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if isinstance(data, dict):
            for asset_name, asset_data in data.items():
                data[asset_name] = self.apply_strategy(asset_data)
        else:
            data = self.apply_strategy(data)
        return data
    
    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        
        df['signal'] = df.apply(lambda row: self.signal_logic(row), axis=1)
        df['position'] = self.position_logic(df, df['signal'])
        
        return df

    @staticmethod
    def default_signal_logic(row: pd.Series) -> int:
        close_price = row['close']
        if close_price > 100:
            return 1
        elif close_price < 100:
            return -1
        return 0

    @staticmethod
    def default_position_logic(df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        return signals.diff().fillna(0)
