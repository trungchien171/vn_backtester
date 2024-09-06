from typing import Any, Union, Callable, Dict
import pandas as pd

class Strategy:
    def __init__(
        self,
        signal_logic: Callable[[pd.Series], int] = None,                     # Custom signal generation logic
        position_logic: Callable[[pd.DataFrame, pd.Series], pd.Series] = None # Custom position logic
    ):
        """
        :param signal_logic: A function that defines how to generate signals (e.g., buy, sell, hold).
        :param position_logic: A function that defines how to generate positions based on signals.
        """
        self.signal_logic = signal_logic or self.default_signal_logic
        self.position_logic = position_logic or self.default_position_logic
    
    def generate_signals(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Generate signals and positions for a given dataset.
        :param data: A DataFrame (single asset) or dictionary of DataFrames (multiple assets).
        :return: DataFrame with signals and positions, or a dictionary of DataFrames.
        """
        if isinstance(data, dict):
            for asset_name, asset_data in data.items():
                data[asset_name] = self.apply_strategy(asset_data)
        else:
            data = self.apply_strategy(data)
        return data
    
    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the strategy to a given DataFrame.
        :param df: A DataFrame with price data containing 'time', 'open', 'high', 'low', 'close', 'volume'.
        :return: DataFrame with signals and positions.
        """
        # Ensure necessary columns exist
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
        
        # Generate signals using user-defined or default logic
        df['Signal'] = df.apply(lambda row: self.signal_logic(row), axis=1)
        
        # Generate positions using user-defined or default logic
        df['Position'] = self.position_logic(df, df['Signal'])
        
        return df

    @staticmethod
    def default_signal_logic(row: pd.Series) -> int:
        """
        Default signal logic (can be overridden by the user).
        :param row: A row of the DataFrame (price data).
        :return: An integer signal (1 for buy, -1 for sell, 0 for hold).
        """
        close_price = row['close']
        if close_price > 100:  # Arbitrary example condition
            return 1  # Buy
        elif close_price < 100:
            return -1  # Sell
        return 0  # Hold

    @staticmethod
    def default_position_logic(df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Default position logic: creates positions based on signal changes.
        :param df: DataFrame with price data and signals.
        :param signals: Series of buy/sell signals (1 for buy, -1 for sell, 0 for hold).
        :return: Series representing positions (buy/sell orders).
        """
        return signals.diff().fillna(0)  # Buy/sell when signal changes

# # Example usage of the Strategy class
# if __name__ == "__main__":
#     # Mock data with columns: 'time', 'open', 'high', 'low', 'close', 'volume'
#     data = pd.DataFrame({
#         'time': pd.date_range(start='2023-01-01', periods=7, freq='D'),
#         'open': [100, 101, 102, 103, 104, 105, 106],
#         'high': [101, 102, 103, 104, 105, 106, 107],
#         'low': [99, 100, 101, 102, 103, 104, 105],
#         'close': [100, 102, 101, 104, 103, 105, 106],
#         'volume': [1000, 1100, 1050, 1150, 1200, 1250, 1300]
#     })
    
#     # Custom signal logic: Buy if close price is greater than 102, sell if less than 102
#     def custom_signal_logic(row: pd.Series) -> int:
#         if row['close'] > 102:
#             return 1  # Buy
#         elif row['close'] < 102:
#             return -1  # Sell
#         return 0  # Hold

#     # Custom position logic: Enter/exit trades based on signal changes
#     def custom_position_logic(df: pd.DataFrame, signals: pd.Series) -> pd.Series:
#         return signals.diff().fillna(0)

#     # Instantiate the strategy with custom logic
#     strategy = Strategy(
#         signal_logic=custom_signal_logic,  # Custom signal generation logic
#         position_logic=custom_position_logic  # Custom position logic
#     )

#     # Generate signals and positions
#     result = strategy.generate_signals(data)
#     print(result)
