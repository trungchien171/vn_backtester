from backtester.data_retrieval import DataRetrieval
from backtester.strat import Strategy
from backtester.backtest import Backtester
import pandas as pd

def custom_signal_logic(row: pd.Series) -> int:
    if row['close'] > row['open']:
        return 1
    elif row['close'] < row['open']:
        return -1
    return 0

def custom_position_logic(df: pd.DataFrame, signals: pd.Series) -> pd.Series:
    return signals.diff().fillna(0)

def main():
    symbol = ['HQC', 'DLG', 'NVL', 'HAG', 'DXG']
    start_date = '2018-01-01'
    end_date = '2022-12-31'
    # start_date = '2023-01-01'
    # end_date = '2024-09-10

    data = DataRetrieval(symbol, start_date, end_date).load_data(interval = '1D')
    weights = {'HQC': 0.2, 'DLG': 0.2, 'NVL': 0.2, 'HAG': 0.2, 'DXG': 0.2}

    strategy = Strategy(
        signal_logic=custom_signal_logic,
        position_logic=custom_position_logic,
    )

    data = strategy.generate_signals(data)
    backtester = Backtester(weights=weights)
    backtester.backtest(data)
    backtester.performance()

if __name__ == "__main__":
    main()