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
    symbol = 'HPG'
    start_date = '2020-01-01'
    end_date = '2020-12-31'

    data = DataRetrieval(symbol, start_date, end_date).load_data(interval = '1D')

    strategy = Strategy(
        signal_logic=custom_signal_logic,
        position_logic=custom_position_logic 
    )

    data = strategy.generate_signals(data)
    print(data)
    backtester = Backtester()
    backtester.backtest(data)
    backtester.performance()

if __name__ == "__main__":
    main()