import pandas as pd
import numpy as np

# Arithmetic Operators
def log_diff(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    inp = inp.where(inp > 0, np.nan)
    out = np.log(inp).diff()
    return out

def ceil(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    out = np.ceil(inp)
    return out

def floor(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    out = np.floor(inp)
    return out

def divide(inp_1: pd.DataFrame, inp_2: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp_1, pd.DataFrame) or not isinstance(inp_2, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")
    if inp_1.shape != inp_2.shape:
        raise ValueError("Both inputs must have the same shape.")
    
    inp_2_transform = inp_2.replace(0, np.nan)
    out = inp_1 / inp_2_transform
    return out

# Time Series Operators
def ts_mean(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not isinstance(window, int) or window < 0:
        raise ValueError("Window must be a non-negative integer.")
    out = inp.rolling(window=window).mean()
    return out

def ts_sum(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not isinstance(window, int) or window < 0:
        raise ValueError("Window must be a non-negative integer.")
    out = inp.rolling(window=window).sum()
    return out

def ts_stddev(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not isinstance(window, int) or window < 0:
        raise ValueError("Window must be a non-negative integer.")
    out = inp.rolling(window=window).std()
    return out

def ts_delta(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not isinstance(window, int) or window < 0:
        raise ValueError("Window must be a non-negative integer.")
    out = inp.diff(window)
    return out

# Cross Sectional Operators
def rank(inp: pd.DataFrame, rate: int) -> pd.DataFrame:
    if isinstance(inp, pd.DataFrame):
        return inp.apply(lambda col: (col.rank(method = 'min') - 1) / (len(col) - 1) if len(col) > 1 else 0, axis=1)
    elif isinstance(inp, pd.Series):
        return (inp.rank(method = 'min') - 1) / (len(inp) - 1) if len(inp) > 1 else 0
    else:
        raise ValueError("Input must be a pandas DataFrame or Series.")

operators = {
    'ts_mean': ts_mean,
    'ts_delta': ts_delta,
    'rank': rank,
    'ts_sum': ts_sum,
    'ts_stddev': ts_stddev,
    'log_diff': log_diff,
    'ceil': ceil,
    'floor': floor,
    'divide': divide,
}
