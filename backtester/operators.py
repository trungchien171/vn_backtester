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

def fraction(inp_1: pd.DataFrame, inp_2: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp_1, pd.DataFrame) or not isinstance(inp_2, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")
    if inp_1.shape != inp_2.shape:
        raise ValueError("Both inputs must have the same shape.")
    
    division = inp_1 / inp_2
    out = out - out.astype(int)
    return out

def log(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    out = np.log(inp)
    return out

def min(inp: pd.DataFrame, *args: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("The inp argument must be a pandas DataFrame.")
    
    out = inp.copy()
    
    for arg in args:
        if not isinstance(arg, pd.DataFrame):
            raise ValueError("All additional arguments must be pandas DataFrames.")
        if arg.shape != inp.shape:
            raise ValueError("All DataFrames must have the same shape.")
        
        out = out.where(out < arg, arg)
    
    return out

def max(inp: pd.DataFrame, *args: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("The inp argument must be a pandas DataFrame.")
    
    out = inp.copy()
    
    for arg in args:
        if not isinstance(arg, pd.DataFrame):
            raise ValueError("All additional arguments must be pandas DataFrames.")
        if arg.shape != inp.shape:
            raise ValueError("All DataFrames must have the same shape.")
        
        out = out.where(out > arg, arg)
    
    return out

def inverse(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    out = 1 / inp
    return out

def exp(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    out = np.exp(inp)
    return out

def mul(inp: pd.DataFrame, *args: pd.DataFrame, filter: bool = False) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("The inp argument must be a pandas DataFrame.")
    
    out = inp.copy()

    if filter:
        out.fillna(1, inplace=True)
    
    for arg in args:
        if not isinstance(arg, pd.DataFrame):
            raise ValueError("All additional arguments must be pandas DataFrames.")
        if arg.shape != inp.shape:
            raise ValueError("All DataFrames must have the same shape.")
        
        out *= arg
    
    return out

def nan_mask(inp1: pd.DataFrame, inp2: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp1, pd.DataFrame) or not isinstance(inp2, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")
    if inp1.shape != inp2.shape:
        raise ValueError("Both inputs must have the same shape.")
    
    out = inp1.copy()
    out[inp2 < 0] = np.nan
    return out

def nan_out(inp: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if lower > upper:
        raise ValueError("Lower bound must be less than or equal to the upper bound.")
    
    out = inp.copy()
    out[(inp < lower) | (inp > upper)] = np.nan
    return out

def power(inp: pd.DataFrame, power: float) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not isinstance(power, (int, float)):
        raise ValueError("The exponent must be a numeric value (int or float).")
    
    out = inp ** power
    return out

def purify(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = inp.replace([np.inf, -np.inf], np.nan)
    return out

def replace(inp: pd.DataFrame, target: pd.Series, dest: pd.Series) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not isinstance(target, pd.Series) or not isinstance(dest, pd.Series):
        raise ValueError("Both target and destination must be pandas Series.")
    if len(target) != len(dest):
        raise ValueError("Both target and destination must have the same length.")
    
    mapping = dict(zip(target, dest))
    out = inp.replace(mapping)
    return out

def round_df(inp: pd.DataFrame, decimals: int=0) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("Decimals must be a non-negative integer.")
    
    out = inp.round(decimals)
    return out

def round_down(inp: pd.DataFrame, num: int=None) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if num is None:
        out = np.floor(inp).astype(int)
    else:
        if not isinstance(num, int) or num <= 0:
            raise ValueError("Number must be a positive integer.")
        out = np.floor(inp / num) * num
    return out

def s_log_1p(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = np.sign(inp) * np.log1p(np.abs(inp))
    return out

def sign(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = np.sign(inp)
    return out

def signed_power(inp: pd.DataFrame, power: float) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not isinstance(power, (int, float)):
        raise ValueError("The exponent must be a numeric value (int or float).")
    
    out = np.sign(inp) * np.abs(inp) ** power
    return out

def sqrt(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = np.sqrt(inp)
    return out

def subtract(inp1: pd.DataFrame, inp2: pd.DataFrame, filter: bool = False) -> pd.DataFrame:
    if not isinstance(inp1, pd.DataFrame) or not isinstance(inp2, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")
    if inp1.shape != inp2.shape:
        raise ValueError("Both inputs must have the same shape.")
    if filter:
        inp1 = inp1.fillna(0)
        inp2 = inp2.fillna(0)
    out = inp1 - inp2
    return out

def to_nan(inp: pd.DataFrame, value: float=0, reverse: bool=False) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    if reverse:
        out = inp.where(inp == value, np.nan)
    else:
        out = inp.replace(value, np.nan)
    
    return out

def clip_with_log(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = np.sign(inp) * np.log1p(np.abs(inp))
    return out

def reverse(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = -inp
    return out

def abs(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = np.abs(inp)
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
    'log_diff': log_diff,
    'ceil': ceil,
    'floor': floor,
    'divide': divide,
    'fraction': fraction,
    'log': log,
    'min': min,
    'max': max,
    'inverse': inverse,
    'exp': exp,
    'mul': mul,
    'nan_mask': nan_mask,
    'nan_out': nan_out,
    'power': power,
    'purify': purify,
    'replace': replace,
    'round_df': round_df,
    'round_down': round_down,
    's_log_1p': s_log_1p,
    'sign': sign,
    'signed_power': signed_power,
    'sqrt': sqrt,
    'subtract': subtract,
    'to_nan': to_nan,
    'clip_with_log': clip_with_log,
    'reverse': reverse,
    'abs': abs,
    'ts_mean': ts_mean,
    'ts_sum': ts_sum,
    'ts_stddev': ts_stddev,
    'ts_delta': ts_delta,
    'rank': rank
}
