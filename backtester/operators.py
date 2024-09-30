# operators.py
import pandas as pd
import numpy as np
import statsmodels.api as sm

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
    out = division - division.astype(int)
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
def cs_rank(inp: pd.DataFrame, ascending: bool=True, na_option='keep', method='first') -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    out = inp.rank(axis=1, ascending=ascending, na_option=na_option, method=method)
    return out

def one_side(inp: pd.DataFrame, side: str = 'long') -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if side not in ['long', 'short']:
        raise ValueError("Side must be either 'long' or 'short'.")
    
    out = inp.copy()
    if side == 'long':
        out[out < 0] = 0
    else:
        out[out > 0] = 0
    return out

def cs_vector_neut(inp1: pd.DataFrame, inp2: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp1, pd.DataFrame) or not isinstance(inp2, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")
    
    if inp1.shape != inp2.shape:
        raise ValueError(f"Both inputs must have the same shape. Got {inp1.shape} and {inp2.shape}.")

    out = pd.DataFrame(index=inp1.index, columns=inp1.columns)
    
    for index in inp1.index:
        row_inp1 = inp1.loc[index].values
        row_inp2 = inp2.loc[index].values
        
        row_inp2_with_const = sm.add_constant(row_inp2)
        
        model = sm.OLS(row_inp1, row_inp2_with_const)
        outs = model.fit()
        
        predicted = outs.predict(row_inp2_with_const)
        
        neutralized_row = row_inp1 - predicted
        
        out.loc[index] = neutralized_row

    return out

operators = {
    'Arithmetic Operators': {
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
        'power': power,
        'round_df': round_df,
        'round_down': round_down,
        'sign': sign,
        'signed_power': signed_power,
        'sqrt': sqrt,
        'subtract': subtract,
        'abs': abs,
        'clip_with_log': clip_with_log,
        'reverse': reverse,
        'purify': purify,
        'replace': replace,
        'nan_mask': nan_mask,
        'nan_out': nan_out,
        'to_nan': to_nan
    },
    'Time Series Operators': {
        'ts_mean': ts_mean,
        'ts_sum': ts_sum,
        'ts_stddev': ts_stddev,
        'ts_delta': ts_delta
    },
    'Cross Sectional Operators': {
        'cs_rank': cs_rank,
        'one_side': one_side,
        'cs_vector_neut': cs_vector_neut
    },
}