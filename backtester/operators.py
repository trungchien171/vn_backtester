# operators.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import gmean
import scipy.stats as stats

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

def cs_vector_proj(inp1: pd.DataFrame, inp2: pd.DataFrame) -> pd.DataFrame:
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
        
        out.loc[index] = predicted

    return out

def cs_winsorize(inp: pd.DataFrame, window: int=4) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = pd.DataFrame(index=inp.index, columns=inp.columns)
    for index in inp.index:
        row = inp.loc[index]
        mean_row = row.mean()
        std_row = row.std()

        lower_bound = mean_row - window * std_row
        upper_bound = mean_row + window * std_row

        row_winsorization = np.clip(row, lower_bound, upper_bound)
        out.loc[index] = row_winsorization
    return out

def cs_zscore(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = (inp - inp.mean()) / inp.std()
    return out

def cs_scale_down(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = pd.DataFrame(index=inp.index, columns=inp.columns)
    for index in inp.index:
        row = inp.loc[index]

        min_val = row.min()
        max_val = row.max()

        if max_val != min_val:
            row_scaled = (row - min_val) / (max_val - min_val)
        else:
            row_scaled = row
        
        out.loc[index] = row_scaled
    return out

def cs_rank_by_side(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = pd.DataFrame(index=inp.index, columns=inp.columns)

    for index in inp.index:
        row = inp.loc[index]

        positive_vals = row[row > 0]
        positive_ranks = positive_vals.rank(ascending=True, method='min')

        negative_vals = row[row < 0]
        negative_ranks = negative_vals.rank(ascending=True, method='min')

        ranks = pd.concat([positive_ranks, negative_ranks])
        out.loc[index] = ranks.reindex(inp.columns)
    return out

def cs_truncate(inp: pd.DataFrame, max_percent: float=0.01, keep_greater: bool=True) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = inp.copy()

    for index in inp.index:
        row = inp.loc[index]

        quantile_val = row.quantile(1 - max_percent) if keep_greater else row.quantile(max_percent)

        if keep_greater:
            out.loc[index] = row.clip(lower=quantile_val)
        else:
            out.loc[index] = row.clip(upper=quantile_val)
    return out

def cs_quantile(inp: pd.DataFrame, driver: str ='gaussian', sigma: float=1.0) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    driver_map = {
        'gaussian': (stats.norm.ppf, {}),
        'uniform': (stats.uniform.ppf, {}),
        'student_t': (stats.t.ppf, {'df': 10}),
        'chi_squared': (stats.chi2.ppf, {'df': 2}),
        'exponential': (stats.expon.ppf, {}),
        'logistic': (stats.logistic.ppf, {}),
        'lognormal': (stats.lognorm.ppf, {'s': 0.954}),
        'pareto': (stats.pareto.ppf, {'b': 2.62}),
        'rayleigh': (stats.rayleigh.ppf, {}),
        'triangular': (stats.triang.ppf, {'c': 0.5}),
        'weibull': (stats.weibull_min.ppf, {'c': 1.5}),
        'cauchy': (stats.cauchy.ppf, {}),
    }

    if driver not in driver_map:
        raise ValueError(f"Invalid driver. Must be one of {list(driver_map.keys())}.")
    
    icdf, params = driver_map[driver]
    out = pd.DataFrame(index=inp.index, columns=inp.columns)

    for index in inp.index:
        row = inp.loc[index]
        ranks = row.rank(pct=True).fillna(0.5)
        out.loc[index] = icdf(ranks, scale=sigma, **params)
    return out

def cs_normalize(inp: pd.DataFrame, std_bound: float=3, linear_normalize: bool=False, std_normalize: bool=False) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = inp.copy()
    
    if std_bound != 0:
        out = cs_winsorize(inp, window=std_bound)

    if linear_normalize:
        min_vals = inp.min(axis=1)
        max_vals = inp.max(axis=1)

        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1

        out = (out.sub(min_vals, axis=0)).div(range_vals, axis=0)

    elif std_normalize:
        mean_vals = inp.mean(axis=1)
        std_vals = inp.std(axis=1)

        std_vals[std_vals == 0] = 1

        out = (inp.sub(mean_vals, axis=0)).div(std_vals, axis=0)

    return out

def cs_market_neutralize(inp: pd.DataFrame, mask: pd.DataFrame = None) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    if mask is not None:
        inp = inp.where(mask.notnull(), other = 0)
    
    row_means = inp.mean(axis=1)
    out = inp.sub(row_means, axis=0)
    return out

# def cs_rank_gmean_amean_diff(inp: pd.DataFrame, *args) -> pd.DataFrame:
#     if not isinstance(inp, pd.DataFrame):
#         raise ValueError("Input must be a pandas DataFrame.")
#     out = pd.DataFrame()

#     for column in args:
#         if column in inp.columns:
#             ranked_values = inp[column].rank()

#             amean = ranked_values.mean()

#             ranked_values_positive = ranked_values[ranked_values > 0]
#             gmean_value = gmean(ranked_values_positive)

#             diff = gmean_value - amean

#             out[column] = [diff]
#         else:
#             raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
#     return out

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
        'cs_vector_neut': cs_vector_neut,
        'cs_vector_proj': cs_vector_proj,
        'cs_winsorize': cs_winsorize,
        'cs_zscore': cs_zscore,
        'cs_scale_down': cs_scale_down,
        'cs_rank_by_side': cs_rank_by_side,
        'cs_truncate': cs_truncate,
        'cs_quantile': cs_quantile,
        'cs_normalize': cs_normalize,
        'cs_market_neutralize': cs_market_neutralize,
        # 'cs_rank_gmean_amean_diff': cs_rank_gmean_amean_diff
    },
}