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

def cs_rank_gmean_amean_diff(inp: pd.DataFrame, *args) -> pd.DataFrame:
    def gmean(x):
        return np.exp(np.log(x).mean())
    
    ranked = inp.rank(axis=0, method='average')
    if args:
        ranked = ranked[list(args)]
    
    gmean_values = ranked.apply(gmean, axis=0)
    amean_values = ranked.mean(axis=0)
    result = gmean_values - amean_values
    return result

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

def days_from_last_change(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = pd.DataFrame(index=inp.index)

    for col in inp.columns:
        change = inp[col].diff().ne(0)

        days_since_change = []
        days_counter = 0

        for has_changed in change:
            if has_changed:
                days_counter = 0
            else:
                days_counter += 1
            days_since_change.append(days_counter)

        out[col] = days_since_change
    return out

def ts_hump_decay(inp: pd.DataFrame, change: float=0.05, relative: bool=False) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = inp.copy()
    
    for index, row in out.iterrows():
        if relative:
            threshold = change * row.max()
        else:
            threshold = change
    
        for col in out.columns:
            if row[col] >= threshold:
                out.at[index, col] = row[col] - (row[col] - threshold) * 0.5

    return out

def ts_inst_tvr(inp: pd.DataFrame, window: int=5) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = pd.DataFrame(index=inp.index, columns=inp.columns)

    for col in inp.columns:
        total_variation = inp[col].diff().abs().rolling(window=window).sum()
        tvr = total_variation / inp[col].rolling(window=window).mean()

        out[col] = tvr
    return out

def ts_min(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not isinstance(window, int) or window < 0:
        raise ValueError("Window must be a non-negative integer.")
    out = inp.rolling(window=window).min()
    return out

def ts_max(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not isinstance(window, int) or window < 0:
        raise ValueError("Window must be a non-negative integer.")
    out = inp.rolling(window=window).max()
    return out

def ts_argmin(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not isinstance(window, int) or window < 0:
        raise ValueError("Window must be a non-negative integer.")
    out = inp.rolling(window=window).apply(lambda x: x.idxmin(), raw=False)
    return out

def ts_argmax(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not isinstance(window, int) or window < 0:
        raise ValueError("Window must be a non-negative integer.")
    out = inp.rolling(window=window).apply(lambda x: x.idxmax(), raw=False)
    return out

def ts_corr(inp1: pd.DataFrame, inp2: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp1, pd.DataFrame) or not isinstance(inp2, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")
    if inp1.shape != inp2.shape:
        raise ValueError("Both inputs must have the same shape.")
    
    out = inp1.rolling(window=window).corr(inp2)
    return out

def ts_cov(inp1: pd.DataFrame, inp2: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp1, pd.DataFrame) or not isinstance(inp2, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")
    if inp1.shape != inp2.shape:
        raise ValueError("Both inputs must have the same shape.")
    
    out = inp1.rolling(window=window).cov(inp2)
    return out

def ts_rank(inp: pd.DataFrame, window: int, constant: int = 0, ascending: bool=True) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    if window < 1:
        raise ValueError("Window must be at least 1.")
    
    order = 1 if ascending else -1
    out = pd.DataFrame(index=inp.index, columns=inp.columns)

    for col in inp.columns:
        rolling_window = inp[col].rolling(window=window)
        ranks = rolling_window.apply(
            lambda x: np.argsort(order * np.argsort(order * x))[-1] + 1 + constant, raw=False
        )
        out[col] = ranks
    return out

def ts_moment(inp: pd.DataFrame, moment_order: int, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if moment_order < 1:
        raise ValueError("Moment order must be at least 1.")
    if window < 1:
        raise ValueError("Window must be at least 1.")
    
    out = inp.rolling(window=window).apply(lambda x: ((x - x.mean()) ** moment_order).mean(), raw=False)
    return out

def if_else(cond: pd.Series, t: any, f: any) -> pd.Series:
    if not isinstance(cond, pd.Series):
        raise ValueError("Condition must be a pandas Series.")
    
    out = cond.apply(lambda x: t if x else f)
    return out

def ts_delta(inp: pd.DataFrame, time: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if time >= len(inp):
        raise ValueError("Time must be less than the number of rows in the DataFrame.")
    
    out = inp - inp.shift(time)
    return out

def slope(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    out = pd.DataFrame(index=inp.index, columns=inp.columns)
    
    for time in range(1, window + 1):
        delta = ts_delta(inp, time)
        out += delta / time
    return out

def ts_co_skewness(inp1: pd.DataFrame, inp2: pd.DataFrame, window: int=5, variant: str="right") -> pd.DataFrame:
    if not isinstance(inp1, pd.DataFrame) or not isinstance(inp2, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")
    
    out = []

    for i in range(len(inp1)):
        if i < window - 1:
            out.append(None)
            continue

        x_window = inp1.iloc[i - window + 1: i + 1]
        y_window = inp2.iloc[i - window + 1: i + 1]

        mean_x = x_window.mean()
        mean_y = y_window.mean()

        sig_x = x_window.std(ddof=0)
        sig_y = y_window.std(ddof=0)

        if variant == "right":
            co_skewness = ((x_window - mean_x) * (y_window - mean_y) ** 3).mean() / (sig_x * sig_y ** 2)
        elif variant == "left":
            co_skewness = ((x_window - mean_x) ** 2 * (y_window - mean_y)).mean() / (sig_x ** 2 * sig_y)
        else:
            raise ValueError("Variant must be either 'right' or 'left'.")
        
        out.append(co_skewness)
        return pd.DataFrame(out, index=inp1.index)
    
def ts_count_nan(inp: pd.DataFrame, window: int=5) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    out = inp.isna().rolling(window=window).sum()
    return out

def ts_decay_exp_window(inp: pd.DataFrame, window: int=5, factor: float=0.5, nan: bool=True) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    weights = np.array([factor ** i for i in range(window)])
    weights = weights[::-1] / np.sum(weights)
    
    def exp_weighted_sum(arr):
        return np.convolve(arr, weights, mode='valid')
    
    if nan:
        out = inp.apply(lambda x: pd.Series(exp_weighted_sum(x.fillna(0).values), index=x.index[window-1:]), axis=0)
        out = out.reindex_like(inp)
    else:
        out = inp.apply(lambda x: pd.Series(exp_weighted_sum(x.values), index=x.index[window-1:]), axis=0)
        out = out.reindex_like(inp)
    
    return out

def ts_decay_linear(inp: pd.DataFrame, window: int, dense: bool=True) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    if dense:
        weights = np.linspace(0, 1, window)
    else:
        weights = np.linspace(1, 0, window) ** 2

    weights /= np.sum(weights)
    
    def apply_linear_decay(arr):
        return np.convolve(arr, weights, mode='valid')
    
    out = inp.apply(lambda x: pd.Series(apply_linear_decay(x.values), index=x.index[window-1:]), axis=0)
    out = out.reindex_like(inp)
    
    return out

def ts_ir(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    def information_ratio(series):
        if len(series) < window:
            return np.nan
        excess_return = series - series.mean()
        return excess_return.mean() / excess_return.std()
    
    return inp.rolling(window=window, min_periods=window).apply(information_ratio, raw=False)

def ts_kurtosis(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    return inp.rolling(window=window, min_periods=window).kurt()

def ts_mean_diff(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    rolling_mean = inp.rolling(window=window).mean()
    mean_diff = inp - rolling_mean
    return mean_diff

def ts_max_diff(inp: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    rolling_max = inp.rolling(window=window).max()
    max_diff = inp - rolling_max
    return max_diff

def ts_min_diff(inp: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    rolling_min = inp.rolling(window=window).min()
    min_diff = inp - rolling_min
    return min_diff

def ts_partial_corr(x: pd.DataFrame, y: pd.DataFrame, z: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame) or not isinstance(y, pd.DataFrame) or not isinstance(z, pd.DataFrame):
        raise ValueError("All inputs must be pandas DataFrames.")
    def partial_corr(x, y, z):
        x_res = x - np.polyval(np.polyfit(z, x, 1), z)
        y_res = y - np.polyval(np.polyfit(z, y, 1), z)
        return np.corrcoef(x_res, y_res)[0, 1]
    
    results = []
    for i in range(window, len(x) + 1):
        x_window = x.iloc[i-window:i]
        y_window = y.iloc[i-window:i]
        z_window = z.iloc[i-window:i]
        results.append(partial_corr(x_window, y_window, z_window))
    
    return pd.DataFrame(results, index=x.index[window-1:])

def ts_percentage(inp: pd.DataFrame, window: int = 5, percentage: float = 0.1) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    rolling_mean = inp.rolling(window=window).mean()
    percentage_change = (inp - rolling_mean) / rolling_mean * 100
    result = percentage_change[percentage_change.abs() > percentage * 100]
    return result

def ts_product(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.rolling(window=window).apply(lambda x: x.prod(), raw=True)
    return result

def ts_scale(inp: pd.DataFrame, window: int = 5, constant: int = 0) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    rolling_mean = inp.rolling(window=window).mean()
    rolling_std = inp.rolling(window=window).std()
    scaled = (inp - rolling_mean) / rolling_std
    result = scaled + constant
    return result

def ts_skewness(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.rolling(window=window).skew()
    return result

def ts_step(inp: pd.DataFrame, step_size: int = 1) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.applymap(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * step_size
    return result

def ts_triple_corr(x: pd.DataFrame, y: pd.DataFrame, z: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame) or not isinstance(y, pd.DataFrame) or not isinstance(z, pd.DataFrame):
        raise ValueError("All inputs must be pandas DataFrames.")
    x_roll = x.rolling(window=window)
    y_roll = y.rolling(window=window)
    z_roll = z.rolling(window=window)
    
    x_mean = x_roll.mean()
    y_mean = y_roll.mean()
    z_mean = z_roll.mean()
    
    x_dev = x - x_mean
    y_dev = y - y_mean
    z_dev = z - z_mean
    
    numerator = (x_dev * y_dev * z_dev).rolling(window=window).mean()
    
    std_dev_product = (x_dev.rolling(window=window).std() * 
                       y_dev.rolling(window=window).std() * 
                       z_dev.rolling(window=window).std())
    
    triple_corr = numerator / std_dev_product
    
    return triple_corr

def ts_zscore(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    rolling_mean = inp.rolling(window=window).mean()
    rolling_std = inp.rolling(window=window).std()
    zscore = (inp - rolling_mean) / rolling_std
    return zscore

def ts_rank_gmean_amean_diff(window: int, inp: pd.DataFrame, *args) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    def gmean(x):
        return np.exp(np.log(x).mean())
    
    ranked = inp.rank(axis=0, method='average')
    rolling_gmean = ranked.rolling(window=window).apply(gmean, raw=True)
    rolling_amean = ranked.rolling(window=window).mean()
    result = rolling_gmean - rolling_amean
    return result

def ts_quantile(inp: pd.DataFrame, window: int = 10, driver: str = "gaussian", sigma: float = 0.5) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if driver == "gaussian":
        weights = np.exp(-0.5 * (np.arange(window) - (window - 1) / 2) ** 2 / sigma ** 2)
        weights /= weights.sum()
        result = inp.rolling(window=window).apply(lambda x: np.quantile(x, weights=weights), raw=True)
    elif driver == "uniform":
        result = inp.rolling(window=window).quantile(0.5)
    else:
        raise ValueError("Unsupported driver. Use 'gaussian' or 'uniform'.")
    return result

def ts_pct_change(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    pct_change = inp.pct_change()
    result = pct_change.rolling(window=window).std()
    return result

def ts_ln_change(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    pct_change = inp.pct_change()
    ln_change = np.log(pct_change + 1)
    result = ln_change.rolling(window=window).mean()
    return result

def ts_shift(inp: pd.DataFrame, shift: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.shift(periods=shift)
    return result

def ts_diff(inp: pd.DataFrame, time_diff: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.diff(periods=time_diff)
    return result

def ts_median(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.rolling(window=window).median()
    return result

# Logical Operators
def convert_float(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.astype(float)
    return result

def equal(inp1: pd.DataFrame, inp2: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp1, pd.DataFrame) or not isinstance(inp2, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")
    result = inp1.eq(inp2)
    return result

def negate(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp * -1
    return result

def less(inp1: pd.DataFrame, inp2: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp1, pd.DataFrame) or not isinstance(inp2, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")
    result = inp1.lt(inp2)
    return result

def is_not_nan(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.notna()
    return result

def is_nan(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.isna()
    return result

def is_finite(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.applymap(np.isfinite)
    return result

# Mathematical Operators
def arc_cos(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.applymap(np.arccos)
    return result

def arc_sin(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.applymap(np.arcsin)
    return result

def arc_tan(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.applymap(np.arctan)
    return result

def tanh(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.applymap(np.tanh)
    return result

def sin(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.applymap(np.sin)
    return result

def cos(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.applymap(np.cos)
    return result

def sigmoid(inp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.applymap(lambda x: 1 / (1 + np.exp(-x)))
    return result

def left_right_tail(inp: pd.DataFrame, maximum: float = -1 * np.inf, minimum: float = -1 * np.inf) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp.applymap(lambda x: x if minimum < x < maximum else np.nan)
    return result

def clamp(inp: pd.DataFrame, upper: float = 0, lower: float = 0, inverse: bool = False, mask: float = 0.0) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if inverse:
        result = inp.applymap(lambda x: np.nan if lower <= x <= upper else x)
    else:
        result = inp.clip(lower=lower, upper=upper)
    result = result.replace(mask, np.nan)
    return result

def prev_diff_value(inp: pd.DataFrame, window: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    def find_prev_diff(series, window):
        for i in range(1, window + 1):
            if i >= len(series):
                return np.nan
            if series.iloc[-1] != series.iloc[-1 - i]:
                return series.iloc[-1 - i]
        return np.nan
    
    result = inp.apply(lambda x: x.rolling(window=window + 1).apply(lambda y: find_prev_diff(y, window), raw=False))
    return result

def get_df(inp: pd.DataFrame, val: int) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    result = inp[inp == val]
    return result

def keep(inp: pd.DataFrame, f: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    if not isinstance(inp, pd.DataFrame) or not isinstance(f, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")
    mask = f.rolling(window=period, min_periods=1).apply(lambda x: x.any(), raw=True).astype(bool)
    result = inp[mask]
    return result

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
        'ts_delta': ts_delta,
        'days_from_last_change': days_from_last_change,
        'ts_hump_decay': ts_hump_decay,
        'ts_inst_tvr': ts_inst_tvr,
        'ts_min': ts_min,
        'ts_max': ts_max,
        'ts_argmin': ts_argmin,
        'ts_argmax': ts_argmax,
        'ts_corr': ts_corr,
        'ts_cov': ts_cov,
        'ts_rank': ts_rank,
        'ts_moment': ts_moment,
        'if_else': if_else,
        'slope': slope,
        'ts_co_skewness': ts_co_skewness,
        'ts_count_nan': ts_count_nan,
        'ts_decay_exp_window': ts_decay_exp_window,
        'ts_decay_linear': ts_decay_linear,
        'ts_ir': ts_ir,
        'ts_kurtosis': ts_kurtosis,
        'ts_mean_diff': ts_mean_diff,
        'ts_max_diff': ts_max_diff,
        'ts_min_diff': ts_min_diff,
        'ts_partial_corr': ts_partial_corr,
        'ts_percentage': ts_percentage,
        'ts_product': ts_product,
        'ts_scale': ts_scale,
        'ts_skewness': ts_skewness,
        'ts_step': ts_step,
        'ts_triple_corr': ts_triple_corr,
        'ts_zscore': ts_zscore,
        'ts_rank_gmean_amean_diff': ts_rank_gmean_amean_diff,
        'ts_quantile': ts_quantile,
        'ts_pct_change': ts_pct_change,
        'ts_ln_change': ts_ln_change,
        'ts_shift': ts_shift,
        'ts_diff': ts_diff,
        'ts_median': ts_median

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
        'cs_rank_gmean_amean_diff': cs_rank_gmean_amean_diff
    },
    'Logical Operators': {
        'convert_float': convert_float,
        'equal': equal,
        'negate': negate,
        'less': less,
        'is_not_nan': is_not_nan,
        'is_nan': is_nan,
        'is_finite': is_finite
    },
    'Mathematical Operators': {
        'arc_cos': arc_cos,
        'arc_sin': arc_sin,
        'arc_tan': arc_tan,
        'tanh': tanh,
        'sin': sin,
        'cos': cos,
        'sigmoid': sigmoid,
        'left_right_tail': left_right_tail,
        'clamp': clamp,
        'prev_diff_value': prev_diff_value,
        'get_df': get_df,
        'keep': keep
    },
}