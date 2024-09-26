# backend.py
import pandas as pd
import numpy as np

def delay(alpha, delay):
    return alpha.shift(delay)
    
def truncation(alpha, percentage):
    alpha_normalized = alpha.div(alpha.sum(axis=1), axis=0)
    alpha_truncated = alpha_normalized.clip(lower=0, upper=percentage)
    alpha_truncated = alpha_truncated.div(alpha_truncated.sum(axis=1), axis=0)
    alpha = alpha_truncated
    return alpha

def decay_linear(alpha, n):
    if n > alpha.shape[0]:
        raise ValueError("n must be less than or equal to the number of rows in the DataFrame.")
    
    weights = np.arange(1, n + 1)
    total_weight = weights.sum()
    
    alpha_decay = pd.DataFrame(index=alpha.index, columns=alpha.columns)
    
    for col in alpha.columns:
        alpha_decay[col] = (
            alpha[col].rolling(window=n)
            .apply(lambda x: np.dot(x, weights) / total_weight, raw=True)
        )
    
    return alpha_decay

def neutralization(alpha, method, region):
    if region == 'US':
        if method == "Market":
            alpha = alpha - alpha.mean()
        if method == "Sector":
            alpha = alpha
        if method == "Industry":
            alpha = alpha
        if method == 'Sub-Industry':
            alpha = alpha
        else:
            alpha = alpha
        return alpha
    if region == 'VN':
        if method == "Market":
            alpha = alpha
        if method == "Sector":
            alpha = alpha
        if method == "Industry":
            alpha = alpha
        if method == 'Sub-Industry':
            alpha = alpha
        else:
            alpha = alpha
        return alpha
    
def simulation_results(alpha, settings):
    try:
        universe = settings['universe']
        variables = dataframes[universe]
        x = eval(alpha, {"__builtins__": None}, variables)
        
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)

        if 'delay' in settings:
            x = delay(x, settings['delay'])
        else:
            raise KeyError("The 'delay' key is missing in settings.")


        if 'decay' in settings:
            decay_days = int(settings['decay'])
            result = decay_linear(x, decay_days)
        else:
            result = x
        
        if 'neutralization' in settings:
            result = neutralization(result, settings['neutralization'], settings['region'])
        
        if 'truncation' in settings:
            truncation_percentage = float(settings['truncation'])
            result = truncation(result, truncation_percentage)   
        return result
    except Exception as e:
        st.error(f"An error occurred: {e}")

datafields = {
        'VN30': {
            'close': 'data/vn_stock/price_volume/close_matrix_top30_20120101-20240101.txt',
            'open': 'data/vn_stock/price_volume/open_matrix_top30_20120101-20240101.txt',
            'high': 'data/vn_stock/price_volume/high_matrix_top30_20120101-20240101.txt',
            'low': 'data/vn_stock/price_volume/low_matrix_top30_20120101-20240101.txt',
            'volume': 'data/vn_stock/price_volume/volume_matrix_top30_20120101-20240101.txt',
    },
        'VN100': {
            'close': 'data/vn_stock/price_volume/close_matrix_top100_20120101-20240101.txt',
            'open': 'data/vn_stock/price_volume/open_matrix_top100_20120101-20240101.txt',
            'high': 'data/vn_stock/price_volume/high_matrix_top100_20120101-20240101.txt',
            'low': 'data/vn_stock/price_volume/low_matrix_top100_20120101-20240101.txt',
            'volume': 'data/vn_stock/price_volume/volume_matrix_top100_20120101-20240101.txt',
    },
        'VNALL': {
            'close': 'data/vn_stock/price_volume/close_matrix_20120101-20240101.txt',
            'open': 'data/vn_stock/price_volume/open_matrix_20120101-20240101.txt',
            'high': 'data/vn_stock/price_volume/high_matrix_20120101-20240101.txt',
            'low': 'data/vn_stock/price_volume/low_matrix_20120101-20240101.txt',
            'volume': 'data/vn_stock/price_volume/volume_matrix_20120101-20240101.txt',
            'vwap': 'data/vn_stock/price_volume/vwap_matrix_20120101-20240101.txt',
            'adv20': 'data/vn_stock/price_volume/adv20_matrix_20120101-20240101.txt',
            'adv60': 'data/vn_stock/price_volume/adv60_matrix_20120101-20240101.txt',
            'adv120': 'data/vn_stock/price_volume/adv120_matrix_20120101-20240101.txt',
            'daily_return': 'data/vn_stock/price_volume/daily_return_matrix_20120101-20240101.txt',
    },
        'US1000': {
    },
}

def load_and_process_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df.set_index('time', inplace=True)
    df = df.astype(float)
    return df

dataframes = {}

for universe, paths in datafields.items():
    dataframes[universe] = {field: load_and_process_data(path) for field, path in paths.items()}
