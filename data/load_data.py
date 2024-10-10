import pandas as pd

datafields = {
        'VN30': {
            'close': 'data/vn_stock/price_volume/close_matrix_top30_20120101-20240101.txt',
            'open': 'data/vn_stock/price_volume/open_matrix_top30_20120101-20240101.txt',
            'high': 'data/vn_stock/price_volume/high_matrix_top30_20120101-20240101.txt',
            'low': 'data/vn_stock/price_volume/low_matrix_top30_20120101-20240101.txt',
            'volume': 'data/vn_stock/price_volume/volume_matrix_top30_20120101-20240101.txt',
            'vwap': 'data/vn_stock/price_volume/vwap_matrix_top30_20120101-20240101.txt',
            'adv20': 'data/vn_stock/price_volume/adv20_matrix_top30_20120101-20240101.txt',
            'adv60': 'data/vn_stock/price_volume/adv60_matrix_top30_20120101-20240101.txt',
            'adv120': 'data/vn_stock/price_volume/adv120_matrix_top30_20120101-20240101.txt',
            'daily_return': 'data/vn_stock/price_volume/daily_return_matrix_top30_20120101-20240101.txt',
            'vn30f1m_open': 'data/vn_stock/price_volume/vn30f1m_open_series_20120101-20240101.txt',
            'vn30f1m_close': 'data/vn_stock/price_volume/vn30f1m_close_series_20120101-20240101.txt',
            'vn30f1m_high': 'data/vn_stock/price_volume/vn30f1m_high_series_20120101-20240101.txt',
            'vn30f1m_low': 'data/vn_stock/price_volume/vn30f1m_low_series_20120101-20240101.txt',
            'vn30f1m_volume': 'data/vn_stock/price_volume/vn30f1m_volume_series_20120101-20240101.txt',
            'close_ridgemodel': 'data/vn_stock/linear_models/ridge_close_top30.txt',
            'open_ridgemodel': 'data/vn_stock/linear_models/ridge_open_top30.txt',
            'high_ridgemodel': 'data/vn_stock/linear_models/ridge_high_top30.txt',
            'low_ridgemodel': 'data/vn_stock/linear_models/ridge_low_top30.txt',
    },
        'VN100': {
            'close': 'data/vn_stock/price_volume/close_matrix_top100_20120101-20240101.txt',
            'open': 'data/vn_stock/price_volume/open_matrix_top100_20120101-20240101.txt',
            'high': 'data/vn_stock/price_volume/high_matrix_top100_20120101-20240101.txt',
            'low': 'data/vn_stock/price_volume/low_matrix_top100_20120101-20240101.txt',
            'volume': 'data/vn_stock/price_volume/volume_matrix_top100_20120101-20240101.txt',
            'vwap': 'data/vn_stock/price_volume/vwap_matrix_top100_20120101-20240101.txt',
            'adv20': 'data/vn_stock/price_volume/adv20_matrix_top100_20120101-20240101.txt',
            'adv60': 'data/vn_stock/price_volume/adv60_matrix_top100_20120101-20240101.txt',
            'adv120': 'data/vn_stock/price_volume/adv120_matrix_top100_20120101-20240101.txt',
            'daily_return': 'data/vn_stock/price_volume/daily_return_matrix_top100_20120101-20240101.txt',
            'vn30f1m_open': 'data/vn_stock/price_volume/vn30f1m_open_series_20120101-20240101.txt',
            'vn30f1m_close': 'data/vn_stock/price_volume/vn30f1m_close_series_20120101-20240101.txt',
            'vn30f1m_high': 'data/vn_stock/price_volume/vn30f1m_high_series_20120101-20240101.txt',
            'vn30f1m_low': 'data/vn_stock/price_volume/vn30f1m_low_series_20120101-20240101.txt',
            'vn30f1m_volume': 'data/vn_stock/price_volume/vn30f1m_volume_series_20120101-20240101.txt',
            'close_ridgemodel': 'data/vn_stock/linear_models/ridge_close_top100.txt',
            'open_ridgemodel': 'data/vn_stock/linear_models/ridge_open_top100.txt',
            'high_ridgemodel': 'data/vn_stock/linear_models/ridge_high_top100.txt',
            'low_ridgemodel': 'data/vn_stock/linear_models/ridge_low_top100.txt',
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
            'vn30f1m_open': 'data/vn_stock/price_volume/vn30f1m_open_series_20120101-20240101.txt',
            'vn30f1m_close': 'data/vn_stock/price_volume/vn30f1m_close_series_20120101-20240101.txt',
            'vn30f1m_high': 'data/vn_stock/price_volume/vn30f1m_high_series_20120101-20240101.txt',
            'vn30f1m_low': 'data/vn_stock/price_volume/vn30f1m_low_series_20120101-20240101.txt',
            'vn30f1m_volume': 'data/vn_stock/price_volume/vn30f1m_volume_series_20120101-20240101.txt',
    },
        'US1000': {
    },
}

def load_and_process_data(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
        df = df.apply(pd.to_numeric, errors='coerce')

        if df.isnull().any().any():
            print(f"Warning: {file_path} contains missing values.")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

dataframes = {}

for universe, paths in datafields.items():
    dataframes[universe] = {field: load_and_process_data(path) for field, path in paths.items()}