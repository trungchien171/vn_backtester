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