# backend.py
import pandas as pd
import numpy as np

__closepath__ = 'data/vn_stock/price_volume/close_matrix_20120101-20240101.txt'
__openpath__ = 'data/vn_stock/price_volume/open_matrix_20120101-20240101.txt'
__highpath__ = 'data/vn_stock/price_volume/high_matrix_20120101-20240101.txt'
__lowpath__ = 'data/vn_stock/price_volume/low_matrix_20120101-20240101.txt'
__volumepath = 'data/vn_stock/price_volume/volume_matrix_20120101-20240101.txt'
__vwapath__ = 'data/vn_stock/price_volume/vwap_matrix_20120101-20240101.txt'
__adv20path__ = 'data/vn_stock/price_volume/adv20_matrix_20120101-20240101.txt'
__adv60path__ = 'data/vn_stock/price_volume/adv60_matrix_20120101-20240101.txt'
__adv120path__ = 'data/vn_stock/price_volume/adv120_matrix_20120101-20240101.txt'
__daily_returnpath__ = 'data/vn_stock/price_volume/daily_return_matrix_20120101-20240101.txt'

def load_and_process_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df.set_index('time', inplace=True)
    df = df.astype(float)
    return df

close = load_and_process_data(__closepath__)
open = load_and_process_data(__openpath__)
high = load_and_process_data(__highpath__)
low = load_and_process_data(__lowpath__)
volume = load_and_process_data(__volumepath)
vwap = load_and_process_data(__vwapath__)
adv20 = load_and_process_data(__adv20path__)
adv60 = load_and_process_data(__adv60path__)
adv120 = load_and_process_data(__adv120path__)
daily_return = load_and_process_data(__daily_returnpath__)
