# backend.py
import pandas as pd
import numpy as np

close_path = 'data/vn_stock/price_volume/close_matrix_200101-240101.txt'
open_path = 'data/vn_stock/price_volume/open_matrix_200101-240101.txt'
high_path = 'data/vn_stock/price_volume/high_matrix_200101-240101.txt'
low_path = 'data/vn_stock/price_volume/low_matrix_200101-240101.txt'
volume_path = 'data/vn_stock/price_volume/volume_matrix_200101-240101.txt'
close = pd.read_csv(close_path, sep='\t')
open = pd.read_csv(open_path, sep='\t')
high = pd.read_csv(high_path, sep='\t')
low = pd.read_csv(low_path, sep='\t')
volume = pd.read_csv(volume_path, sep='\t')
