import pandas as pd
from vnstock3 import Vnstock

all_symbols = Vnstock().stock().listing.all_symbols()['ticker'].tolist()
stock = Vnstock().stock()
data = pd.DataFrame()
for symbol in all_symbols:
    print(f"Downloading data for {symbol}")
    stock = Vnstock().stock(symbol=symbol, source='VCI')
    
    try:
        try:
            df = stock.quote.history(start='2012-01-01', end='2024-01-01', interval='1D')
        except Exception as e:
            print(f"Error: {e}")
            continue
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = df['time'].dt.strftime('%Y-%m-%d')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['volume'] = df['volume'].astype(float)
        df = df[['time', 'volume']].rename(columns={'volume': symbol})
        
        if data.empty:
            data = df.set_index('time')
        else:
            data = data.join(df.set_index('time'), how='outer')
    
    except IndexError:
        print(f"Data for {symbol} not found")
        continue
data.fillna(0, inplace=True)
print(data)
data.to_csv('volume_matrix_20120101-20240101.txt', sep='\t')