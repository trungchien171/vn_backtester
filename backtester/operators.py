import pandas as pd

def ts_mean(x, window):
    return x.rolling(window=window).mean()

def delta(x, period=1):
    return x.diff(period)

def rank(x):
    return x.rank(axis=1)

def ts_sum(x, window):
    return x.rolling(window=window).sum()

def ts_stddev(x, window):
    return x.rolling(window=window).std()

operators = {
    'ts_mean': ts_mean,
    'delta': delta,
    'rank': rank,
    'ts_sum': ts_sum,
    'ts_stddev': ts_stddev
}
