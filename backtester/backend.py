# backend.py
import fields
import re
import pandas as pd
import numpy as np

allowed_variables = {name: getattr(fields, name) for name in dir(fields) 
                     if not name.startswith('__') and isinstance(getattr(fields, name), pd.DataFrame)}

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
        x = eval(alpha, {"__builtins__": None}, allowed_variables)
        
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        
        if 'decay' in settings:
            decay_days = int(settings['decay'])
            result = decay_linear(x, decay_days)
        else:
            result = x
        
        if 'neutralization' in settings:
            result = neutralization(result, settings['neutralization'], settings['region'])
        
        return result
    except Exception as e:
        return str(e)


