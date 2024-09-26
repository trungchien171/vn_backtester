# backend.py
import fields
import pandas as pd
allowed_variables = {name: getattr(fields, name) for name in dir(fields) 
                     if not name.startswith('__') and isinstance(getattr(fields, name), pd.DataFrame)}

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
        result = eval(alpha, {"__builtins__": None}, allowed_variables)
        if 'neutralization' in settings:
            result = neutralization(result, settings['neutralization'], settings['region'])
        return result
    except Exception as e:
        return str(e)