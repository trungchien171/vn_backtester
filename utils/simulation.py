#backend.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from data.load_data import dataframes
from utils.operators import operators

# def pasteurization(alpha, universe):
#     relavant_instruments = datafields[universe].keys()
#     return alpha[relavant_instruments].copy()

def delay(alpha, delay):
    return alpha.shift(delay)
    
def truncation(alpha, percentage):
    alpha_normalized = alpha.div(alpha.sum(axis=1), axis=0)
    alpha_truncated = alpha_normalized.where(alpha_normalized >= 0, other=alpha_normalized)
    alpha_truncated = alpha_truncated.clip(upper = percentage)
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
        flat_operators = {}
        for category, funcs in operators.items():
            flat_operators.update(funcs)
        prices = (variables['close'] + variables['high'] + variables['low']) / 3
        x = eval(alpha, {"__builtins__": None}, {**variables, **flat_operators})
        
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        
        # if settings['pasteurization'] == 'True':
        #     x = pasteurization(x, universe)

        if 'delay' in settings:
            x = delay(x, settings['delay'])
        else:
            raise KeyError("The 'delay' key is missing in settings.")


        if 'decay' in settings:
            decay_days = int(settings['decay'])

            if decay_days > 0:
                result = decay_linear(x, decay_days)
            else:
                result = x
        else:
            result = x
        
        if 'neutralization' in settings:
            result = neutralization(result, settings['neutralization'], settings['region'])
        
        if 'truncation' in settings:
            truncation_percentage = float(settings['truncation'])
            result = truncation(result, truncation_percentage)  

        result = result.fillna(0)

        if settings.get('region') == 'VN':
            result = result.shift(2)
        
        # PnL calculation
        daily_change = prices.diff().fillna(0)
        pnl = (result.shift(1) * daily_change).sum(axis=1)
        pnl_curve = pnl.cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pnl_curve.index, y=pnl_curve, mode='lines', name='PnL', line=dict(color='blue')))
        fig.update_layout(title='PnL Curve', xaxis_title='Date', yaxis_title='PnL')

        # PnL table
        pnl_table = pd.DataFrame(pnl, columns=['PnL'])
        pnl_table.index = pd.to_datetime(pnl_table.index)

        # Turnover table
        money_trading_volume = abs(result.diff().fillna(0) * prices).sum(axis=1)
        booksize = abs(result * prices).sum(axis=1)
        turnover = (money_trading_volume / booksize).replace([np.inf, -np.inf], 0).fillna(0)
        turnover_table = pd.DataFrame(turnover, columns=['Turnover'])
        turnover_table.index = pd.to_datetime(turnover_table.index)

        # Total money traded
        total_money_traded = money_trading_volume.sum()

        # Alpha summary
        summary = {}

        # Sharpe, Turnover, Returns, Fitness, Drawdown, Margin, Long Side, Short Side
        for year in range(pnl_table.index.year.min(), pnl_table.index.year.max() + 1):
            pnl_year = pnl_table['PnL'][pnl_table.index.year == year]
            turnover_year = turnover_table['Turnover'][turnover_table.index.year == year]

            # Sharpe
            mean_daily_pnl = pnl_year.mean()
            std_daily_pnl = pnl_year.std()
            sharpes = mean_daily_pnl / std_daily_pnl * np.sqrt(252) if std_daily_pnl != 0 else 0

            # Turnover
            avg_turnover = turnover_year.mean()

            # Returns
            annualized_pnl = pnl_year.sum() * 252 / len(pnl_year)
            avg_booksize = booksize[pnl_table.index.year == year].mean()
            annual_return = annualized_pnl / (avg_booksize) if avg_booksize != 0 else 0

            # Fitness
            fitness = sharpes * np.sqrt(abs(annual_return) / max(avg_turnover, 0.125))

            # Drawdown
            cum_pnl = pnl_year.cumsum()
            running_max = cum_pnl.cummax()
            drawdown = (running_max - cum_pnl).max()
            largest_drawdown = drawdown / (avg_booksize) if avg_booksize != 0 else 0

            # Margin
            total_money_traded_year = money_trading_volume[pnl_table.index.year == year].sum()
            margin = pnl_year.sum() / total_money_traded_year if total_money_traded_year != 0 else 0

            # Sides
            long_side = result[result > 0][pnl_table.index.year == year].sum().sum()
            short_side = result[result < 0][pnl_table.index.year == year].sum().sum()

            # Summary Table
            summary[year] = {
                'Sharpe': sharpes,
                'Turnover (%)': avg_turnover * 100,
                'Returns (%)': annual_return * 100,
                'Fitness': fitness,
                'Drawdown (%)': largest_drawdown * 100,
                'Margin (%)': margin * 100,
                'Long Side': long_side,
                'Short Side': short_side
            }

            summary_table = pd.DataFrame(summary).T
            summary_table.index.name = 'Year'
            summary_table.index = summary_table.index.astype(str)

            overall_sharpe = summary_table['Sharpe'].mean()
            overall_turnover = summary_table['Turnover (%)'].mean()
            overall_returns = summary_table['Returns (%)'].mean()
            overall_fitness = summary_table['Fitness'].mean()
            overall_drawdown = summary_table['Drawdown (%)'].max()
            overall_margin = summary_table['Margin (%)'].mean()
            overall_long_side = summary_table['Long Side'].sum()
            overall_short_side = summary_table['Short Side'].sum()

            overall_summary = pd.DataFrame(
                {
                    'Sharpe': [overall_sharpe],
                    'Turnover (%)': [overall_turnover],
                    'Returns (%)': [overall_returns],
                    'Fitness': [overall_fitness],
                    'Drawdown (%)': [overall_drawdown],
                    'Margin (%)': [overall_margin],
                    'Long Side': [overall_long_side],
                    'Short Side': [overall_short_side]
                },
                index=['All']
            )
            summary_table = pd.concat([summary_table, overall_summary])
            
        return fig, summary_table
    except Exception as e:
        st.error(f"An error occurred: {e}")
