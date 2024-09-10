import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3

def calculate_total_return(final_portfolio_value, initial_cap):
    return (final_portfolio_value/initial_cap) - 1

def calculate_annualized_return(total_return, num_of_days):
    return np.power((1+total_return), 252/num_of_days) - 1

def calculate_annualized_volatility(daily_returns):
    return daily_returns.std() * np.sqrt(252)

def calculate_sharpe_ratio(annualized_return, annualized_volatility, risk_free_rate=0):
    try:
        return (annualized_return - risk_free_rate) / annualized_volatility
    except ZeroDivisionError:
        return 0
def calculate_sortino_ratio(daily_returns, annualized_return, risk_free_rate=0):
    negative_returns = daily_returns[daily_returns < 0]
    downside_volatility = negative_returns.std() * np.sqrt(252)
    return (
        (annualized_return - risk_free_rate) / downside_volatility
        if downside_volatility > 0
        else np.nan
    )

def calculate_maximum_drawdown(portfolio_values):
    drawdown = portfolio_values / portfolio_values.cummax() - 1
    return drawdown.min()

def calculate_drawdown_periods(portfolio_values: pd.Series):
    rolling_max = portfolio_values.cummax()
    drawdowns = rolling_max - portfolio_values
    drawdowns_periods = (drawdowns / rolling_max).dropna()
    return drawdowns_periods

def calculate_rolling_volatility(daily_returns: pd.Series, window: int = 20):
    rolling_vol = daily_returns.rolling(window).std() * (252 ** 0.5)
    return rolling_vol

def calculate_carmar_ratio(annualized_return, max_drawdown):
    try:
        return annualized_return / abs(max_drawdown)
    except ZeroDivisionError:
        return np.nan
    
def calculate_omega_ratio(daily_returns, target_return=0):
    positive_returns = daily_returns[daily_returns > target_return].sum()
    negative_returns = abs(daily_returns[daily_returns < target_return].sum())
    try:
        return positive_returns / negative_returns
    except ZeroDivisionError:
        return np.nan

def calculate_var(daily_returns, confidence_level=0.95):
    return np.percentile(daily_returns, (1 - confidence_level) * 100)

def calculate_cvar(daily_returns, confidence_level=0.95):
    var_threshold = calculate_var(daily_returns, confidence_level)
    return daily_returns[daily_returns <= var_threshold].mean()

def calculate_stability(portfolio_values):
    time = np.arange(len(portfolio_values))
    log_values = np.log(portfolio_values)
    slope, intercept = np.polyfit(time, log_values, 1)
    return slope

def calculate_tail_ratio(daily_returns):
    percentile_95 = np.percentile(daily_returns, 95)
    percentile_5 = np.percentile(daily_returns, 5)
    return abs(percentile_95) / abs(percentile_5)

def calculate_skewness(daily_returns):
    return daily_returns.skew()

def calculate_kurtosis(daily_returns):
    return daily_returns.kurtosis()

def plot_underwater(portfolio_values: pd.Series) -> None:
    running_max = portfolio_values.cummax()
    drawdown = portfolio_values / running_max - 1

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(drawdown, color='blue', label="Drawdown")
    ax.fill_between(drawdown.index, drawdown, 0, color='blue', alpha=0.3)

    ax.set_title("Underwater Plot (Drawdowns Over Time)")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Time")
    ax.legend()

    plt.tight_layout()
    plot_html = mpld3.fig_to_html(fig)
    plt.close(fig)
    return plot_html
