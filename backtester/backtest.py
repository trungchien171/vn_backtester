from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
from backtester.performance import (
    calculate_total_return,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_maximum_drawdown,
)

class Backtester:
    def __init__(
        self,
        initial_cap: float = 10000.0,
        commission_pct: float = 0.001,
        commission_fixed: float = 1.0,
    ):
        self.initial_cap = initial_cap
        self.commission_pct = commission_pct
        self.commission_fixed = commission_fixed
        self.assets_data: Dict = {}
        self.portfolio_history: Dict = {}
        self.daily_portfolio_values: List[float] = []

    def commission(self, trade_value: float) -> float:
        return max(trade_value * self.commission_pct, self.commission_fixed)
    
    def execute_trade(self, asset: str, signal: int, price: float) -> None:
        if signal > 0 and self.assets_data[asset]["cash"] > 0:
            trade_value = self.assets_data[asset]["cash"]
            commission = self.commission(trade_value)
            shares_to_buy = (trade_value - commission) / price
            self.assets_data[asset]["positions"] += shares_to_buy
            self.assets_data[asset]["cash"] -= trade_value
        elif signal < 0 and self.assets_data[asset]["positions"] > 0:
            trade_value = self.assets_data[asset]["positions"] * price
            commission = self.commission(trade_value)
            self.assets_data[asset]["cash"] += trade_value - commission
            self.assets_data[asset]["positions"] = 0
        
    def update_portfolio(self, asset: str, price: float) -> None:
        self.assets_data[asset]["position_value"] = self.assets_data[asset]["positions"] * price
        self.assets_data[asset]["total_value"] = self.assets_data[asset]["cash"] + self.assets_data[asset]["position_value"]
        self.portfolio_history[asset].append(self.assets_data[asset]["total_value"])

    def backtest(self, data: pd.DataFrame | dict[str, pd.DataFrame]):
        if isinstance(data, pd.DataFrame):
            data = {"SINGLE ASSET": data}

        for asset in data:
            self.assets_data[asset] = {
                "cash": self.initial_cap/len(data),
                "positions": 0,
                "position_value": 0,
                "total_value": 0,
            }

            self.portfolio_history[asset] = []

            for date, row in data[asset].iterrows():
                self.execute_trade(asset, row["signal"], row["close"])
                self.update_portfolio(asset, row["close"])

                if len(self.daily_portfolio_values) < len(data[asset]):
                    self.daily_portfolio_values.append(self.assets_data[asset]["total_value"])
                else:
                    self.daily_portfolio_values[len(self.daily_portfolio_values) - 1] += self.assets_data[asset]["total_value"]
    
    def performance(self, plot: bool = True) -> None:
        if not self.daily_portfolio_values:
            print("No portfolio history to calculate performance")
            return
        
        portfolio_values = pd.Series(self.daily_portfolio_values)
        daily_returns = portfolio_values.pct_change().dropna()
        
        total_returns = calculate_total_return(portfolio_values.iloc[-1], self.initial_cap)
        annualized_return = calculate_annualized_return(total_returns, len(portfolio_values))
        annualized_volatility = calculate_annualized_volatility(daily_returns)
        sharpe_ratio = calculate_sharpe_ratio(annualized_return, annualized_volatility)
        sortino_ratio = calculate_sortino_ratio(daily_returns, annualized_return)
        max_drawdown = calculate_maximum_drawdown(portfolio_values)

        print(f"Final Portfolio Value: {portfolio_values.iloc[-1]:.2f}")
        print(f"Total Return: {total_returns:.2f}")
        print(f"Annualized Return: {annualized_return:.2f}")
        print(f"Annualized Volatility: {annualized_volatility:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown * 100:.2f}%")

        if plot:
            self.plot_performance(portfolio_values, daily_returns)

    def plot_performance(self, portfolio_values: Dict, daily_returns: pd.DataFrame):
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(portfolio_values, label="Portfolio Value")
        plt.title("Portfolio Value Over Time")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(daily_returns, label="Daily Returns")
        plt.title("Daily Returns Over Time")
        plt.legend()

        plt.tight_layout()
        plt.show()