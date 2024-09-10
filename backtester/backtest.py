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
    calculate_drawdown_periods,
    calculate_rolling_volatility
)

class Backtester:
    def __init__(
        self,
        initial_cap: float = 10000.0,
        commission_pct: float = 0.001,
        commission_fixed: float = 1.0,
        leverage: float = 1.0
    ):
        self.initial_cap = initial_cap
        self.commission_pct = commission_pct
        self.commission_fixed = commission_fixed
        self.leverage = leverage
        self.assets_data: Dict = {}
        self.portfolio_history: Dict = {}
        self.daily_portfolio_values: List[float] = []
        self.transaction_log: List[Dict] = []

    def log_transaction(
            self,
            asset: str,
            trade_type: str,
            shares: float,
            price: float,
            commission: float
    ):
        self.transaction_log.append({
            "asset": asset,
            "trade_type": trade_type,
            "shares": shares,
            "price": price,
            "commission": commission
        })

    def commission(self, trade_value: float) -> float:
        if trade_value < 10000:
            commission = self.commission_pct * 1.5
        else:
            commission = self.commission_pct
        return max(trade_value * commission, self.commission_fixed)
    
    def execute_trade(self, asset: str, signal: int, price: float) -> None:
        trade_value = self.assets_data[asset]["cash"] * self.leverage
        if signal > 0 and self.assets_data[asset]["cash"] > 0:
            commission = self.commission(trade_value)
            shares_to_buy = (trade_value - commission) / price
            self.assets_data[asset]["positions"] += shares_to_buy
            self.assets_data[asset]["cash"] -= trade_value / self.leverage
            self.log_transaction(asset, "buy", shares_to_buy, price, commission)
        elif signal < 0 and self.assets_data[asset]["positions"] > 0:
            trade_value = self.assets_data[asset]["positions"] * price * self.leverage
            commission = self.commission(trade_value)
            self.assets_data[asset]["cash"] += (trade_value - commission) / self.leverage
            shares_sold = self.assets_data[asset]["positions"]
            self.assets_data[asset]["positions"] = 0
            self.log_transaction(asset, "sell", shares_sold, price, commission)
        
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
        drawdown_periods = calculate_drawdown_periods(portfolio_values)
        rolling_volatility = calculate_rolling_volatility(daily_returns)

        print(f"Final Portfolio Value: {portfolio_values.iloc[-1]:.2f}")
        print(f"Total Return: {total_returns:.2f}")
        print(f"Annualized Return: {annualized_return:.2f}%")
        print(f"Annualized Volatility: {annualized_volatility:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
        print(f"Average Drawdown Periods: {drawdown_periods.mean():.2f}")
        print(f"Rolling Volatility (Last 20 Days): {rolling_volatility.iloc[-1]:.2f}")

        if plot:
            self.plot_performance(portfolio_values, daily_returns, rolling_volatility)

    def plot_performance(self, portfolio_values: Dict, daily_returns: pd.DataFrame, rolling_volatility: pd.Series = None) -> None:
        plt.figure(figsize=(12, 6))

        plt.subplot(3, 1, 1)
        plt.plot(portfolio_values, label="Portfolio Value", color="orange")
        plt.title("Portfolio Value Over Time")
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(daily_returns, label="Daily Returns")
        plt.title("Daily Returns Over Time")
        plt.legend()

        if rolling_volatility is not None:
            plt.subplot(3, 1, 3)
            plt.plot(rolling_volatility, label="Rolling Volatility (20 Days)", color="red")
            plt.title("Rolling Volatility")
            plt.legend()

        plt.tight_layout()
        plt.show()