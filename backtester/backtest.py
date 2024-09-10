from typing import List, Dict
import jinja2
import mpld3
import webbrowser
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
    
    def performance(self, output_html = "backtest_report.html", plot: bool = True) -> None:
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

        metrics = {
            "Final Portfolio Value": portfolio_values.iloc[-1].round(2),
            "Total Return": f"{total_returns:.2%}",  
            "Annualized Return": f"{annualized_return:.2%}",
            "Annualized Volatility": f"{annualized_volatility:.2%}",
            "Sharpe Ratio": sharpe_ratio.round(2),
            "Sortino Ratio": sortino_ratio.round(2),
            "Max Drawdown": f"{-max_drawdown:.2%}",
            "Average Drawdown Period": drawdown_periods.mean().round(2),
            "Rolling Volatility (Last 20 Days)": f"{rolling_volatility.iloc[-1]:.2%}"
        }

        if plot:
            portfolio_plot_html = self.plot_performance(portfolio_values, daily_returns, rolling_volatility)
        else:
            portfolio_plot_html = ""
        self.generate_html_report(metrics, portfolio_plot_html, output_html)

    def plot_performance(self, portfolio_values: Dict, daily_returns: pd.DataFrame, rolling_volatility: pd.Series = None) -> None:
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))

        axs[0].plot(portfolio_values, label="Portfolio Value", color="orange")
        axs[0].set_title("Portfolio Value Over Time")
        axs[0].legend()

        axs[1].plot(daily_returns, label="Daily Returns")
        axs[1].set_title("Daily Returns Over Time")
        axs[1].legend()

        if rolling_volatility is not None:
            axs[2].plot(rolling_volatility, label="Rolling Volatility (20 Days)", color="red")
            axs[2].set_title("Rolling Volatility")
            axs[2].legend()

        plt.tight_layout()
        plot_html = mpld3.fig_to_html(fig)
        plt.close(fig)
        return plot_html
    
    def generate_html_report(self, metrics: Dict, plot_html: str, output_html: str) -> None:
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Backtest Report</title>
        </head>
        <body>
            <h1>Backtest Performance Report</h1>
            <h2>Metrics</h2>
            <table border="1">
                {% for key, value in metrics.items() %}
                <tr>
                    <th>{{ key }}</th>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Performance Charts</h2>
            <div>{{ plot_html | safe }}</div>
        </body>
        </html>
        """

        template = jinja2.Template(html_template)
        html_content = template.render(metrics=metrics, plot_html=plot_html)

        with open(output_html, "w") as f:
            f.write(html_content)

        print(f"Report saved to {output_html}")
        
        webbrowser.open(output_html)
