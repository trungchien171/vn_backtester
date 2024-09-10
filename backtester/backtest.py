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
    calculate_rolling_volatility,
    plot_underwater,
    calculate_carmar_ratio,
    calculate_omega_ratio,
    calculate_var,
    calculate_cvar,
    calculate_stability,
    calculate_tail_ratio,
    calculate_skewness,
    calculate_kurtosis
)

class Backtester:
    def __init__(
        self,
        initial_cap: float = 10000.0,
        commission_pct: float = 0.0001,
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
            commission = self.commission_pct + 0.0005
        else:
            commission = self.commission_pct
        return max(trade_value * commission, self.commission_fixed)
    
    def execute_trade(self, asset: str, signal: int, price: float) -> None:
        trade_value = self.assets_data[asset]["cash"] * self.leverage
        # tax = 0.001
        if signal > 0 and self.assets_data[asset]["cash"] > 0:
            commission = self.commission(trade_value)
            shares_to_buy = (trade_value - commission) / price
            self.assets_data[asset]["positions"] += shares_to_buy
            self.assets_data[asset]["cash"] -= trade_value / self.leverage
            self.log_transaction(asset, "buy", shares_to_buy, price, commission)
        elif signal < 0 and self.assets_data[asset]["positions"] > 0:
            trade_value = self.assets_data[asset]["positions"] * price * self.leverage
            commission = self.commission(trade_value)
            # income_tax = trade_value * tax
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
        max_drawdown = calculate_maximum_drawdown(portfolio_values)
        sortino_ratio = calculate_sortino_ratio(daily_returns, annualized_return)
        drawdown_periods = calculate_drawdown_periods(portfolio_values)
        rolling_volatility = calculate_rolling_volatility(daily_returns)
        carmar_ratio = calculate_carmar_ratio(annualized_return, max_drawdown)
        omega_ratio = calculate_omega_ratio(daily_returns)
        var = calculate_var(daily_returns)
        cvar = calculate_cvar(daily_returns)
        stability = calculate_stability(portfolio_values)
        tail_ratio = calculate_tail_ratio(daily_returns)
        skewness = calculate_skewness(daily_returns)
        kurtosis = calculate_kurtosis(daily_returns)


        metrics = {
            "Final Portfolio Value": portfolio_values.iloc[-1].round(2),
            "Total Return": f"{total_returns:.2%}",  
            "Annualized Return": f"{annualized_return:.2%}",
            "Annualized Volatility": f"{annualized_volatility:.2%}",
            "Sharpe Ratio": sharpe_ratio.round(2),
            "Max Drawdown": f"{-max_drawdown:.2%}",
            "Sortino Ratio": sortino_ratio.round(2),
            "Average Drawdown Period": drawdown_periods.mean().round(2),
            "Rolling Volatility (Last 20 Days)": f"{rolling_volatility.iloc[-1]:.2%}",
            "Calmar Ratio": carmar_ratio.round(2),
            "Omega Ratio": omega_ratio.round(2),
            "Value at Risk (95%)": f"{var:.2%}",
            "Conditional Value at Risk (95%)": f"{cvar:.2%}",
            "Stability": stability.round(2),
            "Tail Ratio": tail_ratio.round(2),
            "Skewness": skewness.round(2),
            "Kurtosis": kurtosis.round(2)
        }

        if plot:
            portfolio_plot_html = self.plot_performance(portfolio_values, daily_returns, rolling_volatility)
            underwater_plot_html = plot_underwater(portfolio_values)
        else:
            portfolio_plot_html = ""
            underwater_plot_html = ""
        self.generate_html_report(metrics, portfolio_plot_html + underwater_plot_html, output_html)

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
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
            <style>
                body {
                    font-family: 'Roboto', sans-serif;
                    background-color: #f4f4f9;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                }
                h1, h2 {
                    text-align: center;
                    color: #2c3e50;
                }
                h1 {
                    font-size: 2.5em;
                    margin-bottom: 20px;
                }
                h2 {
                    font-size: 1.8em;
                    margin-top: 30px;
                }
                table {
                    width: 100%;
                    max-width: 800px;
                    margin: 20px auto;
                    border-collapse: collapse;
                    background-color: white;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                }
                th, td {
                    padding: 20px 15px;
                    text-align: center; /* Center horizontally */
                    vertical-align: middle; /* Center vertically */
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #2c3e50;
                    color: white;
                    font-weight: 700;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                tr:hover {
                    background-color: #f1f1f1;
                }
                .metrics-table {
                    margin-bottom: 40px;
                }
                .chart-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .chart {
                    padding: 20px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                    width: 100%; /* Allow chart to be full width */
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                }
            </style>
        </head>
        <body>
            <h1>Backtest Report</h1>
            <h2>Metrics</h2>
            <table class="metrics-table">
                {% for key, value in metrics.items() %}
                <tr>
                    <th>{{ key }}</th>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Performance Charts</h2>
            <div class="chart-container">
                <div class="chart">{{ plot_html | safe }}</div>
            </div>
        </body>
        </html>
        """

        template = jinja2.Template(html_template)
        html_content = template.render(metrics=metrics, plot_html=plot_html)

        with open(output_html, "w") as f:
            f.write(html_content)

        print(f"Report saved to {output_html}")
        
        webbrowser.open(output_html)
