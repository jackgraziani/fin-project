import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
import warnings

# --- Configuration ---
TRADING_DAYS_PER_YEAR = 252
LOOKBACK_YEARS = 5
US_MARKET_CLOSE_TIME = time(16, 30)

# Suppress yfinance download messages to keep shell clean as requested
warnings.filterfactory = "ignore"

def return_prev_close_and_current(ticker_string):
    """Retrieves previous day's close and current price."""
    try:
        ticker = yf.Ticker(ticker_string)
        hist = ticker.history(period="2d", auto_adjust=True)
        precision = 4 if ticker_string == "^TNX" else 2

        if len(hist) < 2:
            previous_close = ticker.info.get('previousClose')
            current_price = ticker.info.get('regularMarketPrice') or ticker.info.get('currentPrice')
            if previous_close is None or current_price is None:
                return None
        else:
            previous_close = hist['Close'].iloc[-2]
            current_price = hist['Close'].iloc[-1]

        return [round(float(previous_close), precision), round(float(current_price), precision)]
    except Exception:
        return None

def run_calcs(portfolio_data):
    """Calculates dollar and percent change."""
    tickers = portfolio_data["tickers"]
    num_shares = portfolio_data["num_shares"]
    price_data = [return_prev_close_and_current(t) for t in tickers]
    
    if any(p is None for p in price_data):
        return [0.00, 0.0000]

    total_open = sum(num_shares[i] * price_data[i][0] for i in range(len(tickers)))
    total_curr = sum(num_shares[i] * price_data[i][1] for i in range(len(tickers)))
    
    dollar_change = round(total_curr - total_open, 2)
    percent_change = round(dollar_change / total_open, 4) if total_open != 0 else 0.0
        
    return [dollar_change, percent_change]

def calculate_beta(portfolio_tickers, market_ticker, lookback_years, num_shares):
    """Calculates portfolio Beta."""
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=lookback_years)
    try:
        all_tickers = [market_ticker] + portfolio_tickers
        data = yf.download(all_tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
        
        current_prices = data.iloc[-1]
        values = [num_shares[i] * current_prices[t] for i, t in enumerate(portfolio_tickers)]
        weights = [v / sum(values) for v in values]

        mkt_returns = data[market_ticker].pct_change()
        port_returns = (data[portfolio_tickers].pct_change() * weights).sum(axis=1)

        combined = pd.DataFrame({'P': port_returns, 'M': mkt_returns}).dropna()
        if len(combined) < 126: return 1.0
        return np.polyfit(combined['M'], combined['P'], 1)[0]
    except Exception:
        return 1.0 

def get_viz_data(portfolio_data, benchmark_ticker="^GSPC"): 
    """Gathers all necessary data for visualization."""
    dollar_change, port_return = run_calcs(portfolio_data)
    
    bench_data = {"tickers": [benchmark_ticker], "num_shares": [1]}
    _, bench_return = run_calcs(bench_data)
    
    try:
        tnx = return_prev_close_and_current("^TNX")
        annual_rfr = tnx[1] / 100 if tnx else 0.04
    except:
        annual_rfr = 0.04
        
    daily_rfr = annual_rfr / TRADING_DAYS_PER_YEAR
    beta = calculate_beta(portfolio_data["tickers"], benchmark_ticker, LOOKBACK_YEARS, portfolio_data["num_shares"])
    daily_alpha = port_return - (daily_rfr + beta * (bench_return - daily_rfr))

    return {
        "port_ret": port_return * 100,
        "bench_ret": bench_return * 100,
        "alpha": daily_alpha * 100,
        "dollar": dollar_change
    }

def visualize(metrics):
    """Creates a matplotlib bar chart with a dark theme and clean spacing for negative values."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 6))

    labels = ['Portfolio', 'S&P 500 (Bench)', 'Alpha']
    values = [metrics['port_ret'], metrics['bench_ret'], metrics['alpha']]
    colors = ['#3498db', '#95a5a6', '#2ecc71' if metrics['alpha'] >= 0 else '#e74c3c']

    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.5)
    
    # Clean zero line and grid
    ax.axhline(0, color='white', linewidth=1.2, alpha=0.8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Dynamic Y-axis padding to prevent text-to-border collision
    min_v, max_v = min(values), max(values)
    spread = max(abs(max_v), abs(min_v)) or 1.0
    padding = spread * 0.3
    ax.set_ylim(min_v - padding, max_v + padding)

    ax.set_ylabel('Daily Return (%)', fontsize=12, color='white', labelpad=10)
    ax.set_title(f"Portfolio Performance (Daily Change: ${metrics['dollar']:,.2f})", 
                 fontsize=14, pad=25, color='white', fontweight='bold')
    
    # Remove unnecessary spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for bar in bars:
        height = bar.get_height()
        # Offset logic: Move text up if positive, down if negative
        va = 'bottom' if height >= 0 else 'top'
        y_offset = padding * 0.1 if height >= 0 else -padding * 0.1
        
        ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                f'{height:+.2f}%', ha='center', va=va, 
                color='white', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.show()

def main():
    portfolio_data = {"tickers": [], "num_shares": []}
    
    while True:
        ticker = input('Enter a ticker ("!" to stop): ').upper().strip()
        if ticker == "!":
            break
        if not ticker:
            continue
        try:
            val = input(f"Shares for {ticker}: ")
            num_shares = int(val)
            portfolio_data["tickers"].append(ticker)
            portfolio_data["num_shares"].append(num_shares)
        except ValueError:
            print("Invalid number of shares.")

    if not portfolio_data["tickers"]:
        print("No input detected. Using example portfolio.")
        portfolio_data = {"tickers": ["BKR", "CF", "MRK", "PINS"], "num_shares": [11, 11, 11, 11]}

    metrics = get_viz_data(portfolio_data)
    visualize(metrics)

if __name__ == "__main__":
    main()