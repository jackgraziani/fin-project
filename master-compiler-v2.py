import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, time
import warnings
from typing import Optional, Dict, Any, List, Union # Added from ttfh-v2

# --- GLOBAL CONFIGURATION ---
TRADING_DAYS_PER_YEAR = 252
LOOKBACK_YEARS = 5
US_MARKET_CLOSE_TIME = time(16, 30)

# Suppress yfinance download messages
warnings.filterfactory = "ignore"

# --- FUNDAMENTAL ANALYSIS CONSTANTS (From ttfh-v2) ---
BALANCE_SHEET_MAP = {
    'Total Assets': ['Total Assets'],
    'Current Assets': ['Current Assets'],
    'Current Liabilities': ['Current Liabilities', 'Current Debt'],
    'Cash': ['Cash And Cash Equivalents', 'Cash', 'Cash equivalents and Short Term Investments'],
    'ST Investments': ['Other Short Term Investments', 'Short Term Investments'],
    'Receivables': ['Receivables', 'Net Receivables', 'Accounts Receivable'],
    'Total Debt': ['Total Debt', 'Long Term Debt', 'Total Liabilities Net Minority Interest'],
    'Total Equity': ['Stockholders Equity', 'Total Equity', 'Shareholders Equity']
}

FINANCIALS_MAP = {
    'Total Revenue': ['Total Revenue', 'Revenue', 'Operating Revenue'], 
    'Net Income': ['Net Income', 'Net Income Common Stockholders'],
    'Operating Income': ['Operating Income', 'Operating Expense'], 
    'Pretax Income': ['Pretax Income', 'EBT', 'Earnings Before Taxes'], 
    'Interest Expense': ['Interest Expense'],
    'Interest Income': ['Interest Income', 'Net Interest Income'], 
}

SECTOR_STANDARDS = {
    'Technology':[0.15, 0.08, 0.10, 1.80, 1.20, 0.80, 0.45, 0.80],
    'Communication Services':[0.12, 0.05, 0.08, 1.20, 0.70, 1.20, 0.55, 0.75],
    'Consumer Cyclical':[0.10, 0.04, 0.05, 1.50, 0.80, 1.00, 0.50, 0.75],
    'Financial Services':[0.10, 0.03, 0.05, 1.00, 0.50, 7.00, 0.88, 0.65],
    'Healthcare':[0.12, 0.07, 0.10, 1.80, 1.20, 0.80, 0.45, 0.80],
    'Consumer Defensive':[0.15, 0.03, 0.05, 1.20, 0.50, 1.20, 0.55, 0.75],
    'Energy':[0.10, 0.04, 0.05, 1.20, 0.70, 1.20, 0.55, 0.75],
    'Industrials':[0.12, 0.05, 0.08, 1.50, 0.90, 1.00, 0.50, 0.80],
    'Utilities':[0.08, 0.02, 0.04, 1.00, 0.50, 2.20, 0.65, 0.88],
    'Real Estate':[0.08, 0.05, 0.07, 0.70, 0.20, 1.80, 0.65, 0.75],
    'Basic Materials':[0.10, 0.04, 0.05, 1.50, 1.00, 1.00, 0.50, 0.75]
}

METRICS = [
    "Return on Equity",
    "Annual Revenue Growth",
    "Annual Earnings Growth",
    "Current Ratio",
    "Quick Ratio",
    "Debt-Equity Ratio",
    "Debt-Asset Ratio",
    "Operating Efficiency Ratio"
]

# ==========================================
# SHARED UTILITIES
# ==========================================

def get_current_price_and_volatility(ticker_string):
    """
    Fetches current price and historical annualized volatility for a ticker.
    Returns: (current_price, volatility_decimal)
    """
    try:
        ticker = yf.Ticker(ticker_string)
        # Fetch 1 year of history for volatility calc
        hist = ticker.history(period="1y", auto_adjust=True)
        
        if len(hist) < 2:
            return None, 0.2 # Fallback
            
        current_price = hist['Close'].iloc[-1]
        
        # Calculate historical volatility (annualized standard deviation of log returns)
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
        volatility = log_returns.std() * np.sqrt(252)
        
        return float(current_price), float(volatility)
    except Exception as e:
        print(f"Error fetching data for {ticker_string}: {e}")
        return None, 0.2

def get_risk_free_rate():
    """Fetches the 10-year Treasury Yield as a proxy for RFR."""
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if not hist.empty:
            return hist['Close'].iloc[-1] / 100
        return 0.04 # Fallback 4%
    except:
        return 0.04

# ==========================================
# MODULE 1: PORTFOLIO PERFORMANCE (Daily Alpha)
# ==========================================

def return_prev_close_and_current(ticker_string):
    """Retrieves previous day's close and current price."""
    try:
        ticker = yf.Ticker(ticker_string)
        hist = ticker.history(period="2d", auto_adjust=True)
        precision = 4 if ticker_string == "^TNX" else 2

        if len(hist) < 2:
            # Fallback to info dict if history is insufficient
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

def calculate_beta(portfolio_tickers, market_ticker, lookback_years, num_shares):
    """Calculates portfolio Beta."""
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=lookback_years)
    try:
        all_tickers = [market_ticker] + portfolio_tickers
        data = yf.download(all_tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
        
        if isinstance(data, pd.Series):
             return 1.0

        current_prices = data.iloc[-1]
        
        # Calculate weights based on current value
        values = []
        for i, t in enumerate(portfolio_tickers):
            if t in current_prices:
                values.append(num_shares[i] * current_prices[t])
            else:
                values.append(0)
                
        total_val = sum(values)
        if total_val == 0: return 1.0
        
        weights = [v / total_val for v in values]

        mkt_returns = data[market_ticker].pct_change()
        valid_port_tickers = [t for t in portfolio_tickers if t in data.columns]
        
        if not valid_port_tickers: return 1.0

        port_returns = (data[valid_port_tickers].pct_change() * weights).sum(axis=1)

        combined = pd.DataFrame({'P': port_returns, 'M': mkt_returns}).dropna()
        if len(combined) < 126: return 1.0
        
        beta = np.polyfit(combined['M'], combined['P'], 1)[0]
        return beta
    except Exception as e:
        return 1.0 

def run_portfolio_analysis(portfolio_data, benchmark_ticker="^GSPC"):
    print("\n--- Running Portfolio Performance Analysis ---")
    tickers = portfolio_data["tickers"]
    num_shares = portfolio_data["num_shares"]
    
    # 1. Calculate Portfolio Dollar/Percent Change
    price_data = [return_prev_close_and_current(t) for t in tickers]
    
    if any(p is None for p in price_data):
        print("Error fetching daily price data. Skipping Portfolio Viz.")
        return

    total_open = sum(num_shares[i] * price_data[i][0] for i in range(len(tickers)) if price_data[i])
    total_curr = sum(num_shares[i] * price_data[i][1] for i in range(len(tickers)) if price_data[i])
    
    dollar_change = round(total_curr - total_open, 2)
    port_return = (dollar_change / total_open) if total_open != 0 else 0.0

    # 2. Calculate Benchmark Return
    bench_prices = return_prev_close_and_current(benchmark_ticker)
    bench_return = 0.0
    if bench_prices:
        bench_return = (bench_prices[1] - bench_prices[0]) / bench_prices[0]

    # 3. Calculate Alpha
    rfr = get_risk_free_rate()
    daily_rfr = rfr / TRADING_DAYS_PER_YEAR
    beta = calculate_beta(tickers, benchmark_ticker, LOOKBACK_YEARS, num_shares)
    
    # Jensen's Alpha: Rp - [Rf + B * (Rm - Rf)]
    daily_alpha = port_return - (daily_rfr + beta * (bench_return - daily_rfr))

    metrics = {
        "port_ret": port_return * 100,
        "bench_ret": bench_return * 100,
        "alpha": daily_alpha * 100,
        "dollar": dollar_change
    }
    
    visualize_portfolio(metrics)

def visualize_portfolio(metrics):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 6))

    labels = ['Portfolio', 'S&P 500 (Bench)', 'Alpha']
    values = [metrics['port_ret'], metrics['bench_ret'], metrics['alpha']]
    colors = ['#3498db', '#95a5a6', '#2ecc71' if metrics['alpha'] >= 0 else '#e74c3c']

    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.axhline(0, color='white', linewidth=1.2, alpha=0.8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    min_v, max_v = min(values), max(values)
    spread = max(abs(max_v), abs(min_v)) or 1.0
    padding = spread * 0.3
    ax.set_ylim(min_v - padding, max_v + padding)

    ax.set_ylabel('Daily Return (%)', fontsize=12, color='white', labelpad=10)
    ax.set_title(f"Portfolio Performance (Daily Change: ${metrics['dollar']:,.2f})", 
                 fontsize=14, pad=25, color='white', fontweight='bold')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for bar in bars:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        y_offset = padding * 0.1 if height >= 0 else -padding * 0.1
        ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                f'{height:+.2f}%', ha='center', va=va, 
                color='white', fontweight='bold', fontsize=11)

    print("Displaying Portfolio Analysis (Close window to continue)...")
    plt.tight_layout()
    plt.show()


# ==========================================
# MODULE 2: MONTE CARLO OPTION PRICING
# ==========================================

def black_scholes_call(S, K, T, r, sigma):
    """Calculates the theoretical European Call price for benchmark comparison."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def run_monte_carlo_analysis(ticker):
    print(f"\n--- Running Monte Carlo Analysis for {ticker} ---")
    
    # 1. Fetch Dynamic Data
    S0, SIGMA = get_current_price_and_volatility(ticker)
    if S0 is None:
        print(f"Could not fetch data for {ticker}, skipping Monte Carlo.")
        return

    # 2. Configure Simulation
    RFR = get_risk_free_rate()
    # Assume we are pricing a 1-year Call option that is 5% Out-of-the-Money (OTM)
    K = S0 * 1.05  
    T = 1.0
    N_STEPS = 252
    N_PATHS = 5000 # Reduced slightly for speed in loop
    DT = T / N_STEPS

    # 3. Simulation Engine
    Z = np.random.standard_normal((N_PATHS, N_STEPS))
    daily_returns = np.exp((RFR - 0.5 * SIGMA**2) * DT + SIGMA * np.sqrt(DT) * Z)

    paths = np.zeros((N_PATHS, N_STEPS + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.cumprod(daily_returns, axis=1)
    
    # 4. Longstaff-Schwartz (LSM)
    cashflows = np.maximum(paths[:, -1] - K, 0)
    discount_factor = np.exp(-RFR * DT)

    # Backward Induction
    for t in range(N_STEPS - 1, 0, -1):
        cashflows = cashflows * discount_factor
        S_t = paths[:, t]
        itm_mask = S_t > K
        
        if np.count_nonzero(itm_mask) > 0:
            X = S_t[itm_mask]
            Y = cashflows[itm_mask]
            coeffs = np.polyfit(X, Y, 2)
            continuation_value = np.polyval(coeffs, X)
            exercise_value = X - K
            
            exercise_indices = exercise_value > continuation_value
            full_indices = np.where(itm_mask)[0]
            paths_to_update = full_indices[exercise_indices]
            cashflows[paths_to_update] = exercise_value[exercise_indices]

    option_price = np.mean(cashflows * discount_factor)
    bs_price = black_scholes_call(S0, K, T, RFR, SIGMA)
    breakeven = K + option_price
    
    probability_itm = np.count_nonzero(cashflows > 0) / N_PATHS
    probability_profit = np.count_nonzero(cashflows > option_price) / N_PATHS

    # 5. Visualize
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')

    # Plot subset of paths
    ax.plot(paths[:100].T, lw=0.4, color='white', alpha=0.225)

    ax.axhline(S0, color='white', lw=1, label=f'Current Price (${S0:.2f})', alpha=0.7)
    ax.axhline(K, color='#ff4d4d', linestyle='--', lw=2, label=f'Strike Price (+5% OTM) (${K:.2f})')
    ax.axhline(breakeven, color='#00ff88', linestyle=':', lw=2, label=f'Breakeven (${breakeven:.2f})')

    stats_text = (f"LSM American Price: ${option_price:.2f}\n"
                  f"B-S (Euro) Price:   ${bs_price:.2f}\n"
                  f"Volatility (Hist):  {SIGMA:.1%}\n"
                  f"Prob. of Profit:    {probability_profit:.1%}")
    
    ax.text(0.02, 0.96, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#1e1e1e', edgecolor='#444444', alpha=0.9))

    ax.set_title(f"{ticker}: 1-Year Option Pricing (Monte Carlo)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Trading Days", fontsize=10, color='#cccccc')
    ax.set_ylabel("Price ($)", fontsize=10, color='#cccccc')
    ax.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#444444', fontsize=9)

    print(f"Displaying Monte Carlo for {ticker} (Close window to continue)...")
    plt.tight_layout()
    plt.show()


# ==========================================
# MODULE 3: STOCHASTIC OSCILLATOR
# ==========================================

N_PERIODS = 14   
K_SMOOTH = 3     
D_SMOOTH = 3     
SMA_FILTER = 40  
OVERSOLD_LEVEL = 20
OVERBOUGHT_LEVEL = 80

def run_stochastic_analysis(ticker):
    print(f"\n--- Running Stochastic Oscillator for {ticker} ---")
    
    # 1. Fetch Data
    stock_data = yf.download(ticker, period="1y", interval="1d", progress=False)
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    if stock_data.empty:
        print(f"No data found for {ticker} stochastic analysis.")
        return

    df = stock_data[['High', 'Low', 'Close']].copy()

    # 2. Calculate Indicators
    lowest_low = df['Low'].rolling(window=N_PERIODS).min()
    highest_high = df['High'].rolling(window=N_PERIODS).max()
    fast_k = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    
    df['Slow_%K'] = fast_k.rolling(window=K_SMOOTH).mean()
    df['Slow_%D'] = df['Slow_%K'].rolling(window=D_SMOOTH).mean()
    df['SMA_Trend'] = df['Close'].rolling(window=SMA_FILTER).mean()
    
    df = df.dropna()

    # 3. Generate Signals
    df['Signal'] = 'Hold'
    
    # Buy: Cross up + Oversold + Above SMA Trend
    buy_condition = (
        (df['Slow_%K'] > df['Slow_%D']) &          
        (df['Slow_%K'].shift(1) < df['Slow_%D'].shift(1)) & 
        (df['Slow_%D'] < OVERSOLD_LEVEL) &         
        (df['Close'] > df['SMA_Trend'])            
    )
    df.loc[buy_condition, 'Signal'] = 'Buy'
    
    # Sell: Cross down + Overbought
    sell_condition = (
        (df['Slow_%K'] < df['Slow_%D']) & 
        (df['Slow_%K'].shift(1) > df['Slow_%D'].shift(1)) &
        (df['Slow_%D'] > OVERBOUGHT_LEVEL)
    )
    df.loc[sell_condition, 'Signal'] = 'Sell'

    # 4. Plot
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, height_ratios=[2, 1])
    
    # Plot Price & Trend
    ax1.plot(df.index, df['Close'], color='cyan', alpha=0.9, label='Price')
    ax1.plot(df.index, df['SMA_Trend'], color='yellow', alpha=0.6, linestyle='--', label=f'SMA {SMA_FILTER} (Trend)')
    
    # Plot Signals
    buys = df[df['Signal'] == 'Buy']
    sells = df[df['Signal'] == 'Sell']
    ax1.scatter(buys.index, buys['Close'], marker='^', color='#00ff00', s=100, label='Buy', zorder=5)
    ax1.scatter(sells.index, sells['Close'], marker='v', color='#ff3333', s=100, label='Sell', zorder=5)
    
    ax1.set_title(f"{ticker} - Slow Stochastic Strategy", color='white', fontweight='bold')
    ax1.legend(loc='upper left')
    
    # Plot Oscillator
    ax2.plot(df.index, df['Slow_%K'], color='white', label='Slow %K')
    ax2.plot(df.index, df['Slow_%D'], color='orange', label='Slow %D')
    ax2.axhline(OVERBOUGHT_LEVEL, color='red', linestyle='--', alpha=0.6)
    ax2.axhline(OVERSOLD_LEVEL, color='green', linestyle='--', alpha=0.8)
    ax2.set_ylabel('Oscillator')
    ax2.legend(loc='upper right')
    
    print(f"Displaying Stochastic for {ticker} (Close window to continue)...")
    plt.tight_layout()
    plt.show()

# ==========================================
# MODULE 4: FUNDAMENTAL ANALYSIS (From ttfh-v2)
# ==========================================

def find_financial_item(statement: pd.DataFrame, key_mapping: Union[str, List[str]]) -> Optional[float]:
    """Attempts to find a financial item in the statement's index."""
    search_keys = [key_mapping] if isinstance(key_mapping, str) else key_mapping
    for key in search_keys:
        try:
            if key in statement.index:
                return statement.loc[key].iloc[0]
        except Exception:
            continue
    return None

def roe_func(ticker: str):
    company = yf.Ticker(ticker)
    try:
        roe = company.info.get('returnOnEquity')
        if roe is None:
            financials = company.financials
            balance_sheet = company.balance_sheet
            net_income = find_financial_item(financials, FINANCIALS_MAP['Net Income'])
            total_equity = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Total Equity'])
            if net_income is not None and total_equity is not None and total_equity != 0:
                roe = net_income / total_equity
            else:
                return None
    except Exception: return None
    return float(roe)

def annual_rev_growth_func(ticker: str):
    company = yf.Ticker(ticker)
    try:    
        financials = company.financials
        current_rev = find_financial_item(financials, FINANCIALS_MAP['Total Revenue'])
        previous_rev = find_financial_item(financials.iloc[:, 1:], FINANCIALS_MAP['Total Revenue'])
        if current_rev is None or previous_rev is None or previous_rev == 0: return None
        return float((current_rev - previous_rev) / previous_rev)
    except Exception: return None

def annual_earnings_growth_func(ticker: str):
    company = yf.Ticker(ticker)
    try:
        financials = company.financials
        current_earnings = find_financial_item(financials, FINANCIALS_MAP['Net Income'])
        previous_earnings = find_financial_item(financials.iloc[:, 1:], FINANCIALS_MAP['Net Income'])
        if current_earnings is None or previous_earnings is None or previous_earnings == 0: return None
        return float((current_earnings - previous_earnings) / previous_earnings)
    except Exception: return None

def current_ratio_func(ticker: str):
    company = yf.Ticker(ticker)
    try:
        balance_sheet = company.balance_sheet
        current_assets = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Current Assets'])
        current_liabilities = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Current Liabilities'])
        if current_assets is None or current_liabilities is None or current_liabilities == 0: return None
        return float(current_assets / current_liabilities)
    except Exception: return None

def quick_ratio_func(ticker: str):
    company = yf.Ticker(ticker)
    try:
        balance_sheet = company.balance_sheet
        cash = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Cash']) or 0
        st_investments = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['ST Investments']) or 0
        accounts_receivable = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Receivables']) or 0
        current_liabilities = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Current Liabilities'])
        if current_liabilities is None or current_liabilities == 0: return None
        return float((cash + st_investments + accounts_receivable) / current_liabilities)
    except Exception: return None

def debt_equity_ratio_func(ticker: str):
    company = yf.Ticker(ticker)
    try:
        balance_sheet = company.balance_sheet
        total_debt = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Total Debt'])
        total_equity = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Total Equity'])
        if total_debt is None or total_equity is None or total_equity == 0: return None
        return float(total_debt / total_equity)
    except Exception: return None

def debt_asset_ratio_func(ticker: str):
    company = yf.Ticker(ticker)
    try:
        balance_sheet = company.balance_sheet
        total_debt = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Total Debt'])
        total_assets = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Total Assets'])
        if total_debt is None or total_assets is None or total_assets == 0: return None
        return float(total_debt / total_assets)
    except Exception: return None

def operating_efficiency_ratio_func(ticker: str):
    company = yf.Ticker(ticker)
    try:
        financials = company.financials
        operating_income = find_financial_item(financials, FINANCIALS_MAP['Operating Income'])
        if operating_income is None:
            pretax_income = find_financial_item(financials, FINANCIALS_MAP['Pretax Income'])
            interest_expense = find_financial_item(financials, FINANCIALS_MAP['Interest Expense']) or 0
            interest_income = find_financial_item(financials, FINANCIALS_MAP['Interest Income']) or 0
            if pretax_income is not None:
                operating_income = pretax_income + interest_expense - interest_income
            else:
                return None
        total_rev = find_financial_item(financials, FINANCIALS_MAP['Total Revenue'])
        if total_rev is None or total_rev == 0: return None
        return float(operating_income / total_rev)
    except Exception: return None

def compare_against_sector(ticker):
    sectors = list(SECTOR_STANDARDS.keys())
    company = yf.Ticker(ticker)
    try:
        sector = company.info['sector']
    except:
        return None
        
    if sector in sectors:
        sector_standards = SECTOR_STANDARDS.get(sector)
        unrounded_company_values = [
            roe_func(ticker),
            annual_rev_growth_func(ticker),
            annual_earnings_growth_func(ticker),
            current_ratio_func(ticker),
            quick_ratio_func(ticker),
            debt_equity_ratio_func(ticker),
            debt_asset_ratio_func(ticker),
            operating_efficiency_ratio_func(ticker)
            ]
        company_values = []
        
        for value in unrounded_company_values:
            if value != None:
                company_values.append(round(value, 2))
            else:
                company_values.append(value)
        comparison_output = []
        for i in range(len(company_values)):
            if company_values[i] != None:
                unrounded_difference = company_values[i] - sector_standards[i]
                comparison_output.append(round(unrounded_difference,2))
            else:
                comparison_output.append(None)
    else:
        return None
    return (sector_standards, company_values, comparison_output)

def comparison_data_to_dictionary(ticker):
    company = yf.Ticker(ticker)
    try:
        sector = company.info['sector']
        name = company.info['longName']
    except:
        return {'error': 'Ticker not found or missing info'}

    comparison_data = compare_against_sector(ticker)
    if not comparison_data:
         return {'error': 'Sector not supported or data unavailable'}

    sector_standards = comparison_data[0]
    company_values = comparison_data[1]
    comparison_output = comparison_data[2]
    
    final_output = {
        "Ticker": ticker,
        "Company": name,
        "Sector": sector
        }
        
    for i in range(len(sector_standards)):
        if company_values[i] != None:
            if i <= 4:
                if comparison_output[i] >= 0:
                    intrinsic = "Strong" 
                else:
                    intrinsic = "Weak"
            else:
                if comparison_output[i] > 0:
                    intrinsic = "Weak"
                else:
                    intrinsic = "Strong"
            
            data_list = [sector_standards[i], company_values[i], comparison_output[i], intrinsic]
            data_addition = {METRICS[i]: data_list}
            final_output.update(data_addition)
        else:
            data_addition = {METRICS[i]: [sector_standards[i], "--", "--", "--"]} 
            final_output.update(data_addition)

    return final_output

def visualize_fundamentals(data_output):
    """Adapted from ttfh-v2 for integrated use."""
    if 'error' in data_output:
        print(f"Fundamental Analysis Error: {data_output['error']}")
        return

    print(f"\n--- Running Fundamental Analysis for {data_output.get('Ticker', '')} ---")

    ticker = data_output.get('Ticker', 'N/A')
    company = data_output.get('Company', 'N/A')
    sector = data_output.get('Sector', 'N/A')
    
    columns = ["Metric", "Sector Benchmark", "Company Value", "Difference", "Implication"]
    rows = []
    
    for metric, values in data_output.items():
        if metric in ['Ticker', 'Company', 'Sector', 'error']:
            continue
        row_data = [metric] + [str(v) if v is not None else "--" for v in values]
        rows.append(row_data)

    # Use a separate figure to avoid style conflict
    plt.style.use('default') 
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['text.color'] = 'white'
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')
    ax.axis('tight')

    title_text = f"Financial Analysis: {company} ({ticker})\nSector: {sector}"
    plt.title(title_text, color='white', fontsize=16, pad=20, weight='bold')

    col_widths = [0.30, 0.2, 0.15, 0.15, 0.2]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colWidths=col_widths
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#333333')
        else:
            cell.set_facecolor('black')
            cell.set_text_props(color='white')
            if col == 4:
                text_obj = cell.get_text()
                content = text_obj.get_text()
                if content == "Strong":
                    text_obj.set_color('#00FF00')
                    text_obj.set_weight('bold')
                elif content == "Weak":
                    text_obj.set_color('#FF3333')
                    text_obj.set_weight('bold')

    print(f"Displaying Fundamental Analysis for {ticker} (Close window to continue)...")
    plt.show()

def run_fundamental_analysis(ticker):
    data = comparison_data_to_dictionary(ticker)
    visualize_fundamentals(data)


# ==========================================
# MAIN EXECUTION LOOP
# ==========================================

def main():
    print("=== MULTI-STRATEGY PORTFOLIO ANALYZER ===")
    portfolio_data = {"tickers": [], "num_shares": []}
    
    # 1. Input Loop
    while True:
        ticker = input('Enter a ticker ("!" to stop/run): ').upper().strip()
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
            print("Invalid number of shares. Try again.")

    if not portfolio_data["tickers"]:
        print("\nNo input detected. Using default example ""portfolio.")
        portfolio_data = {"tickers": ["CF", "BKR", "MRK", "PINS"], "num_shares": [11, 11, 11, 11]}

    # 2. Run Aggregate Portfolio Analysis
    run_portfolio_analysis(portfolio_data)

    # 3. Run Individual Ticker Analysis
    print("\n=== STARTING INDIVIDUAL TICKER DEEP DIVES ===")
    for ticker in portfolio_data["tickers"]:
        print(f"\nProcessing {ticker}...")
        
        # Run Stochastic
        try:
            run_stochastic_analysis(ticker)
        except Exception as e:
            print(f"Error running Stochastic for {ticker}: {e}")

        # Run Monte Carlo
        try:
            run_monte_carlo_analysis(ticker)
        except Exception as e:
            print(f"Error running Monte Carlo for {ticker}: {e}")
            
        # Run Fundamentals (New Module 4)
        try:
            run_fundamental_analysis(ticker)
        except Exception as e:
            print(f"Error running Fundamentals for {ticker}: {e}")

    print("\nAll analyses complete.")

if __name__ == "__main__":
    main()