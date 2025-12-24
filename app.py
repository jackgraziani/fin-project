import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, time
import warnings
from typing import Optional, Dict, Any, List, Union

# --- CONFIGURATION ---
st.set_page_config(page_title="FinAnalysis Dashboard", layout="wide")
TRADING_DAYS_PER_YEAR = 252
LOOKBACK_YEARS = 5
warnings.filterfactory = "ignore"

# --- CONSTANTS ---
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
    "Return on Equity", "Annual Revenue Growth", "Annual Earnings Growth",
    "Current Ratio", "Quick Ratio", "Debt-Equity Ratio",
    "Debt-Asset Ratio", "Operating Efficiency Ratio"
]

# ==========================================
# SHARED UTILITIES
# ==========================================

@st.cache_data
def get_current_price_and_volatility(ticker_string):
    try:
        ticker = yf.Ticker(ticker_string)
        hist = ticker.history(period="1y", auto_adjust=True)
        if len(hist) < 2: return None, 0.2
        current_price = hist['Close'].iloc[-1]
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
        volatility = log_returns.std() * np.sqrt(252)
        return float(current_price), float(volatility)
    except Exception: return None, 0.2

@st.cache_data
def get_risk_free_rate():
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if not hist.empty: return hist['Close'].iloc[-1] / 100
        return 0.04
    except: return 0.04

def return_prev_close_and_current(ticker_string):
    try:
        ticker = yf.Ticker(ticker_string)
        hist = ticker.history(period="2d", auto_adjust=True)
        precision = 4 if ticker_string == "^TNX" else 2
        if len(hist) < 2: return None
        return [round(float(hist['Close'].iloc[-2]), precision), round(float(hist['Close'].iloc[-1]), precision)]
    except Exception: return None

@st.cache_data
def calculate_beta(portfolio_tickers, market_ticker, lookback_years, num_shares):
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=lookback_years)
    try:
        all_tickers = [market_ticker] + portfolio_tickers
        data = yf.download(all_tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
        if isinstance(data, pd.Series): return 1.0
        
        # Calculate weights based on last available price
        current_prices = data.iloc[-1]
        values = []
        for i, t in enumerate(portfolio_tickers):
            if t in current_prices: values.append(num_shares[i] * current_prices[t])
            else: values.append(0)
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
    except Exception: return 1.0 

# ==========================================
# MODULE 1: PORTFOLIO PERFORMANCE
# ==========================================

def run_portfolio_analysis(portfolio_data, benchmark_ticker="^GSPC"):
    st.subheader("1) Portfolio Performance (Daily Alpha)")
    tickers = portfolio_data["tickers"]
    num_shares = portfolio_data["num_shares"]
    
    price_data = [return_prev_close_and_current(t) for t in tickers]
    if any(p is None for p in price_data):
        st.error("Error fetching daily price data.")
        return

    total_open = sum(num_shares[i] * price_data[i][0] for i in range(len(tickers)) if price_data[i])
    total_curr = sum(num_shares[i] * price_data[i][1] for i in range(len(tickers)) if price_data[i])
    dollar_change = round(total_curr - total_open, 2)
    port_return = (dollar_change / total_open) if total_open != 0 else 0.0

    bench_prices = return_prev_close_and_current(benchmark_ticker)
    bench_return = (bench_prices[1] - bench_prices[0]) / bench_prices[0] if bench_prices else 0.0

    rfr = get_risk_free_rate()
    daily_rfr = rfr / TRADING_DAYS_PER_YEAR
    beta = calculate_beta(tickers, benchmark_ticker, LOOKBACK_YEARS, num_shares)
    daily_alpha = port_return - (daily_rfr + beta * (bench_return - daily_rfr))

    # Metrics Display
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Return", f"{port_return:.2%}", f"{dollar_change:.2f}")
    c2.metric("Benchmark Return", f"{bench_return:.2%}")
    c3.metric("Alpha", f"{daily_alpha:.2%}")
    c4.metric("Beta", f"{beta:.2f}")

    # Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = ['Portfolio', 'Benchmark', 'Alpha']
    values = [port_return * 100, bench_return * 100, daily_alpha * 100]
    colors = ['#3498db', '#95a5a6', '#2ecc71' if daily_alpha >= 0 else '#e74c3c']
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0, color='white', linewidth=1)
    ax.set_ylabel('Daily Return (%)')
    
    # Dynamic Y-Limit to fix overlap
    max_height = max([abs(v) for v in values]) if values else 1.0
    y_limit = max_height * 1.4
    ax.set_ylim(-y_limit, y_limit)

    for bar in bars:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        xy_text_pos = 5 if height >= 0 else -15 
        ax.annotate(f'{height:.2f}%', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, xy_text_pos),
                    textcoords="offset points",
                    ha='center', va=va, color='white', fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    st.pyplot(fig)

# ==========================================
# MODULE 2: MONTE CARLO OPTION PRICING (LSM)
# ==========================================

def black_scholes_call(S, K, T, r, sigma):
    """Calculates the theoretical European Call price for benchmark comparison."""
    # Note: For non-dividend paying stocks, American Call Price ~= European Call Price
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def run_monte_carlo_analysis(ticker):
    S0, SIGMA = get_current_price_and_volatility(ticker)
    if S0 is None: return None

    RFR = get_risk_free_rate()
    K = S0 * 1.05  
    T = 1.0
    N_STEPS = 252
    N_PATHS = 2000 
    DT = T / N_STEPS

    # --- Simulation Engine ---
    Z = np.random.standard_normal((N_PATHS, N_STEPS))
    daily_returns = np.exp((RFR - 0.5 * SIGMA**2) * DT + SIGMA * np.sqrt(DT) * Z)
    paths = np.zeros((N_PATHS, N_STEPS + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.cumprod(daily_returns, axis=1)
    
    # --- Option Pricing (Longstaff-Schwartz / LSM) ---
    cashflows = np.maximum(paths[:, -1] - K, 0)
    discount_factor = np.exp(-RFR * DT)

    # Backward Induction Loop
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

    # Calculate final prices and metrics
    option_price = np.mean(cashflows * discount_factor)
    bs_price = black_scholes_call(S0, K, T, RFR, SIGMA)
    breakeven = K + option_price
    probability_itm = np.count_nonzero(cashflows > 0) / N_PATHS
    probability_profit = np.count_nonzero(cashflows > option_price) / N_PATHS

    # --- Plotting ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')

    # Plot muted simulation paths
    ax.plot(paths[:100].T, lw=0.4, color='white', alpha=0.225)

    # Reference Levels
    ax.axhline(S0, color='white', lw=1, label=f'Initial Price (${S0:.2f})', alpha=0.7)
    ax.axhline(K, color='#ff4d4d', linestyle='--', lw=2, label=f'Strike Price (${K:.2f})')
    ax.axhline(breakeven, color='#00ff88', linestyle=':', lw=2, label=f'Risk-Adjusted Breakeven (${breakeven:.2f})')

    # --- Annotations & Metadata ---
    stats_text = (f"LSM American Price: ${option_price:.2f}\n"
                  f"B-S (Euro) Price:   ${bs_price:.2f}\n"
                  f"Prob. of Exercise:  {probability_itm:.1%}\n"
                  f"Prob. of Profit:    {probability_profit:.1%}")
    
    ax.text(0.02, 0.96, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#1e1e1e', edgecolor='#444444', alpha=0.9))

    metadata_text = (f"RFR: {RFR:.1%}\n"
                     "Pricing Model: Longstaff-Schwartz (LSM)\n"
                     "Instrument: American Call Option")

    ax.text(0.98, 0.02, metadata_text, transform=ax.transAxes, 
            fontsize=8, color='#888888', verticalalignment='bottom', 
            horizontalalignment='right', style='italic')

    # Chart Polish
    ax.set_title(f"{ticker}: Monte Carlo (LSM) Option Analysis", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Trading Days", fontsize=10, color='#cccccc')
    ax.set_ylabel("Stock Price ($)", fontsize=10, color='#cccccc')
    ax.grid(color='#333333', linestyle='--', alpha=0.5)
    
    ax.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#444444', fontsize=9)
    
    return fig

# ==========================================
# MODULE 3: STOCHASTIC OSCILLATOR
# ==========================================

@st.cache_data
def run_stochastic_analysis(ticker):
    # Parameters strictly matching v3
    N_PERIODS = 14
    K_SMOOTH = 3
    D_SMOOTH = 3
    SMA_FILTER = 40
    OVERSOLD = 20
    OVERBOUGHT = 80

    stock_data = yf.download(ticker, period="1y", interval="1d", progress=False)
    if stock_data.empty: return None

    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)

    df = stock_data[['High', 'Low', 'Close']].copy()
    
    # --- Indicator Calculation ---
    lowest_low = df['Low'].rolling(window=N_PERIODS).min()
    highest_high = df['High'].rolling(window=N_PERIODS).max()
    fast_k = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    
    df['Slow_%K'] = fast_k.rolling(window=K_SMOOTH).mean()
    df['Slow_%D'] = df['Slow_%K'].rolling(window=D_SMOOTH).mean()
    df['SMA_Trend'] = df['Close'].rolling(window=SMA_FILTER).mean()
    
    df = df.dropna()

    # --- Signal Logic (Strict Match to v3) ---
    # Buy: Crossover + Oversold + PRICE > TREND SMA (Buying the dip in uptrend)
    buy_cond = (
        (df['Slow_%K'] > df['Slow_%D']) & 
        (df['Slow_%K'].shift(1) < df['Slow_%D'].shift(1)) & 
        (df['Slow_%D'] < OVERSOLD) &
        (df['Close'] > df['SMA_Trend'])
    )

    # Sell: Crossover + Overbought (No trend filter for selling)
    sell_cond = (
        (df['Slow_%K'] < df['Slow_%D']) & 
        (df['Slow_%K'].shift(1) > df['Slow_%D'].shift(1)) & 
        (df['Slow_%D'] > OVERBOUGHT)
    )
    
    # --- Plotting (Strict Match to Figure_2.png) ---
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 1])
    
    # Plot 1: Price, Trend, Signals
    ax1.plot(df.index, df['Close'], color='cyan', alpha=0.9, label='Price')
    ax1.plot(df.index, df['SMA_Trend'], color='yellow', alpha=0.6, linestyle='--', label=f'SMA {SMA_FILTER} (Trend)')
    
    buys = df[buy_cond]
    sells = df[sell_cond]
    ax1.scatter(buys.index, buys['Close'], marker='^', color='#00ff00', s=100, label='Buy (Trend Filtered)', zorder=5)
    ax1.scatter(sells.index, sells['Close'], marker='v', color='#ff3333', s=100, label='Sell', zorder=5)
    
    ax1.set_title(f"{ticker} - Slow Stochastic Strategy")
    ax1.legend(loc='upper left', fontsize=8)

    # Plot 2: Oscillator
    ax2.plot(df.index, df['Slow_%K'], color='white', label='Slow %K')
    ax2.plot(df.index, df['Slow_%D'], color='orange', label='Slow %D')
    ax2.axhline(OVERBOUGHT, color='red', linestyle='--', alpha=0.6)
    ax2.axhline(OVERSOLD, color='green', linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig

# ==========================================
# MODULE 4: FUNDAMENTAL ANALYSIS
# ==========================================

def find_financial_item(statement: pd.DataFrame, key_mapping: Union[str, List[str]]) -> Optional[float]:
    search_keys = [key_mapping] if isinstance(key_mapping, str) else key_mapping
    for key in search_keys:
        if key in statement.index:
            return statement.loc[key].iloc[0]
    return None

def roe_func(company):
    try:
        roe = company.info.get('returnOnEquity')
        if roe is not None: return float(roe)
        net_income = find_financial_item(company.financials, FINANCIALS_MAP['Net Income'])
        total_equity = find_financial_item(company.balance_sheet, BALANCE_SHEET_MAP['Total Equity'])
        if net_income and total_equity: return net_income / total_equity
    except: return None

def annual_rev_growth_func(company):
    try:    
        curr = find_financial_item(company.financials, FINANCIALS_MAP['Total Revenue'])
        prev = find_financial_item(company.financials.iloc[:, 1:], FINANCIALS_MAP['Total Revenue'])
        if curr and prev: return (curr - prev) / prev
    except: return None

def annual_earnings_growth_func(company):
    try:
        curr = find_financial_item(company.financials, FINANCIALS_MAP['Net Income'])
        prev = find_financial_item(company.financials.iloc[:, 1:], FINANCIALS_MAP['Net Income'])
        if curr and prev: return (curr - prev) / prev
    except: return None

def current_ratio_func(company):
    try:
        assets = find_financial_item(company.balance_sheet, BALANCE_SHEET_MAP['Current Assets'])
        liabs = find_financial_item(company.balance_sheet, BALANCE_SHEET_MAP['Current Liabilities'])
        if assets and liabs: return assets / liabs
    except: return None

def quick_ratio_func(company):
    try:
        cash = find_financial_item(company.balance_sheet, BALANCE_SHEET_MAP['Cash']) or 0
        st_inv = find_financial_item(company.balance_sheet, BALANCE_SHEET_MAP['ST Investments']) or 0
        rec = find_financial_item(company.balance_sheet, BALANCE_SHEET_MAP['Receivables']) or 0
        liabs = find_financial_item(company.balance_sheet, BALANCE_SHEET_MAP['Current Liabilities'])
        if liabs: return (cash + st_inv + rec) / liabs
    except: return None

def debt_equity_ratio_func(company):
    try:
        debt = find_financial_item(company.balance_sheet, BALANCE_SHEET_MAP['Total Debt'])
        equity = find_financial_item(company.balance_sheet, BALANCE_SHEET_MAP['Total Equity'])
        if debt and equity: return debt / equity
    except: return None

def debt_asset_ratio_func(company):
    try:
        debt = find_financial_item(company.balance_sheet, BALANCE_SHEET_MAP['Total Debt'])
        assets = find_financial_item(company.balance_sheet, BALANCE_SHEET_MAP['Total Assets'])
        if debt and assets: return debt / assets
    except: return None

def operating_efficiency_ratio_func(company):
    try:
        op_inc = find_financial_item(company.financials, FINANCIALS_MAP['Operating Income'])
        rev = find_financial_item(company.financials, FINANCIALS_MAP['Total Revenue'])
        if op_inc and rev: return op_inc / rev
    except: return None

@st.cache_data
def get_fundamental_comparison(ticker):
    company = yf.Ticker(ticker)
    try:
        sector = company.info.get('sector')
    except: return None, None

    if sector not in SECTOR_STANDARDS: return None, None

    standards = SECTOR_STANDARDS[sector]
    
    # Calculate company metrics
    values = [
        roe_func(company),
        annual_rev_growth_func(company),
        annual_earnings_growth_func(company),
        current_ratio_func(company),
        quick_ratio_func(company),
        debt_equity_ratio_func(company),
        debt_asset_ratio_func(company),
        operating_efficiency_ratio_func(company)
    ]
    
    # Create comparison table data
    data = []
    for i, metric in enumerate(METRICS):
        val = values[i]
        std = standards[i]
        
        if val is None:
            data.append([metric, std, "N/A", "N/A", "N/A"])
            continue

        diff = val - std
        # Logic for "Strong" vs "Weak"
        # First 5 metrics: Higher is better (usually)
        # Last 3 (Debt/Equity etc): Lower is better (usually)
        if i <= 4:
            condition = diff >= 0
        else:
            condition = diff <= 0 # Lower debt is strong

        strength = "Strong" if condition else "Weak"
        
        data.append([metric, f"{std:.2f}", f"{val:.2f}", f"{diff:.2f}", strength])

    return pd.DataFrame(data, columns=["Metric", "Benchmark", "Actual", "Diff", "Status"]), sector

# ==========================================
# MAIN APP INTERFACE
# ==========================================

st.title("Portfolio Analysis")
st.write(":grey[© Jack Graziani 2025]")

with st.sidebar:
    st.header("Portfolio Configuration")
    default_tickers = "CF, BKR, MRK, PINS"
    default_shares = "11, 11, 11, 11"
    
    ticker_input = st.text_input("Tickers (comma separated)", default_tickers)
    share_input = st.text_input("Shares (comma separated)", default_shares)
    
    run_btn = st.button("Run Analysis", type="primary")

if run_btn:
    # Parse inputs
    t_list = [x.strip().upper() for x in ticker_input.split(',')]
    try:
        s_list = [int(x.strip()) for x in share_input.split(',')]
    except:
        st.error("Shares must be integers.")
        st.stop()
        
    if len(t_list) != len(s_list):
        st.error("Number of tickers must match number of share counts.")
    else:
        # 1. Portfolio Analysis
        portfolio_data = {"tickers": t_list, "num_shares": s_list}
        run_portfolio_analysis(portfolio_data)

        st.markdown("---")
        st.header("2) Buy/Sell Indicators & Options Pricing")

        # 2. Individual Tickers
        tabs = st.tabs(t_list)
        
        for i, ticker in enumerate(t_list):
            with tabs[i]:
                st.subheader(f"{ticker} Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Stochastic Oscillator")
                    fig_stoch = run_stochastic_analysis(ticker)
                    if fig_stoch: st.pyplot(fig_stoch)
                    else: st.warning("No data.")

                with col2:
                    st.write("#### Monte Carlo Simulation (LSM)")
                    # Updated to only expect a figure return
                    fig_mc = run_monte_carlo_analysis(ticker)
                    if fig_mc:
                        st.pyplot(fig_mc)
                
                st.header("3) Financial Health Analysis (Sector Comparison)")
                df_fund, sector = get_fundamental_comparison(ticker)
                
                if df_fund is not None:
                    st.caption(f"Sector: {sector}")
                    
                    # Stylize the dataframe for the web
                    def color_status(val):
                        color = '#2ecc71' if val == 'Strong' else '#e74c3c'
                        return f'color: {color}; font-weight: bold'

                    st.dataframe(
                        df_fund.style.map(color_status, subset=['Status']),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("Fundamental data or sector benchmark unavailable.")