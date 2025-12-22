import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
N_PERIODS = 14   # %K Lookback Period
M_PERIODS = 3    # %D Smoothing Period
OVERSOLD_LEVEL = 20
OVERBOUGHT_LEVEL = 80

# --- 1. DATA RETRIEVAL FUNCTION ---
def fetch_historical_data(ticker):
    # Fetching 60d for enough lookback, but we will plot a clear window
    stock_data = yf.download(ticker, period="60d", interval="1d", progress=False)
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    clean_data = stock_data[['High', 'Low', 'Close']].copy()
    return clean_data

# --- 2. INDICATOR CALCULATION ---
def calculate_stochastic_oscillator(df, n=14, m=3):
    """Calculates the Fast Stochastic Oscillator (%K and %D)."""
    df['LL'] = df['Low'].rolling(window=n).min()
    df['HH'] = df['High'].rolling(window=n).max()

    df['%K'] = 100 * ((df['Close'] - df['LL']) / (df['HH'] - df['LL']))
    df['%D'] = df['%K'].rolling(window=m).mean()
    
    df = df.drop(columns=['LL', 'HH']).round(2)
    return df

# --- 3. TRADING SIGNAL & DIVERGENCE GENERATION ---
def generate_stochastic_signals(df, oversold=20, overbought=80):
    """Generates trading signals and divergence alerts."""
    df['Signal'] = 'Hold'
    
    # Standard Crossover Logic
    buy_condition = (
        (df['%K'].shift(1) < df['%D'].shift(1)) & 
        (df['%K'] > df['%D']) &                   
        (df['%D'] < oversold)                     
    )
    df.loc[buy_condition, 'Signal'] = 'Buy'
    
    sell_condition = (
        (df['%K'].shift(1) > df['%D'].shift(1)) &  
        (df['%K'] < df['%D']) &                   
        (df['%D'] > overbought)                   
    )
    df.loc[sell_condition, 'Signal'] = 'Sell'
    
    # Divergence Detection
    price_slope = df['Close'].diff(5)
    k_slope = df['%K'].diff(5)
    df['Divergence'] = ""
    df.loc[(price_slope < 0) & (k_slope > 0), 'Divergence'] = 'Bullish Div.'
    df.loc[(price_slope > 0) & (k_slope < 0), 'Divergence'] = 'Bearish Div.'
    
    return df

def input_user_ticker():
    still_running = True
    while still_running:
        ticker = input("Enter a ticker: ").upper()
        data = yf.Ticker(ticker)
        history = data.history(period="1d")
        if history.empty:
            print("Please enter a valid ticker.")
        else:
            still_running = False
    return ticker

# --- 4. VISUALIZATION ---
def plot_stochastic_results(df, ticker):
    # Set the style to dark
    plt.style.use('dark_background')
    
    # Create subplots (2 rows: Price and Oscillator)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # --- Plot 1: Price and Buy/Sell Signals ---
    ax1.plot(df.index, df['Close'], color='cyan', alpha=0.8, label='Close Price')
    
    # Plot Buy Signals
    buys = df[df['Signal'] == 'Buy']
    ax1.scatter(buys.index, buys['Close'], marker='^', color='#00ff00', s=100, label='Buy Signal', zorder=5)
    
    # Plot Sell Signals
    sells = df[df['Signal'] == 'Sell']
    ax1.scatter(sells.index, sells['Close'], marker='v', color='#ff3333', s=100, label='Sell Signal', zorder=5)

    # Set Title (Updated specifically)
    ax1.set_title(f"{ticker}: Stochastic Oscillator Analysis", fontsize=16, color='white', pad=20)
    ax1.set_ylabel('Price (USD)')
    
    # Reverted: Auto-scale the Y-axis based on price range instead of starting at 0
    # We add a small 5% margin for visual breathing room
    y_min = df['Close'].min() * 0.95
    y_max = df['Close'].max() * 1.05
    ax1.set_ylim(y_min, y_max) 
    
    ax1.legend(loc='upper left')
    ax1.grid(color='gray', linestyle='--', alpha=0.3)

    # --- Plot 2: Stochastic Oscillator ---
    ax2.plot(df.index, df['%K'], color='white', linewidth=1.2, label='%K (Fast)')
    ax2.plot(df.index, df['%D'], color='orange', linewidth=1.2, label='%D (Slow)')
    
    # Oversold/Overbought levels
    ax2.axhline(OVERBOUGHT_LEVEL, color='#ff3333', linestyle='--', alpha=0.5)
    ax2.axhline(OVERSOLD_LEVEL, color='#00ff00', linestyle='--', alpha=0.5)
    ax2.fill_between(df.index, OVERSOLD_LEVEL, OVERBOUGHT_LEVEL, color='gray', alpha=0.1)

    # Annotate Divergences on the indicator pane
    bullish_div = df[df['Divergence'] == 'Bullish Div.']
    bearish_div = df[df['Divergence'] == 'Bearish Div.']
    ax2.scatter(bullish_div.index, bullish_div['%K'], marker='+', color='#00ff00', label='Bullish Div')
    ax2.scatter(bearish_div.index, bearish_div['%K'], marker='+', color='#ff3333', label='Bearish Div')

    ax2.set_ylabel('Oscillator Value')
    ax2.set_ylim(-5, 105)
    ax2.legend(loc='upper left', fontsize='small', ncol=2)
    ax2.grid(color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    ticker = input_user_ticker()
    price_data = fetch_historical_data(ticker)
    indicator_data = calculate_stochastic_oscillator(price_data, N_PERIODS, M_PERIODS)
    final_results = generate_stochastic_signals(indicator_data, OVERSOLD_LEVEL, OVERBOUGHT_LEVEL)
    
    # Remove NaN rows (from rolling window) for cleaner plotting
    plot_data = final_results.dropna()
    
    plot_stochastic_results(plot_data, ticker)