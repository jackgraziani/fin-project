import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
N_PERIODS = 14   
M_PERIODS = 3    
OVERSOLD_LEVEL = 20
OVERBOUGHT_LEVEL = 80
TREND_FILTER_PERIOD = 30 # Simple Moving Average to identify trend

# --- 1. DATA RETRIEVAL FUNCTION ---
def fetch_historical_data(ticker):
    """Fetches data and handles multi-index columns from yfinance."""
    stock_data = yf.download(ticker, period="100d", interval="1d", progress=False)
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    clean_data = stock_data[['High', 'Low', 'Close']].copy()
    return clean_data

# --- 2. INDICATOR CALCULATION ---
def calculate_stochastic_oscillator(df, n=14, m=3):
    """Calculates the Fast Stochastic Oscillator and Trend Filter."""
    df['LL'] = df['Low'].rolling(window=n).min()
    df['HH'] = df['High'].rolling(window=n).max()

    df['%K'] = 100 * ((df['Close'] - df['LL']) / (df['HH'] - df['LL']))
    df['%D'] = df['%K'].rolling(window=m).mean()
    
    # Add a Trend Filter (SMA 30)
    df['SMA_Trend'] = df['Close'].rolling(window=TREND_FILTER_PERIOD).mean()
    
    df = df.drop(columns=['LL', 'HH']).round(2)
    return df

# --- 3. TRADING SIGNAL & DIVERGENCE GENERATION ---
def generate_stochastic_signals(df, oversold=20, overbought=80):
    """
    Generates signals with a trend filter to avoid 'false' sells in strong uptrends.
    """
    df['Signal'] = 'Hold'
    
    # Buy Condition
    buy_condition = (
        (df['%K'].shift(1) < df['%D'].shift(1)) & 
        (df['%K'] > df['%D']) &                   
        (df['%D'] < oversold)                     
    )
    df.loc[buy_condition, 'Signal'] = 'Buy'
    
    # Sell Condition (Filtered by Trend)
    sell_condition = (
        (df['%K'].shift(1) > df['%D'].shift(1)) &  
        (df['%K'] < df['%D']) &                   
        (df['%D'] > overbought) &
        (df['Close'] < df['SMA_Trend']) 
    )
    df.loc[sell_condition, 'Signal'] = 'Sell'
    
    # Divergence Detection
    price_slope = df['Close'].diff(3)
    k_slope = df['%K'].diff(3)
    df['Divergence'] = ""
    df.loc[(price_slope < 0) & (k_slope > 0) & (df['%K'] < 35), 'Divergence'] = 'Bullish Div.'
    df.loc[(price_slope > 0) & (k_slope < 0) & (df['%K'] > 65), 'Divergence'] = 'Bearish Div.'
    
    return df

def input_user_ticker():
    still_running = True
    while still_running:
        ticker = input("Enter a ticker (e.g., AAPL): ").upper()
        if not ticker: continue
        data = yf.Ticker(ticker)
        history = data.history(period="1d")
        if history.empty:
            print("Please enter a valid ticker.")
        else:
            still_running = False
    return ticker

# --- 4. VISUALIZATION ---
def plot_stochastic_results(df, ticker):
    plt.style.use('dark_background')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, 
                                   gridspec_kw={'height_ratios': [1.5, 1]})
    
    # Plot Price and Trend
    ax1.plot(df.index, df['Close'], color='cyan', alpha=0.6, label='Close Price')
    ax1.plot(df.index, df['SMA_Trend'], color='magenta', linestyle='--', alpha=0.5, label='Trend (SMA 50)')
    
    buys = df[df['Signal'] == 'Buy']
    ax1.scatter(buys.index, buys['Close'], marker='^', color='#00ff00', s=100, label='Filtered Buy', zorder=5)
    
    sells = df[df['Signal'] == 'Sell']
    ax1.scatter(sells.index, sells['Close'], marker='v', color='#ff3333', s=100, label='Filtered Sell', zorder=5)

    ax1.set_title(f"{ticker}: Filtered Stochastic Analysis", fontsize=14, color='white')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left', fontsize='small')
    ax1.grid(color='gray', linestyle='--', alpha=0.2)

    # Plot Oscillator
    ax2.plot(df.index, df['%K'], color='white', linewidth=1, label='%K')
    ax2.plot(df.index, df['%D'], color='orange', linewidth=1, label='%D')
    ax2.axhline(OVERBOUGHT_LEVEL, color='#ff3333', linestyle='--', alpha=0.4)
    ax2.axhline(OVERSOLD_LEVEL, color='#00ff00', linestyle='--', alpha=0.4)
    
    # --- ADDED: Bullish and Bearish Divergence Plotting ---
    bull_div = df[df['Divergence'] == 'Bullish Div.']
    bear_div = df[df['Divergence'] == 'Bearish Div.']
    
    ax2.scatter(bull_div.index, bull_div['%K'], color='#00ff00', marker='+', s=60, label='Bullish Div')
    ax2.scatter(bear_div.index, bear_div['%K'], color='#ff3333', marker='x', s=60, label='Bearish Div')

    ax2.set_ylabel('Oscillator')
    ax2.set_ylim(-5, 105)
    
    # Legend at the top of the pane to avoid overlapping lines
    ax2.legend(loc='upper left', ncol=4, fontsize='x-small', frameon=False)
    ax2.grid(color='gray', linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    plt.show()

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    ticker = input_user_ticker()
    price_data = fetch_historical_data(ticker) 
    indicator_data = calculate_stochastic_oscillator(price_data, N_PERIODS, M_PERIODS)
    final_results = generate_stochastic_signals(indicator_data, OVERSOLD_LEVEL, OVERBOUGHT_LEVEL)
    
    plot_data = final_results.dropna()
    plot_stochastic_results(plot_data, ticker)