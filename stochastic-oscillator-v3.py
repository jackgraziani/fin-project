import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Note: "Slow Stochastic" typically uses 14, 3, 3
N_PERIODS = 14   # Lookback
K_SMOOTH = 3     # First smoothing (Fast %D)
D_SMOOTH = 3     # Second smoothing (Slow %D)
SMA_FILTER = 50  # Trend Filter Period

OVERSOLD_LEVEL = 20
OVERBOUGHT_LEVEL = 80

def fetch_historical_data(ticker):
    # Fetch more data to accommodate the 50 SMA calculation
    stock_data = yf.download(ticker, period="1y", interval="1d", progress=False)
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    return stock_data[['High', 'Low', 'Close']].copy()

def calculate_indicators(df):
    """Calculates Slow Stochastic and SMA Filter."""
    
    # 1. Calculate Fast %K
    lowest_low = df['Low'].rolling(window=N_PERIODS).min()
    highest_high = df['High'].rolling(window=N_PERIODS).max()
    fast_k = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    
    # 2. Calculate Slow %K (which is Fast %K smoothed)
    df['Slow_%K'] = fast_k.rolling(window=K_SMOOTH).mean()
    
    # 3. Calculate Slow %D (which is Slow %K smoothed)
    df['Slow_%D'] = df['Slow_%K'].rolling(window=D_SMOOTH).mean()
    
    # 4. Calculate Trend Filter (SMA)
    df['SMA_Trend'] = df['Close'].rolling(window=SMA_FILTER).mean()
    
    return df.dropna()

def generate_signals(df):
    df['Signal'] = 'Hold'
    
    # --- FILTERED BUY ---
    # Logic: Crossover + Oversold + Price is ABOVE the Trend SMA (Buying the dip)
    buy_condition = (
        (df['Slow_%K'] > df['Slow_%D']) &          # Crossover
        (df['Slow_%K'].shift(1) < df['Slow_%D'].shift(1)) & # Confirmed Cross
        (df['Slow_%D'] < OVERSOLD_LEVEL) &         # Value Area
        (df['Close'] > df['SMA_Trend'])            # Trend Filter (Buy dips in uptrends only)
    )
    df.loc[buy_condition, 'Signal'] = 'Buy'
    
    # --- UNFILTERED SELL ---
    # Logic: Standard crossover for selling (taking profit)
    sell_condition = (
        (df['Slow_%K'] < df['Slow_%D']) & 
        (df['Slow_%K'].shift(1) > df['Slow_%D'].shift(1)) &
        (df['Slow_%D'] > OVERBOUGHT_LEVEL)
    )
    df.loc[sell_condition, 'Signal'] = 'Sell'
    
    return df

def plot_results(df, ticker):
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=[2, 1])
    
    # Plot Price & Trend
    ax1.plot(df.index, df['Close'], color='cyan', alpha=0.9, label='Price')
    ax1.plot(df.index, df['SMA_Trend'], color='yellow', alpha=0.6, linestyle='--', label=f'SMA {SMA_FILTER} (Trend)')
    
    # Plot Signals
    buys = df[df['Signal'] == 'Buy']
    sells = df[df['Signal'] == 'Sell']
    ax1.scatter(buys.index, buys['Close'], marker='^', color='#00ff00', s=100, label='Buy (Trend Filtered)', zorder=5)
    ax1.scatter(sells.index, sells['Close'], marker='v', color='#ff3333', s=100, label='Sell', zorder=5)
    
    ax1.set_title(f"{ticker} - Slow Stochastic Strategy", color='white')
    ax1.legend()
    
    # Plot Oscillator
    ax2.plot(df.index, df['Slow_%K'], color='white', label='Slow %K')
    ax2.plot(df.index, df['Slow_%D'], color='orange', label='Slow %D')
    ax2.axhline(OVERBOUGHT_LEVEL, color='red', linestyle='--', alpha=0.6)
    ax2.axhline(OVERSOLD_LEVEL, color='green', linestyle='--', alpha=0.8)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    ticker = input("Enter ticker (e.g. SPY): ").upper()
    data = fetch_historical_data(ticker)
    data = calculate_indicators(data)
    data = generate_signals(data)
    
    # Plotting only the last 100 periods for clarity
    plot_results(data.iloc[-100:], ticker)