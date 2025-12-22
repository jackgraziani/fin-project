import pandas as pd
import yfinance as yf
import numpy as np

# --- CONFIGURATION ---
TICKER = 'AAPL'  # Example Ticker
N_PERIODS = 14   # %K Lookback Period
M_PERIODS = 3    # %D Smoothing Period
OVERSOLD_LEVEL = 20
OVERBOUGHT_LEVEL = 80

# --- 1. DATA RETRIEVAL FUNCTION ---
def fetch_historical_data(ticker):
    stock_data = yf.download(ticker, period="60d", interval="1d", progress=False)
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    # We use a copy to avoid SettingWithCopy warnings when adding columns later
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
    """Generates trading signals, momentum trends, and divergence alerts."""
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
    
    # --- MOMENTUM INSIGHTS ---
    df['%K_Velocity'] = df['%K'].diff(3) 
    
    def get_trend_description(val):
        if pd.isna(val): return "N/A"
        if val > 5: return "Accel. Up"
        if val < -5: return "Accel. Down"
        return "Steady"
    df['Trend'] = df['%K_Velocity'].apply(get_trend_description)

    # --- DIVERGENCE DETECTION (5-day window) ---
    # Compare slope of Price vs slope of %K
    price_slope = df['Close'].diff(5)
    k_slope = df['%K'].diff(5)
    
    df['Divergence'] = ""
    # Bullish: Price down, %K up
    df.loc[(price_slope < 0) & (k_slope > 0), 'Divergence'] = 'Bullish Div.'
    # Bearish: Price up, %K down
    df.loc[(price_slope > 0) & (k_slope < 0), 'Divergence'] = 'Bearish Div.'
    
    return df

def input_user_ticker():
    still_running = True
    while still_running:
        ticker = input("Enter a ticker: ")
        data = yf.Ticker(ticker)
        history = data.history(period="1d")
        if history.empty == True:
            print("Please enter a valid ticker.")
        else:
            still_running = False
    return ticker

# --- 4. MAIN EXECUTION ---
if __name__ == '__main__':
    # 1. Get the data
    ticker = input_user_ticker()

    price_data = fetch_historical_data(ticker)

    # 2. Calculate the indicator
    indicator_data = calculate_stochastic_oscillator(price_data, N_PERIODS, M_PERIODS)
    
    # 3. Generate the signals
    final_results = generate_stochastic_signals(indicator_data, OVERSOLD_LEVEL, OVERBOUGHT_LEVEL)

    # 4. Print the final output
    print("\n--- Final Stochastic Trading Results ---")
    print(f"Ticker: {ticker} | %K Period: {N_PERIODS} | %D Period: {M_PERIODS}")
    print("\nDisplaying the last 10 periods with Trend and Divergence insights:\n")
    
    # Filter columns for readability
    display_cols = ['Close', '%K', '%D', 'Signal', 'Trend', 'Divergence']
    print(final_results[display_cols].tail(10))

    # Identify the most recent data
    latest = final_results.iloc[-1]
    
    print()
    print("===============")
    print(f"Latest Signal: {latest['Signal']}")
    print(f"Momentum:      {latest['Trend']}")
    print(f"Divergence:    {latest['Divergence'] if latest['Divergence'] != '' else 'None'}")
    print(f"Current %K:    {latest['%K']:.2f}")
    print(f"Current %D:    {latest['%D']:.2f}")
    print("===============")