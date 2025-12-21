import pandas as pd
import numpy as np

# --- CONFIGURATION ---
TICKER = 'AAPL'  # Example Ticker
N_PERIODS = 14   # %K Lookback Period
M_PERIODS = 3    # %D Smoothing Period
OVERSOLD_LEVEL = 20
OVERBOUGHT_LEVEL = 80

# --- 1. DATA RETRIEVAL FUNCTION (PLACEHOLDER) ---
def fetch_historical_data(ticker):
    """
    *** YOU MUST REPLACE THIS FUNCTION WITH YOUR ACTUAL API CALL ***
    
    Uses the Massive.com/Polygon.io API to fetch historical price data.
    The data must be a pandas DataFrame with 'High', 'Low', and 'Close' columns.
    
    For a real implementation, you would use requests/urllib to call the 
    API (e.g., /v2/aggs/ticker/{TICKER}/range/{multiplier}/{timespan}/{from}/{to}).
    """
    print(f"Fetching data for {ticker}...")
    
    # --- SAMPLE DATA (For testing the calculations) ---
    data = {
        'Date': pd.to_datetime([
            '2025-11-01', '2025-11-04', '2025-11-05', '2025-11-06', '2025-11-07', '2025-11-08', '2025-11-11',
            '2025-11-12', '2025-11-13', '2025-11-14', '2025-11-17', '2025-11-18', '2025-11-19', '2025-11-20',
            '2025-11-21', '2025-11-24', '2025-11-25', '2025-11-26', '2025-11-27', '2025-11-28' # Day 20
        ]),
        # Price data simulating a 14-period lookback, a drop, and a bounce
        'High': [102.5, 103.1, 104.5, 105.0, 106.5, 107.0, 108.0, 107.5, 109.0, 110.0, 110.5, 111.0, 112.0, 113.0, 113.5, 114.0, 115.0, 116.0, 116.5, 117.0],
        'Low': [100.0, 100.5, 102.0, 103.0, 104.0, 105.0, 106.0, 105.5, 107.0, 108.0, 108.5, 109.0, 110.0, 111.0, 111.5, 112.0, 113.0, 114.0, 114.5, 115.0],
        'Close': [101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 107.0, 108.5, 109.5, 110.0, 110.5, 111.5, 112.5, 113.0, 113.5, 114.5, 115.5, 116.0, 116.5]
    }
    df = pd.DataFrame(data).set_index('Date')
    return df

# --- 2. INDICATOR CALCULATION ---
def calculate_stochastic_oscillator(df, n=14, m=3):
    """Calculates the Fast Stochastic Oscillator (%K and %D)."""

    # Calculate the Lowest Low (LL) and Highest High (HH) over the 'n' periods
    df['LL'] = df['Low'].rolling(window=n).min()
    df['HH'] = df['High'].rolling(window=n).max()

    # Calculate the raw %K (Fast Stochastic)
    # Formula: 100 * (Close - LLn) / (HHn - LLn)
    df['%K'] = 100 * (
        (df['Close'] - df['LL']) / 
        (df['HH'] - df['LL'])
    )

    # Calculate the %D (Signal Line) as the SMA of %K over 'm' periods
    df['%D'] = df['%K'].rolling(window=m).mean()
    
    # Clean up temporary columns for a cleaner output
    df = df.drop(columns=['LL', 'HH']).round(2)

    return df

# --- 3. TRADING SIGNAL GENERATION ---
def generate_stochastic_signals(df, oversold=20, overbought=80):
    """
    Generates trading signals based on %K and %D crossovers in the 
    overbought/oversold regions.
    """
    # Create a new column for the signals, default to 'Hold'
    df['Signal'] = 'Hold'
    
    # Check for the crossover condition in the oversold region (Buy Signal)
    # Logic: %K crosses ABOVE %D AND both lines are below the oversold threshold
    buy_condition = (
        (df['%K'].shift(1) < df['%D'].shift(1)) &  # %K was below %D yesterday
        (df['%K'] > df['%D']) &                   # %K is above %D today (Crossover)
        (df['%D'] < oversold)                     # Crossover happens in oversold area
    )
    df.loc[buy_condition, 'Signal'] = 'Buy'
    
    # Check for the crossover condition in the overbought region (Sell Signal)
    # Logic: %K crosses BELOW %D AND both lines are above the overbought threshold
    sell_condition = (
        (df['%K'].shift(1) > df['%D'].shift(1)) &  # %K was above %D yesterday
        (df['%K'] < df['%D']) &                   # %K is below %D today (Crossover)
        (df['%D'] > overbought)                   # Crossover happens in overbought area
    )
    df.loc[sell_condition, 'Signal'] = 'Sell'
    
    return df

# --- 4. MAIN EXECUTION ---
if __name__ == '__main__':
    # 1. Get the data
    price_data = fetch_historical_data(TICKER)

    # 2. Calculate the indicator
    indicator_data = calculate_stochastic_oscillator(price_data, N_PERIODS, M_PERIODS)
    
    # 3. Generate the signals
    final_results = generate_stochastic_signals(indicator_data, OVERSOLD_LEVEL, OVERBOUGHT_LEVEL)

    # 4. Print the final output
    print("\n--- Final Stochastic Trading Results ---")
    print(f"Ticker: {TICKER} | %K Period: {N_PERIODS} | %D Period: {M_PERIODS}")
    print("\nDisplaying the last 10 periods with signals:\n")
    print(final_results.tail(10))

    # Identify the most recent signal
    latest_signal = final_results['Signal'].iloc[-1]
    latest_k = final_results['%K'].iloc[-1]
    latest_d = final_results['%D'].iloc[-1]
    
    print("\n------------------------------------------------------")
    print(f"🔥 LATEST SIGNAL: **{latest_signal}**")
    print(f"Current %K: {latest_k:.2f} | Current %D: {latest_d:.2f}")
    print("------------------------------------------------------")