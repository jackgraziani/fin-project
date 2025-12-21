import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, date, timedelta
import warnings

# --- Configuration ---
TRADING_DAYS_PER_YEAR = 252
LOOKBACK_YEARS = 5
US_MARKET_CLOSE_TIME = time(16, 30)

def return_prev_close_and_current(ticker_string):
    """
    Retrieves the previous day's close price and the current price for a given ticker.
    Uses auto_adjust=True to ensure dividend/split consistency.
    """
    try:
        ticker = yf.Ticker(ticker_string)
        # Using 2d history is more reliable than .info for consistent price pairs
        hist = ticker.history(period="2d", auto_adjust=True)
        
        precision = 4 if ticker_string == "^TNX" else 2

        if len(hist) < 2:
            # Fallback for current price if history is lagging or it's a new ticker
            previous_close = ticker.info.get('previousClose')
            current_price = ticker.info.get('regularMarketPrice') or ticker.info.get('currentPrice')
            
            if previous_close is None or current_price is None:
                return None
        else:
            previous_close = hist['Close'].iloc[-2]
            current_price = hist['Close'].iloc[-1]

        return [
            round(float(previous_close), precision), 
            round(float(current_price), precision)
        ]

    except Exception as e:
        print(f"General error retrieving data for {ticker_string}: {e}")
        return None

def run_calcs(portfolio_data):
    """
    Calculates the dollar and percent change for the entire portfolio 
    between the previous close and the current price.
    """
    tickers = portfolio_data["tickers"]
    num_shares = portfolio_data["num_shares"]
    price_data = []
    
    for ticker in tickers:
        price_data.append(return_prev_close_and_current(ticker))
    
    if any(p is None for p in price_data):
        print("Warning: One or more tickers failed to retrieve data. Returning 0 change.")
        return [0.00, 0.0000]

    total_portfolio_at_open = 0
    for i in range(len(tickers)):
        total_portfolio_at_open += num_shares[i] * price_data[i][0]

    total_portfolio_current = 0
    for i in range(len(tickers)):
        total_portfolio_current += num_shares[i] * price_data[i][1]
    
    dollar_change = round(total_portfolio_current - total_portfolio_at_open, 2)
    
    if total_portfolio_at_open == 0:
        percent_change = 0.0000
    else:
        percent_change = round(dollar_change / total_portfolio_at_open, 4)
        
    return [dollar_change, percent_change]

def calculate_beta(portfolio_tickers, market_ticker, lookback_years, num_shares):
    """
    Calculates the Beta of the portfolio using weighted historical daily returns.
    """
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=lookback_years)
    
    try:
        all_tickers = [market_ticker] + portfolio_tickers
        
        # Download historical data with auto_adjust=True for splits/dividends
        data = yf.download(all_tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
        
        if isinstance(data, pd.Series):
             data = data.to_frame()

        # 1. Calculate Weights based on current market values
        # Beta must be weighted by how much dollar value you have in each stock
        current_prices = data.iloc[-1]
        values = []
        for i, t in enumerate(portfolio_tickers):
            values.append(num_shares[i] * current_prices[t])
        
        total_val = sum(values)
        weights = [v / total_val for v in values]

        # 2. Calculate Weighted Portfolio Returns
        mkt_returns = data[market_ticker].pct_change()
        port_returns = (data[portfolio_tickers].pct_change() * weights).sum(axis=1)

        combined_returns = pd.DataFrame({
            'Portfolio_Return': port_returns,
            'Market_Return': mkt_returns
        }).dropna()
        
        if len(combined_returns) < 126: 
             return 1.0

        regression_results = np.polyfit(
            combined_returns['Market_Return'], 
            combined_returns['Portfolio_Return'], 
            1
        )
        return regression_results[0] 

    except Exception as e:
        print(f"Error during Beta calculation: {e}. Defaulting to Beta = 1.0.")
        return 1.0 

def alpha(portfolio_gain, benchmark_ticker, portfolio_data): 
    """
    Calculates the daily Alpha of the portfolio using CAPM:
    Alpha = Portfolio_Return - [Rf + Beta * (Market_Return - Rf)]
    """
    try:
        tnx_data = return_prev_close_and_current("^TNX")
        if tnx_data:
            annual_rfr = tnx_data[1] / 100
        else:
            annual_rfr = 0.04
    except Exception:
        annual_rfr = 0.04 
        
    daily_rfr = annual_rfr / TRADING_DAYS_PER_YEAR
    
    portfolio_data_benchmark = {"tickers": [benchmark_ticker], "num_shares": [1]}   
    benchmark_gain = run_calcs(portfolio_data_benchmark)[1]
    
    # Pass full portfolio_data to account for weights in Beta
    beta = calculate_beta(portfolio_data["tickers"], benchmark_ticker, LOOKBACK_YEARS, portfolio_data["num_shares"])
    
    daily_alpha = portfolio_gain - (daily_rfr + beta * (benchmark_gain - daily_rfr))

    return daily_alpha

def get_last_updated_time(reference_ticker):
    """Returns formatted update time string."""
    try:
        ref_ticker = yf.Ticker(reference_ticker)
        ref_info = ref_ticker.info
        market_timestamp = ref_info.get('regularMarketTime') 
        current_datetime = datetime.now()
        
        current_date_str = current_datetime.strftime("%m/%d/%y") 
        current_time_str = current_datetime.strftime("%I:%M%p").replace(" 0", " ")
        
        if market_timestamp:
            data_datetime = datetime.fromtimestamp(market_timestamp)
            data_date_str = data_datetime.strftime("%m/%d/%y")
            
            if data_datetime.date() == current_datetime.date() and current_datetime.time() < US_MARKET_CLOSE_TIME:
                return f"{current_time_str} on {current_date_str}" 
            else:
                return f"4:30PM on {data_date_str}"

        return f"{current_time_str} on {current_date_str} (Live Run Fallback)"
    except Exception:
        current_datetime = datetime.now()
        return f"{current_datetime.strftime('%I:%M%p')} (Error Fallback)"

def main():
    portfolio_data = {"tickers": [], "num_shares": []}
    
    while True:
        ticker = input('Enter a ticker ("!" to stop): ').upper()
        if ticker == "!":
            break
        try:
            val = input("How many shares: ")
            num_shares = int(val)
            portfolio_data["tickers"].append(ticker)
            portfolio_data["num_shares"].append(num_shares)
        except ValueError:
            print("Invalid input.")

    if not portfolio_data["tickers"]:
        print("\nNo input detected. Using example portfolio.")
        portfolio_data = {"tickers": ["BKR", "CF", "MRK", "PINS"], "num_shares": [11, 11, 11, 11]}

    raw_equity_output = run_calcs(portfolio_data)
    
    dollar_change = raw_equity_output[0]
    percent_change = raw_equity_output[1]
    
    if dollar_change < 0:
        formatted_equity = f"-{round(abs(percent_change)*100, 2)}% (-${abs(dollar_change)})"
    else:
        formatted_equity = f"+{round(percent_change*100, 2)}% (+${dollar_change})"
   
    benchmark_ticker = "^GSPC" 
    raw_alpha = alpha(percent_change, benchmark_ticker, portfolio_data)
    formatted_alpha = f"{raw_alpha*100:+.2f}%"

    print("===============")
    last_updated = get_last_updated_time("SPY")
    print(f"Last updated: {last_updated}")
    print()
    print(f"[Daily] Portfolio Return: {formatted_equity}")
    print(f"[Daily] Alpha: {formatted_alpha}")
    print("===============")

if __name__ == "__main__":
    main()