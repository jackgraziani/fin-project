import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any, List, Union
import json

# --- Helper Functions for Robustness ---

def find_financial_item(
    statement: pd.DataFrame, 
    key_mapping: Union[str, List[str]]
) -> Optional[float]:
    """
    Attempts to find a financial item in the statement's index using a list
    of possible names and returns its value for the most recent period.
    
    Args:
        statement: The financial statement (e.g., balance_sheet or financials) DataFrame.
        key_mapping: A string or a list of possible names for the item.
        
    Returns:
        The value of the financial item for the most recent period, or None if not found.
    """
    # Ensure key_mapping is a list for iteration
    search_keys = [key_mapping] if isinstance(key_mapping, str) else key_mapping
    
    for key in search_keys:
        try:
            # Check if the key exists in the index (case-sensitive check)
            if key in statement.index:
                # Return the value for the most recent period (column index 0)
                return statement.loc[key].iloc[0]
        except Exception:
            # Continue to the next key if an error occurs (e.g., trying to access iloc[0] 
            # on a series that has an NaN or is malformed, though 'in statement.index' 
            # should handle most of this)
            continue
            
    return None

# Mappings for Inconsistent Financial Statement Names
# Note: yfinance often normalizes to certain names, but alternatives are good for safety.
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
    'Total Revenue': ['Total Revenue', 'Revenue', 'Operating Revenue'], # Added 'Operating Revenue' as a potential alias
    'Net Income': ['Net Income', 'Net Income Common Stockholders'],
    # Add new keys for the calculation fallback
    'Operating Income': ['Operating Income', 'Operating Expense'], # Keep this for direct lookup first
    'Pretax Income': ['Pretax Income', 'EBT', 'Earnings Before Taxes'], 
    'Interest Expense': ['Interest Expense'],
    'Interest Income': ['Interest Income', 'Net Interest Income'], # Net Interest Income can also be used as a proxy
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

def roe_func(ticker: str):
    """Calculates or retrieves Return on Equity (ROE)."""
    company = yf.Ticker(ticker)
    try:
        # ROE is often available and reliable in the .info dictionary
        roe = company.info.get('returnOnEquity')
        if roe is None:
            # Fallback calculation if not in .info: Net Income / Total Equity
            financials = company.financials
            balance_sheet = company.balance_sheet
            
            net_income = find_financial_item(financials, FINANCIALS_MAP['Net Income'])
            total_equity = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Total Equity'])
            
            if net_income is not None and total_equity is not None and total_equity != 0:
                roe = net_income / total_equity
            else:
                raise ValueError("Required financial items for ROE not found.")
                
    except Exception as e:
        print(f"ROE data not available (Error: {e})")
        return None
        
    return float(roe)

def annual_rev_growth_func(ticker: str):
    """Calculates Annual Revenue Growth."""
    company = yf.Ticker(ticker)
    try:    
        financials = company.financials
        # Get historical data for the last two periods
        current_rev = find_financial_item(financials, FINANCIALS_MAP['Total Revenue'])
        previous_rev = find_financial_item(financials.iloc[:, 1:], FINANCIALS_MAP['Total Revenue']) # Search in the second column
        
        if current_rev is None or previous_rev is None or previous_rev == 0:
            raise ValueError("Required revenue data for two periods not available or previous revenue is zero.")
            
        annual_rev_growth = (current_rev - previous_rev) / previous_rev
        
    except Exception as e:
        print(f"Could not calculate Annual Revenue Growth (Error: {e})")
        return None
    return float(annual_rev_growth)

def annual_earnings_growth_func(ticker: str):
    """Calculates Annual Earnings Growth (based on Net Income)."""
    company = yf.Ticker(ticker)
    try:
        financials = company.financials
        
        # Get historical data for the last two periods
        current_earnings = find_financial_item(financials, FINANCIALS_MAP['Net Income'])
        previous_earnings = find_financial_item(financials.iloc[:, 1:], FINANCIALS_MAP['Net Income']) # Search in the second column
        
        if current_earnings is None or previous_earnings is None or previous_earnings == 0:
            raise ValueError("Required earnings data for two periods not available or previous earnings is zero.")
            
        annual_earnings_growth = (current_earnings - previous_earnings) / previous_earnings
        
    except Exception as e:
        print(f"Could not calculate Annual Earnings Growth (Error: {e})")
        return None
    return float(annual_earnings_growth)

def current_ratio_func(ticker: str):
    """Calculates Current Ratio (Current Assets / Current Liabilities)."""
    company = yf.Ticker(ticker)
    try:
        balance_sheet = company.balance_sheet
        #print(balance_sheet.index.tolist())
        
        current_assets = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Current Assets'])
        current_liabilities = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Current Liabilities'])

        if current_assets is None or current_liabilities is None:
            raise ValueError("Current Assets or Current Liabilities data not found.")
        if current_liabilities == 0:
            raise ValueError("Current Liabilities is zero, cannot divide.")
            
        current_ratio = current_assets / current_liabilities
        
    except Exception as e:
        print(f"Could not calculate Current Ratio (Error: {e})")
        return None
    return float(current_ratio)

def quick_ratio_func(ticker: str):
    """
    Calculates Quick Ratio (Cash + ST Investments + Receivables) / Current Liabilities).
    The quick assets component is made robust by summing found items.
    """
    company = yf.Ticker(ticker)
    try:
        balance_sheet = company.balance_sheet
        
        cash = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Cash']) or 0
        st_investments = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['ST Investments']) or 0
        accounts_receivable = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Receivables']) or 0
        current_liabilities = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Current Liabilities'])
        
        # Check if at least one quick asset and current liabilities were found
        if current_liabilities is None:
             raise ValueError("Current Liabilities data not found.")
        if current_liabilities == 0:
            raise ValueError("Current Liabilities is zero, cannot divide.")

        quick_assets = cash + st_investments + accounts_receivable
        quick_ratio = quick_assets / current_liabilities
        
    except Exception as e:
        print(f"Could not calculate Quick Ratio (Error: {e})")
        return None
    return float(quick_ratio)

def debt_equity_ratio_func(ticker: str):
    """Calculates Debt-to-Equity Ratio (Total Debt / Total Equity)."""
    company = yf.Ticker(ticker)
    try:
        balance_sheet = company.balance_sheet
        
        total_debt = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Total Debt'])
        total_equity = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Total Equity'])
        
        if total_debt is None or total_equity is None:
            raise ValueError("Total Debt or Total Equity data not found.")
        if total_equity == 0:
            raise ValueError("Total Equity is zero, cannot divide.")
            
        debt_equity_ratio = total_debt / total_equity
        
    except Exception as e:
        print(f"Could not calculate Debt-to-Equity Ratio (Error: {e})")
        return None
    return float(debt_equity_ratio)

def debt_asset_ratio_func(ticker: str):
    """Calculates Debt-to-Asset Ratio (Total Debt / Total Assets)."""
    company = yf.Ticker(ticker)
    try:
        balance_sheet = company.balance_sheet
        
        total_debt = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Total Debt'])
        total_assets = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Total Assets'])
        
        if total_debt is None or total_assets is None:
            raise ValueError("Total Debt or Total Assets data not found.")
        if total_assets == 0:
            raise ValueError("Total Assets is zero, cannot divide.")
            
        debt_asset_ratio = total_debt / total_assets
        
    except Exception as e:
        print(f"Could not calculate Debt-to-Asset Ratio (Error: {e})")
        return None
    return float(debt_asset_ratio)

def operating_efficiency_ratio_func(ticker: str):
    """
    Calculates Operating Efficiency Ratio (Operating Income / Total Revenue).
    Includes a fallback calculation for Operating Income.
    """
    company = yf.Ticker(ticker)
    try:
        financials = company.financials
        
        # 1. Try to find Operating Income directly
        operating_income = find_financial_item(financials, FINANCIALS_MAP['Operating Income'])
        
        # 2. Fallback: Calculate Operating Income if not found
        if operating_income is None:
            pretax_income = find_financial_item(financials, FINANCIALS_MAP['Pretax Income'])
            interest_expense = find_financial_item(financials, FINANCIALS_MAP['Interest Expense']) or 0
            interest_income = find_financial_item(financials, FINANCIALS_MAP['Interest Income']) or 0
            
            if pretax_income is not None:
                # Calculate: Operating Income = Pretax Income + Interest Expense - Interest Income
                # Note: Interest Expense is an EXPENSE (negative impact on Pretax Income), so we ADD it back.
                # Interest Income is INCOME (positive impact), so we SUBTRACT it out.
                operating_income = pretax_income + interest_expense - interest_income
                
            else:
                # Still couldn't find/calculate Operating Income
                raise ValueError("Operating Income, or components needed (Pretax Income/Interest), are missing.")
                
        # Proceed with the ratio calculation
        total_rev = find_financial_item(financials, FINANCIALS_MAP['Total Revenue'])
        
        if total_rev is None:
            raise ValueError("Total Revenue data not found.")
        if total_rev == 0:
            raise ValueError("Total Revenue is zero, cannot divide.")
            
        operating_efficiency_ratio = operating_income / total_rev
        
    except Exception as e:
        print(f"Could not calculate Operating Efficiency Ratio (Error: {e})")
        return None
    return float(operating_efficiency_ratio)

def compare_against_sector(ticker):
    sectors = [
        'Technology',
        'Communication Services',
        'Consumer Cyclical',
        'Financial Services',
        'Healthcare',
        'Consumer Defensive',
        'Energy',
        'Industrials',
        'Utilities',
        'Real Estate',
        'Basic Materials'
        ]
    company = yf.Ticker(ticker)
    sector = company.info['sector']
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
    sector = company.info['sector']
    name = company.info['longName']
    comparison_data = compare_against_sector(ticker)
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
                    intrinsic = "Good"
                else:
                    intrinsic = "Bad"
            else:
                if comparison_output[i] > 0:
                    intrinsic = "Bad"
                else:
                    intrinsic = "Good"
            # if metric is within the first five (refer to METRICS), then being above the benchmark is good. for the other 3, being above is bad.
            data_list = [sector_standards[i], company_values[i], comparison_output[i], intrinsic]
            data_addition = {METRICS[i]: data_list}
            final_output.update(data_addition)
        else:
            data_addition = {METRICS[i]: [sector_standards[i], "--", "--", "--"]} # Data Unavailable
            final_output.update(data_addition)

    return final_output

import math

import math

def print_financial_table(data_output):
    """
    Prints a dictionary of financial metrics in a clean, formatted ASCII table.

    The table is split into a high-level summary and a detailed metric breakdown.
    The detailed metrics (numerical values) are formatted to two decimal places
    for consistency, while also handling 'DU' (Data Unavailable) strings.
    """
    
    # --- 1. Extract High-Level Summary ---
    ticker = data_output.get('Ticker', 'N/A')
    company = data_output.get('Company', 'N/A')
    sector = data_output.get('Sector', 'N/A')

    # --- 2. Configuration for Table Formatting ---
    # Define desired widths for each column
    WIDTH_METRIC = 30
    WIDTH_BENCHMARK = 18
    WIDTH_VALUE = 18
    WIDTH_DIFF = 12
    WIDTH_IMPLICATION = 25
    
    # Calculate the total width of the table border-to-border
    TOTAL_WIDTH = WIDTH_METRIC + WIDTH_BENCHMARK + WIDTH_VALUE + WIDTH_DIFF + WIDTH_IMPLICATION + 5
    
    # Separator line creation
    SEPARATOR = "-" * TOTAL_WIDTH

    print(SEPARATOR)
    print(f"|{' ' * (TOTAL_WIDTH - 2)}|")
    
    # Print the summary header block
    print(f"| {'Ticker:':<10}{ticker:<{TOTAL_WIDTH - 14}}|")
    print(f"| {'Company:':<10}{company:<{TOTAL_WIDTH - 14}}|")
    print(f"| {'Sector:':<10}{sector:<{TOTAL_WIDTH - 14}}|")
    
    print(f"|{' ' * (TOTAL_WIDTH - 2)}|")
    print(SEPARATOR)
    
    # --- 3. Print Detailed Table Header ---
    
    # Header format string for column titles
    HEADER_FORMAT = (
        f"|{{:^{WIDTH_METRIC}}}|{{:^{WIDTH_BENCHMARK}}}"
        f"|{{:^{WIDTH_VALUE}}}|{{:^{WIDTH_DIFF}}}|{{:^{WIDTH_IMPLICATION}}}|"
    )
    
    print(HEADER_FORMAT.format(
        "Financial Metric", 
        "Sector Benchmark", 
        "Company Value", 
        "Difference", 
        "Financial Implication"
    ))
    print(SEPARATOR)

    # Helper function to format value or handle 'DU'
    def format_cell(value, width):
        """Formats a numerical value to two decimal places, or centers 'DU'/'N/A'."""
        if isinstance(value, str) and value.upper() == "--":
            # Center the 'DU' string
            return f"{value.upper():^{width}}"
        try:
            # Format as a float with 2 decimal places
            return f"{value:^{width}.2f}"
        except (ValueError, TypeError):
            # Fallback for non-DU strings or non-numeric types
            return f"{'N/A':^{width}}"

    # --- 4. Print Detailed Table Rows ---

    for metric, values in data_output.items():
        # Skip the header keys
        if metric in ['Ticker', 'Company', 'Sector']:
            continue
            
        # Ensure the value is a list of exactly 4 elements: [Bench, Value, Diff, Implication]
        if isinstance(values, list) and len(values) == 4:
            benchmark, company_value, difference, implication = values
            
            # 1. Format the Metric (Left aligned string)
            metric_cell = f"{metric:<{WIDTH_METRIC}}"
            
            # 2. Format numerical/DU cells
            # We use the helper function for the three numerical columns
            benchmark_cell = format_cell(benchmark, WIDTH_BENCHMARK)
            value_cell = format_cell(company_value, WIDTH_VALUE)
            diff_cell = format_cell(difference, WIDTH_DIFF)
            
            # 3. Format Implication (Center aligned string)
            implication_cell = f"{implication:^{WIDTH_IMPLICATION}}"
            
            # Construct and print the final row
            row = (
                f"|{metric_cell}|{benchmark_cell}|{value_cell}|"
                f"{diff_cell}|{implication_cell}|"
            )
            print(row)
        else:
            # Handle unexpected data format gracefully
            print(f"| {metric:<{WIDTH_METRIC - 1}} | {'ERROR: Data format mismatch':<{TOTAL_WIDTH - WIDTH_METRIC - 3}} |")
            
    print(SEPARATOR)

def ticker_to_dict(ticker):
    check_company_existence = yf.Ticker(ticker)
    try:
        sector = check_company_existence.info['sector']
        name = check_company_existence.info['longName']
        output = comparison_data_to_dictionary(ticker)
        return output
    except Exception as e:
        return {'error': 'Ticker not found'}

def main():
    # print(balance_sheet.index.tolist())
    ticker = "CMPR"
    o = ticker_to_dict(ticker)
    print(o)
    print_financial_table(o)

if __name__ == "__main__":
    main()