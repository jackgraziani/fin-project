import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Union

# --- Helper Functions for Robustness ---

def find_financial_item(
    statement: pd.DataFrame, 
    key_mapping: Union[str, List[str]]
) -> Optional[float]:
    """
    Attempts to find a financial item in the statement's index using a list
    of possible names and returns its value for the most recent period.
    """
    search_keys = [key_mapping] if isinstance(key_mapping, str) else key_mapping
    
    for key in search_keys:
        try:
            if key in statement.index:
                return statement.loc[key].iloc[0]
        except Exception:
            continue
            
    return None

# Mappings for Inconsistent Financial Statement Names
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

def roe_func(ticker: str):
    """Calculates or retrieves Return on Equity (ROE)."""
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
                raise ValueError("Required financial items for ROE not found.")
    except Exception as e:
        print(f"ROE data not available (Error: {e})")
        return None
    return float(roe)

def annual_rev_growth_func(ticker: str):
    company = yf.Ticker(ticker)
    try:    
        financials = company.financials
        current_rev = find_financial_item(financials, FINANCIALS_MAP['Total Revenue'])
        previous_rev = find_financial_item(financials.iloc[:, 1:], FINANCIALS_MAP['Total Revenue'])
        if current_rev is None or previous_rev is None or previous_rev == 0:
            raise ValueError("Required revenue data not available.")
        annual_rev_growth = (current_rev - previous_rev) / previous_rev
    except Exception as e:
        print(f"Could not calculate Annual Revenue Growth (Error: {e})")
        return None
    return float(annual_rev_growth)

def annual_earnings_growth_func(ticker: str):
    company = yf.Ticker(ticker)
    try:
        financials = company.financials
        current_earnings = find_financial_item(financials, FINANCIALS_MAP['Net Income'])
        previous_earnings = find_financial_item(financials.iloc[:, 1:], FINANCIALS_MAP['Net Income'])
        if current_earnings is None or previous_earnings is None or previous_earnings == 0:
            raise ValueError("Required earnings data not available.")
        annual_earnings_growth = (current_earnings - previous_earnings) / previous_earnings
    except Exception as e:
        print(f"Could not calculate Annual Earnings Growth (Error: {e})")
        return None
    return float(annual_earnings_growth)

def current_ratio_func(ticker: str):
    company = yf.Ticker(ticker)
    try:
        balance_sheet = company.balance_sheet
        current_assets = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Current Assets'])
        current_liabilities = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Current Liabilities'])
        if current_assets is None or current_liabilities is None or current_liabilities == 0:
            raise ValueError("Data missing or liabilities zero.")
        current_ratio = current_assets / current_liabilities
    except Exception as e:
        print(f"Could not calculate Current Ratio (Error: {e})")
        return None
    return float(current_ratio)

def quick_ratio_func(ticker: str):
    company = yf.Ticker(ticker)
    try:
        balance_sheet = company.balance_sheet
        cash = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Cash']) or 0
        st_investments = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['ST Investments']) or 0
        accounts_receivable = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Receivables']) or 0
        current_liabilities = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Current Liabilities'])
        if current_liabilities is None or current_liabilities == 0:
             raise ValueError("Liabilities data missing or zero.")
        quick_assets = cash + st_investments + accounts_receivable
        quick_ratio = quick_assets / current_liabilities
    except Exception as e:
        print(f"Could not calculate Quick Ratio (Error: {e})")
        return None
    return float(quick_ratio)

def debt_equity_ratio_func(ticker: str):
    company = yf.Ticker(ticker)
    try:
        balance_sheet = company.balance_sheet
        total_debt = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Total Debt'])
        total_equity = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Total Equity'])
        if total_debt is None or total_equity is None or total_equity == 0:
            raise ValueError("Data missing or equity zero.")
        debt_equity_ratio = total_debt / total_equity
    except Exception as e:
        print(f"Could not calculate Debt-to-Equity Ratio (Error: {e})")
        return None
    return float(debt_equity_ratio)

def debt_asset_ratio_func(ticker: str):
    company = yf.Ticker(ticker)
    try:
        balance_sheet = company.balance_sheet
        total_debt = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Total Debt'])
        total_assets = find_financial_item(balance_sheet, BALANCE_SHEET_MAP['Total Assets'])
        if total_debt is None or total_assets is None or total_assets == 0:
            raise ValueError("Data missing or assets zero.")
        debt_asset_ratio = total_debt / total_assets
    except Exception as e:
        print(f"Could not calculate Debt-to-Asset Ratio (Error: {e})")
        return None
    return float(debt_asset_ratio)

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
                raise ValueError("Operating Income components missing.")
        total_rev = find_financial_item(financials, FINANCIALS_MAP['Total Revenue'])
        if total_rev is None or total_rev == 0:
            raise ValueError("Total Revenue missing or zero.")
        operating_efficiency_ratio = operating_income / total_rev
    except Exception as e:
        print(f"Could not calculate Operating Efficiency Ratio (Error: {e})")
        return None
    return float(operating_efficiency_ratio)

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

def display_matplotlib_output(data_output):
    """
    Renders the output as white text on a dark background using Matplotlib.
    Updated to ensure text fits and use green/red coloring for Strong/Weak.
    """
    if 'error' in data_output:
        print(f"Error: {data_output['error']}")
        return

    # Extract Header Info
    ticker = data_output.get('Ticker', 'N/A')
    company = data_output.get('Company', 'N/A')
    sector = data_output.get('Sector', 'N/A')
    
    # Prepare Table Data
    columns = ["Metric", "Sector Benchmark", "Company Value", "Difference", "Implication"]
    rows = []
    
    for metric, values in data_output.items():
        if metric in ['Ticker', 'Company', 'Sector', 'error']:
            continue
        # Values list: [Bench, Value, Diff, Implication]
        row_data = [metric] + [str(v) if v is not None else "--" for v in values]
        rows.append(row_data)

    # Setup Plot
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['text.color'] = 'white'
    
    fig, ax = plt.subplots(figsize=(14, 7)) # Increased width slightly
    ax.axis('off')
    ax.axis('tight')

    # Add Title Info
    title_text = f"Financial Analysis: {company} ({ticker})\nSector: {sector}"
    plt.title(title_text, color='white', fontsize=16, pad=20, weight='bold')

    # Manual Column Widths (ratios of total width)
    col_widths = [0.30, 0.2, 0.15, 0.15, 0.2]

    # Create Table
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colWidths=col_widths
    )

    # Style Table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2) # Adjust row height

    # Color customization loop
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if row == 0:
            # Header Row Styling
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#333333') # Dark Grey for header
        else:
            # Body Row Styling
            cell.set_facecolor('black')
            cell.set_text_props(color='white')
            
            # 2) Make strong/weak green/red
            # Implication is the 5th column (index 4)
            if col == 4:
                text_obj = cell.get_text()
                content = text_obj.get_text()
                if content == "Strong":
                    text_obj.set_color('#00FF00') # Bright Green
                    text_obj.set_weight('bold')
                elif content == "Weak":
                    text_obj.set_color('#FF3333') # Bright Red
                    text_obj.set_weight('bold')

    plt.show()

def main():
    ticker = input("Enter Stock Ticker (e.g., AAPL): ").strip().upper()
    
    # Process
    if ticker:
        o = comparison_data_to_dictionary(ticker)
        display_matplotlib_output(o)
    else:
        print("No ticker provided.")

if __name__ == "__main__":
    main()