import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. Global Parameters ---
S0 = 100          # Initial stock price
K = 105           # Strike price
T = 1.0           # Time to maturity (1 year)
RFR = 0.04        # Risk-free rate
SIGMA = 0.2       # Volatility (20%)
N_STEPS = 252     # Trading days in a year
N_PATHS = 10000   # Number of simulations
DT = T / N_STEPS

def black_scholes_call(S, K, T, r, sigma):
    """Calculates the theoretical European Call price for benchmark comparison."""
    # Note: For non-dividend paying stocks, American Call Price ~= European Call Price
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def calculations():
    # --- 2. Simulation Engine ---
    # Generate random shocks
    Z = np.random.standard_normal((N_PATHS, N_STEPS))

    # Calculate daily growth factors (Geometric Brownian Motion)
    daily_returns = np.exp((RFR - 0.5 * SIGMA**2) * DT + SIGMA * np.sqrt(DT) * Z)

    # Generate price paths
    paths = np.zeros((N_PATHS, N_STEPS + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.cumprod(daily_returns, axis=1)
    
    # --- 3. Option Pricing (Longstaff-Schwartz / LSM) ---
    # Initialize cashflows at maturity (European payoff)
    cashflows = np.maximum(paths[:, -1] - K, 0)
    
    discount_factor = np.exp(-RFR * DT)

    # Backward Induction Loop
    # We step back from T-1 to 1 to check for early exercise
    for t in range(N_STEPS - 1, 0, -1):
        # Discount cashflows one step back
        cashflows = cashflows * discount_factor
        
        # Select paths that are In-The-Money (ITM) at time t
        S_t = paths[:, t]
        itm_mask = S_t > K
        
        # Only run regression if there are ITM paths
        if np.count_nonzero(itm_mask) > 0:
            X = S_t[itm_mask]
            Y = cashflows[itm_mask] # Discounted future cashflows
            
            # Polynomial Regression (Degree 2) to estimate Continuation Value
            # Returns coefficients [c2, c1, c0] for y = c2*x^2 + c1*x + c0
            coeffs = np.polyfit(X, Y, 2)
            continuation_value = np.polyval(coeffs, X)
            
            # Value if exercised immediately
            exercise_value = X - K
            
            # Identify paths where Exercise > Continuation
            # (Note: For non-div Calls, this condition is rarely met, but required for LSM logic)
            exercise_indices = exercise_value > continuation_value
            
            # Update cashflows for paths where we exercise early
            # We map the subset indices back to the full cashflows array
            full_indices = np.where(itm_mask)[0]
            paths_to_update = full_indices[exercise_indices]
            cashflows[paths_to_update] = exercise_value[exercise_indices]

    # Discount one last time to Present Value (t=0)
    option_price = np.mean(cashflows * discount_factor)
    
    # Calculate Theoretical Black-Scholes Price for comparison
    bs_price = black_scholes_call(S0, K, T, RFR, SIGMA)
    
    breakeven = K + option_price

    # Probability Calculations (Based on realized cashflows)
    probability_itm = np.count_nonzero(cashflows > 0) / N_PATHS
    probability_profit = np.count_nonzero(cashflows > option_price) / N_PATHS

    # --- 4. Visual Implementation (Dark Theme) ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')

    # Plot muted simulation paths
    ax.plot(paths[:100].T, lw=0.4, color='white', alpha=0.225)

    # Reference Levels
    ax.axhline(S0, color='white', lw=1, label=f'Initial Price (${S0})', alpha=0.7)
    ax.axhline(K, color='#ff4d4d', linestyle='--', lw=2, label=f'Strike Price (${K})')
    ax.axhline(breakeven, color='#00ff88', linestyle=':', lw=2, label=f'Risk-Adjusted Breakeven (${breakeven:.2f})')

    # --- 5. Annotations & Metadata ---
    # Main Stats Box - Updated for American Option
    stats_text = (f"LSM American Price: ${option_price:.2f}\n"
                  f"B-S (Euro) Price:   ${bs_price:.2f}\n"
                  f"Prob. of Exercise:  {probability_itm:.1%}\n"
                  f"Prob. of Profit:    {probability_profit:.1%}")
    
    ax.text(0.02, 0.96, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#1e1e1e', edgecolor='#444444', alpha=0.9))

    # Subtle Metadata
    metadata_text = (f"RFR: {RFR:.0%}\n"
                     "Pricing Model: Longstaff-Schwartz (LSM)\n"
                     "Instrument: American Call Option")

    ax.text(0.98, 0.02, metadata_text, transform=ax.transAxes, 
            fontsize=9, color='#888888', verticalalignment='bottom', 
            horizontalalignment='right', style='italic')

    # Chart Polish
    ax.set_title("Monte Carlo (LSM): American Option Analysis", fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel("Trading Days", fontsize=12, color='#cccccc')
    ax.set_ylabel("Stock Price ($)", fontsize=12, color='#cccccc')
    ax.grid(color='#333333', linestyle='--', alpha=0.5)
    
    ax.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#444444', fontsize=10)

    plt.tight_layout()
    plt.show()

def main():
    calculations()

if __name__ == "__main__":
    main()