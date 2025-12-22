import numpy as np
import matplotlib.pyplot as plt

# --- 1. Global Parameters ---
S0 = 100          # Initial stock price
K = 105           # Strike price
T = 1.0           # Time to maturity (1 year)
RFR = 0.05        # Risk-free rate (5%)
SIGMA = 0.2       # Volatility (20%)
N_STEPS = 252     # Trading days in a year
N_PATHS = 10000   # Number of simulations
DT = T / N_STEPS

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
    
    # Terminal prices at the end of the simulation
    ST = paths[:, -1]

    # --- 3. Option Pricing & Risk Metrics ---
    # Calculate payoffs for a European Call
    payoffs = np.maximum(ST - K, 0)

    # Present Value of the average payoff (Risk-Neutral Pricing)
    option_price = np.exp(-RFR * T) * np.mean(payoffs)
    breakeven = K + option_price

    # Probability Calculations
    probability_itm = np.sum(ST > K) / N_PATHS
    probability_profit = np.sum(ST > breakeven) / N_PATHS

    # --- 4. Visual Implementation (Dark Theme) ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#121212') # Deep dark background
    ax.set_facecolor('#121212')

    # Plot muted simulation paths (light grey/white with low opacity)
    ax.plot(paths[:100].T, lw=0.4, color='white', alpha=0.15)

    # Reference Levels (Neon colors for high contrast)
    ax.axhline(S0, color='white', lw=1, label=f'Initial Price (${S0})', alpha=0.7)
    ax.axhline(K, color='#ff4d4d', linestyle='--', lw=2, label=f'Strike Price (${K})')
    ax.axhline(breakeven, color='#00ff88', linestyle=':', lw=2, label=f'Risk-Adjusted Breakeven (${breakeven:.2f})')

    # --- 5. Annotations & Metadata ---
    # Main Stats Box (Top Left)
    stats_text = (f"Simulated Premium: ${option_price:.2f}\n"
                  f"Prob. of Exercise (ITM): {probability_itm:.1%}\n"
                  f"Prob. of Risk-Adjusted Profit: {probability_profit:.1%}")
    
    ax.text(0.02, 0.96, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#1e1e1e', edgecolor='#444444', alpha=0.9))

    # Subtle Metadata (Bottom Right)
    metadata_text = (f"RFR: {RFR:.0%}\n"
                     "Pricing Model: Risk-Neutral (GBM)\n"
                     "Instrument: European Call Option")

    ax.text(0.98, 0.02, metadata_text, transform=ax.transAxes, 
            fontsize=9, color='#888888', verticalalignment='bottom', 
            horizontalalignment='right', style='italic')

    # Chart Polish
    ax.set_title("Monte Carlo Simulation: Option Profitability Analysis", fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel("Trading Days", fontsize=12, color='#cccccc')
    ax.set_ylabel("Stock Price ($)", fontsize=12, color='#cccccc')
    ax.grid(color='#333333', linestyle='--', alpha=0.5)
    
    # Clean Legend
    ax.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='#444444', fontsize=10)

    plt.tight_layout()
    plt.show()

def main():
    calculations()

if __name__ == "__main__":
    main()