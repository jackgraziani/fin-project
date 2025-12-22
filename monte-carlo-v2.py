import numpy as np
import matplotlib.pyplot as plt

# --- 1. Global Parameters ---
S0 = 100          # Initial stock price
K = 105           # Strike price
T = 1.0           # Time to maturity (1 year)
RFR = 0.05        # Risk-free rate (5%)
SIGMA = 0.2       # Volatility (20%)
N_STEPS = 252     # Trading days in a year
N_PATHS = 20000   # Increased paths for smoother Greek estimates
DT = T / N_STEPS

def simulate_price(s_start, r, vol, time, steps, paths, seed=None):
    """Core GBM Simulation Engine"""
    if seed: np.random.seed(seed)
    dt = time / steps
    # Using fixed random seed for 'bumping' ensures variance reduction
    Z = np.random.standard_normal((paths, steps))
    daily_returns = np.exp((r - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z)
    
    price_paths = np.zeros((paths, steps + 1))
    price_paths[:, 0] = s_start
    price_paths[:, 1:] = s_start * np.cumprod(daily_returns, axis=1)
    return price_paths

def price_option(paths, strike, r, time):
    """Calculates discounted expected payoff"""
    st = paths[:, -1]
    payoffs = np.maximum(st - strike, 0)
    return np.exp(-r * time) * np.mean(payoffs)

def calculations():
    # --- 2. Calculate Base Price ---
    # We use a fixed seed for the Z-scores to perform "Finite Difference" 
    # This reduces 'noise' when comparing the bumped price to the base price.
    np.random.seed(42)
    Z_fixed = np.random.standard_normal((N_PATHS, N_STEPS))
    
    def get_price(s, r, vol, t):
        dt = t / N_STEPS
        # Reconstruct paths using the fixed shocks
        daily_rets = np.exp((r - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z_fixed)
        terminal_prices = s * np.prod(daily_rets, axis=1)
        return np.exp(-r * t) * np.mean(np.maximum(terminal_prices - K, 0))

    base_price = get_price(S0, RFR, SIGMA, T)

    # --- 3. Calculate Greeks via Finite Difference ---
    ds = 0.01 * S0  # 1% price bump
    dv = 0.01       # 1% vol bump
    dr = 0.01       # 1% rate bump
    dt = 1/252      # 1 day bump

    # Delta: (P(S+ds) - P(S-ds)) / 2ds
    delta = (get_price(S0 + ds, RFR, SIGMA, T) - get_price(S0 - ds, RFR, SIGMA, T)) / (2 * ds)
    
    # Gamma: (P(S+ds) - 2P(S) + P(S-ds)) / ds^2
    gamma = (get_price(S0 + ds, RFR, SIGMA, T) - 2*base_price + get_price(S0 - ds, RFR, SIGMA, T)) / (ds**2)
    
    # Vega: Change in price for 1% change in Vol
    vega = (get_price(S0, RFR, SIGMA + dv, T) - get_price(S0, RFR, SIGMA - dv, T)) / 2
    
    # Theta: Change in price for 1 day passing (T - dt)
    theta = (get_price(S0, RFR, SIGMA, T - dt) - base_price) 
    
    # Rho: Change in price for 1% change in RFR
    rho = (get_price(S0, RFR + dr, SIGMA, T) - get_price(S0, RFR - dr, SIGMA, T)) / 2

    # --- 4. Visual Implementation ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10), facecolor='#0f0f0f')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
    
    # A. Price Paths Plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#0f0f0f')
    
    # Plot sample paths
    paths = simulate_price(S0, RFR, SIGMA, T, N_STEPS, N_PATHS, seed=42)
    ax1.plot(paths[:100].T, lw=0.5, color='cyan', alpha=0.1)
    
    ax1.axhline(S0, color='white', lw=1, label='Initial Price', alpha=0.5)
    ax1.axhline(K, color='#ff4d4d', ls='--', lw=2, label=f'Strike (${K})')
    breakeven = K + base_price
    ax1.axhline(breakeven, color='#00ff88', ls=':', lw=2, label=f'Breakeven (${breakeven:.2f})')

    # B. Stats Box
    stats_text = (
        f"Option Premium: ${base_price:.2f}\n"
        f"Delta: {delta:.3f} (Hedge Ratio)\n"
        f"Gamma: {gamma:.4f} (Stability)\n"
        f"Vega:  ${vega:.2f} (Vol Sensitivity)\n"
        f"Theta: ${theta:.2f} (Daily Decay)\n"
        f"Rho:   ${rho:.2f} (Rate Sensitivity)"
    )
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, 
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#1e1e1e', edgecolor='#444444'))

    ax1.set_title("Monte Carlo Simulation & Sensitivity Analysis (Greeks)", fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper right')

    # C. Greeks Visualization (Bar Chart)
    ax2 = fig.add_subplot(gs[1, 0])
    greeks_vals = [delta, vega, rho]
    greeks_labels = ['Delta', 'Vega', 'Rho']
    colors = ['#4d94ff', '#ffcc00', '#ff66b3']
    ax2.bar(greeks_labels, greeks_vals, color=colors, alpha=0.8)
    ax2.set_title("Primary Exposure (Direction, Vol, Rates)")
    ax2.grid(axis='y', alpha=0.2)

    # D. Distribution of Terminal Prices
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(paths[:, -1], bins=100, color='white', alpha=0.3, density=True)
    ax3.axvline(K, color='#ff4d4d', lw=2)
    ax3.set_title("Probability Distribution at Expiry")
    ax3.set_xlabel("Price ($)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    calculations()