import numpy as np
import matplotlib.pyplot as plt

# --- 1. Global Parameters ---
S0 = 100          # Initial stock price
K = 105           # Strike price
T = 1.0           # Time to maturity (1 year)
RFR = 0.05        # Risk-free rate
N_STEPS = 252     # Trading days
N_PATHS = 5000    # Reduced for performance in comparison
DT = T / N_STEPS

def simulate_paths(sigma):
    """Generates price paths and option metrics for a given volatility."""
    Z = np.random.standard_normal((N_PATHS, N_STEPS))
    daily_returns = np.exp((RFR - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * Z)
    
    paths = np.zeros((N_PATHS, N_STEPS + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.cumprod(daily_returns, axis=1)
    
    ST = paths[:, -1]
    payoffs = np.maximum(ST - K, 0)
    option_price = np.exp(-RFR * T) * np.mean(payoffs)
    
    prob_itm = np.sum(ST > K) / N_PATHS
    breakeven = K + option_price
    prob_profit = np.sum(ST > breakeven) / N_PATHS
    
    return paths, option_price, prob_itm, prob_profit, breakeven

def main():
    # Run two scenarios
    low_vol = 0.10
    high_vol = 0.40
    
    paths_l, price_l, itm_l, prof_l, break_l = simulate_paths(low_vol)
    paths_h, price_h, itm_h, prof_h, break_h = simulate_paths(high_vol)

    # --- Plotting ---
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.patch.set_facecolor('#0b0e14')

    def style_ax(ax, paths, sigma, price, itm, prof, brk, title_color):
        ax.set_facecolor('#0b0e14')
        # Plot subset of paths
        ax.plot(paths[:80].T, lw=0.5, color='white', alpha=0.2)
        
        # Horizontal lines
        ax.axhline(S0, color='gray', lw=1, alpha=0.5, label='Initial Price')
        ax.axhline(K, color='#ff4d4d', linestyle='--', lw=1.5, label='Strike ($105)')
        ax.axhline(brk, color='#00ff88', linestyle=':', lw=1.5, label=f'Breakeven (${brk:.2f})')
        
        # Stats box
        stats = (f"Volatility: {sigma:.0%}\n"
                 f"Premium: ${price:.2f}\n"
                 f"Prob. ITM: {itm:.1%}\n"
                 f"Prob. Profit: {prof:.1%}")
        
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#161b22', edgecolor='#30363d', alpha=0.8),
                family='monospace', fontsize=10)
        
        ax.set_title(f"{sigma:.0%} Volatility Scenario", fontsize=14, color=title_color, pad=15)
        ax.set_xlabel("Days")
        ax.grid(alpha=0.1)

    style_ax(ax1, paths_l, low_vol, price_l, itm_l, prof_l, break_l, '#3fb950')
    style_ax(ax2, paths_h, high_vol, price_h, itm_h, prof_h, break_h, '#f85149')
    
    ax1.set_ylabel("Stock Price ($)")
    ax2.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()