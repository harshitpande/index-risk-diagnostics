# models/monte_carlo.py
# ─────────────────────────────────────────────────────────────
# Step 8: Risk-conditioned Monte Carlo simulation.
# Regime-conditional drift — locked design decision.
# Volatility from GARCH_Vol (current conditional estimate).
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MC_N_SIMS, MC_HORIZON_DAYS,
    REGIME_LABELS, FEATURES_PKL,
    MONTE_CARLO_PKL, OUTPUTS_DIR
)


def run_monte_carlo() -> dict:
    df = pd.read_pickle(FEATURES_PKL)

    np.random.seed(42)

    # Current state
    S0             = df["Close"].iloc[-1]
    current_vol    = df["GARCH_Vol"].iloc[-1]
    current_regime = int(df["regime"].iloc[-1])

    # Regime-conditional drift
    regime_drift = df.groupby("regime")["log_return"].mean() * 252
    mu           = regime_drift[current_regime]

    # De-annualize
    mu_daily    = mu / 252
    sigma_daily = current_vol / np.sqrt(252)

    print(f"[mc] Simulation date   : {df.index[-1].date()}")
    print(f"[mc] Current level     : {S0:.0f}")
    print(f"[mc] Current regime    : {current_regime} — "
          f"{REGIME_LABELS[current_regime]}")
    print(f"[mc] GARCH_Vol         : {current_vol:.4f} "
          f"({current_vol*100:.1f}% annualized)")
    print(f"[mc] Regime drift (mu) : {mu:.4f} "
          f"({mu*100:.1f}% annualized)")

    # Simulate paths
    paths    = np.zeros((MC_HORIZON_DAYS, MC_N_SIMS))
    paths[0] = S0

    for t in range(1, MC_HORIZON_DAYS):
        shock    = np.random.normal(0, 1, MC_N_SIMS)
        paths[t] = paths[t-1] * np.exp(
            mu_daily
            - 0.5 * sigma_daily**2
            + sigma_daily * shock
        )

    # Percentiles
    end_prices = paths[-1]
    p05 = np.percentile(end_prices, 5)
    p25 = np.percentile(end_prices, 25)
    p50 = np.percentile(end_prices, 50)
    p75 = np.percentile(end_prices, 75)
    p95 = np.percentile(end_prices, 95)

    # Fan series
    trading_days = np.arange(MC_HORIZON_DAYS)
    fan = {
        "p05" : np.percentile(paths, 5,  axis=1),
        "p25" : np.percentile(paths, 25, axis=1),
        "p50" : np.percentile(paths, 50, axis=1),
        "p75" : np.percentile(paths, 75, axis=1),
        "p95" : np.percentile(paths, 95, axis=1),
    }

    print(f"[mc] 5th  pct : {p05:.0f}  ({((p05/S0)-1)*100:.1f}%)")
    print(f"[mc] 50th pct : {p50:.0f}  ({((p50/S0)-1)*100:.1f}%)")
    print(f"[mc] 95th pct : {p95:.0f}  ({((p95/S0)-1)*100:.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.fill_between(trading_days, fan["p05"], fan["p95"],
                    alpha=0.15, color="steelblue", label="5–95%")
    ax.fill_between(trading_days, fan["p25"], fan["p75"],
                    alpha=0.25, color="steelblue", label="25–75%")
    ax.plot(trading_days, fan["p50"],
            color="steelblue", lw=2, label="Median")
    ax.plot(trading_days, fan["p05"],
            color="red", lw=0.8, linestyle="--", label="5th pct")
    ax.plot(trading_days, fan["p95"],
            color="green", lw=0.8, linestyle="--", label="95th pct")
    ax.axhline(S0, color="black", lw=0.8,
               linestyle=":", label="Current level")
    ax.set_title(f"NIFTY 1-Month Fan Chart\n"
                 f"Regime: {REGIME_LABELS[current_regime]}")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Index Level")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.hist(end_prices, bins=80, density=True,
             color="steelblue", alpha=0.6, edgecolor="none")
    ax2.axvline(p05, color="red",    linestyle="--",
                lw=1.2, label=f"5th  : {p05:.0f}")
    ax2.axvline(p50, color="black",  linestyle="-",
                lw=1.2, label=f"50th : {p50:.0f}")
    ax2.axvline(p95, color="green",  linestyle="--",
                lw=1.2, label=f"95th : {p95:.0f}")
    ax2.axvline(S0,  color="orange", linestyle=":",
                lw=1.2, label=f"Now  : {S0:.0f}")
    ax2.set_title("End-of-Month Price Distribution")
    ax2.set_xlabel("NIFTY Index Level")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.suptitle(
        f"Risk-Conditioned Monte Carlo — {df.index[-1].date()}\n"
        f"Vol: {current_vol*100:.1f}%  |  "
        f"Regime: {REGIME_LABELS[current_regime]}  |  "
        f"Drift: {mu*100:.1f}% annualized",
        fontsize=10
    )
    plt.tight_layout()

    plot_path = os.path.join(OUTPUTS_DIR, "step8_monte_carlo.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[mc] Saved plot to {plot_path}")

    output = {
        "simulation_date" : df.index[-1],
        "S0"              : S0,
        "current_regime"  : current_regime,
        "current_vol"     : current_vol,
        "mu"              : mu,
        "paths"           : paths,
        "percentiles"     : {
            "p05": p05, "p25": p25,
            "p50": p50, "p75": p75,
            "p95": p95
        },
        "fan"             : fan,
        "n_sims"          : MC_N_SIMS,
        "horizon_days"    : MC_HORIZON_DAYS,
    }

    with open(MONTE_CARLO_PKL, "wb") as f:
        pickle.dump(output, f)

    print(f"[mc] Saved to {MONTE_CARLO_PKL}")
    return output


if __name__ == "__main__":
    run_monte_carlo()