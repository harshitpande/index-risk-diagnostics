# models/arima.py
# ─────────────────────────────────────────────────────────────
# Step 9: ARIMA diagnostic price forecasting.
# Purpose: uncertainty structure validation ONLY.
# Not a price prediction tool.
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ARIMA_P_RANGE, ARIMA_Q_RANGE, ARIMA_D,
    ARIMA_FORECAST_1M, ARIMA_FORECAST_1Q,
    FEATURES_PKL, ARIMA_OUTPUT_PKL, OUTPUTS_DIR
)


def select_arima_order(log_close: pd.Series) -> tuple:
    results = {}
    for p, q in product(ARIMA_P_RANGE, ARIMA_Q_RANGE):
        try:
            m = ARIMA(log_close, order=(p, ARIMA_D, q)).fit()
            results[(p, ARIMA_D, q)] = {"AIC": m.aic, "BIC": m.bic}
        except Exception:
            pass
    best_order = min(results, key=lambda k: results[k]["AIC"])
    return best_order, results


def run_arima_pipeline() -> dict:
    df        = pd.read_pickle(FEATURES_PKL)
    log_close = np.log(df["Close"])
    last_date = df.index[-1]

    # Stationarity check
    adf_level = adfuller(log_close, autolag="AIC")
    adf_diff  = adfuller(log_close.diff().dropna(), autolag="AIC")

    print(f"[arima] ADF log(Close)       : "
          f"p={adf_level[1]:.4f}  "
          f"{'NON-STATIONARY' if adf_level[1] > 0.05 else 'STATIONARY'}")
    print(f"[arima] ADF diff(log(Close)) : "
          f"p={adf_diff[1]:.4f}  "
          f"{'NON-STATIONARY' if adf_diff[1] > 0.05 else 'STATIONARY'}")

    # Model selection
    best_order, results_grid = select_arima_order(log_close)
    print(f"[arima] Best order (AIC) : ARIMA{best_order}")

    # Fit best model
    arima_model = ARIMA(log_close, order=best_order).fit()

    # 1-month forecast
    fc_1m      = arima_model.get_forecast(steps=ARIMA_FORECAST_1M)
    fc_1m_mean = np.exp(fc_1m.predicted_mean.values)
    fc_1m_ci   = np.exp(fc_1m.conf_int().values)
    dates_1m   = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=ARIMA_FORECAST_1M
    )
    forecast_1m = pd.DataFrame(
        {"Forecast": fc_1m_mean,
         "Lower_95": fc_1m_ci[:, 0],
         "Upper_95": fc_1m_ci[:, 1]},
        index=dates_1m
    )

    # 1-quarter forecast
    fc_1q      = arima_model.get_forecast(steps=ARIMA_FORECAST_1Q)
    fc_1q_mean = np.exp(fc_1q.predicted_mean.values)
    fc_1q_ci   = np.exp(fc_1q.conf_int().values)
    dates_1q   = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=ARIMA_FORECAST_1Q
    )
    forecast_1q = pd.DataFrame(
        {"Forecast": fc_1q_mean,
         "Lower_95": fc_1q_ci[:, 0],
         "Upper_95": fc_1q_ci[:, 1]},
        index=dates_1q
    )

    # Uncertainty width validation
    ci_day1  = forecast_1q["Upper_95"].iloc[0]  - forecast_1q["Lower_95"].iloc[0]
    ci_day21 = forecast_1q["Upper_95"].iloc[20] - forecast_1q["Lower_95"].iloc[20]
    ci_day63 = forecast_1q["Upper_95"].iloc[-1] - forecast_1q["Lower_95"].iloc[-1]

    print(f"[arima] CI width day 1  : {ci_day1:.0f}")
    print(f"[arima] CI width day 21 : {ci_day21:.0f}")
    print(f"[arima] CI width day 63 : {ci_day63:.0f}")

    if ci_day1 < ci_day21 < ci_day63:
        print("[arima] PASS — uncertainty widens correctly.")
    else:
        print("[arima] WARN — unexpected uncertainty structure.")

    # Plot
    history = df["Close"][df.index >= df.index[-1] - pd.DateOffset(months=12)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, fc_df, title in zip(
        axes,
        [forecast_1m, forecast_1q],
        ["1-Month Forecast (21 days)",
         "1-Quarter Forecast (63 days)"]
    ):
        ax.plot(history.index, history.values,
                color="steelblue", lw=0.9, label="NIFTY (actual)")
        ax.plot(fc_df.index, fc_df["Forecast"],
                color="black", lw=1.2,
                label=f"ARIMA{best_order} forecast")
        ax.fill_between(fc_df.index,
                        fc_df["Lower_95"], fc_df["Upper_95"],
                        color="gray", alpha=0.3,
                        label="95% confidence interval")
        ax.axvline(last_date, color="red", lw=0.8,
                   linestyle="--", label="Forecast origin")
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Index Level")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    plt.suptitle(
        f"NIFTY 50 — ARIMA Diagnostic Forecast  |  {last_date.date()}\n"
        f"Model: ARIMA{best_order}  |  "
        f"Purpose: uncertainty structure validation, not prediction",
        fontsize=10
    )
    plt.tight_layout()

    plot_path = os.path.join(OUTPUTS_DIR, "step9_arima_forecast.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[arima] Saved plot to {plot_path}")

    output = {
        "model"           : arima_model,
        "best_order"      : best_order,
        "results_grid"    : results_grid,
        "forecast_1m"     : forecast_1m,
        "forecast_1q"     : forecast_1q,
        "forecast_origin" : last_date,
        "ci_widths"       : {
            "day1" : ci_day1,
            "day21": ci_day21,
            "day63": ci_day63
        }
    }

    with open(ARIMA_OUTPUT_PKL, "wb") as f:
        pickle.dump(output, f)

    print(f"[arima] Saved to {ARIMA_OUTPUT_PKL}")
    return output


if __name__ == "__main__":
    run_arima_pipeline()