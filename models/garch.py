# models/garch.py
# ─────────────────────────────────────────────────────────────
# Step 6: GJR-GARCH(1,1) volatility modeling.
# Fits model on full return series.
# Adds GARCH_Vol column to features dataframe.
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from arch import arch_model

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    GARCH_P, GARCH_O, GARCH_Q, GARCH_DIST,
    FEATURES_PKL, GARCH_OUTPUT_PKL
)


def fit_garch(df: pd.DataFrame) -> tuple:
    returns = df["log_return"].dropna() * 100

    model = arch_model(
        returns,
        mean = "Constant",
        vol  = "GARCH",
        p    = GARCH_P,
        o    = GARCH_O,
        q    = GARCH_Q,
        dist = GARCH_DIST
    )

    result = model.fit(update_freq=0, disp="off")
    return result


def compute_persistence(result) -> float:
    params      = result.params
    alpha       = params["alpha[1]"]
    beta        = params["beta[1]"]
    gamma       = params["gamma[1]"]
    persistence = alpha + beta + 0.5 * gamma
    return persistence


def run_garch_pipeline() -> pd.DataFrame:
    df = pd.read_pickle(FEATURES_PKL)

    print("[garch] Fitting GJR-GARCH(1,1)...")
    result      = fit_garch(df)
    persistence = compute_persistence(result)

    print(f"[garch] Persistence : {persistence:.4f}")

    if persistence >= 1.0:
        print("[garch] WARNING — process is non-stationary. "
              "Persistence >= 1.0. Review before proceeding.")
    else:
        print("[garch] OK — process is stationary.")

    # Unit conversion — annualized decimal
    df["GARCH_Vol"] = (
        result.conditional_volatility / 100 * np.sqrt(252)
    )

    # Save
    garch_output = {
        "model"               : result,
        "params"              : result.params,
        "persistence"         : persistence,
        "conditional_vol_raw" : result.conditional_volatility,
    }

    with open(GARCH_OUTPUT_PKL, "wb") as f:
        pickle.dump(garch_output, f)

    df.to_pickle(FEATURES_PKL)

    print(f"[garch] GARCH_Vol range : "
          f"{df['GARCH_Vol'].min():.4f} – {df['GARCH_Vol'].max():.4f}")
    print(f"[garch] Saved to {GARCH_OUTPUT_PKL}")

    return df


if __name__ == "__main__":
    run_garch_pipeline()