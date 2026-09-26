# pipeline/export_json.py
# Step 14: JSON Export Layer — dashboard data contract
#
# Reads the pipeline's existing outputs (features.pkl, monte_carlo_output.pkl,
# early_warning_signals.pkl) and writes the three files the React/ECharts
# frontend consumes: data/dashboard/{snapshot,timeseries,montecarlo}.json.
#
# No new computation happens here — every field is a direct read or a
# unit/label conversion of something an upstream pipeline step already
# produced. See docs/json_contract.md for the authoritative field spec.

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FEATURES_PKL,
    MONTE_CARLO_PKL,
    EWS_PKL,
    REGIME_LABELS,
    THRESHOLDS,
    DASHBOARD_DIR,
    SNAPSHOT_JSON,
    TIMESERIES_JSON,
    MONTECARLO_JSON,
)


# Fixed characterization phrases for the snapshot `reasoning` sentence. These are
# defined once here, not generated — see build_reasoning(). Stress is a catch-all
# regime (deep drawdown OR high vol), so it carries one phrase per trigger.
REGIME_CHARACTERIZATION = {
    0: "an expansion / low-risk phase",
    1: "a normal-risk correction, not systemic stress",
    2: {"drawdown": "drawdown-driven stress", "volatility": "volatility-driven stress"},
    3: "systemic stress",
}


# ─────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    return pd.read_pickle(FEATURES_PKL)


def load_monte_carlo() -> dict:
    with open(MONTE_CARLO_PKL, "rb") as f:
        return pickle.load(f)


def load_early_warning() -> pd.DataFrame:
    return pd.read_pickle(EWS_PKL)


# ─────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────

def regime_label(regime_int) -> str:
    """Map a rule-based/predicted regime int (0-3) to its canonical label."""
    return REGIME_LABELS[int(regime_int)]


def iso(date_like) -> str:
    """Format any date-like value as an ISO 'YYYY-MM-DD' string."""
    return pd.Timestamp(date_like).strftime("%Y-%m-%d")


def clean_float(x):
    """NaN/np.floating -> plain float, or None if missing (contract: null, never 0)."""
    if x is None or pd.isna(x):
        return None
    return float(x)


def _pct(x, decimals=1) -> str:
    """Decimal -> percentage string, e.g. -0.111 -> '-11.1%'."""
    return f"{x * 100:.{decimals}f}%"


def _dd_band(dd) -> str:
    """Which drawdown band (per config.THRESHOLDS) the current drawdown falls into."""
    dd_shallow = THRESHOLDS["DD_SHALLOW"]
    dd_moderate = THRESHOLDS["DD_MODERATE"]
    if dd >= dd_shallow:
        return f"shallow (≥ {_pct(dd_shallow, 0)})"
    elif dd >= dd_moderate:
        return f"moderate ({_pct(dd_shallow, 0)} to {_pct(dd_moderate, 0)})"
    else:
        return f"deep (< {_pct(dd_moderate, 0)})"


def _vol_band(vol) -> str:
    """Which volatility band (per config.THRESHOLDS) the current realized_vol falls into."""
    vol_low = THRESHOLDS["VOL_LOW"]
    vol_high = THRESHOLDS["VOL_HIGH"]
    if vol <= vol_low:
        return f"low (≤ {_pct(vol_low, 0)})"
    elif vol <= vol_high:
        return f"elevated ({_pct(vol_low, 0)}–{_pct(vol_high, 0)})"
    else:
        return f"high (> {_pct(vol_high, 0)})"


def _characterization(regime_int, dd) -> str:
    """
    Fixed characterization phrase for a regime. Stress (2) is reachable via deep
    drawdown or high vol, so it picks its phrase by trigger: deep drawdown
    (< DD_MODERATE, same boundary as the 'deep' band) is the defining driver;
    otherwise assign_regimes guarantees vol > VOL_HIGH.
    """
    phrase = REGIME_CHARACTERIZATION[int(regime_int)]
    if int(regime_int) == 2:
        return phrase["drawdown" if dd < THRESHOLDS["DD_MODERATE"] else "volatility"]
    return phrase


def build_reasoning(regime_int, dd, vol):
    """
    Deterministic, rule-based sentence explaining today's regime classification.

    Part (a) is generated from the real current drawdown/volatility values and the
    real calibrated thresholds in config.THRESHOLDS. Part (b) is a fixed
    characterization phrase (REGIME_CHARACTERIZATION) — not generated; for Stress,
    one of two fixed phrases chosen by its trigger (see _characterization()). No
    LLM/API involved anywhere in this function.
    """
    if pd.isna(dd) or pd.isna(vol):
        return None

    regime_str = regime_label(regime_int)
    dd_band = _dd_band(dd)
    vol_band = _vol_band(vol)
    characterization = _characterization(regime_int, dd)

    return (
        f"Classified {regime_str}: drawdown of {_pct(dd)} is in the {dd_band} band, "
        f"with volatility of {_pct(vol)} in the {vol_band} band — {characterization}."
    )


# ─────────────────────────────────────────────────────────────
# BUILDERS
# ─────────────────────────────────────────────────────────────

def build_snapshot(features_df: pd.DataFrame, ews_df: pd.DataFrame) -> dict:
    latest = features_df.iloc[-1]
    current_regime = int(latest["regime"])
    current_spell = latest["regime_spell"]
    days_in_regime = int((features_df["regime_spell"] == current_spell).sum())

    latest_date = features_df.index[-1]
    if ews_df.index[-1] != latest_date:
        print(f"[export_json] WARNING: features.pkl last date ({latest_date.date()}) "
              f"!= early_warning_signals.pkl last date ({ews_df.index[-1].date()}) "
              f"— snapshot signals may be stale relative to price/regime data.")

    ews_latest = ews_df.iloc[-1]

    return {
        "date": iso(latest_date),
        "current_regime": regime_label(current_regime),
        "days_in_regime": days_in_regime,
        "realized_vol": clean_float(latest["realized_vol"]),
        "drawdown": clean_float(latest["drawdown"]),
        "reasoning": build_reasoning(current_regime, latest["drawdown"], latest["realized_vol"]),
        "signals": {
            "stress": bool(ews_latest["stress_signal"]),
            "crisis": bool(ews_latest["crisis_alert"]),
            "escalation": bool(ews_latest["escalation_signal"]),
        },
    }


def build_timeseries(features_df: pd.DataFrame) -> list:
    rows = []
    for date, row in features_df.iterrows():
        rows.append({
            "date": iso(date),
            "close": clean_float(row["Close"]),
            "regime": regime_label(row["regime"]),
            "drawdown": clean_float(row["drawdown"]),
            "realized_vol": clean_float(row["realized_vol"]),
            "garch_vol": clean_float(row.get("GARCH_Vol")),
        })
    return rows


def build_montecarlo(mc: dict, features_df: pd.DataFrame) -> dict:
    simulation_date = pd.Timestamp(mc["simulation_date"])
    horizon_days = int(mc["horizon_days"])

    # Historical leg: ~horizon_days trading days up to and including simulation_date,
    # pulled from features.pkl (monte_carlo_output.pkl carries no historical prices).
    hist_slice = features_df.loc[:simulation_date].tail(horizon_days)
    historical = [
        {"date": iso(date), "close": clean_float(row["Close"])}
        for date, row in hist_slice.iterrows()
    ]

    # Forecast leg: mc["fan"] has horizon_days entries; index 0 is the
    # deterministic anchor (today, S0) and indices 1..horizon_days-1 are the
    # genuine simulated forward days. All horizon_days entries are kept,
    # dated simulation_date .. simulation_date + (horizon_days-1) business days.
    fan = mc["fan"]
    forecast_dates = pd.bdate_range(start=simulation_date, periods=horizon_days)
    forecast = []
    for i, date in enumerate(forecast_dates):
        forecast.append({
            "date": iso(date),
            "ptile_5": clean_float(fan["p05"][i]),
            "ptile_25": clean_float(fan["p25"][i]),
            "ptile_50": clean_float(fan["p50"][i]),
            "ptile_75": clean_float(fan["p75"][i]),
            "ptile_95": clean_float(fan["p95"][i]),
        })

    return {
        "generated_on": iso(simulation_date),
        "current_vol": clean_float(mc["current_vol"]),
        "current_regime": regime_label(mc["current_regime"]),
        "horizon_days": horizon_days,
        "historical": historical,
        "forecast": forecast,
    }


# ─────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────

def write_json(obj, path) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    os.replace(tmp_path, path)
    print(f"[export_json] Saved {path}")


# ─────────────────────────────────────────────────────────────
# ORCHESTRATION
# ─────────────────────────────────────────────────────────────

def run_export() -> None:
    print("[export_json] Loading pipeline outputs...")
    features_df = load_features()
    mc = load_monte_carlo()
    ews_df = load_early_warning()

    os.makedirs(DASHBOARD_DIR, exist_ok=True)

    snapshot = build_snapshot(features_df, ews_df)
    write_json(snapshot, SNAPSHOT_JSON)

    timeseries = build_timeseries(features_df)
    write_json(timeseries, TIMESERIES_JSON)

    montecarlo = build_montecarlo(mc, features_df)
    write_json(montecarlo, MONTECARLO_JSON)

    print(f"[export_json] Snapshot date        : {snapshot['date']}")
    print(f"[export_json] Current regime       : {snapshot['current_regime']} "
          f"({snapshot['days_in_regime']} day(s))")
    print(f"[export_json] Active signals        : "
          f"{[k for k, v in snapshot['signals'].items() if v] or 'NONE'}")
    print(f"[export_json] Timeseries rows       : {len(timeseries)}")
    print(f"[export_json] Monte Carlo forecast   : {len(montecarlo['forecast'])} day(s), "
          f"generated {montecarlo['generated_on']}")


if __name__ == "__main__":
    run_export()
