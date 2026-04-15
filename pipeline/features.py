# pipeline/features.py
# ─────────────────────────────────────────────────────────────
# Steps 1-5: Data ingestion, feature engineering,
# risk diagnostics, regime identification, regime analysis.
#
# Time-aware fetch logic:
#   Before 3:30 PM IST → fetches up to and including previous trading day
#   After  3:30 PM IST → fetches up to and including today
#
# Missing-day recovery:
#   Every run fetches the full series from START_DATE.
#   Any day missed by a previous run is automatically recovered
#   with its correct official closing price from yfinance.
#   No estimated or carried-forward values are ever written.
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import sys
import os
from datetime import datetime, time as dtime

import pytz

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TICKER, START_DATE, ROLLING_VOL_WINDOW,
    ROLLING_MEAN_WINDOW, ROLLING_SKEW_WINDOW,
    ROLLING_KURT_WINDOW, THRESHOLDS, STRESS_PERIODS,
    FEATURES_PKL
)

# ─────────────────────────────────────────────────────────────
# MARKET CLOSE CONSTANT
# NSE closes at 15:30 IST. yfinance publishes the official
# closing price shortly after. We use 15:30 as the cutoff.
# ─────────────────────────────────────────────────────────────
NSE_CLOSE_TIME = dtime(15, 30)
IST = pytz.timezone("Asia/Kolkata")


def get_target_date() -> pd.Timestamp:
    """
    Determine the correct target date for data fetching
    based on the local machine time converted to IST.

    Logic:
        Before 15:30 IST → target = yesterday
            (today's market has not closed; fetching today would
             return a live intraday price, not the official close)
        After  15:30 IST → target = today
            (market is closed; yfinance has the official close)

    Note:
        yfinance's end parameter is exclusive, so we pass
        target_date + 1 day to ensure target_date is included.

    Returns:
        pd.Timestamp — the target date (midnight, no timezone)
    """
    now_ist = datetime.now(IST)
    current_time_ist = now_ist.time()

    if current_time_ist < NSE_CLOSE_TIME:
        # Before market close — use previous calendar day.
        # yfinance will return the last available trading day
        # on or before that date, so weekends and holidays
        # are handled automatically.
        target = pd.Timestamp(now_ist.date()) - pd.Timedelta(days=1)
        reason = "before 15:30 IST — using previous trading day"
    else:
        # After market close — use today.
        target = pd.Timestamp(now_ist.date())
        reason = "after 15:30 IST — using today's close"

    print(f"[features] Local time (IST) : {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[features] Target date      : {target.date()}  ({reason})")

    return target


def fetch_data() -> pd.DataFrame:
    """
    Fetch the full NIFTY 50 daily close series from START_DATE
    up to and including the target date.

    Full-series fetch guarantees that any day missed by a
    previous pipeline run is recovered with its correct
    official closing price. No estimation is performed.

    Returns:
        pd.DataFrame — index: Date (DatetimeIndex), column: Close
    """
    target_date = get_target_date()

    # yfinance end is exclusive — add one day to include target_date
    end_date = target_date + pd.Timedelta(days=1)

    print(f"[features] Fetching {TICKER} from {START_DATE} to {target_date.date()} ...")

    raw = yf.download(
        TICKER,
        start=START_DATE,
        end=str(end_date.date()),
        auto_adjust=True,
        progress=False
    )

    if raw.empty:
        raise ValueError(
            f"[features] ERROR: yfinance returned empty data for {TICKER}. "
            f"Check ticker, date range, and internet connection."
        )

    # Flatten MultiIndex columns produced by some yfinance versions
    raw = raw.reset_index()
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

    df = raw[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # ── Verify target date is present ───────────────────────
    # If today is a market holiday or weekend, yfinance will
    # simply not return a row for that date. This is correct
    # behaviour — do not raise an error, just report it.
    if target_date not in df.index:
        last_available = df.index[-1].date()
        print(
            f"[features] NOTE: Target date {target_date.date()} not in fetched data. "
            f"Likely a market holiday or weekend. "
            f"Last available date: {last_available}"
        )
    else:
        print(
            f"[features] Last date fetched : {df.index[-1].date()}"
        )
        print(
            f"[features] Last close fetched: {df['Close'].iloc[-1]:,.2f}"
        )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived features from the raw Close series.

    Features added:
        log_return      — daily log return
        realized_vol    — 30-day rolling annualised volatility (decimal)
        running_peak    — cumulative maximum close
        drawdown        — (Close - peak) / peak  [always <= 0]
        rolling_mean_21 — 21-day rolling mean of log return
        rolling_skew_63 — 63-day rolling skewness of log return
        rolling_kurt_63 — 63-day rolling excess kurtosis of log return
        event_flag      — 1 during defined stress periods, else 0
    """
    df = df.copy()

    # Log return
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Realized volatility — annualised, expressed as decimal
    # e.g. 0.25 = 25% annualised volatility
    df["realized_vol"] = (
        df["log_return"]
        .rolling(ROLLING_VOL_WINDOW)
        .std() * np.sqrt(252)
    )

    # Running peak and drawdown
    df["running_peak"] = df["Close"].cummax()
    df["drawdown"]     = (df["Close"] - df["running_peak"]) / df["running_peak"]

    # Rolling distribution statistics
    df["rolling_mean_21"] = df["log_return"].rolling(ROLLING_MEAN_WINDOW).mean()
    df["rolling_skew_63"] = df["log_return"].rolling(ROLLING_SKEW_WINDOW).skew()
    df["rolling_kurt_63"] = df["log_return"].rolling(ROLLING_KURT_WINDOW).kurt()

    # Event flag — binary marker for known stress periods
    df["event_flag"] = 0
    for name, (start, end) in STRESS_PERIODS.items():
        mask = (df.index >= start) & (df.index <= end)
        df.loc[mask, "event_flag"] = 1

    # Drop rows with NaN (warmup period for rolling windows)
    df = df.dropna()

    return df


def assign_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a discrete market regime to each trading day.

    Regime definitions (sequential if-elif — order matters):
        0 — Calm / Expansion      : drawdown >= DD_SHALLOW AND vol <= VOL_LOW
        1 — Pullback / Normal Risk: drawdown >= DD_MODERATE AND vol <= VOL_HIGH
        3 — Crisis                : drawdown <  DD_MODERATE AND vol >  VOL_HIGH
        2 — High-Volatility Stress: all remaining cases

    Thresholds sourced from config.THRESHOLDS.
    regime_spell — integer counter that increments on every
                   regime change, enabling spell-duration analysis.
    """
    df = df.copy()

    dd_shallow  = THRESHOLDS["DD_SHALLOW"]
    dd_moderate = THRESHOLDS["DD_MODERATE"]
    vol_low     = THRESHOLDS["VOL_LOW"]
    vol_high    = THRESHOLDS["VOL_HIGH"]

    def _regime(row):
        dd  = row["drawdown"]
        vol = row["realized_vol"]
        if dd >= dd_shallow and vol <= vol_low:
            return 0   # Calm / Expansion
        elif dd >= dd_moderate and vol <= vol_high:
            return 1   # Pullback / Normal Risk
        elif dd < dd_moderate and vol > vol_high:
            return 3   # Crisis
        else:
            return 2   # High-Volatility Stress

    df["regime"]       = df.apply(_regime, axis=1)
    df["regime_spell"] = (
        df["regime"] != df["regime"].shift()
    ).cumsum()

    return df


def run_features_pipeline() -> pd.DataFrame:
    """
    Master entry point for Steps 1-5.
    Fetches data, engineers features, assigns regimes,
    and saves the result to FEATURES_PKL.
    """
    print("[features] ── Step 1-5: Features Pipeline ──────────────")

    print("[features] Step 1-2: Fetching data...")
    df = fetch_data()

    print("[features] Step 3-4: Engineering features...")
    df = engineer_features(df)

    print("[features] Step 5: Assigning regimes...")
    df = assign_regimes(df)

    df.to_pickle(FEATURES_PKL)

    print(f"[features] ── Pipeline complete ────────────────────────")
    print(f"[features] Saved to : {FEATURES_PKL}")
    print(f"[features] Shape    : {df.shape}")
    print(f"[features] Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"[features] Columns  : {df.columns.tolist()}")
    print(f"[features] Last row :")
    print(df.iloc[-1].to_string())

    return df


if __name__ == "__main__":
    run_features_pipeline()