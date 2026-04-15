# pipeline/run_daily.py
# Master daily runner -- Steps 1-14 fully integrated.
# Loads saved models -- NEVER retrains.
# Run manually : python pipeline/run_daily.py
# Run via PS1  : powershell -ExecutionPolicy Bypass -File .\run_daily.ps1
#
# Step 11 design:
#   Batch inference across ALL dates in features.pkl.
#   Any date missing from regime_probs.pkl is detected and
#   computed with its correct historical features.
#   Missed pipeline runs (holidays, machine off) are recovered
#   automatically on the next run. No estimation, no carry-forward.

import sys
import os
import logging
import shutil
import pickle
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LOGS_DIR = os.path.join(ROOT, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

RUN_DATE = datetime.now().strftime('%Y-%m-%d')
LOG_FILE = os.path.join(LOGS_DIR, f'run_{RUN_DATE}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.join(ROOT, 'outputs')
ARCHIVE_DIR = os.path.join(OUTPUTS_DIR, 'archive', RUN_DATE)
DATA_DIR    = os.path.join(ROOT, 'data')

CHART_FILES = [
    'regime_probability_ts.png',
    'stress_signal_dashboard.png',
    'monte_carlo_fanchart.png',
    'full_system_dashboard.png',
    'tier1_evaluation.png',
    'tier2_evaluation.png',
    'threshold_calibration.png',
]


def archive_outputs():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    archived = 0
    for fname in CHART_FILES:
        src = os.path.join(OUTPUTS_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ARCHIVE_DIR, fname))
            archived += 1
    log.info(f"Archived {archived} chart(s) to outputs/archive/{RUN_DATE}/")


def run_step(label, func, *args, **kwargs):
    log.info("-" * 55)
    log.info(f"START  {label}")
    try:
        result = func(*args, **kwargs)
        log.info(f"OK     {label}")
        return result
    except Exception as e:
        log.error(f"FAILED {label} -- {e}", exc_info=True)
        raise


def run_daily():
    log.info("=" * 60)
    log.info("NIFTY Risk Diagnostics -- Daily Pipeline")
    log.info(f"Run date : {RUN_DATE}")
    log.info(f"Log file : {LOG_FILE}")
    log.info("=" * 60)

    from models.gru_regime          import LOOKBACK, REGIME_LABELS
    from models.gru_regime          import predict_regime_probabilities_batch
    from pipeline.features          import run_features_pipeline
    from models.garch               import run_garch_pipeline
    from models.gru_volatility      import run_gru_inference
    from models.monte_carlo         import run_monte_carlo
    from models.arima               import run_arima_pipeline
    from early_warning.signals      import compute_early_warning_signals
    from pipeline.evaluation        import run_evaluation
    from visualization.dashboards   import run_dashboards

    # Steps 1-5: Feature engineering
    run_step("Steps 1-5  | Feature Engineering",       run_features_pipeline)

    # Step 6: GARCH
    run_step("Step 6     | GJR-GARCH Volatility",      run_garch_pipeline)

    # Step 7: GRU volatility inference
    run_step("Step 7     | GRU Volatility Inference",  run_gru_inference)

    # Step 8: Monte Carlo
    run_step("Step 8     | Monte Carlo Simulation",    run_monte_carlo)

    # Step 9: ARIMA
    run_step("Step 9     | ARIMA Diagnostic Forecast", run_arima_pipeline)

    # ------------------------------------------------------------------
    # Step 11: GRU regime classifier — batch inference
    #
    # Design:
    #   1. Load full features.pkl (rebuilt fresh by Step 1-5 above)
    #   2. Load existing regime_probs.pkl to find already-computed dates
    #   3. Identify ALL dates in features.pkl missing from regime_probs.pkl
    #   4. Run batch inference for every missing date using its correct
    #      60-day lookback window from the historical feature matrix
    #   5. Append new rows, deduplicate, sort, save
    #
    # This recovers any date missed by a previous pipeline run
    # (machine off, holiday, crash) with correct official closing
    # prices from features.pkl. No estimation is performed.
    # ------------------------------------------------------------------
    def run_regime_step():
        features_path     = os.path.join(DATA_DIR, 'features.pkl')
        model_path        = os.path.join(DATA_DIR, 'gru_regime_model.keras')
        scaler_path       = os.path.join(DATA_DIR, 'regime_scaler_X.pkl')
        regime_probs_path = os.path.join(DATA_DIR, 'regime_probs.pkl')

        with open(features_path, 'rb') as f:
            features_df = pickle.load(f)
        features_df.index = pd.to_datetime(features_df.index)

        # ── Load existing regime_probs.pkl ──────────────────
        if os.path.exists(regime_probs_path):
            with open(regime_probs_path, 'rb') as f:
                existing = pickle.load(f)
            existing.index = pd.to_datetime(existing.index)
            already_computed = set(existing.index)
        else:
            existing = pd.DataFrame()
            already_computed = set()

        # ── Find all dates needing inference ─────────────────
        # Dates in features_df that require a full LOOKBACK window:
        # the first valid date for inference is features_df.index[LOOKBACK]
        # because we need LOOKBACK rows of history before that date.
        TRAIN_CUTOFF = pd.Timestamp('2024-01-01')
        all_feature_dates = features_df.index[
            (features_df.index >= TRAIN_CUTOFF)
        ]
        missing_dates = [d for d in all_feature_dates
                         if d not in already_computed]

        if not missing_dates:
            log.info("  Regime probs: already up to date, no missing dates.")
            # Still return existing so downstream steps have the full df
            return existing

        log.info(f"  Regime probs: {len(missing_dates)} date(s) to compute.")
        log.info(f"  First missing: {missing_dates[0].date()}")
        log.info(f"  Last missing : {missing_dates[-1].date()}")

        # ── Batch inference for all missing dates ────────────
        new_rows = predict_regime_probabilities_batch(
            features_df, model_path, scaler_path, missing_dates
        )

        # ── Combine, deduplicate, sort ────────────────────────
        if not existing.empty:
            updated = pd.concat([existing, new_rows]).sort_index()
        else:
            updated = new_rows.sort_index()

        updated = updated[~updated.index.duplicated(keep='last')]

        with open(regime_probs_path, 'wb') as f:
            pickle.dump(updated, f)

        latest       = updated.iloc[-1]
        stress_crisis = float(latest['P_regime_2']) + float(latest['P_regime_3'])
        pred_regime  = int(latest['predicted_regime'])

        log.info(f"  Regime probs saved  : {len(updated)} rows")
        log.info(f"  Latest date         : {updated.index[-1].date()}")
        log.info(f"  Predicted regime    : {pred_regime} "
                 f"({REGIME_LABELS[pred_regime]})")
        log.info(f"  P(Stress+Crisis)    : {stress_crisis:.4f}")

        return updated

    prob_df = run_step("Step 11    | GRU Regime Classifier", run_regime_step)

    # ------------------------------------------------------------------
    # Step 12: Early warning signals
    # compute_early_warning_signals expects prob_df with P_regime_0..3
    # Returns enriched DataFrame -- save to early_warning_signals.pkl
    # ------------------------------------------------------------------
    def run_ews_step():
        ews_df = compute_early_warning_signals(prob_df)

        # Preserve actual_regime and predicted_regime columns if present
        if 'actual_regime' in prob_df.columns and 'actual_regime' not in ews_df.columns:
            ews_df['actual_regime'] = prob_df['actual_regime']
        if 'predicted_regime' in prob_df.columns and 'predicted_regime' not in ews_df.columns:
            ews_df['predicted_regime'] = prob_df['predicted_regime']

        ews_path = os.path.join(DATA_DIR, 'early_warning_signals.pkl')
        with open(ews_path, 'wb') as f:
            pickle.dump(ews_df, f)

        # Log current signal state
        latest = ews_df.iloc[-1]
        active = []
        for sig in ['stress_signal', 'crisis_alert', 'escalation_signal']:
            if sig in latest and latest[sig]:
                active.append(sig.upper())
        log.info(f"  EWS updated: {len(ews_df)} rows, "
                 f"combined={latest.get('stress_combined_prob', 0):.4f}, "
                 f"active={active if active else 'NONE'}")
        return ews_df

    run_step("Step 12    | Early Warning System", run_ews_step)

    # Step 13: Model evaluation
    run_step("Step 13    | Model Evaluation",     run_evaluation)

    # Step 14: Visualization
    run_step("Step 14    | Visualization Layer",  run_dashboards)

    # Archive
    archive_outputs()

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Outputs : {OUTPUTS_DIR}")
    log.info(f"  Archive : {ARCHIVE_DIR}")
    log.info(f"  Log     : {LOG_FILE}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_daily()