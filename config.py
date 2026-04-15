# config.py
# ─────────────────────────────────────────────────────────────
# Central configuration for NIFTY Risk Diagnostics pipeline.
# ALL constants, paths, and hyperparameters live here.
# Every module imports from this file — nothing is hardcoded
# in individual modules.
# ─────────────────────────────────────────────────────────────

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR  = os.path.join(BASE_DIR, "outputs")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
PIPELINE_DIR = os.path.join(BASE_DIR, "pipeline")

# ── Data ─────────────────────────────────────────────────────
TICKER     = "^NSEI"
START_DATE = "2007-09-17"
SPLIT_DATE = "2024-01-01"

# ── Feature Engineering ──────────────────────────────────────
ROLLING_VOL_WINDOW  = 30
ROLLING_MEAN_WINDOW = 21
ROLLING_SKEW_WINDOW = 63
ROLLING_KURT_WINDOW = 63

# ── Regime Thresholds ────────────────────────────────────────
THRESHOLDS = {
    "DD_SHALLOW"  : -0.05,
    "DD_MODERATE" : -0.15,
    "DD_DEEP"     : -0.30,
    "VOL_LOW"     :  0.12,
    "VOL_HIGH"    :  0.25,
}

# ── Regime Labels ────────────────────────────────────────────
# Short labels used in all models, signals, and visualizations.
# Full descriptive labels available in REGIME_LABELS_FULL for reports.
REGIME_LABELS = {
    0: "Calm",
    1: "Pullback",
    2: "Stress",
    3: "Crisis"
}

REGIME_LABELS_FULL = {
    0: "Calm / Expansion",
    1: "Pullback / Normal Risk",
    2: "High-Volatility Stress",
    3: "Crisis"
}

# ── Regime Colors ────────────────────────────────────────────
# Used consistently across all visualization modules.
REGIME_COLORS = {
    0: "#2ecc71",
    1: "#f39c12",
    2: "#e74c3c",
    3: "#1a1a2e"
}

# Probability band colors (darker shades for stacked area charts)
PROB_COLORS = {
    0: "#27ae60",
    1: "#e67e22",
    2: "#c0392b",
    3: "#922b21"
}

# ── Train/Test Split ─────────────────────────────────────────
# Fixed cutoff — never move this. Locked design decision.
TRAIN_CUTOFF = "2024-01-01"

# ── Stress Episodes ──────────────────────────────────────────
# Used in evaluation (Tier 2), visualization shading,
# and early warning lead-time analysis.
# Single source of truth — do not redefine in individual modules.
STRESS_EPISODES = {
    "GFC_Acute_Crash"         : ("2008-09-15", "2009-03-09"),
    "Euro_Crisis"             : ("2011-08-01", "2011-12-31"),
    "Taper_Tantrum"           : ("2013-05-22", "2013-09-30"),
    "ILandFS_NBFC"            : ("2018-09-01", "2018-12-31"),
    "COVID_Crash"             : ("2020-02-20", "2020-03-23"),
    "COVID_Recovery_Vol"      : ("2020-03-24", "2020-11-09"),
    "Post_COVID_Macro_Stress" : ("2021-10-18", "2022-11-30"),
}

# Stress periods for event_flag column in features.py only.
# Separate from STRESS_EPISODES — used for feature engineering,
# not for model evaluation or visualization.
STRESS_PERIODS = {
    "GFC_acute_crash"         : ("2008-09-15", "2009-03-09"),
    "Euro_crisis"             : ("2011-08-01", "2011-12-31"),
    "Taper_tantrum"           : ("2013-05-22", "2013-09-30"),
    "ILandFS_NBFC_crisis"     : ("2018-09-01", "2018-12-31"),
    "COVID_crash"             : ("2020-02-20", "2020-03-23"),
    "COVID_recovery_vol"      : ("2020-03-24", "2020-11-09"),
    "Post_COVID_macro_stress" : ("2021-10-18", "2022-11-30"),
}

# ── GARCH ────────────────────────────────────────────────────
GARCH_P    = 1
GARCH_O    = 1
GARCH_Q    = 1
GARCH_DIST = "normal"

# ── GRU Volatility (Step 7) ──────────────────────────────────
GRU_FEATURES = [
    "log_return",
    "realized_vol",
    "GARCH_Vol",
    "drawdown",
    "regime"
]
GRU_TARGET      = "GARCH_Vol"
GRU_LOOKBACK    = 60
GRU_HORIZON     = 1
GRU_UNITS       = 32
GRU_DENSE_UNITS = 16
GRU_BATCH_SIZE  = 32
GRU_MAX_EPOCHS  = 50
GRU_PATIENCE    = 5

# ── Monte Carlo ───────────────────────────────────────────────
MC_N_SIMS       = 10_000
MC_HORIZON_DAYS = 21

# ── ARIMA ────────────────────────────────────────────────────
ARIMA_P_RANGE     = [0, 1, 2]
ARIMA_Q_RANGE     = [0, 1, 2]
ARIMA_D           = 1
ARIMA_FORECAST_1M = 21
ARIMA_FORECAST_1Q = 63

# ── Early Warning Signal Thresholds ──────────────────────────
# Starting points calibrated via Step 13 precision-recall sweep.
# Do not change without re-running Step 13 calibration.
STRESS_PROB_THRESHOLD     = 0.40
STRESS_SUSTAIN_DAYS       = 2
STRESS_OVERRIDE_THRESHOLD = 0.60
CRISIS_PROB_THRESHOLD     = 0.25
CRISIS_SUSTAIN_DAYS       = 2
CRISIS_OVERRIDE_THRESHOLD = 0.50
ESCALATION_WINDOW         = 5

# ── GRU Regime Classifier (Step 11) ─────────────────────────
REGIME_LOOKBACK         = 60
REGIME_MIN_SPELL_DAYS   = 3     # empirically calibrated; methodology grounded
                                # in Hamilton (1989) and Pagan & Sossounov (2003)
REGIME_HORIZON          = 1
REGIME_N_CLASSES        = 4
REGIME_RANDOM_SEED      = 42
REGIME_GRU_UNITS        = 64
REGIME_DENSE_UNITS      = 32
REGIME_DROPOUT_RATE     = 0.2
REGIME_EPOCHS           = 50
REGIME_BATCH_SIZE       = 32
REGIME_VALIDATION_SPLIT = 0.2

# ── Dashboard Theme ──────────────────────────────────────────
# Used by visualization/dashboards.py and dashboard_app.py
DARK_BG  = "#0d0d1a"
PANEL_BG = "#11112a"
GRID_CLR = "#222244"
TEXT_CLR = "#dde0f0"
ACCENT   = "#7b89d4"

# ── Stress Episodes Short Labels ─────────────────────────────
# Short display names for chart axis labels in dashboards
STRESS_EPISODES_SHORT = {
    "GFC"         : ("2008-09-15", "2009-03-09"),
    "Euro Crisis" : ("2011-08-01", "2011-12-31"),
    "Taper"       : ("2013-05-22", "2013-09-30"),
    "IL&FS"       : ("2018-09-01", "2018-12-31"),
    "COVID Crash" : ("2020-02-20", "2020-03-23"),
    "COVID Vol"   : ("2020-03-24", "2020-11-09"),
    "Post-COVID"  : ("2021-10-18", "2022-11-30"),
}

# ── File Paths ───────────────────────────────────────────────
FEATURES_PKL      = os.path.join(DATA_DIR, "features.pkl")
GARCH_OUTPUT_PKL  = os.path.join(DATA_DIR, "garch_output.pkl")
GRU_7A_OUTPUT_PKL = os.path.join(DATA_DIR, "gru_7a_output.pkl")
GRU_7J_OUTPUT_PKL = os.path.join(DATA_DIR, "gru_7j_output.pkl")
GRU_MODEL_7A      = os.path.join(DATA_DIR, "gru_best_model.keras")
GRU_MODEL_7J      = os.path.join(DATA_DIR, "gru_best_model_7j.keras")
MONTE_CARLO_PKL   = os.path.join(DATA_DIR, "monte_carlo_output.pkl")
ARIMA_OUTPUT_PKL  = os.path.join(DATA_DIR, "arima_output.pkl")
REGIME_PROBS_PKL  = os.path.join(DATA_DIR, "regime_probs.pkl")
REGIME_MODEL_PATH = os.path.join(DATA_DIR, "gru_regime_model.keras")
REGIME_SCALER_PKL = os.path.join(DATA_DIR, "regime_scaler_X.pkl")
EWS_PKL           = os.path.join(DATA_DIR, "early_warning_signals.pkl")
COST_RATIO_JSON   = os.path.join(DATA_DIR, "cost_ratio_config.json")
EVAL_RESULTS_JSON = os.path.join(DATA_DIR, "evaluation_results.json")
THRESHOLD_CAL_PKL = os.path.join(DATA_DIR, "threshold_calibration.pkl")