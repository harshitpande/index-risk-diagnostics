# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An equity risk diagnostic and early-warning system for the NIFTY 50 index (India). It is explicitly
**not** a price predictor — every design decision (drift/volatility conditioning, dual-tier evaluation,
epistemic-humility framing) exists to keep the system honest about uncertainty rather than confident
about direction. `METHODOLOGY.md` is the authoritative, heavily-cited spec for every formula, threshold,
and modeling choice in the pipeline — consult it before changing any model, threshold, or metric, since
most values here are empirically derived (not arbitrary) and documented there with their rationale.

## Current repo state — read before working here

The repo is **mid-migration**. Commit `dda5791` removed the old Streamlit/matplotlib dashboard
(`dashboard_app.py`, `visualization/dashboards.py`) in favor of a JSON-export → React/ECharts
frontend. The JSON-export layer now exists (`pipeline/export_json.py`), but the React frontend does
not exist yet in this repo.

- **The pipeline runs end-to-end.** Step 14 was rewired from the deleted `visualization.dashboards`
  import to `pipeline/export_json.run_export`, so `python pipeline/run_daily.py` completes Steps 1–14
  and writes `data/dashboard/{snapshot,timeseries,montecarlo}.json`. Steps 1–13 remain independently
  runnable.
- `REQUIREMENTS.md` is the locked spec for the *new* dashboard (single-page, ECharts, dark chart cards on
  white chrome, 4 charts + signal status bar) — read it before building any frontend/visualization work.
- `docs/json_schema_notes.md` captures what the old dashboard's data-shaping code did, as a reference for
  designing the new JSON export contract. It documents a `get_current_state()`-style snapshot schema and
  notes that `signal_strength` is intentionally excluded from the new dashboard per REQUIREMENTS.md §4.
- The README's "Repository Structure" and "Live Dashboard" sections describe the pre-migration layout
  (`dashboard_app.py`, `run_daily.ps1`, `schedule_task.xml`) — those files no longer exist in this repo.
  Trust the actual file tree and `docs/json_schema_notes.md` over the README when they conflict.

## Commands

There is no test suite, linter, or build step in this repo — it's a data pipeline of standalone scripts.

```bash
# Full daily pipeline (Steps 1-14) — runs end-to-end
python pipeline/run_daily.py

# Individual steps (each is independently runnable; each loads/writes its own data/*.pkl)
python pipeline/features.py          # Steps 1-5: fetch data, engineer features, assign regimes
python models/garch.py               # Step 6: GJR-GARCH volatility
python models/gru_volatility.py      # Step 7: GRU volatility inference (loads saved model, no training)
python models/monte_carlo.py         # Step 8: regime-conditional Monte Carlo
python models/arima.py               # Step 9: ARIMA diagnostic forecast
python early_warning/signals.py      # Step 12: early warning signals (also plots a standalone dashboard)
python pipeline/evaluation.py        # Step 13: dual-tier evaluation + threshold calibration
python pipeline/export_json.py        # Step 14: JSON export → data/dashboard/*.json (dashboard data contract)

# Retrain the GRU regime classifier from scratch (the only script that calls .fit())
python models/gru_regime.py

# Recompute the empirical cost ratio / class weights from features.pkl
python cost_ratio_analysis.py
```

`pip install -r requirements.txt` to set up the environment (Python 3.11, per `.python-version`). Note:
`requirements.txt` is UTF-16-encoded — some tools mis-handle this; if editing it manually, preserve or
convert the encoding deliberately rather than assuming UTF-8.

## Architecture

### Config is the single source of truth

`config.py` holds every constant, path, threshold, and hyperparameter — regime thresholds, GARCH orders,
GRU architecture sizes, early-warning thresholds, stress episode date ranges, file paths. Every module
imports from it rather than hardcoding values. **Exception**: `cost_ratio_analysis.py`,
`generate_regime_probs.py`, and `diagnose_test_window.py` are standalone diagnostic/utility scripts that
redefine constants locally (e.g. `REGIME_LABELS`, `TRAIN_CUTOFF`) and use relative `'data/...'` paths
instead of importing from `config.py` — keep any constant changes there in sync with `config.py` by hand.

### Pipeline is 14 sequential steps, each reading/writing pickle files under `data/`

Steps communicate exclusively through `data/*.pkl` and `data/*.json` files (paths all defined in
`config.py`), not in-memory handoffs across process boundaries — each script can be run standalone as
long as its upstream `.pkl` inputs already exist:

```
1-5  pipeline/features.py    → features.pkl        (log returns, realized vol, drawdown, skew/kurt,
                                                      rule-based regime labels — see METHODOLOGY.md §2-4)
6    models/garch.py         → garch_output.pkl, adds GARCH_Vol column to features.pkl
7    models/gru_volatility.py→ gru_7j_output.pkl    (inference only — loads gru_best_model_7j.keras)
8    models/monte_carlo.py   → monte_carlo_output.pkl (regime-conditional GBM, 10k paths, 21-day horizon)
9    models/arima.py         → arima_output.pkl     (diagnostic only — validates CI width grows ~sqrt(h))
11   models/gru_regime.py    → regime_probs.pkl     (inference only — loads gru_regime_model.keras;
                                                      batch-recovers any date missing from regime_probs.pkl)
12   early_warning/signals.py→ early_warning_signals.pkl (3 signals computed from regime_probs.pkl)
13   pipeline/evaluation.py  → evaluation_results.json, threshold_calibration.pkl (dual-tier metrics)
14   pipeline/export_json.py  → data/dashboard/{snapshot,timeseries,montecarlo}.json (dashboard data contract)
```

Steps 6, 7, 11 are inference-only in production: the `.keras` models are trained offline
(`models/gru_regime.py`'s `if __name__ == '__main__'` block is the training entry point) and never
retrained by the daily pipeline — `pipeline/run_daily.py` only calls `load_model()`.

### JSON export layer (Step 14) feeds the not-yet-built React frontend

`pipeline/export_json.py` reads `features.pkl`, `monte_carlo_output.pkl`, and
`early_warning_signals.pkl` and writes `data/dashboard/{snapshot,timeseries,montecarlo}.json` — the
data contract the planned React/ECharts frontend will consume. It performs no new computation: every
field is a direct read or a unit/label conversion of an upstream output. `docs/json_contract.md` is
the authoritative field spec.

### Trained model files are not in git

Only the four model artifacts (gru_best_model_7j.keras, gru_regime_model.keras, regime_scaler_X.pkl, cost_ratio_config.json) are committed to data/ (force-added past the data/ ignore rule). The intermediate pipeline outputs (features.pkl, garch_output.pkl, regime_probs.pkl, etc.) are not tracked — they are regenerated from scratch on every pipeline run. The dashboard JSON (data/dashboard/*.json) is tracked and is committed by CI each run, since the frontend consumes it and cannot regenerate it.

### Two-tier model relationship: GARCH+GRU-volatility feed the GRU-regime classifier

`models/gru_regime.py`'s `build_feature_matrix()` uses `GARCH_Vol` (and optionally a `gru_vol_forecast`
column, if Step 7's output has been merged back into `features.pkl`) as an input feature — the regime
classifier is architecturally downstream of the volatility models, not independent of them. See
METHODOLOGY.md §7-8 for the rationale.

### Time-aware, gap-recovering data ingestion

`pipeline/features.py`'s `get_target_date()` compares the current time against NSE's 15:30 IST close to
decide whether "today" is a valid fetch target (avoids ingesting an intraday price as if it were a
close). Every run re-fetches the *full* series from `START_DATE` (2007-09-17) rather than incrementally,
so a missed scheduled run (holiday, machine off) self-heals on the next run — this pattern repeats in
Step 11's batch inference (`run_daily.py`'s `run_regime_step()`), which diffs `features.pkl` dates against
`regime_probs.pkl` and backfills any gap using correct historical lookback windows, never estimation or
carry-forward.

### Regime classification is rule-based, not learned, at the labeling stage

`assign_regimes()` in `pipeline/features.py` assigns one of 4 regimes (Calm/Pullback/Stress/Crisis) via a
deterministic sequential if-elif on `drawdown` and `realized_vol` against `config.THRESHOLDS` — this is
intentional (transparent, reproducible ground truth) and separate from the *learned* GRU regime
classifier, which predicts tomorrow's rule-based label from a 60-day feature history. Don't conflate the
two: `regime` (rule-based, in `features.pkl`) is the training target; `predicted_regime`/`P_regime_*`
(learned, in `regime_probs.pkl`) is the model output.
