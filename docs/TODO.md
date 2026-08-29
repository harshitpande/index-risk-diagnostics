# TODO / Deferred Items — NIFTY 50 Risk Diagnostics

> Running backlog of observations noted during the rebuild but deliberately deferred, so they aren't lost and aren't addressed at the wrong time. Each item records *what*, *why it was deferred*, and *when/what triggers addressing it*. Remove an item when done (git history keeps the record).

---

## Code / architecture cleanup

### 1. `REGIME_COLORS` defined in three places
- **What:** `REGIME_COLORS` is independently defined in `config.py`, `models/gru_regime.py`, and `diagnose_test_window.py`. Not confirmed identical — the hex values may have drifted between copies.
- **Why deferred:** out of scope during the `REGIME_LABELS` dedup; widening into it mid-refactor breaks scope discipline.
- **Trigger / resolution:** address during the matplotlib-stripping cleanup. Note: colours are *presentation*, and the JSON contract keeps colour out of the data (frontend owns the palette, REQUIREMENTS.md §3). So the right fix may be to **delete** these Python-side colour dicts entirely once the old matplotlib plotting is removed, not consolidate them. Decide then.

### 2. `cost_ratio_analysis.py` line ~160 ternary — add a clarifying comment
- **What:** `'Stress' if previous['regime']==2 else 'Crisis'`. An automated read flagged this as a "latent bug" for not handling regimes 0/1.
- **Why deferred / status:** **Not a bug** — confirmed deliberate. It is part of false-alarm measurement, only ever called in a context where the regime is already 2 or 3, so the two-way branch is correct and sufficient. No code change needed.
- **Trigger / resolution:** add a one-line comment at that line (e.g. `# safe: only reached when regime is 2 or 3 — false-alarm check, not a general mapping`) so a future reader/agent doesn't "fix" a non-bug. Do at next natural edit of that file.

### 3. Matplotlib plotting code embedded in kept modules
- **What:** `models/monte_carlo.py`, `models/arima.py`, `pipeline/evaluation.py`, `early_warning/signals.py`, and `models/gru_regime.py` mix reusable numeric logic (keep) with matplotlib/seaborn plotting calls that only served the old PNG/Streamlit display layer.
- **Why deferred:** each is a surgical edit inside a file we depend on; deserves its own careful, diff-reviewed session, separate from the labels/units work.
- **Trigger / resolution:** dedicated "strip old plotting" session, ideally after the JSON export layer works (so nothing depends on the old plots).

### 4. Stale `data/*.pkl` outputs committed from the old pipeline
- **What:** `data/` holds pickle outputs from the final old-architecture run (last "Daily dashboard update" 2026-07-30). The new architecture emits JSON to `data/dashboard/` instead.
- **Why deferred:** some `.pkl` files may still be read as inputs during the pipeline refactor transition; don't delete prematurely.
- **Trigger / resolution:** resolve during / at the end of the pipeline refactor (step 3), once the JSON export is the source of truth and nothing reads the stale pkls.

---

## Environment / tooling

### 5. No Python environment set up in the fresh clone
- **What:** `C:\dev\nifty_risk_system` has no venv yet. Heavy deps (TensorFlow/Keras, etc.) won't import until an environment exists.
- **Why deferred:** not needed for the lightweight verification/refactor steps done so far (config imports, syntax checks).
- **Trigger / resolution:** required before running the full pipeline end-to-end and before the GitHub Actions workflow (step 4). Set up a clean venv + a corrected `requirements.txt` (see item 6) at that point.

### 6. `requirements.txt` needs splitting / de-pinning + `tensorflow` is missing
- **What:** `requirements.txt` still carries Streamlit deps (removable), is exact-pinned due to a now-irrelevant Streamlit-Cloud pyarrow/numpy ABI issue (can likely be loosened), is UTF-16 encoded (trips naive edits), and does **not** list `tensorflow` despite the GRU modules importing it.
- **Why deferred:** tied to the environment setup (item 5) and the CI workflow; better done as one deliberate dependency pass.
- **Trigger / resolution:** during environment setup / step 4 (GitHub Actions), produce a clean, correct requirements file the CI runner can rely on.

### 7. `inspect_units.py` — throwaway diagnostic
- **What:** temporary script created to verify pkl value units.
- **Why deferred:** may be re-run while building the export layer.
- **Trigger / resolution:** delete at the end of step 3.

---

## Documentation

### 8. `README.md` describes the old architecture
- **What:** README still documents venv setup, `streamlit run`, Task Scheduler, Google Drive model download — all removed.
- **Why deferred:** low urgency; content isn't harmful, just stale.
- **Trigger / resolution:** rewrite once the new architecture is working. Carry forward the regime-table and methodology content rather than starting from scratch.

### 9. `METHODOLOGY.md` §13–14 describe the dead Streamlit/Task-Scheduler stack
- **What:** §1–12 (data ingestion → evaluation methodology) are reusable; §13 "Dashboard and Visualization Layer" and §14 "Production Pipeline Architecture" describe the retired architecture.
- **Why deferred:** needs the new architecture (JSON contract + Actions + React) to be settled before rewriting.
- **Trigger / resolution:** **planned first Cowork task** — bounded, multi-file, read-then-write. Do once the JSON contract and pipeline shape are stable.

---

### 10. Forecast dates use generic business days, not the NSE trading calendar

What: montecarlo.json forecast dates are built with pd.bdate_range (Mon–Fri), which includes Indian market holidays as if they were trading days. Matches existing arima.py behavior.
Why deferred: cosmetically minor over a 21-day horizon; doesn't affect the risk story. Accepted for MVP.
Trigger: revisit if forecast-date precision matters; would use an NSE holiday calendar instead of bdate_range.

### 11. data/dashboard/*.json must be un-ignored for CI to commit them

What: the export layer writes to data/dashboard/, but data/ is gitignored — so the JSON outputs won't be tracked by default. The GitHub Actions architecture depends on committing these JSON files back to the repo for the frontend to read.
Why deferred: it's a step-4 (GitHub Actions) concern, not needed for local export testing.
Trigger: address when building the GitHub Actions workflow — add un-ignore rules for data/dashboard/*.json (same pattern as the model-artifacts un-ignore).

## Notes on maintaining this file
- Add items as they surface; don't fix them mid-unrelated-task (scope discipline).
- Record the *why deferred* and *trigger*, not just the *what* — the context is what makes the item actionable later.
- Remove completed items (git history preserves them).
