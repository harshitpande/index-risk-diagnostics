# Index Risk Diagnostics
## End-to-End Equity Risk Diagnostic and Early Warning System — NIFTY 50

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Pipeline-Daily%20Automated-brightgreen)

> **Central hypothesis:** Risk is more predictable than price. This system
> diagnoses risk states — it does not predict markets.

---

## Overview

This project builds an institutional-grade equity risk diagnostic and early
warning system for the NIFTY 50 index — India's benchmark equity index,
representing the 50 largest companies listed on the National Stock Exchange.

The system answers one question every trading day:

> *"Given the recent history of returns, volatility, drawdowns, and regimes —
> what is the current market risk state, and how is it likely to evolve?"*

It is not a trading signal generator. It is not a price prediction model.
It is a risk diagnostic system — the same category of tool used by central
banks, the IMF, the BIS, and institutional risk managers.

**Scope:** Equity markets only. Specifically the NIFTY 50 index. Does not
cover debt, currency, or credit markets.

---

## Live Dashboard

> 🚀 **[Live Streamlit Dashboard](URL)** — updates daily after NSE market close

**Reference Snapshot: April 9, 2026**
All figures in the associated Medium article reflect this pipeline run.
The live dashboard shows current values.

---

## Architecture

The pipeline executes 14 steps sequentially, each module reading inputs
from disk and writing outputs to disk:

```
Data Ingestion (yfinance ^NSEI)
        ↓
Feature Engineering
(log returns, realized vol, drawdown, rolling skew/kurt)
        ↓
GJR-GARCH Volatility Estimation
(conditional volatility, leverage effect, persistence)
        ↓
GRU Volatility Forecasting
(60-day lookback, 1-day ahead, MAE -49% vs naive)
        ↓
Regime Classification
(4 regimes: Calm → Pullback → Stress → Crisis)
        ↓
GRU Regime Classifier
(softmax probabilities, 60-day lookback)
        ↓
Early Warning Signal Computation
(3 signals: Stress, Crisis Alert, Escalation)
        ↓
Monte Carlo Scenario Simulation
(10,000 paths, regime-conditional drift and volatility)
        ↓
ARIMA Diagnostic Forecast
(uncertainty structure validation only)
        ↓
Model Evaluation
(dual-tier: out-of-sample + in-sample consistency)
        ↓
Dashboard Generation
(4 PNG outputs + Streamlit live dashboard)
```

---

## Four Market Regimes

| Regime | Drawdown | Realized Vol | Description |
|--------|----------|-------------|-------------|
| 0 — Calm | ≥ -5% | ≤ 12% | Market near peak, low uncertainty |
| 1 — Pullback | ≥ -15% | ≤ 25% | Routine correction, contained risk |
| 2 — Stress | Residual | Residual | Elevated uncertainty, deteriorating conditions |
| 3 — Crisis | < -15% | > 25% | Simultaneous deep drawdown and elevated volatility |

Volatility thresholds are empirically derived from the NIFTY dataset:
12% = 40th percentile, 25% = 75th percentile of realized volatility
(2007–2026).

---

## Early Warning System

Three signals, calibrated to an empirical cost ratio of **6.5:1**
(average crisis drawdown 19.51% vs false alarm cost 3.0%):

| Signal | Condition | Purpose |
|--------|-----------|---------|
| Stress Signal | P(Stress+Crisis) > 0.40 for 2 days OR > 0.60 single day | Monitoring |
| Crisis Alert | P(Crisis) > 0.25 for 2 days OR > 0.50 single day | Confirmation |
| Escalation Signal | P(Crisis) strictly rising for 5 consecutive days | Trend detection |

---

## Model Performance

### Tier 1 — Out-of-Sample (2024–present, genuine generalization test)

| Metric | Value | Note |
|--------|-------|------|
| Overall Accuracy | 84.1% | Not the primary metric |
| Transition Accuracy | 93.9% | Primary Tier 1 metric |
| Calm Recall | 77.5% | n=222 |
| Pullback Recall | 91.4% | n=327 |
| Stress Recall | 0.0% | n=11, statistically insufficient |
| Crisis Recall | Not evaluable | n=0, structural property of test window |

### Tier 2 — In-Sample Consistency (2007–2026)

| Episode | Status | Lead Time |
|---------|--------|-----------|
| GFC Acute Crash (2008–2009) | CAPTURED | 28 days |
| Euro Crisis (2011) | CAPTURED | 0 days |
| Taper Tantrum (2013) | CAPTURED | 0 days |
| IL&FS NBFC (2018) | MISSED | — |
| COVID Crash (2020) | CAPTURED | 0 days |
| COVID Recovery Vol (2020) | CAPTURED | 15 days |
| Post-COVID Macro Stress (2021–2022) | CAPTURED | 0 days |

**Episodes captured: 6/7 | Average lead time: 6.1 days**

> ⚠ IL&FS miss is a known and documented limitation. The 2018 crisis was
> a credit/liquidity event at the NBFC sector level. NIFTY 50 index-level
> volatility thresholds were never breached. An index-level volatility
> diagnostic will structurally miss credit events that have not yet
> transmitted to broad market volatility.

---

## Intellectual Foundation

Every design decision in this system is traceable to one of three frameworks:

| Framework | Source | Design Influence |
|-----------|--------|-----------------|
| Risk-first framing | Bodie, Kane and Marcus — *Investments* (2018) | Prioritize risk estimation over return prediction |
| Hybrid modeling discipline | Abdullah Karasan — *ML for Financial Risk Management* (O'Reilly, 2021) | GARCH + GRU complementary, not competing |
| Epistemic humility | John C. Bogle — *The Little Book of Common Sense Investing* (2017) | Probabilistic outputs, never point predictions |

**Academic references:** Hamilton (1989), Glosten/Jagannathan/Runkle (1993),
Cho et al. (2014), Pagan & Sossounov (2003), Ang & Bekaert (2002),
Kaminsky/Lizondo/Reinhart (1998), Black (1976), Reinhart & Rogoff (2009)

---

## Repository Structure

```
index-risk-diagnostics/
├── config.py                    # Single source of truth — all constants
├── dashboard_app.py             # Streamlit dashboard (6 tabs)
├── cost_ratio_analysis.py       # Empirical cost ratio derivation
├── diagnose_test_window.py      # Test window diagnostic analysis
├── generate_regime_probs.py     # Regenerate regime_probs.pkl from saved model
├── run_daily.ps1                # PowerShell automation wrapper
├── schedule_task.xml            # Windows Task Scheduler configuration
├── requirements.txt             # Pinned dependencies
├── pipeline/
│   ├── features.py              # Steps 1–5: data ingestion + feature engineering
│   ├── evaluation.py            # Step 13: dual-tier model evaluation
│   └── run_daily.py             # Master daily runner — all 14 steps
├── models/
│   ├── garch.py                 # Step 6: GJR-GARCH volatility estimation
│   ├── gru_volatility.py        # Step 7: GRU volatility forecasting
│   ├── monte_carlo.py           # Step 8: regime-conditional Monte Carlo
│   ├── arima.py                 # Step 9: ARIMA diagnostic forecasting
│   └── gru_regime.py            # Step 11: GRU regime transition classifier
├── early_warning/
│   └── signals.py               # Step 12: early warning signal computation
└── visualization/
    └── dashboards.py            # Step 14: chart and dashboard generation
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Windows (for Task Scheduler automation) or any OS for manual runs
- Internet connection (yfinance data fetch)

### Installation

```bash
# Clone the repository
git clone https://github.com/harshitpande/index-risk-diagnostics.git
cd index-risk-diagnostics

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Download Model Files

The trained model files are not stored in this repository.
Download them from Google Drive and place in the `data/` folder:

📁 **[Model Files — Google Drive](https://drive.google.com/drive/folders/1XcTdikMvv1vyrfTJzGoRVNDKF1CJWjas?usp=drive_link)**

Files to download into `data/`:
- `gru_best_model_7j.keras`
- `gru_regime_model.keras`
- `regime_scaler_X.pkl`
- `cost_ratio_config.json`

### Run the Pipeline

```bash
# Run full pipeline manually
python pipeline/run_daily.py
```

### Launch Dashboard

```bash
streamlit run dashboard_app.py
```

### Automate Daily Execution (Windows)

The pipeline is designed to run automatically at 5:00 PM IST every weekday
after NSE market close. Import the provided Task Scheduler configuration:

```powershell
schtasks /create /xml schedule_task.xml /tn "index-risk-diagnostics-daily"
```

---

## Data

- **Source:** Yahoo Finance via yfinance (`^NSEI` ticker)
- **Start date:** September 17, 2007
- **Effective start:** December 17, 2007 (after 63-day rolling window warmup)
- **First GRU sequence:** approximately March 2008 (after 60-day lookback warmup)
- **Training cutoff:** January 1, 2024 (fixed, never moved)
- **Update frequency:** Daily, automated at 5:00 PM IST

---

## Known Limitations

1. **IL&FS miss (2018):** Credit/liquidity events not yet transmitted to
   index-level volatility are outside this system's detection scope.

2. **Crisis recall not evaluable:** The 2024–present test window contains
   zero Crisis days by definition. This is a structural property of the
   test period, not a model failure.

3. **Equity markets only:** Debt, credit, and currency stress are not
   captured.

4. **India-specific:** Trained and calibrated on NIFTY 50 data. Performance
   on other indices requires retraining and recalibration.

5. **GRU volatility integration:** The two-stage architecture (GRU vol
   forecast as input to regime classifier) is partially implemented.
   Full integration is planned for a future release.

---

## Medium Article

📖 **[Full technical write-up on Medium](URL)**

The article explains every design decision, threshold calibration,
theoretical foundation, and live diagnostic results from April 9, 2026.

---

## License

MIT License — see [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Harshit Pande

---

## Author

**Harshit Pande**
Building institutional-grade risk diagnostic systems for emerging market
equity indices.

[GitHub](https://github.com/harshitpande) |
[Medium](URL) |
[LinkedIn](URL)
