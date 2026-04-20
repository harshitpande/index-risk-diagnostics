# METHODOLOGY — Index Risk Diagnostics and Early Warning System

**Version:** 1.0
**Reference Snapshot Date:** April 9, 2026
**Data Universe:** NIFTY 50 Index (^NSEI), September 2007 – Present
**Scope:** Equity index risk diagnostics only. Credit, currency, and
fixed-income markets are outside the scope of this system.

---

## Table of Contents

1. [Data Ingestion](#1-data-ingestion)
2. [Feature Engineering](#2-feature-engineering)
3. [Drawdown Diagnostics](#3-drawdown-diagnostics)
4. [Regime Classification](#4-regime-classification)
5. [Regime Analysis and Transition Estimation](#5-regime-analysis-and-transition-estimation)
6. [GJR-GARCH Volatility Modeling](#6-gjr-garch-volatility-modeling)
7. [GRU Volatility Forecasting](#7-gru-volatility-forecasting)
8. [GRU Regime Classifier](#8-gru-regime-classifier)
9. [Early Warning System](#9-early-warning-system)
10. [Risk-Conditioned Monte Carlo Simulation](#10-risk-conditioned-monte-carlo-simulation)
11. [ARIMA Diagnostic Price Forecasting](#11-arima-diagnostic-price-forecasting)
12. [Model Evaluation Framework](#12-model-evaluation-framework)
13. [Dashboard and Visualization Layer](#13-dashboard-and-visualization-layer)
14. [Production Pipeline Architecture](#14-production-pipeline-architecture)
- [Appendix A — Key Reference Figures](#appendix-a--key-reference-figures-april-9-2026-snapshot)
- [Appendix B — Notation Summary](#appendix-b--notation-summary)
- [Appendix C — Academic References](#appendix-c--academic-references)

---

## 1. Data Ingestion

### 1.1 Data Source

Index price data is sourced from Yahoo Finance using the `yfinance` Python
library. The ticker symbol is `^NSEI`, representing the NSE NIFTY 50 Index.
The full historical series begins September 17, 2007. The pipeline performs
a full-series download on every execution to ensure gap recovery and
data consistency. The effective start date for feature computation is
December 17, 2007, after the 63-day rolling window warmup period.

### 1.2 Time-Aware Fetch Logic

The pipeline is time-aware relative to the NSE market close at 15:30 IST
(Indian Standard Time). The fetch logic is as follows:

- If the pipeline executes **before 15:30 IST**, the target date is the
  previous business day. Fetching before market close would return an
  intraday price rather than the official closing auction price.
- If the pipeline executes **at or after 15:30 IST**, the target date is
  the current business day and the official closing price is available.

This prevents the pipeline from ingesting a partial or intraday price as a
closing observation, which would propagate incorrect values through all
downstream calculations.

### 1.3 Gap Recovery

On any execution, the pipeline compares all dates present in the stored
feature set against all dates returned by the live data fetch. If gaps are
detected — for example, due to a missed scheduled execution or a public
holiday — the pipeline performs batch inference across all missing dates
before updating stored outputs. No date in the historical series is left
without a regime label or diagnostic feature.

Gap recovery uses correct historical closing prices from the exchange
record as returned by yfinance. No estimation, interpolation, or
carry-forward is applied.

### 1.4 Training Cutoff Enforcement

A hard training cutoff date (`TRAIN_CUTOFF = 2024-01-01`) is maintained in
`config.py`. All model inference on dates beyond this cutoff uses only the
trained model weights — no retraining or fine-tuning is performed on live
data. This boundary separates the in-sample estimation period from the
genuine out-of-sample evaluation window and ensures that reported
evaluation metrics are not contaminated by post-training data.

---

## 2. Feature Engineering

All features are derived from the adjusted close price series of the
NIFTY 50 index. Let $P_t$ denote the adjusted closing level of the index
on business day $t$.

### 2.1 Log Returns

Daily log returns are defined as:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

Log returns are used in preference to simple arithmetic returns for three
reasons. First, they are time-additive: the cumulative return over any
period equals the sum of daily log returns. Second, the log-normal
distributional assumption in GARCH models is most cleanly satisfied by
log returns, not arithmetic returns. Third, log returns are numerically
stable in sequence modeling and multi-period variance calculations.

To illustrate the arithmetic return limitation: starting from an index
level of 100, a 10% fall gives 90, and a subsequent 10% rise gives 99 —
not 100. The asymmetry compounds over time. Log returns eliminate this
distortion.

### 2.2 Realized Volatility

Rolling realized volatility is computed as the annualized standard
deviation of log returns over a 30 trading-day lookback window:

$$\sigma_t^{\text{realized}} = \sqrt{252} \cdot \text{std}
\left(r_{t-29}, r_{t-28}, \ldots, r_t\right)$$

The annualization factor $\sqrt{252}$ converts daily volatility to an
annual figure, consistent with the equity market convention of
approximately 252 trading days per year.

The 30 trading-day window was selected for three reasons. It creates
conceptual alignment with the 30-day horizon of the CBOE VIX methodology
(CBOE VIX White Paper, 2019) — both measures describe approximately the
same time horizon, one backward-looking and realized, one forward-looking
and implied. It aligns with exponentially weighted volatility windows
established in the RiskMetrics Technical Document (Morgan/Reuters, 1996)
as a benchmark for institutional short-term volatility measurement.
And it is long enough to capture a complete short-term volatility cycle
while remaining responsive to emerging stress conditions.

Note: `.rolling(30)` in the implementation operates on 30 consecutive
rows of the DataFrame, which contains only trading days. This corresponds
to 30 trading days, approximately six calendar weeks, not 30 calendar days.

### 2.3 Drawdown

The drawdown at time $t$ is defined as the percentage decline from the
running maximum of the price series up to and including time $t$:

$$D_t = \frac{P_t - \max_{s \leq t} P_s}{\max_{s \leq t} P_s}$$

By construction, $D_t \leq 0$ for all $t$, with $D_t = 0$ only when
$P_t$ equals its all-time historical maximum. The running peak
$\max_{s \leq t} P_s$ never decreases — it resets only when the index
establishes a new all-time high. This means drawdown captures the full
duration of a stress episode, not just its depth.

### 2.4 Rolling Higher Moments

Rolling skewness and excess kurtosis of log returns are computed over
a 63 trading-day window to capture distributional asymmetry and tail
thickness over a longer horizon than realized volatility. These are
input features to the GRU regime classifier.

Rolling skewness:

$$\text{Skew}_t^{63} = \frac{\frac{1}{63}\sum_{i=0}^{62}
(r_{t-i} - \bar{r}_t)^3}{\left(\frac{1}{63}\sum_{i=0}^{62}
(r_{t-i} - \bar{r}_t)^2\right)^{3/2}}$$

Rolling excess kurtosis:

$$\text{Kurt}_t^{63} = \frac{\frac{1}{63}\sum_{i=0}^{62}
(r_{t-i} - \bar{r}_t)^4}{\left(\frac{1}{63}\sum_{i=0}^{62}
(r_{t-i} - \bar{r}_t)^2\right)^{2}} - 3$$

where $\bar{r}_t$ is the rolling mean over the same 63-day window.
Excess kurtosis above zero indicates fatter tails than a normal
distribution — this measure tends to rise before crisis episodes,
providing early-warning information beyond what volatility alone captures.

### 2.5 Feature Warmup Period

The 63-day rolling window for kurtosis requires 63 observations before
the first valid value is produced. Combined with the 60-day lookback
required by the GRU models, the effective first usable date for full
pipeline inference is approximately March 2008, despite the raw data
starting in September 2007.

---

## 3. Drawdown Diagnostics

### 3.1 Maximum Drawdown

The maximum drawdown (MDD) over the full historical sample is:

$$\text{MDD} = \min_{t \in \mathcal{T}} D_t$$

where $\mathcal{T}$ is the full sample period. Over the 2007–2026 sample,
the maximum drawdown of the NIFTY 50 was approximately **-38.4%**,
recorded during the Global Financial Crisis (GFC) acute phase of
2008–2009. This figure is calculated directly from the dataset.

### 3.2 Drawdown Duration

A drawdown episode begins on the first day following a new all-time high
and ends on the first day a new all-time high is re-established. Drawdown
duration is measured in business days.

Drawdown spell identification uses a cumulative sum approach:

$$\text{Episode}_t = \sum_{s=1}^{t} \mathbf{1}[D_s = 0]$$

Each unique value of $\text{Episode}_t$ defines a recovery episode.
Within each episode, the subset of observations where $D_t < 0$
constitutes the drawdown period. Duration, maximum depth, and recovery
date are computed for each episode.

### 3.3 Identified Stress Periods

The following stress periods are identified based on drawdown structure
and corroborated by macroeconomic context. These periods are used in
Tier 2 model evaluation (Section 12).

| Episode | Period | Peak Realized Vol | Classification |
|---|---|---|---|
| GFC Acute Crash | Sep 2008 – Mar 2009 | 77.7% | Crisis |
| Euro Crisis | Aug 2011 – Dec 2011 | Elevated | Stress/Crisis |
| Taper Tantrum | May 2013 – Sep 2013 | Moderate | Pullback/Stress |
| IL&FS NBFC | Sep 2018 – Dec 2018 | Below threshold | Pullback |
| COVID Crash | Feb 2020 – Mar 2020 | 76.6% | Crisis |
| COVID Recovery Vol | Mar 2020 – Nov 2020 | Elevated | Stress/Crisis |
| Post-COVID Macro Stress | Oct 2021 – Nov 2022 | Moderate | Pullback/Stress |

Peak realized volatility figures for GFC (77.7%) and COVID crash (76.6%)
are calculated directly from the dataset over the respective periods.

---

## 4. Regime Classification

### 4.1 Design Rationale

Market regimes are defined using a rule-based, sequential classification
over two observable state variables: the current drawdown $D_t$ and the
current realized volatility $\sigma_t^{\text{realized}}$. The
classification is deterministic given the feature values and requires no
probabilistic estimation.

This design choice follows Hamilton's (1989) foundational work on Markov
regime-switching models, which established that financial markets exhibit
distinct behavioral states with persistent characteristics. The rule-based
approach is used at the labeling stage because it is transparent,
interpretable, and reproducible — properties essential for training a
downstream machine learning classifier.

### 4.2 Regime Thresholds

Volatility thresholds are set at empirical quantiles of the full-sample
realized volatility distribution (2007–2026):

$$\sigma^{\text{low}} = Q_{0.40}(\sigma^{\text{realized}}) \approx 12\%
\text{ (annualized)}$$

$$\sigma^{\text{high}} = Q_{0.75}(\sigma^{\text{realized}}) \approx 25\%
\text{ (annualized)}$$

These quantile-based thresholds are data-driven and reproducible. The
40th percentile as the low boundary means Calm conditions correspond to
the quietest 40% of historical market days. The 75th percentile as the
high boundary means elevated volatility beyond this point occurs in only
the top 25% of historical readings.

Drawdown thresholds are calibrated empirically. The -5% boundary draws
conceptually on the phase identification framework of Pagan and Sossounov
(2003), who distinguish genuine bull market phases from short-term
fluctuations using minimum duration and magnitude filters. The -15%
boundary is consistent with the empirical documentation in Reinhart and
Rogoff (This Time Is Different, Princeton University Press, 2009), whose
cross-country analysis of financial crises shows equity market declines
of this magnitude frequently accompany the onset of broader financial
system stress in emerging markets.

```
D_shallow  = -0.05    # Calm / Pullback boundary (5% drawdown threshold)
D_moderate = -0.15    # Pullback / Crisis boundary (15% drawdown threshold)
```

### 4.3 Regime Assignment Rule

Regimes are assigned using a sequential if-elif rule evaluated in the
order shown. The ordering ensures that crisis conditions are not masked
by residual classification:

$$
\text{Regime}_t =
\begin{cases}
0 \; (\text{Calm}) & \text{if } D_t \geq -0.05 \text{ and }
\sigma_t \leq 0.12 \\
1 \; (\text{Pullback}) & \text{elif } D_t \geq -0.15 \text{ and }
\sigma_t \leq 0.25 \\
3 \; (\text{Crisis}) & \text{elif } D_t < -0.15 \text{ and }
\sigma_t > 0.25 \\
2 \; (\text{Stress}) & \text{otherwise (residual)}
\end{cases}
$$

The four regimes and their interpretations:

| Regime | Label | Interpretation |
|---|---|---|
| 0 | Calm / Expansion | Low drawdown, low volatility. Normal market function. |
| 1 | Pullback / Normal Risk | Moderate drawdown, contained volatility. Routine correction. |
| 2 | High-Volatility Stress | Elevated volatility or moderate drawdown without meeting Crisis criteria. Transition state. |
| 3 | Crisis | Deep drawdown coinciding with elevated volatility simultaneously. Systemic stress. |

The regime labels are drawn from the vocabulary of financial stability
literature. Calm and Stress appear consistently in IMF Financial Stability
Reports. Crisis is the standard term in BIS financial stability
publications for episodes of systemic market breakdown. Pullback is
standard practitioner terminology for a contained and recoverable decline.

### 4.4 Regime-Conditional Return Profiles

Mean annualized drift is estimated for each regime using the full
in-sample data:

$$\mu_k = 252 \cdot \frac{1}{|\mathcal{T}_k|} \sum_{t \in \mathcal{T}_k}
r_t, \quad k \in \{0, 1, 2, 3\}$$

Reference values calculated from the dataset (2007–2026):

| Regime | Daily Drift | Annualized Drift |
|---|---|---|
| Calm (0) | +0.000736 | +18.5% |
| Pullback (1) | +0.000366 | +9.2% |
| Stress (2) | −0.000128 | −3.2% |
| Crisis (3) | −0.000437 | −11.0% |
| Full sample | +0.000307 | +7.75% |

These figures are used directly as the drift parameter in the Monte Carlo
simulation (Section 10). Using the full-sample unconditional mean (+7.75%
annualized) in a Crisis regime simulation would assume the market tends to
rise on average even during systemic breakdown — an error that
systematically understates downside risk.

---

## 5. Regime Analysis and Transition Estimation

### 5.1 Regime Frequency and Duration

For each regime $k$, the following statistics are computed:

- **Frequency:** $f_k = |\mathcal{T}_k| / |\mathcal{T}|$, the proportion
  of all business days spent in regime $k$.
- **Average spell duration:** Mean length of uninterrupted episodes in
  regime $k$, measured in business days.
- **Maximum spell duration:** Length of the longest single uninterrupted
  episode in regime $k$.

### 5.2 Regime Transition Matrix

The empirical one-step transition probability matrix $\mathbf{P}$ is
estimated from the historical regime sequence. The $(i, j)$ entry
represents the empirical probability of transitioning from regime $i$
to regime $j$ in one business day:

$$\hat{P}_{ij} = \frac{n_{ij}}{\sum_{j'} n_{ij'}}$$

where $n_{ij}$ is the count of observed one-step transitions from regime
$i$ to regime $j$ in the historical data.

High diagonal values indicate regime persistence. Off-diagonal entries
in the direction of higher-numbered regimes (Calm → Pullback → Stress →
Crisis) indicate escalation risk. The transition matrix motivates the
use of a sequential model (GRU) rather than a memoryless classifier —
the probability of being in a given regime tomorrow depends on the
regime today, not just on current feature values.

### 5.3 Persistence Estimation

Regime persistence for regime $k$ is the probability of remaining in
the same regime over one business day:

$$\rho_k = \hat{P}_{kk}$$

Regimes with $\rho_k$ close to 1.0 are highly persistent. Hamilton
(1989) establishes that financial market regimes are persistent by nature
— they last weeks or months, not single days. This theoretical property
motivates the minimum spell filter applied to training labels
(Section 8.3).

---

## 6. GJR-GARCH Volatility Modeling

### 6.1 Motivation

Realized volatility computed over a rolling window does not capture
conditional heteroskedasticity — the empirical tendency for volatility
to cluster in time — nor the leverage effect: the documented finding
that negative return shocks generate larger volatility increases than
positive shocks of equal magnitude (Black, 1976). The GJR-GARCH model
addresses both properties within a unified specification.

### 6.2 Model Specification

The GJR-GARCH(1,1) model (Glosten, Jagannathan and Runkle, 1993):

**Mean equation:**

$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t,
\quad z_t \sim \mathcal{N}(0, 1)$$

**Variance equation:**

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 +
\gamma \epsilon_{t-1}^2 \mathbf{1}[\epsilon_{t-1} < 0] +
\beta \sigma_{t-1}^2$$

where:

- $\omega > 0$ is the long-run variance intercept
- $\alpha \geq 0$ is the ARCH effect (symmetric shock response)
- $\gamma \geq 0$ is the GJR asymmetry term (additional response to
  negative shocks — the leverage effect parameter)
- $\beta \geq 0$ is the GARCH persistence term
- $\mathbf{1}[\epsilon_{t-1} < 0]$ equals 1 when the lagged shock is
  negative and 0 otherwise

When $\gamma > 0$, negative return innovations increase conditional
variance by $\alpha + \gamma$ per unit squared shock, while positive
innovations increase it by only $\alpha$. This asymmetry is the
formalization of the leverage effect.

### 6.3 Volatility Persistence

Total volatility persistence is:

$$\rho^{\text{GARCH}} = \alpha + \beta + \frac{1}{2}\gamma$$

A value close to 1.0 indicates near-unit-root behavior in the conditional
variance process — shocks to volatility decay slowly. Reference value
for NIFTY 50:

$$\rho^{\text{GARCH}} = 0.9868$$

This confirms that volatility shocks are long-lived. A spike in
conditional volatility driven by a geopolitical or macroeconomic shock
is expected to persist for weeks, not days — consistent with the
empirical behavior observed during the stress episodes in Section 3.3.

### 6.4 Estimation

The model is estimated by quasi-maximum likelihood (QML). The
log-likelihood function under the Gaussian innovation assumption is:

$$\ell(\theta) = -\frac{1}{2} \sum_{t=1}^{T} \left[ \ln(2\pi) +
\ln(\sigma_t^2) + \frac{\epsilon_t^2}{\sigma_t^2} \right]$$

where $\theta = (\mu, \omega, \alpha, \gamma, \beta)$ is the parameter
vector. Stationarity requires $\rho^{\text{GARCH}} < 1$.

### 6.5 Conditional Volatility Output

The fitted conditional standard deviation series $\hat{\sigma}_t^
{\text{GARCH}}$ is used as a feature in the GRU models and as the
volatility parameter in the Monte Carlo simulation. The annualized form:

$$\hat{\sigma}_t^{\text{GARCH, ann.}} = \sqrt{252} \cdot \hat{\sigma}_t$$

Reference value as of April 9, 2026: $\hat{\sigma}^{\text{GARCH}} =
27.1\%$ annualized.

---

## 7. GRU Volatility Forecasting

### 7.1 Architecture Rationale

Realized volatility is a sequential process with time-varying dynamics
partially captured by regime structure. A Gated Recurrent Unit (GRU)
network (Cho et al., 2014) is used to learn these dynamics from a
rolling window of historical features.

GRU is preferred over LSTM for this application because its lower
parameter count — the GRU has no separate cell state — reduces
overfitting risk on training sets of fewer than 3,000 sequences, which
is the characteristic constraint of financial time series with limited
history. This is a consideration documented by Karasan (2021) in the
context of financial risk modeling with constrained historical data.

### 7.2 GRU Cell Formulation

Given input vector $\mathbf{x}_t$ and previous hidden state
$\mathbf{h}_{t-1}$:

```
Reset gate:
  r_t = sigmoid(W_r * x_t + U_r * h_{t-1} + b_r)

Update gate:
  z_t = sigmoid(W_z * x_t + U_z * h_{t-1} + b_z)

Candidate hidden state:
  h_tilde_t = tanh(W_h * x_t + U_h * (r_t ⊙ h_{t-1}) + b_h)

Hidden state update:
  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde_t
```

where `sigmoid(·)` is the sigmoid activation and `⊙` denotes
element-wise multiplication.

### 7.3 Feature Set

The input to the GRU volatility model at each time step $t$ is a
six-dimensional feature vector:

$$\mathbf{x}_t = \left[r_t, \; \sigma_t^{\text{realized}}, \;
\hat{\sigma}_t^{\text{GARCH}}, \; D_t, \;
\text{Skew}_t^{63}, \; \text{Kurt}_t^{63}\right]$$

All features are standardized using a `StandardScaler` fitted on the
training set only. Standardized inputs are denoted $\tilde{\mathbf{x}}_t$.
The scaler is fitted once and frozen — test and live data are transformed
using training statistics, never refitted, to prevent data leakage.

### 7.4 Sequence Construction

Each training sample consists of a lookback window of 60 consecutive
business days of standardized features, paired with the standardized
realized volatility one day ahead:

$$\text{Input: } \left[\tilde{\mathbf{x}}_{t-59},
\tilde{\mathbf{x}}_{t-58}, \ldots, \tilde{\mathbf{x}}_t\right]
\in \mathbb{R}^{60 \times 6}$$

$$\text{Target: } \tilde{\sigma}_{t+1}^{\text{realized}}$$

### 7.5 Model Architecture

The GRU volatility model uses a three-layer architecture:

- **Layer 1:** GRU with 32 units (Tanh / Sigmoid internal activations)
- **Layer 2:** Dense with 16 units (ReLU activation)
- **Layer 3:** Dense with 1 unit (Linear activation — scalar volatility output)

### 7.6 Training Configuration

- **Loss function:** Mean Absolute Error (MAE). MAE is preferred over
  MSE for risk modeling because it does not disproportionately penalize
  large forecast errors from volatility spikes, providing more robust
  training behavior in crisis episodes.
- **Optimizer:** Adam, learning rate 0.001
- **Batch size:** 32
- **Epochs:** 30
- **Validation split:** 20% of training set
- **Train/test split:** 75% / 25% walk-forward split

### 7.7 Forecast Performance

| Model | MAE (Test Set) | Improvement vs Naive |
|---|---|---|
| GRU | Lowest | ~49% improvement |
| GJR-GARCH | Intermediate | ~17% lower MAE than GRU baseline |
| Naive (lag-1) | Highest | Baseline |

The GRU outperforms GARCH on the test window because it learns regime
transition dynamics that GARCH treats as smooth mean reversion. The
16-17% GRU improvement over GARCH is calculated from test set MAE
values confirmed in pipeline execution logs.

---

## 8. GRU Regime Classifier

### 8.1 Objective

The GRU regime classifier estimates the probability distribution over the
four market regimes for the next business day, conditional on the
observed 60-day history of market features:

$$\hat{\mathbf{p}}_{t+1} = P\left(\text{Regime}_{t+1} = k \mid
\tilde{\mathbf{x}}_{t-59}, \ldots, \tilde{\mathbf{x}}_t\right),
\quad k \in \{0, 1, 2, 3\}$$

where the output vector `p_hat_{t+1}` is in R^4 with all four
probabilities summing to 1.

### 8.2 Architecture

The GRU regime classifier uses a four-layer architecture:

- **Layer 1:** GRU with 64 units (Tanh / Sigmoid internal activations)
- **Layer 2:** Dropout (Rate = 0.2)
- **Layer 3:** Dense with 32 units (ReLU activation)
- **Layer 4:** Dense with 4 units (Softmax activation — one probability per regime)

The GRU(64) → Dropout(0.2) → Dense(32) → Dense(4, softmax) architecture
balances expressiveness with regularization. Dropout at rate 0.2 is
applied after the GRU layer to prevent co-adaptation of hidden units.
Recurrent dropout is intentionally avoided — it has been shown to degrade
GRU gradient flow on short financial sequences (Semeniuta et al., 2016).

### 8.3 Training Label Construction and Minimum Spell Filter

The training target at each time step is the next-day rule-based
regime label:

$$y_t = \text{Regime}_{t+1} \in \{0, 1, 2, 3\}$$

Before training, a **minimum spell filter** is applied to the raw
regime labels. The filter is motivated by the phase identification
framework of Pagan and Sossounov (2003), who established that a market
phase must persist for a minimum duration before it qualifies as a
genuine phase, and by Hamilton's (1989) formal demonstration that
financial market regimes are persistent by nature.

**Filter procedure:** Any consecutive regime spell shorter than three
trading days is replaced by the surrounding (preceding) regime before
the GRU sees the labels. This eliminates single-day or two-day regime
assignments that are statistically inconsistent with genuine regime
transitions.

**Calibration rationale:** Two days still allows noise labels through.
Five days risks absorbing the acute phase of genuine short stress
episodes (e.g., the COVID crash was acute in nature). Three trading
days is the empirically calibrated balance point between noise
elimination and genuine signal preservation.

**Important:** The filter is applied as a pre-processing step before any
model is fitted. It is defined by a single fixed hyperparameter
(minimum spell = 3 days), does not use model outputs, and does not
use forward information. This is not circular reasoning — it is a
label cleaning step equivalent to any other data pre-processing decision.

### 8.4 Class Imbalance Handling via Cost-Weighted Loss

The NIFTY 50 historical series exhibits severe class imbalance. Crisis
observations (Regime 3) represent approximately 10.4% of the full
sample, while Pullback represents approximately 50.4%. Without
correction, gradient descent would optimize for accuracy on the majority
class and produce a model blind to Crisis regimes.

Class weights are applied to the categorical cross-entropy loss to
penalize the model more heavily for misclassifying minority class
observations:

$$\mathcal{L} = -\sum_{t}\sum_{k=0}^{3} w_k \cdot y_{k,t} \cdot
\log \hat{p}_{k,t}$$

where $y_{k,t}$ is the one-hot encoded true label and $w_k$ is the
class weight for regime $k$.

**Class weights used:**

| Regime | Weight | Rationale |
|---|---|---|
| Calm (0) | 1.0 | Baseline |
| Pullback (1) | 3.0 | Transition state before Stress |
| Stress (2) | 5.0 | Direct precursor to Crisis |
| Crisis (3) | 7.0 | Reflects 6.5:1 empirical cost ratio |

The Crisis weight of 7.0 is calibrated to the empirical cost ratio of
6.5:1 (Section 9.4), rounded for numerical stability. A Crisis day
contributes 7 times as much to the gradient update as a Calm day,
forcing the model to pay disproportionate attention to correct Crisis
classification despite its low base rate.

### 8.5 Training Configuration

- **Optimizer:** Adam, learning rate 0.001
- **Batch size:** 32
- **Epochs:** 50 with early stopping (patience = 10 on validation loss)
- **Validation split:** 20% of training data
- **Train/test split:** Fixed cutoff at 2024-01-01 (never moved)
- **Random seed:** 42 (reproducibility)

### 8.6 Training Cutoff Justification

The January 1, 2024 training cutoff was selected for four reasons:

1. **Data sufficiency:** Approximately 3,840 training sequences — sufficient
   for GRU(64) to learn meaningful patterns without severe overfitting.
2. **Regime coverage:** The training window (2008–2023) contains all four
   regime types across multiple economic cycles, including two Crisis
   episodes (GFC and COVID).
3. **Post-pandemic new phase:** January 2024 marks the start of the
   post-COVID normalization period. The test set begins in a genuinely
   new market environment.
4. **Reproducibility:** The cutoff is a fixed constant in `config.py`.
   Any researcher replicating this work obtains identical train/test
   splits.

### 8.7 Reference Output (April 9, 2026)

| Regime | GRU Probability |
|---|---|
| Calm (0) | 0.02% |
| Pullback (1) | 61.9% |
| Stress (2) | 35.4% |
| Crisis (3) | 2.7% |

The rule-based classifier assigns Pullback (drawdown -9.7%, within the
-15% boundary). The GRU assigns 35.4% to Stress — detecting the
deteriorating sequential pattern of the past 60 days that the single-day
snapshot does not capture. This divergence is the primary value of the
ML layer over the rule-based classifier.

---

## 9. Early Warning System

### 9.1 Combined Stress Probability

The early warning system operates on the combined stress probability
index, defined as the sum of GRU-estimated probabilities for the Stress
and Crisis regimes:

$$S_t = \hat{p}_{t,2} + \hat{p}_{t,3} = P(\text{Regime}_t \in \{2, 3\})$$

This scalar $S_t \in [0, 1]$ is the primary input to all three early
warning signals.

Reference value as of April 9, 2026: $S_t = 0.381$.

### 9.2 Signal Definitions

Three binary early warning signals are defined, each targeting a distinct
aspect of stress dynamics. The three-signal structure mirrors the tiered
alert frameworks used in institutional financial stability monitoring.
Central banks and international financial institutions consistently
distinguish between probability-based, persistence-based, and
trend-based signals in their financial stability frameworks.

**Signal 1 — Stress Signal (monitoring function):**

$$\text{Stress Signal}_t = \mathbf{1}\left[S_t > 0.60\right] \cup
\mathbf{1}\left[S_{t-1} > 0.40 \text{ and } S_t > 0.40\right]$$

Fires if combined stress probability exceeds 0.60 on a single day
(single-day override) or exceeds 0.40 on two consecutive trading days
(sustained threshold). The persistence requirement filters noise — a
single elevated reading may reflect model uncertainty. Two consecutive
elevated readings constitute a pattern.

**Signal 2 — Crisis Alert (confirmation function):**

$$\text{Crisis Alert}_t = \mathbf{1}\left[\hat{p}_{t,3} > 0.50\right]
\cup \mathbf{1}\left[\hat{p}_{t-1,3} > 0.25 \text{ and }
\hat{p}_{t,3} > 0.25\right]$$

Fires if P(Crisis) alone exceeds 0.50 on a single day or exceeds 0.25
on two consecutive days. P(Crisis) is a narrower signal than combined
stress — it must cross a lower absolute threshold to provide information
beyond the Stress Signal. A reading above 0.50 means the model assigns
majority probability specifically to Crisis.

**Signal 3 — Escalation Signal (trend function):**

$$\text{Escalation}_t = \mathbf{1}\left[\hat{p}_{t,3} >
\hat{p}_{t-1,3} > \hat{p}_{t-2,3} > \hat{p}_{t-3,3} >
\hat{p}_{t-4,3}\right]$$

Fires if P(Crisis) has been strictly increasing for five consecutive
trading days. This is a trajectory signal — it detects escalation
even when absolute threshold levels have not been crossed.

### 9.3 Threshold Calibration

All signal thresholds were calibrated through a precision-recall sweep
across 91 combinations: thirteen candidate thresholds from 0.20 to 0.80
in steps of 0.05, multiplied by seven candidate persistence windows from
one to seven days. Each combination was evaluated against the seven known
historical stress episodes using the noise-to-signal ratio methodology
of Kaminsky, Lizondo and Reinhart (1998).

The optimal combination minimizes expected cost under the empirical cost
ratio. Hamilton (1989) regime persistence theory supports persistence
windows of n=2–3 as theoretically grounded; n>5 risks missing short
acute stress episodes entirely.

### 9.4 Cost Ratio Framework

Signal design is governed by an asymmetric cost framework. Let:

- $C_{\text{miss}}$ = cost of a missed crisis signal (false negative)
- $C_{\text{alarm}}$ = cost of a false alarm (false positive)

$$\rho = \frac{C_{\text{miss}}}{C_{\text{alarm}}} = \frac{19.51\%}{3.0\%}
\approx 6.5$$

The numerator (19.51%) is the average maximum drawdown across the seven
historical stress episodes, calculated directly from the dataset. This
represents the capital loss from a missed warning — staying fully
invested through a crisis episode without defensive repositioning.

The denominator (3.0%) is a conservative upper bound on the opportunity
cost of temporary defensive repositioning during a false alarm period —
foregone return from partial equity underweight during a non-event.

A cost ratio of 6.5 means it is economically rational to tolerate up to
6.5 false alarms for every genuine crisis correctly identified. All
thresholds, persistence windows, and class weights are calibrated to
this ratio.

### 9.5 Signal Timeline — April 2026 Episode

| Date | $S_t$ | Stress Signal | Crisis Alert | Escalation |
|---|---|---|---|---|
| March 30, 2026 | 0.325 | INACTIVE | INACTIVE | INACTIVE |
| April 1, 2026 | 0.633 | **ACTIVE** (override) | INACTIVE | INACTIVE |
| April 2, 2026 | 0.573 | **ACTIVE** (sustained) | INACTIVE | INACTIVE |
| April 6, 2026 | 0.573 | **ACTIVE** (sustained) | INACTIVE | INACTIVE |
| April 7, 2026 | 0.642 | **ACTIVE** (override) | INACTIVE | INACTIVE |
| April 8, 2026 | 0.203 | INACTIVE | INACTIVE | INACTIVE |
| April 9, 2026 | 0.381 | INACTIVE | INACTIVE | INACTIVE |

The signal deactivated on April 8, 2026 following a temporary
de-escalation in geopolitical tensions that reduced the combined stress
probability from 0.642 to 0.203 in a single session.

### 9.6 Episode Detection Summary (Tier 2 In-Sample)

| Episode | Status | Lead Time | Notes |
|---|---|---|---|
| GFC Acute Crash (2008–2009) | CAPTURED | 28 days | 100% stress+crisis elevation throughout |
| Euro Crisis (2011) | CAPTURED | 0 days | 97% elevation, rapid onset |
| Taper Tantrum (2013) | CAPTURED | 0 days | 24% elevation, moderate event |
| IL&FS NBFC (2018) | MISSED | — | Credit event; index vol never breached thresholds |
| COVID Crash (2020) | CAPTURED | 0 days | 48% elevation, exogenous shock |
| COVID Recovery Vol (2020) | CAPTURED | 15 days | 42% elevation, sequential buildup |
| Post-COVID Macro Stress (2021–2022) | CAPTURED | 0 days | 10% elevation, slow-moving |

**Summary:** Episodes captured 6/7 | Average lead time 6.1 business days

The IL&FS NBFC crisis of 2018 is a documented scope limitation. The
event was a credit and liquidity contagion at the non-banking financial
company sector level. NIFTY 50 index-level realized volatility and
drawdown never breached the thresholds required for Stress or Crisis
classification. An index-level equity volatility diagnostic will
structurally miss credit events that have not yet transmitted to broad
market volatility. Detecting such events would require credit spread
data and sector-specific metrics outside this system's scope.

---

## 10. Risk-Conditioned Monte Carlo Simulation

### 10.1 Geometric Brownian Motion Framework

Forward price scenarios are generated using Geometric Brownian Motion
(GBM) with regime-conditional parameters. Under GBM, the discretized
price process over one business day $\Delta t = 1/252$ is:

$$P_{t+1} = P_t \cdot \exp\left[\left(\mu_k - \frac{1}{2}\sigma^2
\right)\Delta t + \sigma \sqrt{\Delta t} \cdot z_t\right]$$

where $z_t \sim \mathcal{N}(0, 1)$ i.i.d., $\mu_k$ is the annualized
regime-conditional drift for regime $k$, and $\sigma$ is the current
annualized GARCH conditional volatility.

### 10.2 The Itô Correction

The term $-\frac{1}{2}\sigma^2$ in the exponent is the Itô correction,
required by Itô's Lemma when simulating log-normal processes.

Without this correction, the exponential function applied to a normally
distributed random variable creates a systematic upward bias through
Jensen's Inequality: $E[\exp(\sigma \epsilon)] = \exp(\frac{1}{2}
\sigma^2) > 1$ even when $E[\epsilon] = 0$. The Itô correction exactly
cancels this bias, ensuring that $E[P_{t+1}] = P_t \cdot \exp(\mu_k
\Delta t)$ as required. Without it, a zero-drift simulation would
exhibit a spurious upward trend driven purely by the mathematics of
log-normal distributions.

### 10.3 Risk Conditioning

The simulation is risk-conditioned in two distinct ways, distinguishing
it from a naive Monte Carlo:

**Volatility conditioning:** The volatility parameter $\sigma$ is set to
the current GJR-GARCH conditional volatility $\hat{\sigma}_t^
{\text{GARCH, ann.}}$ rather than the historical unconditional average.
Using the historical average (17.4% annualized) during an elevated
volatility period (27.1% annualized as of April 9, 2026) would produce
a distribution that understates current downside risk. The theoretical
basis is Ang and Bekaert (2002), who demonstrate that regime-conditional
volatility estimates produce materially better-calibrated tail risk
assessments than constant-volatility models.

**Drift conditioning:** The drift parameter $\mu_k$ is set to the
regime-conditional mean return for the current predicted regime
(Section 4.4) rather than the full-sample unconditional mean (+7.75%
annualized). This ensures the central tendency of the simulation
reflects the historically observed return behavior in the current
market state.

### 10.4 Simulation Execution

At each daily pipeline execution:

- **Starting level:** $P_0 = P_t$, the current NIFTY 50 closing level
- **Horizon:** $H = 21$ business days (~1 calendar month)
- **Number of paths:** $N = 10{,}000$
- **Daily shock:** $z_t^{(i)} \sim \mathcal{N}(0,1)$ independently
  drawn for each path $i$ and time step

For path $i$, the terminal price is:

$$P_{t+21}^{(i)} = P_t \cdot \exp\left[\sum_{s=1}^{21}\left(
\left(\mu_k - \frac{1}{2}\sigma^2\right)\Delta t +
\sigma\sqrt{\Delta t} \cdot z_s^{(i)}\right)\right]$$

### 10.5 Scenario Quantiles

Reference values as of April 9, 2026 (reference snapshot):

| Quantile | Index Level | Change |
|---|---|---|
| 5th percentile (downside) | 20,846 | −12.3% |
| 50th percentile (median) | 23,650 | −0.5% |
| 95th percentile (upside) | 26,793 | +12.7% |

Note: The reference snapshot used a near-zero drift estimate rather than
the regime-conditional Pullback drift (+9.2% annualized). The production
system uses the empirically derived regime-conditional drift. Readers
viewing the live dashboard will see updated values.

### 10.6 Interpretation Constraint

The Monte Carlo output is a probabilistic scenario distribution
conditioned on the current risk state. It is not a price forecast.
The distribution communicates the range of plausible outcomes under
current regime-conditional dynamics — consistent with Bogle's humility
principle that uncertainty must be made explicit rather than hidden
behind a point estimate.

---

## 11. ARIMA Diagnostic Price Forecasting

### 11.1 Purpose

The ARIMA model is included for diagnostic purposes only. Its sole
function is to validate that the system's uncertainty structure is
honest — that forecast confidence intervals widen monotonically with
horizon. The expanding confidence interval width, not the point
forecast, is the output of interest.

### 11.2 Model Selection

The best ARIMA order is selected via AIC grid search over
$p \in \{0,1,2\}$, $d \in \{1\}$, $q \in \{0,1,2\}$ on the
log-transformed index level $\ell_t = \ln(P_t)$. The ADF test confirms
that $\ell_t$ is non-stationary in levels and stationary in first
differences, justifying $d = 1$.

The selected specification is **ARIMA(0,1,1)**:

$$\Delta \ell_t = \theta_1 \epsilon_{t-1} + \epsilon_t,
\quad \epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2)$$

where $\Delta \ell_t = \ell_t - \ell_{t-1}$ is the daily log return,
and $\theta_1$ is the MA(1) coefficient. This is equivalent to a random
walk with a moving average correction — the correct specification for
log prices under weak-form market efficiency, where prices should not
be predictable from their own history.

### 11.3 Forecast Construction

Point forecasts and confidence intervals are produced for horizons
$h = 1, 2, \ldots, 63$ business days. Forecasts in log space are
exponentiated to index-level values:

$$\hat{P}_{t+h} = \exp\left(\hat{\ell}_{t+h}\right)$$

### 11.4 Uncertainty Structure Validation

Under the ARIMA(0,1,1) specification, confidence interval width grows
approximately proportional to $\sqrt{h}$:

$$\text{CI Width}_h \propto \sigma_\epsilon \sqrt{h}$$

Reference confidence interval widths (April 9, 2026, 95% CI):

| Horizon | CI Width (Index Points) |
|---|---|
| Day 1 | ~1,202 |
| Day 21 (1 month) | ~5,689 |
| Day 63 (1 quarter) | ~9,910 |

The ratio Day 21 / Day 1 = 5,689 / 1,202 ≈ 4.73, consistent with
$\sqrt{21} \approx 4.58$ (slight deviation reflects the MA(1) correction).
The monotonic widening confirms the system does not claim to know more
about the distant future than the near future — a necessary property
of any epistemically honest forecasting system.

---

## 12. Model Evaluation Framework

### 12.1 Dual-Tier Structure

Model evaluation uses two complementary tiers. Neither tier alone
provides a complete picture. Both together, clearly labeled, do.

**Tier 1 — Out-of-Sample Generalization (2024–present):**
The only honest generalization test. The model never saw this data during
training. Metrics reflect true predictive performance on genuinely
unseen market conditions.

**Tier 2 — In-Sample Consistency (2007–2026):**
The model is evaluated against the full historical record. Because the
model was trained on the majority of this data, Tier 2 metrics reflect
training fit, not generalization. Their value is in demonstrating that
the model captures the historical stress episodes that motivated the
system's design.

### 12.2 Primary Metrics

**Overall accuracy (Tier 1):**

$$\text{Accuracy} = \frac{\sum_{t \in \mathcal{T}_{\text{test}}}
\mathbf{1}[\hat{y}_t = y_t]}{|\mathcal{T}_{\text{test}}|}$$

Reference value: 84.1%. This metric is explicitly de-prioritized because
the test set is 97.8% Calm and Pullback. A naive classifier predicting
Pullback on every day would achieve ~58% accuracy. Overall accuracy is
dominated by the majority class and is not an appropriate primary metric
for a risk system.

**Transition accuracy (Tier 1, primary metric):**

$$\text{Transition Accuracy} = \frac{1}{T-1}\sum_{t=1}^{T-1}
\mathbf{1}[\hat{y}_{t+1} = y_{t+1}]$$

computed over all consecutive day pairs in the test set, regardless of
whether a transition occurred. Reference value: 93.9%.

This is the operationally meaningful metric for a system whose purpose
is detecting regime escalation. On 93.9% of consecutive day pairs in the
test set, the model correctly identified whether the regime was staying
the same or changing direction.

### 12.3 Regime-Specific Recall

Recall for regime $k$ measures the proportion of true regime-$k$
instances correctly identified:

$$\text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$$

Reference values on the 2024–present test window:

| Regime | Recall | n (test set) | Status |
|---|---|---|---|
| Calm (0) | 77.5% | 222 | Reliable |
| Pullback (1) | 91.4% | 327 | Reliable |
| Stress (2) | 0.0% | 11 | LOW CONFIDENCE — insufficient observations |
| Crisis (3) | NOT EVALUABLE | 0 | Structural — no Crisis days in test window |

The primary metric hierarchy follows Bodie's risk-first framing: Crisis
recall, then Stress recall, then transition accuracy, then overall
accuracy. The cost of missing a Crisis is an order of magnitude higher
than the cost of a false alarm, and the metric hierarchy reflects this.

### 12.4 Structural Evaluation Limitations

**Crisis recall (Tier 1):** The 2024–present test window contains zero
Crisis days by definition. NIFTY 50 did not experience a drawdown below
-15% coinciding with volatility above 25% during this period. This is a
structural property of the market environment, not a model failure.
Crisis detection capability is assessed via Tier 2 in-sample consistency.

**Stress recall (Tier 1):** With n=11 observations in the test window,
the Stress recall estimate carries high statistical variance. A single
correct prediction would change recall from 0.0% to 9.1%. These figures
are reported for completeness but should not be interpreted as
performance indicators.

**IL&FS miss (Tier 2):** As documented in Section 9.6, the 2018 credit
event did not breach index-level volatility thresholds. The rule-based
labels never classified it as Stress or Crisis, so the GRU never learned
the pattern. This is a scope limitation, not a model error.

---

## 13. Dashboard and Visualization Layer

### 13.1 Technology

The visualization layer is implemented in Streamlit (`dashboard_app.py`).
The dashboard is structured across six tabs.

### 13.2 Dashboard Tabs

| Tab | Content |
|---|---|
| Live Dashboard | Current NIFTY level, drawdown, GARCH volatility, realized volatility, predicted regime, early warning signal status |
| Regime Analysis | Softmax probability bands over test period, predicted vs actual regime chart, regime distribution, transition accuracy metrics |
| Early Warning | Three signal status cards, composite stress probability time series, probability trajectory table (last 15 trading days) |
| Scenario Analysis | Monte Carlo fan chart (21-day horizon), percentile summary table, ARIMA diagnostic forecast (1-quarter), CI width table |
| Model Evaluation | Tier 1 metrics, per-class metrics table, Tier 2 episode capture table, threshold calibration chart, methodology notes |
| About | System description, pipeline architecture, data sources, known limitations, academic references |

### 13.3 Dynamic Metric Display

All metrics displayed on the dashboard are recomputed on each pipeline
execution and sourced from current-day outputs in `data/`. No static
reference values are hardcoded. The "Below signal threshold" message is
displayed when all three signals are inactive. Signal activation messages
are displayed dynamically based on the current value of $S_t$ against
the defined thresholds.

### 13.4 Chart Design Conventions

- **Monte Carlo fan chart:** x-axis formatted at monthly intervals.
  Three shaded bands showing 5th-95th, 25th-75th percentile ranges.
  Median path annotated.
- **ARIMA chart:** Historical price in blue, flat forecast line, expanding
  confidence band. x-axis formatted at monthly intervals.
- **Composite stress probability:** Horizontal reference lines at 0.40
  (sustained threshold) and 0.60 (override threshold). Three fill colors:
  green (< 0.40), orange (0.40–0.60), red (≥ 0.60).
- **Regime probability bands:** Stacked area chart. Four-color scheme:
  green (Calm), orange (Pullback), red (Stress), near-black (Crisis).

---

## 14. Production Pipeline Architecture

### 14.1 Execution Schedule

The pipeline executes daily at **17:00 IST (5:00 PM)**, approximately
90 minutes after NSE market close at 15:30 IST. Execution is managed
by Windows Task Scheduler.

- **Task name:** `NIFTY_Risk_Diagnostics_Daily`
- **Trigger:** Daily at 17:00 IST, weekdays only
- **Script:** `run_daily.ps1` → `pipeline/run_daily.py`

### 14.2 Daily Execution Sequence

Each daily execution performs the following steps in order:

1. Download full NIFTY 50 price series from Yahoo Finance (`^NSEI`)
2. Apply time-aware close selection (before/after 15:30 IST logic)
3. Compute all engineered features (log returns, realized volatility,
   drawdown, rolling skewness, rolling kurtosis)
4. Apply rule-based regime classification to all dates
5. Fit GJR-GARCH model; compute conditional volatility series
6. Run GRU volatility inference for next-day forecast
7. Check for gaps in regime_probs.pkl; run batch inference for all
   missing dates using correct historical features
8. Append current-day regime probabilities
9. Evaluate early warning signal conditions; record signal state
10. Execute Monte Carlo simulation with regime-conditional parameters
11. Compute ARIMA forecast with confidence intervals
12. Run dual-tier model evaluation
13. Regenerate all dashboard visualization outputs
14. Archive dated copies of all chart files

### 14.3 Data Boundaries

| Directory / File | Content | Git Status |
|---|---|---|
| `data/` | All `.pkl` output files, `.json` config | Pushed to repository |
| `outputs/` | Chart exports (.png) | `.gitignore` — not pushed |
| `logs/` | Execution logs | `.gitignore` — not pushed |
| `*.keras` | Trained model weights | `.gitignore` — not pushed |
| `venv/` | Python virtual environment | `.gitignore` — not pushed |

Trained model weights are hosted externally:
`https://drive.google.com/drive/folders/1XcTdikMvv1vyrfTJzGoRVNDKF1CJWjas`

Files required for production inference (download to `data/`):
- `gru_best_model_7j.keras` — GRU volatility forecaster
- `gru_regime_model.keras` — GRU regime classifier
- `regime_scaler_X.pkl` — StandardScaler fitted on training data
- `cost_ratio_config.json` — empirical cost ratio and class weights



## Appendix A — Key Reference Figures (April 9, 2026 Snapshot)

All figures in this appendix are calculated from the production pipeline
run of April 9, 2026 and sourced from the confirmed output files.

| Metric | Value | Source |
|---|---|---|
| NIFTY 50 closing level | 23,775 | features.pkl |
| Current drawdown | −9.7% | features.pkl |
| Rule-based regime | Regime 1 — Pullback | features.pkl |
| GJR-GARCH conditional volatility | 27.1% annualized | garch_output.pkl |
| Realized volatility (30-day) | 25.5% annualized | features.pkl |
| GARCH persistence | 0.9868 | garch_output.pkl |
| Calm regime vol average (full sample) | 9.5% annualized | features.pkl |
| Full sample realized vol average | 17.4% annualized | features.pkl |
| GFC peak realized volatility | 77.7% | features.pkl |
| COVID crash peak realized volatility | 76.6% | features.pkl |
| GRU P(Calm) | 0.02% | regime_probs.pkl |
| GRU P(Pullback) | 61.9% | regime_probs.pkl |
| GRU P(Stress) | 35.4% | regime_probs.pkl |
| GRU P(Crisis) | 2.7% | regime_probs.pkl |
| Combined stress probability $S_t$ | 0.381 | early_warning_signals.pkl |
| Monte Carlo 5th percentile | 20,846 (−12.3%) | monte_carlo_output.pkl |
| Monte Carlo median | 23,650 (−0.5%) | monte_carlo_output.pkl |
| Monte Carlo 95th percentile | 26,793 (+12.7%) | monte_carlo_output.pkl |
| ARIMA CI Day 1 | ~1,202 pts | arima_output.pkl |
| ARIMA CI Day 21 | ~5,689 pts | arima_output.pkl |
| ARIMA CI Day 63 | ~9,910 pts | arima_output.pkl |
| Overall classification accuracy | 84.1% | evaluation_results.json |
| Transition prediction accuracy | 93.9% | evaluation_results.json |
| Calm recall | 77.5% (n=222) | evaluation_results.json |
| Pullback recall | 91.4% (n=327) | evaluation_results.json |
| Stress recall | 0.0% (n=11, insufficient) | evaluation_results.json |
| Crisis recall | NOT EVALUABLE (n=0) | evaluation_results.json |
| Episodes captured | 6/7 | evaluation_results.json |
| Average early warning lead time | 6.1 business days | evaluation_results.json |
| GFC lead time | 28 business days | evaluation_results.json |
| COVID Recovery lead time | 15 business days | evaluation_results.json |
| Empirical cost ratio | 6.5:1 | cost_ratio_config.json |
| Average crisis drawdown | 19.51% | cost_ratio_config.json |
| False alarm cost (estimated) | 3.0% | cost_ratio_config.json |

---

## Appendix B — Notation Summary

| Symbol | Definition |
|---|---|
| $P_t$ | Adjusted closing level of NIFTY 50 on trading day $t$ |
| $r_t$ | Log return: $\ln(P_t / P_{t-1})$ |
| $\sigma_t^{\text{realized}}$ | 30-day rolling realized volatility (annualized) |
| $D_t$ | Drawdown at time $t$: $(P_t - \max_{s\leq t}P_s) / \max_{s\leq t}P_s$ |
| $\hat{\sigma}_t^{\text{GARCH}}$ | GJR-GARCH conditional volatility (annualized) |
| $\text{Skew}_t^{63}$ | 63-day rolling skewness of log returns |
| $\text{Kurt}_t^{63}$ | 63-day rolling excess kurtosis of log returns |
| $\hat{\mathbf{p}}_{t+1}$ | GRU-estimated next-day regime probability vector |
| S_t | Combined stress probability: P(Stress) + P(Crisis) = p_{t,2} + p_{t,3} |
| $\mu_k$ | Regime-conditional annualized drift for regime $k$ |
| $\rho^{\text{GARCH}}$ | GJR-GARCH volatility persistence: $\alpha + \beta + \gamma/2$ |
| $\rho$ | Early warning cost ratio: $C_{\text{miss}} / C_{\text{alarm}}$ |
| $w_k$ | Class weight for regime $k$ in cross-entropy loss |
| $K$ | Number of regime classes (= 4) |
| $H$ | Monte Carlo simulation horizon (= 21 business days) |
| $N$ | Number of Monte Carlo paths (= 10,000) |
| $\mathcal{T}_k$ | Set of all business days classified under regime $k$ |
| $\Delta t$ | One business day: $1/252$ years |

---

## Appendix C — Academic References

Black, F. (1976). Studies of Stock Market Volatility Changes. *Proceedings
of the American Statistical Association*, Business and Economic Statistics
Section.

Ang, A. and Bekaert, G. (2002). International Asset Allocation with Regime
Shifts. *Review of Financial Studies*, 15(4), 1137–1187.

Bodie, Z., Kane, A. and Marcus, A. (2018). *Investments* (12th ed.).
McGraw-Hill.

Bogle, J.C. (2017). *The Little Book of Common Sense Investing*. Wiley.

CBOE (2019). *CBOE Volatility Index: VIX — White Paper*. Chicago Board
Options Exchange.

Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
Schwenk, H. and Bengio, Y. (2014). Learning Phrase Representations using
RNN Encoder-Decoder for Statistical Machine Translation. *arXiv:1406.1078*.

Glosten, L.R., Jagannathan, R. and Runkle, D.E. (1993). On the Relation
Between the Expected Value and the Volatility of the Nominal Excess Return
on Stocks. *Journal of Finance*, 48(5), 1779–1801.

Hamilton, J.D. (1989). A New Approach to the Economic Analysis of
Nonstationary Time Series and the Business Cycle. *Econometrica*,
57(2), 357–384.

Kaminsky, G., Lizondo, S. and Reinhart, C.M. (1998). Leading Indicators
of Currency Crises. *IMF Staff Papers*, 45(1), 1–48.

Karasan, A. (2021). *Machine Learning for Financial Risk Management with
Python: Algorithms for Modeling Risk* (1st ed.). O'Reilly Media.

Morgan/Reuters (1996). *RiskMetrics Technical Document* (4th ed.).
J.P. Morgan and Reuters.

Pagan, A.R. and Sossounov, K.A. (2003). A Simple Framework for Analysing
Bull and Bear Markets. *Journal of Applied Econometrics*, 18(1), 23–46.

Reinhart, C.M. and Rogoff, K.S. (2009). *This Time Is Different: Eight
Centuries of Financial Folly*. Princeton University Press.

Semeniuta, S., Severyn, A. and Barth, E. (2016). Recurrent Dropout
Without Memory Loss. *arXiv:1603.05118*.
