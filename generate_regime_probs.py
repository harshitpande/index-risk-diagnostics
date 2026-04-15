# generate_regime_probs.py
# Regenerates regime_probs.pkl from the already-trained model.
# Run this instead of retraining gru_regime.py from scratch.

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# ── Constants (must match gru_regime.py exactly)
LOOKBACK       = 60
HORIZON        = 1
TRAIN_CUTOFF   = '2024-01-01'
N_CLASSES      = 4
MIN_SPELL_DAYS = 3
REGIME_LABELS  = {0: 'Calm', 1: 'Pullback', 2: 'Stress', 3: 'Crisis'}

FEATURE_COLS = [
    'log_return', 'realized_vol', 'GARCH_Vol',
    'drawdown', 'rolling_skew_63', 'rolling_kurt_63'
]

# ── Load
print("Loading features.pkl...")
features = pd.read_pickle('data/features.pkl')
features.index = pd.to_datetime(features.index)
features = features.sort_index()
print(f"  {features.index.min().date()} → {features.index.max().date()}  n={len(features)}")

print("Loading saved model and scaler...")
model    = load_model('data/gru_regime_model.keras')
with open('data/regime_scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)

# ── Label smoothing (identical to training)
def smooth_regime_labels(regime_series, min_spell=MIN_SPELL_DAYS):
    smoothed = regime_series.copy()
    values   = smoothed.values.copy()
    n        = len(values)
    i = 0
    while i < n:
        current = values[i]
        j = i
        while j < n and values[j] == current:
            j += 1
        if (j - i) < min_spell and i > 0:
            values[i:j] = values[i - 1]
        i = j
    smoothed[:] = values
    return smoothed

# ── Prepare features
df = features[FEATURE_COLS + ['regime']].dropna().copy()
df['regime_smooth'] = smooth_regime_labels(df['regime'])
df['target']        = df['regime_smooth'].shift(-HORIZON)
df = df.dropna(subset=['target'])
df['target'] = df['target'].astype(int)

# ── Scale (fit on train, transform all)
X_all   = df[FEATURE_COLS].values
y_all   = df['target'].values
dates   = df.index

train_mask = df.index < TRAIN_CUTOFF
X_scaled   = X_all.copy().astype(float)
X_scaled[train_mask]  = scaler_X.fit_transform(X_all[train_mask])
X_scaled[~train_mask] = scaler_X.transform(X_all[~train_mask])

# ── Sequences
X_seq, y_seq, seq_dates = [], [], []
for i in range(LOOKBACK, len(X_scaled) - HORIZON + 1):
    X_seq.append(X_scaled[i - LOOKBACK:i])
    y_seq.append(y_all[i + HORIZON - 1])
    seq_dates.append(dates[i + HORIZON - 1])

X_seq      = np.array(X_seq)
y_seq      = np.array(y_seq)
seq_dates  = pd.DatetimeIndex(seq_dates)

# ── Test split
cutoff_idx  = (seq_dates < TRAIN_CUTOFF).sum()
X_test      = X_seq[cutoff_idx:]
y_test_raw  = y_seq[cutoff_idx:]
test_dates  = seq_dates[cutoff_idx:]

print(f"\nTest sequences : {len(X_test)}")
print(f"Test period    : {test_dates.min().date()} → {test_dates.max().date()}")

# ── Predict
print("Running inference...")
probs   = model.predict(X_test, verbose=0)
y_pred  = probs.argmax(axis=1)

# ── Build and save DataFrame
prob_df = pd.DataFrame(
    probs,
    index   = test_dates,
    columns = [f'P_regime_{i}' for i in range(N_CLASSES)]
)
prob_df['predicted_regime'] = y_pred
prob_df['actual_regime']    = y_test_raw

prob_df.to_pickle('data/regime_probs.pkl')
print(f"\nSaved → data/regime_probs.pkl")
print(f"Columns: {list(prob_df.columns)}")
print(f"\nSample (last 5 rows):")
print(prob_df.tail().to_string())

# ── Quick sanity check
print(f"\nPredicted regime distribution (test set):")
vc = pd.Series(y_pred).value_counts().sort_index()
for r, n in vc.items():
    print(f"  Regime {r} ({REGIME_LABELS[r]:8s}): {n:4d} ({n/len(y_pred)*100:.1f}%)")

print(f"\nCurrent P(Regime) vector (latest observation):")
latest = prob_df.iloc[-1]
for i in range(N_CLASSES):
    print(f"  P(Regime {i} — {REGIME_LABELS[i]:8s}) = {latest[f'P_regime_{i}']:.4f}")
print(f"  Predicted: Regime {int(latest['predicted_regime'])} "
      f"({REGIME_LABELS[int(latest['predicted_regime'])]})")
print(f"  Actual   : Regime {int(latest['actual_regime'])} "
      f"({REGIME_LABELS[int(latest['actual_regime'])]})")