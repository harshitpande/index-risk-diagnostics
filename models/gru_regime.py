# models/gru_regime.py
# ══════════════════════════════════════════════════════════════
# Step 11: GRU Regime Transition Classifier
# Two-Stage Architecture: Vol Forecast → Regime Classification
#
# Theoretical grounding:
#   - Hamilton (1989): regime persistence assumption
#   - Karasan (2021): hybrid statistical + ML pipeline
#   - Ang & Bekaert (2002): regime transition dynamics
#
# Label smoothing basis:
#   - Pagan & Sossounov (2003): minimum phase length filter
#   - MSCI Barra Risk Model Handbook: regime flicker prevention
#   - Hamilton (1989): persistence-consistent label construction
# ══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────

LOOKBACK          = 60          # trading days of history per sequence
MIN_SPELL_DAYS    = 3           # empirically calibrated; methodology grounded in
                                # Hamilton (1989) persistence and Pagan & Sossounov (2003)
HORIZON           = 1           # 1-day ahead regime prediction
TRAIN_CUTOFF      = '2024-01-01'
N_CLASSES         = 4
RANDOM_SEED       = 42
GRU_UNITS         = 64
DENSE_UNITS       = 32
DROPOUT_RATE      = 0.2
EPOCHS            = 50
BATCH_SIZE        = 32
VALIDATION_SPLIT  = 0.2

REGIME_LABELS = {
    0: 'Calm',
    1: 'Pullback',
    2: 'Stress',
    3: 'Crisis'
}

REGIME_COLORS = {
    0: '#2ecc71',   # green
    1: '#f39c12',   # orange
    2: '#e74c3c',   # red
    3: '#1a1a2e'    # near-black
}

# ──────────────────────────────────────────────────────────────
# SECTION 1: LABEL SMOOTHING
# Minimum 3-day spell filter (Hamilton 1989; Pagan & Sossounov 2003)
# ──────────────────────────────────────────────────────────────

def smooth_regime_labels(regime_series: pd.Series,
                         min_spell: int = MIN_SPELL_DAYS) -> pd.Series:
    """
    Apply minimum-spell filter to regime labels.

    Methodology:
        Identifies regime spells shorter than min_spell trading days
        and replaces them with the surrounding regime. This eliminates
        'regime flicker' — single-day or two-day regime assignments
        that are statistically inconsistent with the persistence
        assumption of Hamilton (1989) Markov Switching Models.

    Reference:
        Pagan & Sossounov (2003), J. Applied Econometrics 18(1):
        'A minimum turning point filter is applied to eliminate
         spurious phase reversals.'
        MSCI Barra Risk Model Handbook: minimum 3-day persistence
        before reclassification.

    Args:
        regime_series : pd.Series of integer regime labels {0,1,2,3}
        min_spell     : minimum number of days for a valid regime spell

    Returns:
        pd.Series of smoothed regime labels
    """
    smoothed = regime_series.copy()
    n = len(smoothed)
    values = smoothed.values.copy()

    i = 0
    while i < n:
        current = values[i]
        # Find end of current spell
        j = i
        while j < n and values[j] == current:
            j += 1
        spell_len = j - i

        # If spell is shorter than minimum, replace with preceding regime
        if spell_len < min_spell and i > 0:
            replacement = values[i - 1]
            values[i:j] = replacement

        i = j

    smoothed[:] = values

    # Report filtering impact
    changed = (smoothed != regime_series).sum()
    pct_changed = changed / len(regime_series) * 100
    print(f"  Label smoothing: {changed} labels reassigned "
          f"({pct_changed:.2f}% of dataset)")
    print(f"  Min spell filter: {min_spell} trading days "
          f"(Pagan & Sossounov 2003)")

    return smoothed


# ──────────────────────────────────────────────────────────────
# SECTION 2: FEATURE CONSTRUCTION (TWO-STAGE)
# Stage 1 output (GRU vol forecast) fed as input to Stage 2
# ──────────────────────────────────────────────────────────────

def build_feature_matrix(features_df: pd.DataFrame,
                         vol_forecast_col: str = 'gru_vol_forecast'
                         ) -> pd.DataFrame:
    """
    Construct the feature matrix for the regime classifier.

    Two-Stage Architecture:
        Stage 1: GRU volatility model (gru_best_model_7j.keras)
                 Output: gru_vol_forecast — the forward volatility estimate
        Stage 2: GRU regime classifier (this module)
                 Inputs include Stage 1 output as a feature

    Feature rationale:
        log_return      : contemporaneous return signal
        realized_vol    : backward-looking realized volatility (slow)
        GARCH_Vol       : conditional volatility expectation (statistical)
        gru_vol_forecast: forward volatility estimate (ML, Stage 1 output)
                          Adds information about where vol is GOING,
                          not just where it IS — the key signal for
                          regime escalation detection
        drawdown        : current drawdown from peak (severity signal)
        rolling_skew_63 : 63-day return skewness (tail asymmetry signal)
        rolling_kurt_63 : 63-day return kurtosis (fat tail signal)
                          Higher moments are early-warning features:
                          kurtosis rises before crisis, not during it

    Returns:
        DataFrame with feature columns, NaN rows dropped
    """
    required_cols = [
        'log_return', 'realized_vol', 'GARCH_Vol',
        'drawdown', 'rolling_skew_63', 'rolling_kurt_63',
        'regime'
    ]

    # Check for Stage 1 output
    if vol_forecast_col in features_df.columns:
        feature_cols = [
            'log_return', 'realized_vol', 'GARCH_Vol',
            vol_forecast_col,
            'drawdown', 'rolling_skew_63', 'rolling_kurt_63'
        ]
        print(f"  Two-stage architecture: '{vol_forecast_col}' found ✓")
    else:
        # Fallback: use GARCH_Vol as proxy until Stage 1 output is available
        feature_cols = [
            'log_return', 'realized_vol', 'GARCH_Vol',
            'drawdown', 'rolling_skew_63', 'rolling_kurt_63'
        ]
        print(f"  WARNING: '{vol_forecast_col}' not found.")
        print(f"  Falling back to 6-feature set (GARCH_Vol as vol proxy).")
        print(f"  To use full two-stage architecture, run gru_volatility.py")
        print(f"  first and add gru_vol_forecast column to features.pkl")

    df = features_df[feature_cols + ['regime']].copy()
    df = df.dropna()

    print(f"  Features used       : {feature_cols}")
    print(f"  Rows after dropna   : {len(df)}")

    return df, feature_cols


# ──────────────────────────────────────────────────────────────
# SECTION 3: SEQUENCE CONSTRUCTION
# ──────────────────────────────────────────────────────────────

def create_sequences(X: np.ndarray, y: np.ndarray,
                     lookback: int = LOOKBACK,
                     horizon: int = HORIZON):
    """
    Build rolling sequences for GRU input.

    For each timestep t, the input is X[t-lookback:t] and the
    target is y[t+horizon-1] — the regime one step ahead of
    the last observation in the window.

    This is consistent with the label definition:
        target = regime.shift(-1)
    i.e., "given the last 60 days, predict tomorrow's regime"
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(X) - horizon + 1):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i + horizon - 1])
    return np.array(X_seq), np.array(y_seq)


# ──────────────────────────────────────────────────────────────
# SECTION 4: CLASS WEIGHT LOADING
# ──────────────────────────────────────────────────────────────

def load_class_weights(config_path: str = 'data/cost_ratio_config.json',
                       fallback_crisis_weight: float = 10.0) -> dict:
    """
    Load empirically derived class weights from cost ratio analysis.
    Falls back to conservative defaults if config not found.
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        weights = {int(k): v for k, v in config['class_weights'].items()}
        print(f"  Class weights loaded from {config_path}")
        print(f"  Empirical cost ratio: {config['empirical_cost_ratio']}:1")
    else:
        weights = {0: 1.0, 1: 3.0, 2: 5.0, 3: fallback_crisis_weight}
        print(f"  WARNING: Config not found. Using fallback weights.")
        print(f"  Run cost_ratio_analysis.py first for empirical weights.")

    for k, v in weights.items():
        print(f"  Regime {k} ({REGIME_LABELS[k]:8s}) weight: {v:.1f}")

    return weights


# ──────────────────────────────────────────────────────────────
# SECTION 5: MODEL ARCHITECTURE
# ──────────────────────────────────────────────────────────────

def build_regime_classifier(n_features: int,
                             lookback: int = LOOKBACK,
                             n_classes: int = N_CLASSES,
                             gru_units: int = GRU_UNITS,
                             dense_units: int = DENSE_UNITS,
                             dropout_rate: float = DROPOUT_RATE) -> tf.keras.Model:
    """
    GRU regime transition classifier.

    Architecture rationale:
        Single GRU layer (64 units): captures temporal dependencies
        in regime features. Deeper stacks overfit on the relatively
        small financial time series (Karasan 2021 recommends parsimonious
        architectures for financial regime classification).

        Dropout (0.2): regularises without destroying temporal structure.
        Note: recurrent dropout is intentionally avoided — it degrades
        GRU gradient flow on short sequences (Semeniuta et al. 2016).

        Dense(32, relu): non-linear combination of GRU hidden state.

        Dense(4, softmax): outputs P(Regime=k) for k in {0,1,2,3}.
        Softmax ensures probabilities sum to 1, enabling direct
        interpretation as regime transition probabilities.

    Loss function:
        categorical_crossentropy with class weights passed at fit()
        time. This is equivalent to a cost-sensitive loss where the
        gradient contribution of Crisis samples is scaled by the
        empirical cost ratio (derived in cost_ratio_analysis.py).

    Reference:
        Karasan (2021): hybrid GARCH + ML for financial regime detection
        Cho et al. (2014): GRU architecture
        Semeniuta et al. (2016): recurrent dropout considerations
    """
    model = Sequential([
        GRU(gru_units,
            input_shape=(lookback, n_features),
            return_sequences=False,
            name='gru_temporal'),
        Dropout(dropout_rate, name='dropout_regularisation'),
        Dense(dense_units, activation='relu', name='dense_hidden'),
        Dense(n_classes, activation='softmax', name='regime_probabilities')
    ], name='GRU_Regime_Classifier')

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ──────────────────────────────────────────────────────────────
# SECTION 6: TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────

def train_regime_classifier(features_df: pd.DataFrame,
                             save_path: str = 'data/gru_regime_model.keras',
                             train_cutoff: str = TRAIN_CUTOFF):
    """
    Full training pipeline for regime transition classifier.

    Train/test split:
        Fixed date cutoff (2024-01-01) — identical to GRU vol model
        for consistency. Test set grows daily.

    Returns:
        model, history, X_test, y_test_cat, y_test_raw, test_dates, scaler_X
    """
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n" + "=" * 60)
    print("STEP 11: GRU REGIME TRANSITION CLASSIFIER — TRAINING")
    print("=" * 60)

    # ── 1. Build feature matrix
    print("\n[1] Feature construction")
    df, feature_cols = build_feature_matrix(features_df)

    # ── 2. Apply label smoothing
    print("\n[2] Label smoothing")
    df['regime_smooth'] = smooth_regime_labels(df['regime'])

    # Distribution check post-smoothing
    print("\n  Regime distribution after smoothing:")
    vc = df['regime_smooth'].value_counts().sort_index()
    total = len(df)
    for r, count in vc.items():
        print(f"    Regime {r} ({REGIME_LABELS[r]:8s}): "
              f"{count:5d} days ({count/total*100:.1f}%)")

    # ── 3. Define target: next-period regime
    df['target'] = df['regime_smooth'].shift(-HORIZON)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    # ── 4. Train/test split
    print(f"\n[3] Train/test split at {train_cutoff}")
    X_all = df[feature_cols].values
    y_all = df['target'].values
    dates_all = df.index

    train_mask = df.index < train_cutoff
    test_mask  = df.index >= train_cutoff

    # ── 5. Scale features (fit on train only)
    print("\n[4] Feature scaling (StandardScaler, fit on train)")
    scaler_X = StandardScaler()
    X_all_scaled = X_all.copy().astype(float)
    X_all_scaled[train_mask] = scaler_X.fit_transform(X_all[train_mask])
    X_all_scaled[test_mask]  = scaler_X.transform(X_all[test_mask])

    # ── 6. Sequence construction
    print(f"\n[5] Sequence construction (lookback={LOOKBACK}, horizon={HORIZON})")
    X_seq, y_seq = create_sequences(X_all_scaled, y_all, LOOKBACK, HORIZON)

    # Align dates with sequences
    seq_dates = dates_all[LOOKBACK:][:len(y_seq)]

    # Split on cutoff
    cutoff_idx = (seq_dates < train_cutoff).sum()
    X_train = X_seq[:cutoff_idx]
    X_test  = X_seq[cutoff_idx:]
    y_train_raw = y_seq[:cutoff_idx]
    y_test_raw  = y_seq[cutoff_idx:]
    test_dates  = seq_dates[cutoff_idx:]

    y_train_cat = to_categorical(y_train_raw, num_classes=N_CLASSES)
    y_test_cat  = to_categorical(y_test_raw,  num_classes=N_CLASSES)

    print(f"  Train sequences: {len(X_train)} (up to {train_cutoff})")
    print(f"  Test sequences : {len(X_test)}  (from {train_cutoff} to present)")
    print(f"  Input shape    : {X_train.shape}")

    # Train distribution
    print(f"\n  Train class distribution:")
    for r in range(N_CLASSES):
        n = (y_train_raw == r).sum()
        print(f"    Regime {r} ({REGIME_LABELS[r]:8s}): {n:5d} ({n/len(y_train_raw)*100:.1f}%)")

    # ── 7. Load class weights
    print("\n[6] Class weights (empirical cost ratio)")
    class_weights = load_class_weights()

    # ── 8. Build and train model
    print("\n[7] Model architecture")
    n_features = X_train.shape[2]
    model = build_regime_classifier(n_features)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(save_path, monitor='val_loss',
                        save_best_only=True, verbose=0)
    ]

    print(f"\n[8] Training (max {EPOCHS} epochs, early stopping patience=10)")
    history = model.fit(
        X_train, y_train_cat,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n  Model saved → {save_path}")

    return model, history, X_test, y_test_cat, y_test_raw, test_dates, scaler_X


# ──────────────────────────────────────────────────────────────
# SECTION 7: EVALUATION
# ──────────────────────────────────────────────────────────────

def evaluate_regime_classifier(model, X_test, y_test_cat,
                                y_test_raw, test_dates,
                                save_prefix: str = 'outputs/regime_classifier'):
    """
    Evaluation with institutional-grade metrics.

    Primary metric: Crisis recall (sensitivity to regime=3)
    The system must not miss Crisis transitions — a missed Crisis
    signal is an operational failure, not a modelling imperfection.

    Secondary metrics:
        - Stress recall (regime=2): pre-crisis detection
        - Transition matrix accuracy: correct escalation/de-escalation
        - Confusion matrix: full error decomposition
    """
    print("\n" + "=" * 60)
    print("STEP 11: EVALUATION")
    print(f"Test set: {test_dates.min().date()} → {test_dates.max().date()}")
    print(f"n = {len(y_test_raw)}")
    print("=" * 60)

    # ── Predictions
    probs = model.predict(X_test, verbose=0)  # shape: (n, 4)
    y_pred = probs.argmax(axis=1)

    # ── Classification report
    # labels= ensures all 4 classes are reported even if absent from test set.
    # This is intentional: Crisis absence from 2024-2026 test window is a
    # meaningful finding (no Crisis regime in the test period), not a bug.
    # zero_division=0 suppresses warnings for classes with no true positives.
    print("\n[1] Classification Report")
    present_classes = sorted(np.unique(np.concatenate([y_test_raw, y_pred])))
    print(classification_report(
        y_test_raw, y_pred,
        labels=list(range(N_CLASSES)),
        target_names=[REGIME_LABELS[i] for i in range(N_CLASSES)],
        zero_division=0,
        digits=3
    ))
    absent = [REGIME_LABELS[i] for i in range(N_CLASSES)
              if i not in np.unique(y_test_raw)]
    if absent:
        print(f"  NOTE: {absent} absent from test set actuals "
              f"(test period: 2024-01-01 → present)")
        print(f"  This is a valid finding — no {absent} episode "
              f"in the test window, not a data error.")

    # ── Regime-specific recall (primary institutional metric)
    from sklearn.metrics import recall_score, precision_score
    print("[2] Regime-Specific Recall (primary metric)")
    for regime_id in range(N_CLASSES):
        binary_true = (y_test_raw == regime_id).astype(int)
        binary_pred = (y_pred == regime_id).astype(int)
        recall = recall_score(binary_true, binary_pred, zero_division=0)
        precision = precision_score(binary_true, binary_pred, zero_division=0)
        n_true = binary_true.sum()
        print(f"  Regime {regime_id} ({REGIME_LABELS[regime_id]:8s}): "
              f"Recall={recall:.3f}  Precision={precision:.3f}  "
              f"N={n_true}")

    # ── Confusion matrix
    print("\n[3] Confusion Matrix")
    cm = confusion_matrix(y_test_raw, y_pred)
    print("  Rows=Actual, Cols=Predicted")
    print(f"  {'':12s}", end='')
    for r in range(N_CLASSES):
        print(f"  Pred_{REGIME_LABELS[r][:4]}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"  Act_{REGIME_LABELS[i][:8]}", end='')
        for val in row:
            print(f"  {val:8d}", end='')
        print()

    # ── Transition accuracy
    print("\n[4] Regime Transition Accuracy")
    actual_transitions   = (y_test_raw[1:] != y_test_raw[:-1]).sum()
    predicted_at_actual  = y_pred[:-1][y_test_raw[1:] != y_test_raw[:-1]]
    actual_at_transition = y_test_raw[1:][y_test_raw[1:] != y_test_raw[:-1]]
    transition_acc = (predicted_at_actual == actual_at_transition).mean()
    print(f"  Total actual transitions : {actual_transitions}")
    print(f"  Correctly predicted      : {(predicted_at_actual == actual_at_transition).sum()}")
    print(f"  Transition accuracy      : {transition_acc:.3f}")

    # ── Visualisations
    _plot_evaluation(y_test_raw, y_pred, probs, test_dates, cm, save_prefix)

    return probs, y_pred


def _plot_evaluation(y_true, y_pred, probs, test_dates, cm, save_prefix):
    """Generate evaluation charts."""

    fig, axes = plt.subplots(3, 1, figsize=(15, 14))
    fig.suptitle('GRU Regime Classifier — Step 11 Evaluation',
                 fontsize=14, fontweight='bold')

    # ── Plot 1: Actual vs Predicted Regimes
    ax1 = axes[0]
    ax1.plot(test_dates, y_true,  alpha=0.7, label='Actual Regime',    lw=1.5,
             color='#2c3e50')
    ax1.plot(test_dates, y_pred,  alpha=0.7, label='Predicted Regime', lw=1.0,
             color='#e74c3c', linestyle='--')
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels([REGIME_LABELS[i] for i in range(4)])
    ax1.set_title('Actual vs Predicted Regime (Test Set)', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── Plot 2: Softmax probability bands over time
    ax2 = axes[1]
    colors_list = [REGIME_COLORS[i] for i in range(4)]
    labels_list = [REGIME_LABELS[i] for i in range(4)]
    ax2.stackplot(test_dates, probs.T, labels=labels_list,
                  colors=colors_list, alpha=0.75)
    ax2.set_title('Regime Transition Probabilities — P(Regime|History)',
                  fontsize=11)
    ax2.set_ylabel('Probability')
    ax2.legend(loc='upper left', fontsize=8, ncol=4)
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)

    # ── Plot 3: Confusion matrix heatmap
    ax3 = axes[2]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm,
        annot=True, fmt='.2f', cmap='Blues',
        xticklabels=[REGIME_LABELS[i] for i in range(4)],
        yticklabels=[REGIME_LABELS[i] for i in range(4)],
        ax=ax3, linewidths=0.5
    )
    ax3.set_title('Normalised Confusion Matrix\n(Rows = Actual, Cols = Predicted)',
                  fontsize=11)
    ax3.set_xlabel('Predicted Regime')
    ax3.set_ylabel('Actual Regime')

    plt.tight_layout()
    out_path = f'{save_prefix}_evaluation.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Chart saved → {out_path}")


# ──────────────────────────────────────────────────────────────
# SECTION 8: INFERENCE (DAILY RUNNER)
# ──────────────────────────────────────────────────────────────

def predict_regime_probabilities(features_df: pd.DataFrame,
                                 model_path: str = 'data/gru_regime_model.keras',
                                 scaler_path: str = 'data/regime_scaler_X.pkl'
                                 ) -> dict:
    """
    Daily inference: load saved model, return P(Regime) vector.

    Called by pipeline/run_daily.py — never retrains.
    Returns softmax probability vector for the most recent observation.

    Returns:
        {
            'regime_probs': [P(Calm), P(Pullback), P(Stress), P(Crisis)],
            'predicted_regime': int,
            'predicted_label': str,
            'crisis_probability': float,
            'stress_or_crisis_probability': float
        }
    """
    import pickle

    model = load_model(model_path)

    with open(scaler_path, 'rb') as f:
        scaler_X = pickle.load(f)

    df, feature_cols = build_feature_matrix(features_df)
    X = df[feature_cols].values
    X_scaled = scaler_X.transform(X)

    # Use last LOOKBACK days
    if len(X_scaled) < LOOKBACK:
        raise ValueError(f"Insufficient data: need {LOOKBACK} rows, got {len(X_scaled)}")

    X_input = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, len(feature_cols))
    probs = model.predict(X_input, verbose=0)[0]

    predicted_regime = int(probs.argmax())

    return {
        'regime_probs':                   probs.tolist(),
        'predicted_regime':               predicted_regime,
        'predicted_label':                REGIME_LABELS[predicted_regime],
        'crisis_probability':             float(probs[3]),
        'stress_or_crisis_probability':   float(probs[2] + probs[3]),
    }

def predict_regime_probabilities_batch(features_df: pd.DataFrame,
                                       model_path: str,
                                       scaler_path: str,
                                       target_dates: list) -> pd.DataFrame:
    """
    Batch inference for a list of target dates.
    
    For each date in target_dates, constructs the 60-day lookback
    window ending on that date, runs inference, and returns a
    DataFrame of regime probabilities.
    
    Used by run_daily.py to recover missed dates automatically.
    Every date uses its correct historical features from features.pkl.
    No estimation or carry-forward is performed.
    
    Args:
        features_df  : full features DataFrame from features.pkl
        model_path   : path to saved .keras model
        scaler_path  : path to saved scaler .pkl
        target_dates : list of pd.Timestamp dates needing inference
        
    Returns:
        pd.DataFrame with columns:
            P_regime_0, P_regime_1, P_regime_2, P_regime_3,
            predicted_regime, actual_regime
        indexed by target_dates
    """
    import pickle

    model    = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler_X = pickle.load(f)

    # Build full scaled feature matrix once
    df, feature_cols = build_feature_matrix(features_df)
    X_all        = df[feature_cols].values.astype(float)
    X_all_scaled = scaler_X.transform(X_all)
    dates_index  = df.index

    rows = []
    skipped = []

    for target_date in target_dates:
        # Find position of target_date in the feature matrix
        if target_date not in dates_index:
            skipped.append(target_date)
            continue

        pos = dates_index.get_loc(target_date)

        # Need exactly LOOKBACK rows before this position
        if pos < LOOKBACK:
            skipped.append(target_date)
            continue

        # Construct lookback window ending at pos (exclusive of target)
        # Window: [pos-LOOKBACK : pos]
        # This is identical to the single-day inference logic
        X_input = X_all_scaled[pos - LOOKBACK : pos]
        X_input = X_input.reshape(1, LOOKBACK, len(feature_cols))

        probs = model.predict(X_input, verbose=0)[0]
        predicted_regime = int(probs.argmax())

        # Actual regime from features.pkl for this date
        actual_regime = int(features_df.loc[target_date, 'regime']) \
            if 'regime' in features_df.columns else -1

        rows.append({
            'date':            target_date,
            'P_regime_0':      float(probs[0]),
            'P_regime_1':      float(probs[1]),
            'P_regime_2':      float(probs[2]),
            'P_regime_3':      float(probs[3]),
            'predicted_regime': predicted_regime,
            'actual_regime':   actual_regime,
        })

    if skipped:
        print(f"  Batch inference: skipped {len(skipped)} date(s) "
              f"(insufficient lookback or not in feature matrix): "
              f"{[d.date() for d in skipped]}")

    if not rows:
        return pd.DataFrame(columns=[
            'P_regime_0','P_regime_1','P_regime_2','P_regime_3',
            'predicted_regime','actual_regime'
        ])

    result = pd.DataFrame(rows).set_index('date')
    result.index.name = 'Date'
    print(f"  Batch inference complete: {len(result)} date(s) computed.")
    return result

# ──────────────────────────────────────────────────────────────
# MAIN: TRAINING RUN
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pickle

    print("Loading features.pkl...")
    features = pd.read_pickle('data/features.pkl')
    features.index = pd.to_datetime(features.index)
    features = features.sort_index()

    print(f"Dataset: {features.index.min().date()} → {features.index.max().date()}")
    print(f"Rows: {len(features)}")
    print(f"Columns: {list(features.columns)}")

    # Train
    (model, history, X_test, y_test_cat,
     y_test_raw, test_dates, scaler_X) = train_regime_classifier(features)

    # Save scaler
    with open('data/regime_scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    print("Scaler saved → data/regime_scaler_X.pkl")

    # Evaluate
    probs, y_pred = evaluate_regime_classifier(
        model, X_test, y_test_cat, y_test_raw, test_dates
    )

    # Save probability output for Step 12
    prob_df = pd.DataFrame(
        probs,
        index=test_dates,
        columns=[f'P_regime_{i}' for i in range(N_CLASSES)]
    )
    prob_df['predicted_regime'] = y_pred
    prob_df['actual_regime']    = y_test_raw
    prob_df.to_pickle('data/regime_probs.pkl')
    print("Regime probabilities saved → data/regime_probs.pkl")

    # Training curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'],     label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['accuracy'],     label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle('GRU Regime Classifier — Training History', fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/regime_classifier_training.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Training curve saved → outputs/regime_classifier_training.png")