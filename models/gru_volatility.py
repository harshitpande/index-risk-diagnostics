# models/gru_volatility.py
# ─────────────────────────────────────────────────────────────
# Step 7: GRU volatility forecasting.
# Loads saved model for inference — never retrains.
# Two models: 7a (realized_vol target), 7j (GARCH_Vol target).
# Production model: 7j (GARCH_Vol target).
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
os_environ_set = False

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    GRU_FEATURES, GRU_TARGET, GRU_LOOKBACK,
    GRU_HORIZON, SPLIT_DATE,
    FEATURES_PKL, GRU_MODEL_7J, GRU_7J_OUTPUT_PKL
)


def create_sequences(X: np.ndarray,
                     y: np.ndarray,
                     lookback: int) -> tuple:
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def prepare_data(df: pd.DataFrame) -> tuple:
    gru_df = df[GRU_FEATURES].copy()
    gru_df["target"] = gru_df[GRU_TARGET].shift(-GRU_HORIZON)
    gru_df = gru_df.dropna()

    train = gru_df[gru_df.index <  SPLIT_DATE]
    test  = gru_df[gru_df.index >= SPLIT_DATE]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(train[GRU_FEATURES])
    y_train = scaler_y.fit_transform(train[["target"]])
    X_test  = scaler_X.transform(test[GRU_FEATURES])
    y_test  = scaler_y.transform(test[["target"]])

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, GRU_LOOKBACK)
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  GRU_LOOKBACK)

    return (X_train_seq, y_train_seq,
            X_test_seq,  y_test_seq,
            scaler_X, scaler_y, test)


def run_gru_inference() -> dict:
    df = pd.read_pickle(FEATURES_PKL)

    print("[gru] Preparing data...")
    (X_train_seq, y_train_seq,
     X_test_seq, y_test_seq,
     scaler_X, scaler_y, test) = prepare_data(df)

    print(f"[gru] Loading model from {GRU_MODEL_7J}...")
    gru_model = load_model(GRU_MODEL_7J)

    print("[gru] Running inference...")
    y_pred_scaled = gru_model.predict(X_test_seq, verbose=0)
    y_pred        = scaler_y.inverse_transform(y_pred_scaled)
    y_actual      = scaler_y.inverse_transform(y_test_seq)

    # Baselines
    naive_baseline = test[GRU_TARGET].iloc[GRU_LOOKBACK:-1].values.reshape(-1, 1)
    garch_baseline = test[GRU_TARGET].iloc[GRU_LOOKBACK:].values.reshape(-1, 1)

    mae_gru   = mean_absolute_error(y_actual, y_pred)
    mae_naive = mean_absolute_error(y_actual[1:], naive_baseline)
    mae_garch = mean_absolute_error(y_actual, garch_baseline)

    print(f"[gru] MAE GRU         : {mae_gru:.4f}")
    print(f"[gru] MAE Naive lag   : {mae_naive:.4f}")
    print(f"[gru] MAE GARCH       : {mae_garch:.4f}")
    print(f"[gru] GRU vs Naive    : "
          f"{((mae_naive - mae_gru)/mae_naive*100):.1f}% improvement")
    print(f"[gru] GRU vs GARCH    : "
          f"{((mae_garch - mae_gru)/mae_garch*100):.1f}% improvement")

    # Current volatility forecast
    latest_seq    = X_test_seq[-1:]
    latest_pred_s = gru_model.predict(latest_seq, verbose=0)
    latest_pred   = scaler_y.inverse_transform(latest_pred_s)[0][0]
    print(f"[gru] Next-day GARCH_Vol forecast : {latest_pred:.4f} "
          f"({latest_pred*100:.1f}% annualized)")

    output = {
        "y_actual"      : y_actual,
        "y_pred"        : y_pred,
        "mae_gru"       : mae_gru,
        "mae_naive"     : mae_naive,
        "mae_garch"     : mae_garch,
        "latest_pred"   : latest_pred,
        "target"        : GRU_TARGET,
        "split_date"    : SPLIT_DATE,
        "test_size"     : len(y_actual),
    }

    with open(GRU_7J_OUTPUT_PKL, "wb") as f:
        pickle.dump(output, f)

    print(f"[gru] Saved to {GRU_7J_OUTPUT_PKL}")
    return output


if __name__ == "__main__":
    run_gru_inference()