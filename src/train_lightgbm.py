"""I train LightGBM quantile regression to produce probabilistic price forecasts.

This is my main model. Instead of predicting a single price, I train one model
per quantile [0.1, 0.5, 0.9] so I get a forecast DISTRIBUTION. The P10-P90 band
is an 80% prediction interval: for a well-calibrated model, 80% of real prices
should land inside it. That uncertainty is exactly what a trading or battery
team needs for risk-aware decisions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.utils_io import configure_logging, ensure_dir, load_parquet, save_parquet

LOGGER = logging.getLogger(__name__)

# === CONFIGURATION ===

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
TARGET_COL = "target_intraday_eur_mwh"
QUANTILES = [0.1, 0.5, 0.9]

BASE_PARAMS = {
    "objective": "quantile",   # I minimise pinball loss, not MSE
    "n_estimators": 600,
    "learning_rate": 0.05,     # slow learning + early stopping generalises better
    "num_leaves": 63,
    "min_child_samples": 20,   # guards against overfitting rare price spikes
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
}


# === HELPER FUNCTIONS ===

def _feature_cols(df: pd.DataFrame) -> list[str]:
    """I return all model inputs, excluding datetime, target, and raw target."""
    drop = {"datetime", TARGET_COL, "intraday_price_eur_mwh"}
    return [c for c in df.columns if c not in drop]


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """I compute pinball loss, the proper scoring rule for a quantile forecast."""
    err = y_true - y_pred
    return float(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))


def _coverage(y_true, lo, hi) -> float:
    """I compute the fraction of actual prices that fall inside [lo, hi]."""
    return float(((y_true >= lo) & (y_true <= hi)).mean())


# === MAIN TRAINING FUNCTION ===

def train_lgbm_main() -> None:
    """I train one LightGBM model per quantile and save predictions + metrics."""
    configure_logging()
    train = load_parquet(PROCESSED_DIR / "train.parquet")
    valid = load_parquet(PROCESSED_DIR / "valid.parquet")
    feats = _feature_cols(train)

    X_train, y_train = train[feats], train[TARGET_COL]
    X_valid, y_valid = valid[feats], valid[TARGET_COL]
    LOGGER.info("Training on %d rows, %d features", len(X_train), len(feats))

    preds = pd.DataFrame({"datetime": valid["datetime"], "y_true": y_valid.values})
    for q in QUANTILES:
        model = lgb.LGBMRegressor(**BASE_PARAMS, alpha=q)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds[f"q_{q}"] = model.predict(X_valid)
        LOGGER.info("Trained Q%.2f (best iteration %s)", q, model.best_iteration_)

    # P50 is the central forecast; I score it like a normal point forecast.
    mae = mean_absolute_error(y_valid, preds["q_0.5"])
    coverage80 = _coverage(y_valid.values, preds["q_0.1"].values, preds["q_0.9"].values)
    metrics = {
        "mae_p50": round(mae, 3),
        "pinball_q10": round(_pinball_loss(y_valid.values, preds["q_0.1"].values, 0.1), 3),
        "pinball_q50": round(_pinball_loss(y_valid.values, preds["q_0.5"].values, 0.5), 3),
        "pinball_q90": round(_pinball_loss(y_valid.values, preds["q_0.9"].values, 0.9), 3),
        "coverage_80": round(coverage80, 3),
        "n_train": len(X_train),
        "n_valid": len(X_valid),
    }

    ensure_dir(REPORTS_DIR)
    save_parquet(preds, REPORTS_DIR / "lgbm_valid_predictions.parquet")
    with open(REPORTS_DIR / "lgbm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    LOGGER.info("=== Results ===")
    LOGGER.info("P50 MAE: %.2f EUR/MWh", metrics["mae_p50"])
    LOGGER.info("80%% interval coverage: %.1f%% (target 80%%)", coverage80 * 100)


# === ENTRY POINT ===

if __name__ == "__main__":
    train_lgbm_main()
