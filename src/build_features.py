"""I turn raw market data into a machine-learning feature table.

This is the heart of the project. A model cannot learn from raw prices alone.
I build features that capture the real drivers of intraday prices:
- lagged prices (yesterday's same hour is the strongest single predictor)
- rolling statistics (recent average and volatility)
- the day-ahead price and the intraday-vs-day-ahead spread
- generation, load, residual load and renewable share (the merit order)
- calendar features with cyclical encoding and German public holidays

The golden rule of this file is NO LEAKAGE: when predicting hour t, I only
use information that was actually available before hour t. Every price-based
feature is shifted by at least one hour.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils_io import configure_logging, load_parquet, save_parquet

LOGGER = logging.getLogger(__name__)

# === CONFIGURATION ===

PROCESSED_DIR = Path("data/processed")
TARGET_COL = "target_intraday_eur_mwh"

# I prefer real SMARD files but fall back to demo files so this runs day one.
SOURCES = {
    "intraday": ("smard_intraday_hourly.parquet", "demo_intraday_hourly.parquet"),
    "day_ahead": ("smard_day_ahead_hourly.parquet", "demo_day_ahead_hourly.parquet"),
    "features": ("smard_features_hourly.parquet", "demo_features_hourly.parquet"),
}


# === HELPER FUNCTIONS ===

def _pick(real_name: str, demo_name: str) -> Path:
    """I return the real file if it exists, otherwise the demo file."""
    real = PROCESSED_DIR / real_name
    return real if real.exists() else PROCESSED_DIR / demo_name


def _load_all_sources() -> pd.DataFrame:
    """I load intraday price, day-ahead price and generation, merged on datetime.

    I use the intraday price as the base table because it defines exactly
    which hours I want to forecast. I left-join everything else onto it.
    """
    idf = load_parquet(_pick(*SOURCES["intraday"]))
    daf = load_parquet(_pick(*SOURCES["day_ahead"]))
    gen = load_parquet(_pick(*SOURCES["features"]))

    df = (
        idf.merge(daf, on="datetime", how="left")
           .merge(gen, on="datetime", how="left")
           .sort_values("datetime")
           .reset_index(drop=True)
    )
    LOGGER.info("Merged sources: %d rows, %d columns", len(df), df.shape[1])
    return df


def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """I add lagged prices, rolling stats, momentum, and the day-ahead spread.

    Lags are the #1 predictor in electricity price forecasting. lag_24 is the
    same hour yesterday (daily seasonality); lag_168 is the same hour last
    week (weekly seasonality). Every lag uses past values only, so no leakage.
    """
    p = df["intraday_price_eur_mwh"]

    for k in (1, 2, 3, 6, 12, 24, 48, 168):
        df[f"lag_{k}"] = p.shift(k)

    # Rolling mean/std measure the recent price level and volatility.
    # I shift by 1 first so the window ends at t-1, never including hour t.
    for w in (6, 24, 168):
        shifted = p.shift(1)
        df[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        df[f"roll_std_{w}"] = shifted.rolling(w).std()

    # Momentum: how fast the price moved over the last 1 and 3 hours.
    df["change_1h"] = df["lag_1"] - df["lag_2"]
    df["change_3h"] = df["lag_1"] - df["lag_3"]

    # The day-ahead price for hour t is published the day before, so it is
    # safe to use with no lag. The spread uses lagged intraday to stay clean.
    df["da_price_eur_mwh"] = df["da_price_eur_mwh"]
    df["id_da_spread_lag1"] = df["lag_1"] - df["da_price_eur_mwh"].shift(1)
    return df


def _add_generation_features(df: pd.DataFrame) -> pd.DataFrame:
    """I add wind/solar totals, renewable share, and residual load.

    Residual load (demand minus renewables) is the most compact summary of
    where we sit on the merit order curve: low residual load => cheap plants
    set the price => low prices, and vice versa.
    """
    if "wind_onshore_mw" in df and "wind_offshore_mw" in df:
        df["wind_total_mw"] = df["wind_onshore_mw"].fillna(0) + df["wind_offshore_mw"].fillna(0)
    if "wind_total_mw" in df and "solar_mw" in df:
        df["renewable_total_mw"] = df["wind_total_mw"] + df["solar_mw"].fillna(0)
    if "renewable_total_mw" in df and "total_load_mw" in df:
        # +1 in the denominator avoids division by zero on rare bad rows.
        df["renewable_share"] = df["renewable_total_mw"] / (df["total_load_mw"] + 1)
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """I add hour, weekday, month, weekend/holiday flags and cyclical encoding.

    Cyclical (sine/cosine) encoding tells the model that hour 23 and hour 0
    are neighbours. Plain integers would treat them as 23 units apart, which
    is wrong for a daily cycle.
    """
    dt = df["datetime"].dt
    df["hour"] = dt.hour
    df["weekday"] = dt.dayofweek
    df["month"] = dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    try:
        import holidays as hol
        de = hol.Germany()
        df["is_holiday"] = df["datetime"].dt.date.map(lambda d: int(d in de))
    except Exception:
        # If the holidays library is missing, I degrade gracefully to 0.
        df["is_holiday"] = 0

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


# === MAIN BUILD FUNCTION ===

def build_main() -> None:
    """I build the full feature matrix and save it to model_dataset.parquet."""
    configure_logging()
    df = _load_all_sources()

    # The target is simply the intraday price at hour t. The model predicts
    # this using only features that end at t-1 (plus the known day-ahead price).
    df[TARGET_COL] = df["intraday_price_eur_mwh"]

    df = _add_price_features(df)
    df = _add_generation_features(df)
    df = _add_calendar_features(df)

    # I drop the early rows where the longest lag (168h) is still NaN, and any
    # row missing the target. I forward-fill short gaps in slow-moving
    # generation features first so I do not throw away good price rows.
    gen_cols = [c for c in df.columns if c.endswith("_mw") or c == "renewable_share"]
    df[gen_cols] = df[gen_cols].ffill(limit=3)
    before = len(df)
    df = df.dropna(subset=[TARGET_COL, "lag_168", "roll_std_168"]).reset_index(drop=True)
    LOGGER.info("Dropped %d warm-up/NaN rows, %d rows remain", before - len(df), len(df))

    save_parquet(df, PROCESSED_DIR / "model_dataset.parquet")
    feature_cols = [c for c in df.columns
                    if c not in ("datetime", TARGET_COL, "intraday_price_eur_mwh")]
    LOGGER.info("Built %d features. Target: %s", len(feature_cols), TARGET_COL)


# === ENTRY POINT ===

if __name__ == "__main__":
    build_main()
