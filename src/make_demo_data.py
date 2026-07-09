"""I generate realistic synthetic German market data so the app works instantly.

I exist so the dashboard can be demonstrated BEFORE the real SMARD download
finishes (which takes several minutes). I copy the statistical shape of real
German prices: daily seasonality, weekly patterns, solar mid-day dips,
occasional spikes, and even negative price hours.

Once real data is fetched, the app automatically prefers it over my output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils_io import configure_logging, save_parquet

LOGGER = logging.getLogger(__name__)

# === CONFIGURATION ===

PROCESSED_DIR = Path("data/processed")
DEMO_DAYS = 90
RNG = np.random.default_rng(42)  # I fix the seed so the demo is reproducible


# === GENERATION ===

def _make_demo_frame() -> pd.DataFrame:
    """I build 90 days of hourly synthetic prices and generation data.

    I model price as: base level + daily shape (morning/evening peaks)
    + weekend discount + wind-driven noise + rare spikes. This mirrors the
    real drivers so the dashboard charts look like genuine market data.
    """
    end = pd.Timestamp.now(tz="Europe/Berlin").floor("h")
    idx = pd.date_range(end=end, periods=DEMO_DAYS * 24, freq="h")
    n = len(idx)
    hour = idx.hour.to_numpy()
    weekday = idx.weekday.to_numpy()

    # Daily demand shape: low at night, peaks ~8h and ~19h.
    daily_shape = 18 * np.sin((hour - 3) * np.pi / 12) + 8 * np.sin((hour - 16) * np.pi / 6)
    weekend = np.where(weekday >= 5, -12.0, 0.0)

    # Wind: slow-moving random walk (weather systems last days, not hours).
    wind = np.clip(np.cumsum(RNG.normal(0, 350, n)) + 15000, 1000, 45000)
    # Solar: zero at night, bell curve at midday, stronger randomness (clouds).
    solar_shape = np.clip(np.sin((hour - 6) * np.pi / 12), 0, None)
    solar = solar_shape * RNG.uniform(8000, 30000, n)

    load = 55000 + 8000 * np.sin((hour - 7) * np.pi / 12) + weekend * 400 + RNG.normal(0, 1500, n)
    residual = load - wind - solar

    # Price follows residual load (the merit order in one line) plus spikes.
    price = 35 + residual / 900 + daily_shape + weekend + RNG.normal(0, 6, n)
    spikes = RNG.random(n) < 0.01
    price = np.where(spikes, price + RNG.uniform(40, 120, n), price)

    # Negative price events: on sunny+windy midday hours renewables can
    # exceed demand, and inflexible plants pay to keep producing. Real German
    # data had 400+ negative hours in 2024, so the demo must show some too.
    surplus = (residual < np.quantile(residual, 0.03)) & (solar > np.quantile(solar, 0.7))
    price = np.where(surplus, RNG.uniform(-60, -5, n), price)

    da_price = price + RNG.normal(0, 9, n)  # day-ahead misses intraday news

    return pd.DataFrame({
        "datetime": idx,
        "intraday_price_eur_mwh": np.round(price, 2),
        "da_price_eur_mwh": np.round(da_price, 2),
        "wind_onshore_mw": np.round(wind * 0.82, 0),
        "wind_offshore_mw": np.round(wind * 0.18, 0),
        "solar_mw": np.round(solar, 0),
        "total_load_mw": np.round(load, 0),
        "residual_load_mw": np.round(residual, 0),
    })


def make_demo_main() -> None:
    """I write the demo files in the same paths/format the real fetch uses."""
    configure_logging()
    df = _make_demo_frame()
    save_parquet(
        df[["datetime", "intraday_price_eur_mwh"]],
        PROCESSED_DIR / "demo_intraday_hourly.parquet",
    )
    save_parquet(
        df[["datetime", "da_price_eur_mwh"]],
        PROCESSED_DIR / "demo_day_ahead_hourly.parquet",
    )
    save_parquet(
        df[["datetime", "wind_onshore_mw", "wind_offshore_mw", "solar_mw",
            "total_load_mw", "residual_load_mw"]],
        PROCESSED_DIR / "demo_features_hourly.parquet",
    )
    LOGGER.info("Demo data ready (%d hours). Run: streamlit run app/Home.py", len(df))


# === ENTRY POINT ===

if __name__ == "__main__":
    make_demo_main()
