"""I fetch electricity market data from SMARD.de (Bundesnetzagentur).

I use SMARD because it provides official German market data including the
intraday continuous trading price, which is my forecasting target. SMARD is
a fully public API: no API key, no registration, no rate-limit headaches.

SMARD organises every dataset in weekly chunks. I first ask for the list of
available week-start timestamps (the "index"), then download each weekly
chunk one by one and stitch them together.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.utils_io import configure_logging, ensure_dir, save_parquet

LOGGER = logging.getLogger(__name__)

# === CONFIGURATION ===

SMARD_BASE = "https://www.smard.de/app/chart_data"
REGION = "DE-LU"            # Germany-Luxembourg bidding zone
RESOLUTION = "hour"
TARGET_TZ = "Europe/Berlin"  # German electricity contracts use local delivery time
SLEEP_BETWEEN_CALLS = 0.3    # I am polite to the server
MAX_RETRIES = 3              # I retry on temporary server errors
DATA_YEARS = 3               # 3 years is plenty for a first model and fetches faster

FILTERS = {
    "intraday_price": 4170,   # Intraday price EUR/MWh — MY TARGET VARIABLE
    "day_ahead_price": 4169,  # Day-ahead price EUR/MWh — strongest feature
    "wind_onshore": 4067,     # Wind onshore generation MW
    "wind_offshore": 1225,    # Wind offshore generation MW
    "solar": 4068,            # Solar PV generation MW
    "total_load": 410,        # Total electricity consumption MW
    "residual_load": 4359,    # Demand minus renewables MW — key price driver
}

COLUMN_NAMES = {
    "intraday_price": "intraday_price_eur_mwh",
    "day_ahead_price": "da_price_eur_mwh",
    "wind_onshore": "wind_onshore_mw",
    "wind_offshore": "wind_offshore_mw",
    "solar": "solar_mw",
    "total_load": "total_load_mw",
    "residual_load": "residual_load_mw",
}

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# === HELPER FUNCTIONS ===

def _get_with_retry(url: str) -> dict | None:
    """I fetch a URL with up to MAX_RETRIES attempts and exponential backoff.

    SMARD sometimes returns 503 when its servers are busy. I wait 1s, 2s,
    then 4s between attempts so I never hammer an overloaded server.
    I return None instead of raising so the caller can skip a bad chunk
    without killing a 30-minute download run.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            LOGGER.warning("HTTP %d for %s (attempt %d)", resp.status_code, url, attempt + 1)
        except requests.RequestException as exc:
            LOGGER.warning("Request failed: %s (attempt %d)", exc, attempt + 1)
        time.sleep(2 ** attempt)
    return None


def _fetch_index(filter_id: int) -> list[int]:
    """I fetch the list of available weekly start timestamps for one filter.

    SMARD stores data in weekly files. I need this list first so I know
    which weekly files exist. Each entry is a Unix-millisecond timestamp
    marking the start of one week.
    """
    url = f"{SMARD_BASE}/{filter_id}/{REGION}/index_{RESOLUTION}.json"
    data = _get_with_retry(url)
    if data is None or "timestamps" not in data:
        raise RuntimeError(
            f"Could not fetch index for filter {filter_id}. "
            "Check your internet connection or whether smard.de is reachable."
        )
    return data["timestamps"]


def _fetch_chunk(filter_id: int, timestamp_ms: int, col_name: str) -> pd.DataFrame:
    """I fetch one week of data for one filter and return a tidy DataFrame.

    SMARD returns a list of [unix_ms, value] pairs where value can be null
    for hours not yet reported. I drop nulls here, convert UTC milliseconds
    to Europe/Berlin local time, and return an empty DataFrame for empty
    chunks so the caller can skip them cleanly.
    """
    url = f"{SMARD_BASE}/{filter_id}/{REGION}/{filter_id}_{REGION}_{RESOLUTION}_{timestamp_ms}.json"
    data = _get_with_retry(url)
    if data is None or not data.get("series"):
        return pd.DataFrame(columns=["datetime", col_name])

    rows = [(ts, val) for ts, val in data["series"] if val is not None]
    if not rows:
        return pd.DataFrame(columns=["datetime", col_name])

    df = pd.DataFrame(rows, columns=["ts_ms", col_name])
    # I convert Unix milliseconds (UTC) to Europe/Berlin local time because
    # German electricity contracts are defined in local delivery time.
    df["datetime"] = (
        pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(TARGET_TZ)
    )
    return df[["datetime", col_name]]


def _fetch_full_series(filter_id: int, col_name: str, years: int) -> pd.DataFrame:
    """I download all weekly chunks for one filter within the last N years.

    I log progress every 25 chunks because a full fetch takes a few minutes
    (one polite request per week of data). I deduplicate on datetime at the
    end because adjacent weekly files overlap by one boundary row.
    """
    timestamps = _fetch_index(filter_id)

    # I keep only weeks newer than my cutoff so a 3-year fetch does not
    # download 7 years of files I will never use.
    cutoff_ms = int((pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=years)).timestamp() * 1000)
    timestamps = [ts for ts in timestamps if ts >= cutoff_ms]
    LOGGER.info("Filter %d (%s): %d weekly chunks to fetch", filter_id, col_name, len(timestamps))

    chunks: list[pd.DataFrame] = []
    for i, ts in enumerate(timestamps):
        chunk = _fetch_chunk(filter_id, ts, col_name)
        if not chunk.empty:
            chunks.append(chunk)
        if (i + 1) % 25 == 0:
            LOGGER.info("  ...%d/%d chunks done", i + 1, len(timestamps))
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not chunks:
        raise RuntimeError(f"No data received for filter {filter_id} ({col_name}).")

    df = pd.concat(chunks, ignore_index=True)
    # I keep the last value when a timestamp appears twice — weekly files
    # overlap at their boundary by design, so this duplicate is expected.
    df = (
        df.drop_duplicates(subset="datetime", keep="last")
          .sort_values("datetime")
          .reset_index(drop=True)
    )
    LOGGER.info(
        "Filter %d done: %d rows from %s to %s",
        filter_id, len(df), df.datetime.min(), df.datetime.max(),
    )
    return df


# === MAIN FETCH FUNCTIONS ===

def fetch_intraday_prices(years: int = DATA_YEARS) -> pd.DataFrame:
    """I fetch SMARD intraday prices (filter 4170) — my forecasting target.

    The intraday price is the volume-weighted average of continuous trades
    in the hours before physical delivery. It reflects the real-time
    supply/demand balance, including every wind forecast revision.
    I save both a raw copy (for audit) and a cleaned copy (for modelling).
    """
    df = _fetch_full_series(FILTERS["intraday_price"], COLUMN_NAMES["intraday_price"], years)
    save_parquet(df, RAW_DIR / "smard_intraday_raw.parquet")
    save_parquet(df, PROCESSED_DIR / "smard_intraday_hourly.parquet")
    return df


def fetch_day_ahead_prices(years: int = DATA_YEARS) -> pd.DataFrame:
    """I fetch SMARD day-ahead prices (filter 4169) — my strongest feature.

    The day-ahead price for hour t is published at ~12:42 the day before,
    so it is always known before intraday trading and is safe to use as a
    feature with no lag.
    """
    df = _fetch_full_series(FILTERS["day_ahead_price"], COLUMN_NAMES["day_ahead_price"], years)
    save_parquet(df, RAW_DIR / "smard_day_ahead_raw.parquet")
    save_parquet(df, PROCESSED_DIR / "smard_day_ahead_hourly.parquet")
    return df


def fetch_generation_and_load(years: int = DATA_YEARS) -> pd.DataFrame:
    """I fetch wind, solar, load and residual load, merged on datetime.

    These are the most important non-price features for intraday forecasting.
    Residual load (demand minus renewables) is especially powerful: it tells
    me directly how much expensive dispatchable generation is needed, which
    is the position on the merit order curve.
    """
    keys = ["wind_onshore", "wind_offshore", "solar", "total_load", "residual_load"]
    merged: pd.DataFrame | None = None
    for key in keys:
        df = _fetch_full_series(FILTERS[key], COLUMN_NAMES[key], years)
        # I use an outer join so I can see where any single source has gaps.
        merged = df if merged is None else merged.merge(df, on="datetime", how="outer")

    merged = merged.sort_values("datetime").reset_index(drop=True)
    save_parquet(merged, RAW_DIR / "smard_generation_raw.parquet")
    save_parquet(merged, PROCESSED_DIR / "smard_features_hourly.parquet")
    return merged


def fetch_smard_main() -> None:
    """I run the full SMARD fetch: intraday + day-ahead prices + generation."""
    configure_logging()
    LOGGER.info("=== SMARD fetch starting (last %d years, region %s) ===", DATA_YEARS, REGION)
    fetch_intraday_prices()
    fetch_day_ahead_prices()
    fetch_generation_and_load()
    LOGGER.info("=== SMARD fetch complete. Now run: streamlit run app/Home.py ===")


# === ENTRY POINT ===

if __name__ == "__main__":
    fetch_smard_main()
