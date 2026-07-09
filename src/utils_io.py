"""I provide small file helpers so every other module saves and loads data the same way.

I centralise Parquet reading/writing here because if I ever change the storage
format (for example to DuckDB), I only have to change this one file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


# === HELPER FUNCTIONS ===

def ensure_dir(path: str | Path) -> Path:
    """I create the directory if it does not exist yet and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """I save a DataFrame to a Parquet file, creating parent directories as needed.

    I use Parquet instead of CSV because it stores datatypes (including
    timezone-aware datetimes) exactly, loads ~10x faster, and takes ~5x
    less disk space for numerical time series.
    """
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False)
    LOGGER.info("Saved %d rows to %s", len(df), p)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """I load a Parquet file and return a DataFrame with a clear error if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run the fetch or demo step first (see README)."
        )
    return pd.read_parquet(p)


def configure_logging() -> None:
    """I configure the root logger with timestamp + level + module name.

    Every entry-point function calls me first so all pipeline output looks
    identical regardless of which file is being run.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
