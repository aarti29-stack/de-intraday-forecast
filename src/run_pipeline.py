"""I run the whole Phase-1 pipeline end to end with one command.

Order matters: each stage reads what the previous stage wrote. I keep every
stage call here so I can comment one out during development without touching
the individual files.
"""

from __future__ import annotations

import logging

from src.utils_io import configure_logging
from src.build_features import build_main
from src.split_data import split_main
from src.train_lightgbm import train_lgbm_main

LOGGER = logging.getLogger(__name__)


def run_all_main() -> None:
    """I run feature building, splitting, and model training in order."""
    configure_logging()
    LOGGER.info("=== STAGE 1/3: Build features ===")
    build_main()
    LOGGER.info("=== STAGE 2/3: Split train/validation ===")
    split_main()
    LOGGER.info("=== STAGE 3/3: Train LightGBM quantile model ===")
    train_lgbm_main()
    LOGGER.info("Pipeline finished. Launch the app: streamlit run app/Home.py")


if __name__ == "__main__":
    run_all_main()
