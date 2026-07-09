"""I split the feature table into training, calibration, and validation sets BY TIME.

For time series I must never shuffle rows. If I trained on random rows I would
let the model peek at the future to predict the past, which inflates accuracy
and collapses in production. So I always order by time and slice forward.

I now produce THREE slices, not two, because I am adding conformal calibration:

  - train  (oldest 60%) : the LightGBM quantile models learn ONLY from this.
  - calib  (middle 20%) : the models never train on this. I use it purely to
                          MEASURE how wrong the models' uncertainty bands are
                          on data they have never seen. This is the "measuring
                          tape" that conformal prediction needs.
  - valid  (newest 20%) : the final, clean judge. NOTHING fits or calibrates on
                          it. I only ever read it once, at the very end, to
                          report honest before/after coverage.

The golden rule: each later slice is strictly NEWER in time than the one before
it. That mirrors real life — I train on the past, calibrate on a more recent
past, and am judged on the most recent data, exactly as a deployed model would
face genuinely future hours.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils_io import configure_logging, load_parquet, save_parquet

LOGGER = logging.getLogger(__name__)

# === CONFIGURATION ===

PROCESSED_DIR = Path("data/processed")

# I keep the three fractions here as named constants so the split is obvious at
# a glance and trivial to change. They must sum to 1.0. I deliberately keep the
# validation fraction identical (0.20) to my old two-way split so my final
# "after" coverage is measured on a validation set the same SIZE as before —
# this keeps the before/after comparison fair.
TRAIN_FRACTION = 0.60   # oldest 60% — the models learn here
CALIB_FRACTION = 0.20   # middle 20% — I measure miscalibration here
VALID_FRACTION = 0.20   # newest 20% — clean final judge, touched once


# === MAIN ===

def split_main() -> None:
    """I split model_dataset.parquet into train/calib/valid parquet files by time."""
    configure_logging()

    # I assert the fractions are sane up front. A silent off-by-a-bit here would
    # quietly poison every downstream metric, so I fail loudly instead.
    total = TRAIN_FRACTION + CALIB_FRACTION + VALID_FRACTION
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split fractions must sum to 1.0 but sum to {total:.4f}. "
            "Fix TRAIN_FRACTION / CALIB_FRACTION / VALID_FRACTION."
        )

    df = load_parquet(PROCESSED_DIR / "model_dataset.parquet").sort_values("datetime")
    n = len(df)

    # I convert fractions into row cut-points. Because the frame is sorted by
    # time, slicing by position [:cut1], [cut1:cut2], [cut2:] gives me three
    # contiguous, strictly time-ordered blocks with no overlap and no shuffle.
    cut_train = int(n * TRAIN_FRACTION)
    cut_calib = int(n * (TRAIN_FRACTION + CALIB_FRACTION))

    train = df.iloc[:cut_train].reset_index(drop=True)
    calib = df.iloc[cut_train:cut_calib].reset_index(drop=True)
    valid = df.iloc[cut_calib:].reset_index(drop=True)

    save_parquet(train, PROCESSED_DIR / "train.parquet")
    save_parquet(calib, PROCESSED_DIR / "calib.parquet")
    save_parquet(valid, PROCESSED_DIR / "valid.parquet")

    # I log the exact date boundaries of all three slices. Seeing the dates is
    # the fastest way to confirm by eye that the split is purely chronological:
    # train ends before calib starts, calib ends before valid starts.
    LOGGER.info(
        "Train: %d rows (%s to %s)",
        len(train), train.datetime.min(), train.datetime.max(),
    )
    LOGGER.info(
        "Calib: %d rows (%s to %s)",
        len(calib), calib.datetime.min(), calib.datetime.max(),
    )
    LOGGER.info(
        "Valid: %d rows (%s to %s)",
        len(valid), valid.datetime.min(), valid.datetime.max(),
    )


# === ENTRY POINT ===

if __name__ == "__main__":
    split_main()
