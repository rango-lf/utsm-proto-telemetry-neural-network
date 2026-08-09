"""
model.py
--------
Two things live here:

1. Data plumbing for `lap{N}_distgrid.csv` files: column configuration,
   file discovery, computing per-lap summary statistics, and resolving
   the regression target (the lap's actual total energy).

2. build_model(): returns a scikit-learn RandomForestRegressor.

------------------------------------------------------------------------
WHY AGGREGATE STATISTICS INSTEAD OF A FLAT VECTOR?
------------------------------------------------------------------------
The previous version flattened every row of the distgrid CSV into one
huge input vector and resampled all laps to the same length. That was
necessary for a neural network but has two problems:
  * It invented data points when a lap was shorter than 200 rows.
  * Feature importances were meaningless ("position 47 speed").

A Random Forest doesn't need a fixed-length sequence -- it just needs
a fixed set of numbers per lap. So instead we summarise each feature
column with four statistics (mean, max, min, std) over the lap. A lap
with 80 rows and one with 500 rows both produce the same 24-number
input vector, with no resampling needed. Feature importances then read
naturally: "mean speed matters more than max motor temperature."

------------------------------------------------------------------------
ASSUMPTIONS -- check these against real distgrid.csv output
------------------------------------------------------------------------
  * FEATURE_COLUMNS -- the per-track-position columns to summarise.
  * STAT_FUNCTIONS  -- which statistics to compute per column.
                       Default: mean, max, min, std (4 stats × 6 cols
                       = 24 input features total).
  * TARGET_MODE     -- how to get the lap's actual total energy:
       "cumulative_column" : last value of TARGET_COLUMN in the CSV.
       "sum_column"        : sum of TARGET_COLUMN over the lap.
       "labels_file"       : separate data/lap_labels.csv with columns
                             lap_id,total_energy_wh.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# --------------------------------------------------------------------------
# 1) CONFIG -- edit to match actual lap{N}_distgrid.csv columns
# --------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "speed_m_s",              # speed in m/s
    "accel_m_s2",             # acceleration in m/s^2
    "grade_pct",              # grade/slope in percent
    "imu_total_g",            # total acceleration in g
    "power_w",                # power in watts
    "cum_energy_wh",          # cumulative energy (for normalization/context)
]

# Statistics computed per column to form the fixed-length input vector.
# Adding or removing entries here changes the input dimension automatically.
STAT_FUNCTIONS = ["mean", "max", "min", "std"]

# How to obtain the regression target (actual total lap energy).
# One of: "cumulative_column", "sum_column", "labels_file"
TARGET_MODE = "cumulative_column"
TARGET_COLUMN = "cum_energy_wh"  # last value = total lap energy

# Filename pattern produced by the pre-processing script
FILENAME_PATTERN = re.compile(r"lap(\d+)_distgrid\.csv$", re.IGNORECASE)


# --------------------------------------------------------------------------
# 2) Data loading helpers
# --------------------------------------------------------------------------

def discover_lap_files(data_dir):
    """Find every lap{N}_distgrid.csv in data_dir.
    Returns [(lap_id, Path), ...] sorted by lap_id."""
    data_dir = Path(data_dir)
    laps = []
    for path in data_dir.glob("*distgrid.csv"):
        m = FILENAME_PATTERN.search(path.name)
        if m:
            laps.append((int(m.group(1)), path))
    laps.sort(key=lambda x: x[0])
    return laps


def feature_names():
    """Return the ordered list of feature names for the input vector.
    e.g. ["speed_m_s_mean", "speed_m_s_max", ..., "cum_energy_wh_std"]
    Used by evaluate.py to label the feature importance chart."""
    return [f"{col}_{stat}" for col in FEATURE_COLUMNS for stat in STAT_FUNCTIONS]


def load_lap_features(path, feature_columns=None):
    """Load one lap{N}_distgrid.csv -> fixed-length stats vector.

    For each column in FEATURE_COLUMNS, computes mean / max / min / std
    across all rows in the lap. The result is always the same length
    regardless of how many rows the lap CSV has.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    df = pd.read_csv(path)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: missing expected columns {missing}. "
            f"Update FEATURE_COLUMNS in model.py to match your distgrid output. "
            f"Columns found: {list(df.columns)}"
        )

    stats = []
    for col in feature_columns:
        series = df[col].to_numpy(dtype=np.float64)
        for stat in STAT_FUNCTIONS:
            if stat == "mean":
                stats.append(series.mean())
            elif stat == "max":
                stats.append(series.max())
            elif stat == "min":
                stats.append(series.min())
            elif stat == "std":
                stats.append(series.std())
            else:
                raise ValueError(f"Unknown stat function: {stat!r}")

    return np.array(stats, dtype=np.float32)


def load_lap_target(lap_id, path, data_dir):
    """Resolve the regression target (actual total lap energy) given TARGET_MODE."""
    if TARGET_MODE in ("cumulative_column", "sum_column"):
        df = pd.read_csv(path)
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"{path.name}: TARGET_COLUMN '{TARGET_COLUMN}' not found.")
        if TARGET_MODE == "cumulative_column":
            return float(df[TARGET_COLUMN].iloc[-1])
        return float(df[TARGET_COLUMN].sum())

    elif TARGET_MODE == "labels_file":
        labels_path = Path(data_dir) / LABELS_FILE
        if not labels_path.exists():
            raise FileNotFoundError(
                f"TARGET_MODE='labels_file' but {labels_path} does not exist. "
                f"Create it with columns: lap_id,total_energy_wh"
            )
        labels = pd.read_csv(labels_path).set_index("lap_id")
        if lap_id not in labels.index:
            raise KeyError(f"lap_id {lap_id} not found in {labels_path}")
        return float(labels.loc[lap_id, "total_energy_wh"])

    raise ValueError(f"Unknown TARGET_MODE: {TARGET_MODE}")


class LapDistGridDataset:
    """One sample = one lap. X = stats vector, y = actual total lap energy.

    Unlike the previous neural-network version, this class returns plain
    numpy arrays rather than PyTorch tensors, since scikit-learn works
    directly with numpy.
    """

    def __init__(self, data_dir, lap_ids=None):
        data_dir = Path(data_dir)
        all_laps = discover_lap_files(data_dir)

        if lap_ids is not None:
            wanted = set(lap_ids)
            all_laps = [(lid, p) for lid, p in all_laps if lid in wanted]
        if not all_laps:
            raise RuntimeError(f"No lap{{N}}_distgrid.csv files found in {data_dir}")

        self.lap_ids = [lid for lid, _ in all_laps]
        self.X = np.stack([load_lap_features(p) for _, p in all_laps])
        self.y = np.array(
            [load_lap_target(lid, p, data_dir) for lid, p in all_laps],
            dtype=np.float32,
        )
        self.feature_names = feature_names()

    def __len__(self):
        return len(self.lap_ids)


# --------------------------------------------------------------------------
# 3) Model
# --------------------------------------------------------------------------

def build_model(n_estimators=200, max_depth=None, min_samples_leaf=1, seed=42):
    """Return a configured RandomForestRegressor.

    Key hyperparameters (all can be overridden via train.py CLI flags):

      n_estimators      -- number of trees. More trees = more stable but
                           slower to train. 200 is a safe default.
      max_depth         -- maximum depth of each tree. None = grow until
                           leaves are pure (can overfit on small datasets;
                           try 5-10 if val score is much worse than train).
      min_samples_leaf  -- minimum samples required at a leaf node.
                           Increasing this (e.g. to 3-5) regularises the
                           model and helps with small datasets.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        oob_score=True,   # free out-of-bag estimate: a rough val score
        n_jobs=-1,        # use all CPU cores
        random_state=seed,
    )