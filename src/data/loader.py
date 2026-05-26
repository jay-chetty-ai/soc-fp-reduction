"""CICIDS2017 dataset loading and fixture creation."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# Map day names (found in CICIDS2017 filenames) to capture dates.
# The dataset covers Monday 03/07/2017 through Friday 07/07/2017.
_DAY_TO_DATE: dict[str, str] = {
    "monday": "03/07/2017",
    "tuesday": "04/07/2017",
    "wednesday": "05/07/2017",
    "thursday": "06/07/2017",
    "friday": "07/07/2017",
}


def _infer_timestamps(filename: str, n_rows: int, rng: np.random.Generator) -> list[str]:
    """Generate synthetic timestamps for rows loaded from a named CICIDS2017 CSV.

    The ML version of CICIDS2017 strips the original Timestamp column. We
    reconstruct it by mapping the filename to the capture date (Monday =
    2017-07-03, ..., Friday = 2017-07-07) and spreading rows uniformly
    across the 08:00-18:00 work window.

    Args:
        filename: CSV filename (e.g. "Friday-WorkingHours-Morning.pcap_ISCX.csv").
        n_rows: Number of rows that need timestamps.
        rng: NumPy random generator for reproducible minute offsets.

    Returns:
        List of timestamp strings in "%d/%m/%Y %H:%M" format.
    """
    lower = Path(filename).stem.lower()
    date_str = "03/07/2017"  # fallback: Monday
    for day_key, date in _DAY_TO_DATE.items():
        if day_key in lower:
            date_str = date
            break
    base = pd.to_datetime(date_str, dayfirst=True) + pd.Timedelta(hours=8)
    minutes = rng.integers(0, 600, n_rows)
    return [
        (base + pd.Timedelta(minutes=int(m))).strftime("%d/%m/%Y %H:%M")
        for m in minutes
    ]


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load CICIDS2017 CSV files from the raw data directory.

    Concatenates all CSVs found in config["data"]["raw_dir"]. Strips
    whitespace from column names (a known CICIDS2017 artefact). Because the
    Machine Learning CSV release omits the original Timestamp column, a
    synthetic Timestamp is derived from each file's name so that downstream
    temporal features and train/test splits work correctly.

    Args:
        config: Parsed config.yaml as a dict.

    Returns:
        Combined DataFrame with clean column names and a Timestamp column.

    Raises:
        FileNotFoundError: If no CSV files are found in the raw directory.
    """
    raw_dir = Path(config["data"]["raw_dir"])
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            "Run scripts/download_data.py to download CICIDS2017."
        )
    rng = np.random.default_rng(42)
    frames: list[pd.DataFrame] = []
    for f in csv_files:
        logger.info("Loading %s", f.name)
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        if "Timestamp" not in df.columns:
            df["Timestamp"] = _infer_timestamps(f.name, len(df), rng)
        logger.info("  %d rows, %d columns", len(df), len(df.columns))
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Total combined rows: %d, columns: %d", len(combined), len(combined.columns)
    )
    return combined


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that a DataFrame conforms to the CICIDS2017 schema.

    Checks for the Label column, a minimum column count, no true duplicates,
    and no whitespace-padded column names. The Timestamp column is optional
    at this stage (it may have been added by load_dataset or be present in
    the synthetic fixture).

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: On any schema violation.
    """
    if "Label" not in df.columns:
        raise ValueError(
            "Schema validation failed: 'Label' column is missing. "
            "Check that the loaded CSV is a CICIDS2017 file."
        )
    if len(df.columns) < 79:
        raise ValueError(
            f"Schema validation failed: expected >= 79 columns, "
            f"found {len(df.columns)}."
        )
    col_list = list(df.columns)
    dupes = {c for c in col_list if col_list.count(c) > 1}
    if dupes:
        raise ValueError(
            f"Schema validation failed: duplicate column names: {sorted(dupes)}"
        )
    padded = [c for c in df.columns if c != c.strip()]
    if padded:
        raise ValueError(
            f"Schema validation failed: {len(padded)} column names have whitespace "
            "padding. Call load_dataset() which strips column names automatically."
        )


def create_fixture_subset(
    df: pd.DataFrame,
    n: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a stratified random subset for use as a test fixture.

    Stratification preserves the class distribution from the full dataset.

    Args:
        df: Source DataFrame (must have a 'Label' column).
        n: Number of rows to sample.
        random_state: Reproducibility seed.

    Returns:
        Stratified subset of n rows, index reset to 0..n-1.

    Raises:
        ValueError: If the source DataFrame has fewer than n rows.
    """
    if len(df) < n:
        raise ValueError(
            f"Source DataFrame has {len(df)} rows; cannot create a {n}-row fixture."
        )
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n, random_state=random_state)
    _, indices = next(sss.split(df, df["Label"]))
    subset = df.iloc[indices].reset_index(drop=True)
    dist = subset["Label"].value_counts().to_dict()
    logger.info(
        "Fixture subset: %d rows. Class distribution: %s",
        len(subset),
        dist,
    )
    return subset
