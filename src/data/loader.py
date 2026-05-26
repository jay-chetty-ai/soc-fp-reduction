"""CICIDS2017 dataset loading and fixture creation."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load CICIDS2017 CSV files from the raw data directory.

    Concatenates all CSVs found in config["data"]["raw_dir"]. Strips
    whitespace from column names (a known CICIDS2017 artefact).

    Args:
        config: Parsed config.yaml as a dict.

    Returns:
        Combined DataFrame with clean column names.

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
    frames: list[pd.DataFrame] = []
    for f in csv_files:
        logger.info("Loading %s", f.name)
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        logger.info("  %d rows, %d columns", len(df), len(df.columns))
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total combined rows: %d, columns: %d", len(combined), len(combined.columns))
    return combined


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that a DataFrame conforms to the CICIDS2017 schema.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: On any schema violation (missing Label, too few columns,
                    duplicate column names, or whitespace-padded names).
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
