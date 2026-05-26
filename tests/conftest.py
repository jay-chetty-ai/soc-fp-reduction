"""Shared pytest fixtures for all Epic tests.

If data/fixtures/fixture_10k.csv does not exist (i.e., the real CICIDS2017
dataset has not been downloaded yet), a synthetic fixture is generated and
saved there. All tests then use that fixture consistently.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# CICIDS2017 column definitions (78 numeric features + Timestamp + Label)
# ---------------------------------------------------------------------------

CICIDS2017_FEATURE_COLS = [
    "Destination Port", "Protocol", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]  # 78 columns

FIXTURE_PATH = Path("data/fixtures/fixture_10k.csv")
CONFIG_PATH = Path("config.yaml")
N_ROWS = 10_000
N_DAYS = 5
_BASE_DATE = pd.Timestamp("2017-07-03")


def _generate_synthetic_fixture(n: int = N_ROWS, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic CICIDS2017-compatible DataFrame.

    Produces clear class separation so LightGBM achieves PR-AUC >= 0.85.
    80% BENIGN, 20% attack (4 types). Distributed evenly across 5 days.
    """
    rng = np.random.default_rng(seed)
    rows_per_day = n // N_DAYS  # 2000
    frames = []

    for day_idx in range(N_DAYS):
        date = _BASE_DATE + pd.Timedelta(days=day_idx)
        n_benign = int(rows_per_day * 0.80)
        n_attack = rows_per_day - n_benign

        # BENIGN: moderate rates, mixed ports
        benign = {col: np.zeros(n_benign) for col in CICIDS2017_FEATURE_COLS}
        benign["Destination Port"] = rng.choice([80, 443, 22, 53, 8080], n_benign).astype(float)
        benign["Protocol"] = rng.choice([6, 17], n_benign).astype(float)
        benign["Flow Duration"] = rng.lognormal(10, 2, n_benign).clip(0)
        benign["Total Fwd Packets"] = rng.integers(1, 20, n_benign).astype(float)
        benign["Total Backward Packets"] = rng.integers(1, 20, n_benign).astype(float)
        benign["Total Length of Fwd Packets"] = rng.lognormal(6, 1.5, n_benign).clip(0)
        benign["Total Length of Bwd Packets"] = rng.lognormal(6, 1.5, n_benign).clip(0)
        benign["Flow Bytes/s"] = rng.lognormal(8, 2, n_benign).clip(0)
        benign["Flow Packets/s"] = rng.lognormal(4, 1.5, n_benign).clip(0)
        benign["SYN Flag Count"] = rng.integers(0, 3, n_benign).astype(float)
        benign["ACK Flag Count"] = rng.integers(1, 10, n_benign).astype(float)
        benign["FIN Flag Count"] = rng.integers(0, 3, n_benign).astype(float)
        benign["RST Flag Count"] = rng.integers(0, 2, n_benign).astype(float)
        benign["Init_Win_bytes_forward"] = rng.integers(4096, 65535, n_benign).astype(float)
        benign["Init_Win_bytes_backward"] = rng.integers(4096, 65535, n_benign).astype(float)
        benign_df = pd.DataFrame(benign)
        benign_df["Label"] = "BENIGN"

        # DoS Hulk: very high packet rate, short duration, port 80
        n_per = n_attack // 4
        dos = {col: np.zeros(n_per) for col in CICIDS2017_FEATURE_COLS}
        dos["Destination Port"] = np.full(n_per, 80.0)
        dos["Protocol"] = np.full(n_per, 6.0)
        dos["Flow Duration"] = rng.lognormal(3, 1, n_per).clip(0)
        dos["Total Fwd Packets"] = rng.integers(100, 500, n_per).astype(float)
        dos["Total Backward Packets"] = rng.integers(0, 5, n_per).astype(float)
        dos["Flow Bytes/s"] = rng.lognormal(14, 1, n_per).clip(0)
        dos["Flow Packets/s"] = rng.lognormal(10, 1, n_per).clip(0)
        dos["SYN Flag Count"] = rng.integers(5, 50, n_per).astype(float)
        dos["ACK Flag Count"] = rng.integers(0, 3, n_per).astype(float)
        dos["Init_Win_bytes_forward"] = rng.integers(8192, 16384, n_per).astype(float)
        dos["Init_Win_bytes_backward"] = rng.integers(0, 512, n_per).astype(float)
        dos_df = pd.DataFrame(dos)
        dos_df["Label"] = "DoS Hulk"

        # PortScan: random high ports, very few packets
        ps = {col: np.zeros(n_per) for col in CICIDS2017_FEATURE_COLS}
        ps["Destination Port"] = rng.integers(1024, 65535, n_per).astype(float)
        ps["Protocol"] = np.full(n_per, 6.0)
        ps["Flow Duration"] = rng.lognormal(2, 0.5, n_per).clip(0)
        ps["Total Fwd Packets"] = rng.integers(1, 3, n_per).astype(float)
        ps["Total Backward Packets"] = rng.integers(0, 1, n_per).astype(float)
        ps["Flow Bytes/s"] = rng.lognormal(3, 1, n_per).clip(0)
        ps["Flow Packets/s"] = rng.lognormal(2, 1, n_per).clip(0)
        ps["SYN Flag Count"] = rng.integers(1, 3, n_per).astype(float)
        ps["RST Flag Count"] = rng.integers(0, 3, n_per).astype(float)
        ps["Init_Win_bytes_forward"] = rng.integers(1024, 4096, n_per).astype(float)
        ps["Init_Win_bytes_backward"] = rng.integers(0, 256, n_per).astype(float)
        ps_df = pd.DataFrame(ps)
        ps_df["Label"] = "PortScan"

        # DDoS: extremely high bytes/s, UDP
        ddos = {col: np.zeros(n_per) for col in CICIDS2017_FEATURE_COLS}
        ddos["Destination Port"] = rng.choice([80, 443, 53], n_per).astype(float)
        ddos["Protocol"] = np.full(n_per, 17.0)
        ddos["Flow Duration"] = rng.lognormal(6, 1, n_per).clip(0)
        ddos["Total Fwd Packets"] = rng.integers(500, 2000, n_per).astype(float)
        ddos["Total Backward Packets"] = rng.integers(0, 10, n_per).astype(float)
        ddos["Flow Bytes/s"] = rng.lognormal(16, 1, n_per).clip(0)
        ddos["Flow Packets/s"] = rng.lognormal(12, 1, n_per).clip(0)
        ddos["SYN Flag Count"] = rng.integers(0, 2, n_per).astype(float)
        ddos["Init_Win_bytes_forward"] = rng.integers(0, 1024, n_per).astype(float)
        ddos_df = pd.DataFrame(ddos)
        ddos_df["Label"] = "DDoS"

        # Bot: periodic, mid-range, non-standard ports
        n_bot = n_attack - 3 * n_per
        bot = {col: np.zeros(n_bot) for col in CICIDS2017_FEATURE_COLS}
        bot["Destination Port"] = rng.choice([6667, 4444, 1337], n_bot).astype(float)
        bot["Protocol"] = np.full(n_bot, 6.0)
        bot["Flow Duration"] = rng.lognormal(8, 1.5, n_bot).clip(0)
        bot["Total Fwd Packets"] = rng.integers(5, 30, n_bot).astype(float)
        bot["Total Backward Packets"] = rng.integers(5, 30, n_bot).astype(float)
        bot["Flow Bytes/s"] = rng.lognormal(6, 1.5, n_bot).clip(0)
        bot["Flow Packets/s"] = rng.lognormal(3, 1, n_bot).clip(0)
        bot["SYN Flag Count"] = rng.integers(1, 5, n_bot).astype(float)
        bot["ACK Flag Count"] = rng.integers(2, 15, n_bot).astype(float)
        bot["Init_Win_bytes_forward"] = rng.integers(8192, 32768, n_bot).astype(float)
        bot["Init_Win_bytes_backward"] = rng.integers(8192, 32768, n_bot).astype(float)
        bot_df = pd.DataFrame(bot)
        bot_df["Label"] = "Bot"

        day_df = pd.concat([benign_df, dos_df, ps_df, ddos_df, bot_df], ignore_index=True)

        # Add timestamps spread across the work day (08:00-18:00)
        n_day = len(day_df)
        minutes = rng.integers(0, 600, n_day)
        day_df["Timestamp"] = [
            (date + pd.Timedelta(hours=8) + pd.Timedelta(minutes=int(m))).strftime("%d/%m/%Y %H:%M")
            for m in minutes
        ]
        day_df = day_df.sample(frac=1, random_state=seed + day_idx).reset_index(drop=True)
        frames.append(day_df)

    combined = pd.concat(frames, ignore_index=True)
    ordered = CICIDS2017_FEATURE_COLS + ["Timestamp", "Label"]
    return combined[ordered]


def _load_or_generate_fixture() -> pd.DataFrame:
    if FIXTURE_PATH.exists() and FIXTURE_PATH.stat().st_size > 0:
        df = pd.read_csv(FIXTURE_PATH, low_memory=False)
        logging.getLogger(__name__).info(
            "Loaded existing fixture from %s (%d rows).", FIXTURE_PATH, len(df)
        )
        return df
    logging.getLogger(__name__).warning(
        "Fixture not found at %s; generating synthetic data.", FIXTURE_PATH
    )
    df = _generate_synthetic_fixture()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIXTURE_PATH, index=False)
    logging.getLogger(__name__).info(
        "Synthetic fixture saved to %s (%d rows).", FIXTURE_PATH, len(df)
    )
    return df


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def fixture_df() -> pd.DataFrame:
    return _load_or_generate_fixture()


@pytest.fixture(scope="session")
def fixture_features(fixture_df) -> pd.DataFrame:
    from src.data.features import add_temporal_features, clean_features
    df = clean_features(fixture_df)
    return add_temporal_features(df)


@pytest.fixture(scope="session")
def fixture_train(fixture_features) -> pd.DataFrame:
    from src.data.features import temporal_train_test_split
    train_df, _ = temporal_train_test_split(fixture_features, test_day=5)
    return train_df


@pytest.fixture(scope="session")
def fixture_test(fixture_features) -> pd.DataFrame:
    from src.data.features import temporal_train_test_split
    _, test_df = temporal_train_test_split(fixture_features, test_day=5)
    return test_df


@pytest.fixture(scope="session")
def mock_lgb_model(fixture_train) -> lgb.Booster:
    """Lightweight LightGBM trained on 500 rows of fixture_train."""
    from src.data.features import encode_labels, get_feature_columns
    sample = fixture_train.head(500)
    feat_cols = get_feature_columns(sample)
    X = sample[feat_cols]
    y = encode_labels(sample)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "is_unbalance": True,
        "verbose": -1,
        "n_jobs": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
    }
    ds = lgb.Dataset(X, label=y)
    return lgb.train(params, ds, num_boost_round=100, callbacks=[lgb.log_evaluation(-1)])


@pytest.fixture(scope="session")
def mock_shap_values(mock_lgb_model, fixture_test) -> np.ndarray:
    """Pre-computed SHAP values for mock_lgb_model on first 50 rows of fixture_test."""
    from src.data.features import get_feature_columns
    from src.models.explainer import build_explainer, explain_batch
    feat_cols = get_feature_columns(fixture_test)
    sample = fixture_test[feat_cols].head(50)
    explainer = build_explainer(mock_lgb_model)
    return explain_batch(explainer, sample)


@pytest.fixture(scope="session")
def sample_uncertain_alert(fixture_test) -> pd.Series:
    return fixture_test.iloc[0]


@pytest.fixture
def tmp_model_path(tmp_path) -> Path:
    return tmp_path / "test_model.pkl"


@pytest.fixture
def tmp_audit_path(tmp_path) -> Path:
    return tmp_path / "audit.jsonl"


@pytest.fixture
def mock_stage2_response() -> dict:
    with open("tests/fixtures/stage2_response.json") as f:
        return json.load(f)


@pytest.fixture
def mock_adversarial_response() -> dict:
    with open("tests/fixtures/adversarial_response.json") as f:
        return json.load(f)
