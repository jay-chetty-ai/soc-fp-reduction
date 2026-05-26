"""Build and persist the FAISS RAG index from CICIDS2017 training data.

Usage:
    python scripts/build_rag_index.py [--config config.yaml] [--sample-size N]

The index covers all rows from the temporal training split (days 1-4).
With the full 2.8M-row dataset this embeds roughly 2.25M alerts; on an
RTX 2070 SUPER that takes ~15-20 minutes. Pass --sample-size to cap the
number of rows embedded (useful for quick demos or CPU-only machines).

Outputs (paths from config.yaml):
    rag.faiss_index_path    - FAISS IndexFlatIP binary
    rag.training_df_path    - Parquet of the embedded training rows (for
                              label lookups during retrieval)

The row order in training_df matches the FAISS index exactly: FAISS
vector i corresponds to training_df.iloc[i].
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.features import (
    add_temporal_features,
    clean_features,
    get_feature_columns,
    temporal_train_test_split,
)
from src.data.loader import load_dataset, validate_schema
from src.llm.embeddings import alert_to_text, embed_alerts, load_embedding_model
from src.llm.retrieval import build_index, save_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS RAG index from training data.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Embed at most N training rows (stratified by label). "
            "Omit to embed the full training set."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("Loading dataset from %s...", config["data"]["raw_dir"])
    df = load_dataset(config)
    validate_schema(df)

    logger.info("Engineering features...")
    df = clean_features(df)
    df = add_temporal_features(df)

    logger.info("Applying temporal split (days 1-4 = training)...")
    train_df, _ = temporal_train_test_split(df, test_day=5)
    logger.info("Training set: %d rows.", len(train_df))

    if args.sample_size is not None and args.sample_size < len(train_df):
        logger.info(
            "Sampling %d rows (stratified by label) from %d training rows...",
            args.sample_size,
            len(train_df),
        )
        from sklearn.model_selection import StratifiedShuffleSplit
        y = (train_df["Label"] != "BENIGN").astype(int)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.sample_size, random_state=42)
        _, keep_idx = next(sss.split(train_df, y))
        train_df = train_df.iloc[keep_idx].reset_index(drop=True)
        logger.info("Sampled %d rows.", len(train_df))

    device = config["rag"].get("device", "auto")
    embedding_model_name = config["rag"]["embedding_model"]
    logger.info("Loading embedding model '%s' (device=%s)...", embedding_model_name, device)
    embedding_model = load_embedding_model(embedding_model_name, device=device)

    logger.info("Embedding %d training alerts...", len(train_df))
    texts = [alert_to_text(row) for _, row in train_df.iterrows()]
    embeddings = embed_alerts(embedding_model, texts)
    logger.info("Embeddings shape: %s dtype=%s", embeddings.shape, embeddings.dtype)

    logger.info("Building FAISS index...")
    index = build_index(embeddings)

    index_path = Path(config["rag"]["faiss_index_path"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    save_index(index, index_path)
    logger.info("FAISS index saved to %s (%d vectors).", index_path, index.ntotal)

    training_df_path = Path(config["rag"]["training_df_path"])
    training_df_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.reset_index(drop=True).to_parquet(training_df_path, index=False)
    logger.info(
        "Training DataFrame saved to %s (%d rows, %d columns).",
        training_df_path,
        len(train_df),
        len(train_df.columns),
    )

    dist = train_df["Label"].value_counts().to_dict()
    logger.info("Label distribution in index: %s", dist)
    logger.info("RAG index build complete.")


if __name__ == "__main__":
    main()
