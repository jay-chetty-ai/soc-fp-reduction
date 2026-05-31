"""Build and persist the FAISS RAG index from CICIDS2017 training and validation data.

Usage:
    python scripts/build_rag_index.py [--config config.yaml] [--sample-size N]

The index covers all rows from both the training split (70%) and the validation split
(15%) of the per-label stratified split. Including the validation set ensures all attack
families are available as retrieval candidates -- with a temporal day-5 hold-out, attack
types that only appear on Friday (DDoS, PortScan, Bot) would be absent from the index.

With the full 2.8M-row dataset this embeds roughly 2.45M alerts; on an
RTX 2070 SUPER that takes ~15-20 minutes. Pass --sample-size to cap the
number of rows embedded (useful for quick demos or CPU-only machines).

Outputs (paths from config.yaml):
    rag.faiss_index_path    - FAISS IndexFlatIP binary
    rag.training_df_path    - Parquet of the indexed rows (for
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
    per_day_stratified_split,
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
    parser = argparse.ArgumentParser(description="Build FAISS RAG index from training and validation data.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Embed at most N rows (stratified by label) from the train+val set. "
            "Omit to embed all train+val rows."
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

    logger.info("Applying per-label stratified split (70/15/15)...")
    train_df, val_df, _ = per_day_stratified_split(df, train_ratio=0.70, val_ratio=0.15, random_state=42)

    # Combine train and val for the RAG index -- the test split is never indexed
    index_df = pd.concat([train_df, val_df], ignore_index=True)
    logger.info(
        "Rows for FAISS index: train=%d + val=%d = %d total.",
        len(train_df),
        len(val_df),
        len(index_df),
    )

    if args.sample_size is not None and args.sample_size < len(index_df):
        logger.info(
            "Sampling %d rows (stratified by label) from %d train+val rows...",
            args.sample_size,
            len(index_df),
        )
        from sklearn.model_selection import StratifiedShuffleSplit
        y = (index_df["Label"] != "BENIGN").astype(int)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.sample_size, random_state=42)
        _, keep_idx = next(sss.split(index_df, y))
        index_df = index_df.iloc[keep_idx].reset_index(drop=True)
        logger.info("Sampled %d rows.", len(index_df))

    device = config["rag"].get("device", "auto")
    embedding_model_name = config["rag"]["embedding_model"]
    logger.info("Loading embedding model '%s' (device=%s)...", embedding_model_name, device)
    embedding_model = load_embedding_model(embedding_model_name, device=device)

    batch_size = config["rag"].get("embedding_batch_size", 64)
    logger.info("Embedding %d alerts (batch_size=%d)...", len(index_df), batch_size)
    texts = [alert_to_text(row) for _, row in index_df.iterrows()]
    embeddings = embed_alerts(embedding_model, texts, batch_size=batch_size)
    logger.info("Embeddings shape: %s dtype=%s", embeddings.shape, embeddings.dtype)

    logger.info("Building FAISS index...")
    index = build_index(embeddings)

    index_path = Path(config["rag"]["faiss_index_path"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    save_index(index, index_path)
    logger.info("FAISS index saved to %s (%d vectors).", index_path, index.ntotal)

    training_df_path = Path(config["rag"]["training_df_path"])
    training_df_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.reset_index(drop=True).to_parquet(training_df_path, index=False)
    logger.info(
        "Index DataFrame saved to %s (%d rows, %d columns).",
        training_df_path,
        len(index_df),
        len(index_df.columns),
    )

    dist = index_df["Label"].value_counts().to_dict()
    logger.info("Label distribution in index: %s", dist)
    logger.info("RAG index build complete.")


if __name__ == "__main__":
    main()
