"""Sentence-transformer embeddings for alert text representation."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Features included in the alert text representation. Chosen for their
# discriminative power in distinguishing attack traffic from benign flows.
_TEXT_FEATURES: list[str] = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "SYN Flag Count",
    "ACK Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]


def load_embedding_model(model_name: str, device: str = "auto"):
    """Load a sentence-transformer embedding model.

    Args:
        model_name: HuggingFace model identifier, e.g.
            "sentence-transformers/all-MiniLM-L6-v2".
        device: "cuda", "cpu", or "auto" (CUDA if available, else CPU).

    Returns:
        Loaded SentenceTransformer instance.
    """
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        import torch
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    logger.info("Loading embedding model %s on %s.", model_name, resolved_device)
    model = SentenceTransformer(model_name, device=resolved_device)
    return model


def alert_to_text(alert: pd.Series) -> str:
    """Convert a single alert row to a plain-text representation for embedding.

    Only includes features with non-null values. Unknown features are skipped.

    Args:
        alert: A pd.Series representing one network flow alert. May come
            from the feature-engineered DataFrame (numeric columns only).

    Returns:
        Non-empty string with feature name/value pairs.
    """
    parts: list[str] = []
    for feat in _TEXT_FEATURES:
        if feat in alert.index:
            val = alert[feat]
            if pd.notna(val):
                parts.append(f"{feat}={val:.4g}" if isinstance(val, float) else f"{feat}={val}")

    if not parts:
        # Fallback: use all available numeric values
        for k, v in alert.items():
            if isinstance(v, (int, float)) and pd.notna(v):
                parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")

    return "Network flow alert: " + ", ".join(parts)


def embed_alerts(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of alert text strings.

    Args:
        model: Loaded SentenceTransformer instance.
        texts: List of alert text representations.
        batch_size: Number of texts per encoding batch. Set from
            config["rag"]["embedding_batch_size"] when embedding large corpora.

    Returns:
        Float32 array of shape (len(texts), embedding_dim).
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    result = np.array(embeddings, dtype=np.float32)
    logger.info(
        "Embedded %d texts -> shape %s.", len(texts), result.shape
    )
    return result
