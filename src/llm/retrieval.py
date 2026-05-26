"""FAISS-based RAG retrieval for historical alert dispositions."""

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from pre-normalised embeddings.

    Embeddings must already be L2-normalised (as produced by embed_alerts with
    normalize_embeddings=True). Inner product of unit vectors equals cosine
    similarity, so scores returned by retrieve_similar are in [-1, 1] and in
    practice in [0, 1] for semantically related text.

    Args:
        embeddings: Float32 array of shape (n, dim).

    Returns:
        Fitted faiss.IndexFlatIP with n vectors.
    """
    n, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("Built FAISS index: %d vectors, dim=%d.", n, dim)
    return index


def save_index(index: faiss.IndexFlatIP, path: Path) -> None:
    """Persist a FAISS index to disk.

    Args:
        index: Fitted FAISS index.
        path: Destination file path. Parent directories are created if absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info("Saved FAISS index to %s.", path)


def load_index(path: Path) -> faiss.IndexFlatIP:
    """Load a FAISS index from disk.

    Args:
        path: Path to the saved index file.

    Returns:
        Loaded FAISS index.

    Raises:
        FileNotFoundError: If the index file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found at {path}.")
    index = faiss.read_index(str(path))
    logger.info("Loaded FAISS index from %s (%d vectors).", path, index.ntotal)
    return index


def retrieve_similar(
    index: faiss.IndexFlatIP,
    query: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve the top-k most similar vectors from the FAISS index.

    Args:
        index: FAISS IndexFlatIP built from normalised embeddings.
        query: Float32 array of shape (embedding_dim,) or (1, embedding_dim).
        k: Number of nearest neighbours to return.

    Returns:
        Tuple of (similarities, indices), each a 1-D array of length k.
        Similarities are cosine values clipped to [0.0, 1.0].
    """
    q = np.array(query, dtype=np.float32)
    if q.ndim == 1:
        q = q[np.newaxis, :]
    raw_scores, idx = index.search(q, k)
    # Clip cosine similarities to [0, 1] (negative values indicate dissimilar text)
    similarities = np.clip(raw_scores[0], 0.0, 1.0)
    indices = idx[0]
    logger.debug("Retrieved %d neighbours; top similarity=%.4f.", k, float(similarities[0]))
    return similarities, indices
