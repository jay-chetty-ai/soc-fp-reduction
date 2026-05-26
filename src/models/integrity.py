"""Model artifact integrity verification via SHA-256 (Security Control S4)."""

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024  # 1 MB read chunks for large model files


class ModelIntegrityError(Exception):
    """Raised when a model artifact's hash does not match its stored checksum."""

    def __init__(self, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Model integrity check failed.\n"
            f"  Expected SHA-256: {expected}\n"
            f"  Actual  SHA-256: {actual}"
        )


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file using chunked reads."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def save_hash(artifact_path: Path, checksums_path: Path) -> str:
    """Compute and persist the SHA-256 hash of a model artifact.

    Reads any existing checksums from checksums_path, updates or inserts the
    entry for artifact_path, and writes back to disk.

    Args:
        artifact_path: Path to the saved model file.
        checksums_path: Path to the checksums JSON file.

    Returns:
        The computed SHA-256 hex digest.
    """
    artifact_path = Path(artifact_path)
    checksums_path = Path(checksums_path)
    digest = _sha256_file(artifact_path)
    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if checksums_path.exists():
        existing = json.loads(checksums_path.read_text())
    existing[artifact_path.name] = digest
    checksums_path.write_text(json.dumps(existing, indent=2))
    logger.info("Saved hash for %s: %s", artifact_path.name, digest[:16] + "...")
    return digest


def verify_hash(artifact_path: Path, checksums_path: Path) -> None:
    """Verify that a model artifact matches its stored SHA-256 hash.

    Args:
        artifact_path: Path to the model file to verify.
        checksums_path: Path to the checksums JSON file.

    Raises:
        FileNotFoundError: If checksums_path does not exist.
        KeyError: If no checksum is stored for this artifact.
        ModelIntegrityError: If the computed hash does not match the stored hash.
    """
    artifact_path = Path(artifact_path)
    checksums_path = Path(checksums_path)
    if not checksums_path.exists():
        raise FileNotFoundError(
            f"Checksums file not found at {checksums_path}. "
            "Save the model with save_model() before loading."
        )
    stored = json.loads(checksums_path.read_text())
    if artifact_path.name not in stored:
        raise KeyError(
            f"No checksum stored for '{artifact_path.name}'. "
            "Model may have been saved without integrity tracking."
        )
    expected = stored[artifact_path.name]
    actual = _sha256_file(artifact_path)
    if actual != expected:
        raise ModelIntegrityError(expected=expected, actual=actual)
    logger.info("Integrity check passed for %s.", artifact_path.name)
