"""SHA-256 hash-chained audit logger (Security Control S3)."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_GENESIS_HASH: str = hashlib.sha256(b"GENESIS").hexdigest()


class AuditEntry(BaseModel):
    """Structured record of a pipeline triage decision."""

    timestamp: str
    alert_id: str
    stage: str
    verdict: str
    confidence: Optional[float] = None
    model_version: str
    prompt_hash: Optional[str] = None
    response_hash: Optional[str] = None
    band: Optional[str] = None
    previous_entry_hash: str


class FeedbackEntry(BaseModel):
    """Structured record of an analyst override."""

    timestamp: str
    alert_id: str
    analyst_id: str
    override_verdict: str
    original_verdict: str
    rationale: str
    previous_entry_hash: str


class AuditLogger:
    """Append-only audit logger with SHA-256 hash chain.

    Each entry stores the SHA-256 hash of the previous serialized entry,
    forming a tamper-evident chain. The first entry references a GENESIS hash.
    """

    def __init__(self, log_path: Path) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash: str = self._read_last_hash()

    def _read_last_hash(self) -> str:
        """Return the hash of the last log entry, or the GENESIS hash if empty."""
        if not self._path.exists() or self._path.stat().st_size == 0:
            return _GENESIS_HASH
        last_line = ""
        with open(self._path) as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
        if not last_line:
            return _GENESIS_HASH
        return hashlib.sha256(last_line.encode()).hexdigest()

    def _append(self, entry: BaseModel) -> None:
        line = entry.model_dump_json()
        with open(self._path, "a") as f:
            f.write(line + "\n")
        self._last_hash = hashlib.sha256(line.encode()).hexdigest()

    def log_decision(
        self,
        alert_id: str,
        stage: str,
        verdict: str,
        model_version: str,
        confidence: Optional[float] = None,
        prompt_hash: Optional[str] = None,
        response_hash: Optional[str] = None,
        band: Optional[str] = None,
    ) -> None:
        """Append a triage decision to the audit log."""
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            alert_id=alert_id,
            stage=stage,
            verdict=verdict,
            confidence=confidence,
            model_version=model_version,
            prompt_hash=prompt_hash,
            response_hash=response_hash,
            band=band,
            previous_entry_hash=self._last_hash,
        )
        self._append(entry)
        logger.info(
            "Audit: alert_id=%s stage=%s verdict=%s", alert_id, stage, verdict
        )

    def log_feedback(
        self,
        alert_id: str,
        analyst_id: str,
        override_verdict: str,
        original_verdict: str,
        rationale: str,
    ) -> None:
        """Append an analyst feedback override to the audit log."""
        entry = FeedbackEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            alert_id=alert_id,
            analyst_id=analyst_id,
            override_verdict=override_verdict,
            original_verdict=original_verdict,
            rationale=rationale,
            previous_entry_hash=self._last_hash,
        )
        self._append(entry)
        logger.info(
            "Feedback: alert_id=%s analyst=%s override=%s",
            alert_id,
            analyst_id,
            override_verdict,
        )

    def validate_chain(self) -> bool:
        """Verify the hash chain integrity of the audit log.

        Returns:
            True if the chain is valid.

        Raises:
            ValueError: On the first broken link in the chain.
        """
        if not self._path.exists():
            return True
        entries: list[str] = []
        with open(self._path) as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    entries.append(stripped)
        if not entries:
            return True
        first = json.loads(entries[0])
        if first["previous_entry_hash"] != _GENESIS_HASH:
            raise ValueError(
                f"Hash chain break at entry 0: "
                f"expected GENESIS hash {_GENESIS_HASH[:16]}..., "
                f"got {first['previous_entry_hash'][:16]}..."
            )
        for i in range(1, len(entries)):
            expected = hashlib.sha256(entries[i - 1].encode()).hexdigest()
            actual = json.loads(entries[i])["previous_entry_hash"]
            if actual != expected:
                raise ValueError(
                    f"Hash chain break at entry {i}: "
                    f"expected={expected[:16]}... actual={actual[:16]}..."
                )
        return True
