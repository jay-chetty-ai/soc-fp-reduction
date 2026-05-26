"""Retroactive IOC check for auto-closed FP alerts (tripwire)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AutoFPRecord(BaseModel):
    """Record of an alert that was auto-closed as false positive."""

    alert_id: str
    alert_fields: dict[str, Any]
    closed_at: datetime


class TripwireStore:
    """Store of auto-closed FP alert records with optional file persistence.

    Records are kept in memory and, when a path is given, appended to a JSON
    Lines file on disk so they survive process restarts. Each line is a JSON
    object matching the AutoFPRecord schema.

    Args:
        path: Optional path to the JSON Lines persistence file. If None the
              store is in-memory only (suitable for tests).
    """

    def __init__(self, path: Path | None = None) -> None:
        self._records: list[AutoFPRecord] = []
        self._path = Path(path) if path is not None else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Read any existing records from the persistence file on startup."""
        if self._path is None or not self._path.exists():
            return
        loaded = 0
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._records.append(AutoFPRecord.model_validate_json(line))
                    loaded += 1
                except Exception as exc:
                    logger.warning("Skipping malformed tripwire record: %s", exc)
        if loaded:
            logger.info("Tripwire: loaded %d existing records from %s.", loaded, self._path)

    def record(self, rec: AutoFPRecord) -> None:
        """Add a record to the store and persist it to disk if a path is configured."""
        self._records.append(rec)
        if self._path is not None:
            with open(self._path, "a") as f:
                f.write(rec.model_dump_json() + "\n")

    def all_records(self) -> list[AutoFPRecord]:
        """Return all stored records (oldest first)."""
        return list(self._records)


def record_auto_fp(
    alert_id: str,
    alert_fields: dict[str, Any],
    store: TripwireStore,
    timestamp: datetime | None = None,
) -> None:
    """Record an auto-closed FP alert in the tripwire store.

    Args:
        alert_id: Unique alert identifier.
        alert_fields: Feature dict for the alert (used for IOC matching).
        store: TripwireStore instance to append to.
        timestamp: Override the closed_at timestamp (defaults to UTC now).
    """
    ts = timestamp or datetime.now(tz=timezone.utc)
    rec = AutoFPRecord(alert_id=alert_id, alert_fields=alert_fields, closed_at=ts)
    store.record(rec)
    logger.debug("Tripwire: recorded auto-FP alert %s at %s.", alert_id, ts)


def check_ioc(
    ioc: dict[str, Any],
    store: TripwireStore,
    lookback_days: int = 7,
) -> list[str]:
    """Return alert IDs of auto-FP records that match the IOC within the lookback window.

    A record matches the IOC if all key-value pairs in `ioc` appear in the
    record's alert_fields with equal values.

    Args:
        ioc: Dict of field name to value that defines the indicator of compromise.
        store: TripwireStore containing previously auto-closed alerts.
        lookback_days: Only consider records closed within this many days.

    Returns:
        List of matching alert IDs; empty if no matches.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
    matches: list[str] = []
    for rec in store.all_records():
        if rec.closed_at < cutoff:
            continue
        if all(str(rec.alert_fields.get(k)) == str(v) for k, v in ioc.items()):
            matches.append(rec.alert_id)
    if matches:
        logger.warning(
            "Tripwire triggered: IOC %s matched %d auto-FP alerts: %s",
            ioc,
            len(matches),
            matches,
        )
    return matches
