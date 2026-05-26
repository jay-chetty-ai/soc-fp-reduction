"""Retroactive IOC check for auto-closed FP alerts (tripwire)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AutoFPRecord(BaseModel):
    """Record of an alert that was auto-closed as false positive."""

    alert_id: str
    alert_fields: dict[str, Any]
    closed_at: datetime


class TripwireStore:
    """In-memory store of auto-closed FP alert records.

    In production this would be backed by a database. For the POC it keeps
    records in a list and supports JSON serialisation to disk.
    """

    def __init__(self) -> None:
        self._records: list[AutoFPRecord] = []

    def record(self, record: AutoFPRecord) -> None:
        """Add an auto-FP record to the store."""
        self._records.append(record)

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
    record = AutoFPRecord(alert_id=alert_id, alert_fields=alert_fields, closed_at=ts)
    store.record(record)
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
