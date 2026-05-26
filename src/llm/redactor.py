"""Data minimization for LLM API calls (Security Control S6).

Only a defined allowlist of fields is sent to the Claude API. Sensitive fields
(IPs, ports that could identify internal hosts, credentials) are stripped.
The redacted set is logged in the audit trail.
"""

import logging

logger = logging.getLogger(__name__)

# Features sent to the Claude API. Excludes raw IP addresses, flow IDs, and
# any fields that could directly identify internal hosts or individuals.
_FIELD_ALLOWLIST: frozenset[str] = frozenset([
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Destination Port",
    "hour_of_day",
    "day_of_week",
])


def redact_alert(alert_fields: dict) -> dict:
    """Return only allowlisted fields from an alert dict.

    Args:
        alert_fields: Raw alert feature dict (may include IP addresses, IDs).

    Returns:
        Filtered dict containing only fields in the allowlist.
    """
    redacted = {k: v for k, v in alert_fields.items() if k in _FIELD_ALLOWLIST}
    removed = set(alert_fields.keys()) - set(redacted.keys())
    if removed:
        logger.info("Redacted %d sensitive/non-allowlisted fields: %s", len(removed), sorted(removed))
    return redacted


def get_allowlist() -> frozenset[str]:
    """Return the current field allowlist (for audit logging)."""
    return _FIELD_ALLOWLIST
