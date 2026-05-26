"""Prompt injection mitigation (Security Control S1).

All alert data inserted into LLM prompts is untrusted input. This module
sanitizes field values before prompt assembly and wraps them in XML delimiters
so the LLM can distinguish instructions from data.
"""

import html
import logging
import re

logger = logging.getLogger(__name__)

# Patterns that are characteristic of prompt injection attempts.
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"IGNORE\s+ALL\s+PREVIOUS\s+INSTRUCTIONS?", "[REDACTED_INJECTION]"),
    (r"IGNORE\s+PREVIOUS\s+INSTRUCTIONS?", "[REDACTED_INJECTION]"),
    (r"DISREGARD\s+ALL\s+(?:PREVIOUS\s+)?INSTRUCTIONS?", "[REDACTED_INJECTION]"),
    (r"YOU\s+ARE\s+NOW\s+(?:A\s+)?(?:AN?\s+)?(?:DIFFERENT|NEW)\s+AI", "[REDACTED_INJECTION]"),
    (r"<\|im_start\|>", "[REDACTED_INJECTION]"),
    (r"<\|im_end\|>", "[REDACTED_INJECTION]"),
    (r"<\|system\|>", "[REDACTED_INJECTION]"),
    (r"\{\{.*?\}\}", "[REDACTED_INJECTION]"),  # template-injection patterns
]


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_value(value: str) -> str:
    """Sanitize a single string field value for prompt inclusion.

    Strips ASCII control characters (except tab and newline), applies HTML
    escaping to block XML delimiter injection, then removes known prompt
    injection phrases.

    Args:
        value: Raw field value from the alert.

    Returns:
        Sanitized string safe for insertion into a structured prompt.
    """
    text = _CONTROL_CHAR_RE.sub("", str(value))
    text = html.escape(text, quote=True)
    for pattern, replacement in _INJECTION_PATTERNS:
        original = text
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.DOTALL)
        if text != original:
            logger.warning(
                "Injection pattern detected and neutralised: pattern=%r value=%r",
                pattern,
                original[:120],
            )
    return text


def sanitize_alert_dict(alert_fields: dict) -> dict[str, str]:
    """Sanitize all string fields in an alert dictionary.

    Non-string values are converted to strings first, then sanitized.

    Args:
        alert_fields: Dict of alert feature names to values.

    Returns:
        New dict with the same keys and sanitized string values.
    """
    return {k: sanitize_value(str(v)) for k, v in alert_fields.items()}
