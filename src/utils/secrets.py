"""Secrets management and API key handling (Security Control S2)."""

import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv

# Load .env once at import time so callers don't need to manage it.
# Tests that patch os.environ with patch.dict(..., clear=True) still work
# because patch.dict overrides the already-loaded values for the test's scope.
load_dotenv()

_API_KEY_PATTERN = re.compile(r"sk-ant-[^\s\"']+")


class RedactionFilter(logging.Filter):
    """Logging filter that redacts Anthropic API keys from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact API key patterns from the log record message before emission."""
        record.msg = redact_secrets(str(record.msg))
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    redact_secrets(str(a)) if isinstance(a, str) else a
                    for a in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: redact_secrets(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
        return True


def redact_secrets(msg: str) -> str:
    """Replace any Anthropic API key patterns in a string with [REDACTED]."""
    return _API_KEY_PATTERN.sub("[REDACTED]", msg)


def load_api_key() -> str:
    """Load the Anthropic API key from the environment.

    The .env file is loaded at module import time, so the key is available
    via os.environ by the time this function is called.

    Raises:
        ValueError: If the key is missing or does not match the expected format.
    """
    key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
        )
    if not key.startswith("sk-ant-"):
        raise ValueError(
            "ANTHROPIC_API_KEY appears malformed: expected prefix 'sk-ant-', "
            f"got a key starting with '{key[:7]}...'. Check your .env file."
        )
    return key


def configure_redaction_filter() -> None:
    """Attach a RedactionFilter to the root logger and all existing handlers.

    Call once at application startup before any logging of potentially
    sensitive data.
    """
    root = logging.getLogger()
    redaction = RedactionFilter()
    root.addFilter(redaction)
    for handler in root.handlers:
        handler.addFilter(redaction)
