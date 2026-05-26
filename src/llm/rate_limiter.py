"""Rate limiting, circuit breaker, and exponential backoff (Security Control S7)."""

import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Sliding-window rate limiter for Claude API calls.

    Tracks call timestamps in two deques (hourly and daily). Thread-safety is
    not guaranteed; use a single instance per process.

    Args:
        max_per_hour: Maximum API calls allowed in any 60-minute window.
        max_per_day: Maximum API calls allowed in any 24-hour window.
    """

    max_per_hour: int
    max_per_day: int
    _hourly: deque = field(default_factory=deque, repr=False, compare=False)
    _daily: deque = field(default_factory=deque, repr=False, compare=False)

    def acquire(self) -> bool:
        """Attempt to consume one API call slot.

        Returns:
            True if the call is permitted; False if a limit is exhausted.
        """
        now = time.monotonic()
        cutoff_hour = now - 3600.0
        cutoff_day = now - 86400.0

        # Evict expired timestamps
        while self._hourly and self._hourly[0] < cutoff_hour:
            self._hourly.popleft()
        while self._daily and self._daily[0] < cutoff_day:
            self._daily.popleft()

        if len(self._hourly) >= self.max_per_hour:
            logger.warning(
                "Rate limit: hourly quota exhausted (%d/%d).",
                len(self._hourly),
                self.max_per_hour,
            )
            return False
        if len(self._daily) >= self.max_per_day:
            logger.warning(
                "Rate limit: daily quota exhausted (%d/%d).",
                len(self._daily),
                self.max_per_day,
            )
            return False

        self._hourly.append(now)
        self._daily.append(now)
        return True


@dataclass
class CircuitBreaker:
    """Open the circuit when the uncertain-band ratio exceeds a threshold.

    When too many alerts fall into the uncertain band, further API calls may
    indicate a systematic model failure. The circuit breaker prevents runaway
    API spend in that scenario.

    Args:
        threshold: Fraction of uncertain alerts (0–1) that triggers the breaker.
    """

    threshold: float

    def check(self, uncertain_count: int, total_count: int) -> bool:
        """Return True (circuit open, halt calls) if the ratio exceeds threshold.

        Args:
            uncertain_count: Number of alerts routed to the uncertain band.
            total_count: Total alerts processed.

        Returns:
            True if the circuit should open (stop further calls); False otherwise.
        """
        if total_count == 0:
            return False
        ratio = uncertain_count / total_count
        open_ = ratio > self.threshold
        if open_:
            logger.warning(
                "Circuit breaker tripped: uncertain ratio %.2f exceeds threshold %.2f.",
                ratio,
                self.threshold,
            )
        return open_


def compute_backoff(base: float, attempt: int, max_wait: float = 30.0) -> float:
    """Compute exponential backoff with full jitter, capped at max_wait.

    Formula: min(base * 2^attempt + Uniform(0, 1), max_wait)

    Args:
        base: Base delay in seconds.
        attempt: Zero-indexed retry attempt number.
        max_wait: Upper bound on the returned delay in seconds.

    Returns:
        Delay in seconds; always in (0.0, max_wait].
    """
    delay = min(base * (2 ** attempt) + random.uniform(0, 1), max_wait)
    return delay
