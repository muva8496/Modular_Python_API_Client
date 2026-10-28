"""Retry policy and exponential backoff.

delay(attempt) = min(max_delay, base_delay * 2**(attempt-1))
with "equal jitter": half fixed, half random, so many clients do not retry in lockstep.
A server-supplied Retry-After always wins if it asks for longer than we would wait.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Optional, TypeVar

from .errors import APIError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 0.5
    max_delay: float = 30.0
    max_retry_after: float = 120.0
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0 or self.max_retry_after < 0:
            raise ValueError("delays must be >= 0")


def compute_delay(
    attempt: int,
    policy: RetryPolicy,
    retry_after: Optional[float] = None,
    rng: Optional[random.Random] = None,
) -> float:
    """Seconds to wait after ``attempt`` (1-based) failed."""
    if attempt < 1:
        raise ValueError("attempt is 1-based")
    rng = rng or random
    ceiling = min(policy.max_delay, policy.base_delay * (2 ** (attempt - 1)))
    delay = ceiling / 2 + rng.uniform(0, ceiling / 2) if policy.jitter else ceiling
    if retry_after is not None and retry_after > 0:
        delay = max(delay, min(retry_after, policy.max_retry_after))
    return delay


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse a Retry-After header: delta-seconds or an HTTP-date. None if unusable."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
        return seconds if seconds >= 0 else None
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def execute_with_retry(
    operation: Callable[[int], T],
    policy: RetryPolicy,
    *,
    can_retry: bool = True,
    sleep: Callable[[float], None] = time.sleep,
    rng: Optional[random.Random] = None,
    on_retry: Optional[Callable[[int, APIError, float], None]] = None,
) -> T:
    """Run ``operation(attempt)`` until it succeeds or the policy gives up.

    Only retryable APIError subclasses are retried, and only when ``can_retry``
    is true (callers set it false for non-idempotent requests).
    """
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation(attempt)
        except APIError as exc:
            exc.attempts = attempt
            if not (can_retry and exc.retryable) or attempt == policy.max_attempts:
                raise
            delay = compute_delay(attempt, policy, exc.retry_after, rng)
            if on_retry:
                on_retry(attempt, exc, delay)
            sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover