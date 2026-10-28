"""HTTP transport: one place that talks to httpx, retries, maps errors and records health data."""
from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

from .config import ClientConfig
from .errors import APIError, NetworkError, error_from_response
from .retry import RetryPolicy, execute_with_retry, parse_retry_after

logger = logging.getLogger("api_client.transport")

# Safe to repeat without side effects. POST/PATCH are NOT retried unless the caller says so.
IDEMPOTENT_METHODS = {"GET", "HEAD", "OPTIONS", "PUT", "DELETE"}


@dataclass
class AttemptRecord:
    """One row of the API health dataset: a single HTTP attempt."""

    call_id: str
    timestamp_utc: str
    method: str
    url: str
    attempt: int
    outcome: str  # success | retryable_error | fatal_error
    status_code: Optional[int]
    error_type: Optional[str]
    latency_ms: float
    next_delay_s: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Transport:
    def __init__(
        self,
        config: ClientConfig,
        *,
        http_transport: Optional[httpx.BaseTransport] = None,
        policy: Optional[RetryPolicy] = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.config = config
        self.policy = policy or config.retry_policy()
        self._sleep = sleep
        self._rng = rng
        self._log: list[AttemptRecord] = []
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout(),
            headers=config.headers(),
            transport=http_transport,
            follow_redirects=True,
        )

    # -- lifecycle ---------------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Transport":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- health data -------------------------------------------------------------
    @property
    def health_log(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._log]

    def _record(self, **kwargs: Any) -> AttemptRecord:
        rec = AttemptRecord(timestamp_utc=datetime.now(timezone.utc).isoformat(), **kwargs)
        self._log.append(rec)
        logger.info(
            "api_attempt",
            extra={
                "call_id": rec.call_id,
                "method": rec.method,
                "url": rec.url,
                "attempt": rec.attempt,
                "outcome": rec.outcome,
                "status_code": rec.status_code,
                "error_type": rec.error_type,
                "latency_ms": rec.latency_ms,
            },
        )
        return rec

    # -- public API --------------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json: Any = None,
        idempotent: Optional[bool] = None,
    ) -> httpx.Response:
        """Send a request with retries. Raises a mapped ``APIError`` on failure.

        ``idempotent`` overrides the default per-method rule (e.g. a POST that is
        safe to repeat because it carries an idempotency key).
        """
        method = method.upper()
        can_retry = idempotent if idempotent is not None else method in IDEMPOTENT_METHODS
        call_id = uuid.uuid4().hex[:8]
        full_url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        last: dict[str, Optional[AttemptRecord]] = {"rec": None}

        def attempt_once(attempt: int) -> httpx.Response:
            start = time.perf_counter()
            try:
                response = self._client.request(method, path, params=params, json=json)
            except httpx.TimeoutException as exc:
                err: APIError = NetworkError(f"Timeout: {exc.__class__.__name__}", url=full_url)
                self._fail(call_id, method, full_url, attempt, start, err, last)
                raise err from exc
            except httpx.TransportError as exc:
                err = NetworkError(f"Connection failure: {exc.__class__.__name__}", url=full_url)
                self._fail(call_id, method, full_url, attempt, start, err, last)
                raise err from exc

            latency = (time.perf_counter() - start) * 1000
            if response.status_code >= 400:
                err = error_from_response(
                    response,
                    retry_after=parse_retry_after(response.headers.get("Retry-After")),
                    attempts=attempt,
                )
                last["rec"] = self._record(
                    call_id=call_id, method=method, url=full_url, attempt=attempt,
                    outcome="retryable_error" if err.retryable else "fatal_error",
                    status_code=response.status_code, error_type=type(err).__name__,
                    latency_ms=round(latency, 2),
                )
                raise err
            last["rec"] = self._record(
                call_id=call_id, method=method, url=full_url, attempt=attempt,
                outcome="success", status_code=response.status_code, error_type=None,
                latency_ms=round(latency, 2),
            )
            return response

        def on_retry(attempt: int, exc: APIError, delay: float) -> None:
            if last["rec"] is not None:
                last["rec"].next_delay_s = round(delay, 3)
            logger.warning(
                "api_retry",
                extra={"call_id": call_id, "attempt": attempt,
                       "error_type": type(exc).__name__, "next_delay_s": round(delay, 3)},
            )

        try:
            return execute_with_retry(
                attempt_once, self.policy, can_retry=can_retry,
                sleep=self._sleep, rng=self._rng, on_retry=on_retry,
            )
        except APIError as exc:
            logger.error(
                "api_giving_up",
                extra={"call_id": call_id, "error_type": type(exc).__name__,
                       "status_code": exc.status_code, "attempts": exc.attempts},
            )
            raise

    def _fail(
        self, call_id: str, method: str, url: str, attempt: int, start: float,
        err: APIError, last: dict[str, Optional[AttemptRecord]],
    ) -> None:
        last["rec"] = self._record(
            call_id=call_id, method=method, url=url, attempt=attempt,
            outcome="retryable_error", status_code=None, error_type=type(err).__name__,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )