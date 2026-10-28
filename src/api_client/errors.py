"""Custom exceptions and status-code mapping.

Hierarchy
---------
APIError
├── ClientError            non-retryable 4xx (caller's fault)
│   ├── BadRequestError    400, 422
│   ├── AuthenticationError 401, 403
│   └── NotFoundError      404
├── RateLimitError         429                (retryable)
├── ServerError            5xx                (retryable)
├── NetworkError           timeouts, connection failures, 408 (retryable)
└── ResponseValidationError  2xx but malformed body (not retryable)
"""
from __future__ import annotations

from typing import Optional

import httpx


class APIError(Exception):
    """Base class for every error this library raises."""

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        url: Optional[str] = None,
        attempts: int = 1,
        retry_after: Optional[float] = None,
        body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.url = url
        self.attempts = attempts
        self.retry_after = retry_after
        self.body = body

    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.url:
            parts.append(f"url={self.url}")
        parts.append(f"attempts={self.attempts}")
        return " | ".join(parts)


class ClientError(APIError):
    """4xx that retrying cannot fix."""


class BadRequestError(ClientError):
    pass


class AuthenticationError(ClientError):
    pass


class NotFoundError(ClientError):
    pass


class RateLimitError(APIError):
    retryable = True


class ServerError(APIError):
    retryable = True


class NetworkError(APIError):
    retryable = True


class ResponseValidationError(APIError):
    """The server answered 2xx but the payload was not what we expected."""


_STATUS_MAP: dict[int, type[APIError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: NotFoundError,
    408: NetworkError,
    422: BadRequestError,
    429: RateLimitError,
}

_BODY_SNIPPET = 300


def error_class_for_status(status_code: int) -> type[APIError]:
    """Return the exception class that represents ``status_code``."""
    if status_code in _STATUS_MAP:
        return _STATUS_MAP[status_code]
    if 500 <= status_code <= 599:
        return ServerError
    if 400 <= status_code <= 499:
        return ClientError
    return APIError


def error_from_response(
    response: httpx.Response,
    *,
    retry_after: Optional[float] = None,
    attempts: int = 1,
) -> APIError:
    """Build the right exception for a failed (>= 400) response."""
    cls = error_class_for_status(response.status_code)
    try:
        body = response.text[:_BODY_SNIPPET]
    except Exception:  # pragma: no cover - unreadable body
        body = None
    return cls(
        f"{cls.__name__}: HTTP {response.status_code} {response.reason_phrase}",
        status_code=response.status_code,
        url=str(response.request.url) if response.request else None,
        attempts=attempts,
        retry_after=retry_after,
        body=body,
    )