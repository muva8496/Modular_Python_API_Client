"""Configuration: everything tunable lives here, nothing is hardcoded in the transport."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import httpx
from dotenv import load_dotenv

from .retry import RetryPolicy

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com"
NAGER_DATE_URL = "https://date.nager.at"

# Approximate city-centre coordinates (verify before relying on them for anything precise).
CITY_COORDS: dict[str, tuple[float, float]] = {
    "Nairobi": (-1.2921, 36.8219),
    "Mombasa": (-4.0435, 39.6682),
    "Kisumu": (-0.0917, 34.7680),
    "Nakuru": (-0.3031, 36.0800),
    "Eldoret": (0.5143, 35.2698),
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class ClientConfig:
    base_url: str
    max_attempts: int = 5
    timeout_s: float = 15.0
    connect_timeout_s: float = 5.0
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 30.0
    jitter: bool = True
    api_key: Optional[str] = None
    user_agent: str = "vellamo-api-client/0.1"
    extra_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, base_url: str, *, load_env_file: bool = True) -> "ClientConfig":
        if load_env_file:
            load_dotenv()
        return cls(
            base_url=base_url,
            max_attempts=_env_int("API_MAX_ATTEMPTS", 5),
            timeout_s=_env_float("API_TIMEOUT_S", 15.0),
            connect_timeout_s=_env_float("API_CONNECT_TIMEOUT_S", 5.0),
            backoff_base_s=_env_float("API_BACKOFF_BASE_S", 0.5),
            backoff_cap_s=_env_float("API_BACKOFF_CAP_S", 30.0),
            api_key=os.getenv("API_KEY") or None,
        )

    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=self.max_attempts,
            base_delay=self.backoff_base_s,
            max_delay=self.backoff_cap_s,
            jitter=self.jitter,
        )

    def timeout(self) -> httpx.Timeout:
        return httpx.Timeout(self.timeout_s, connect=self.connect_timeout_s)

    def headers(self) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers