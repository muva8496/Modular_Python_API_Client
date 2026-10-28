"""Public clients. One base class, one thin subclass per third-party API."""
from __future__ import annotations

import datetime as dt
import random
import time
from typing import Any, Callable, Optional

import httpx

from .config import NAGER_DATE_URL, OPEN_METEO_ARCHIVE_URL, ClientConfig
from .errors import ResponseValidationError
from .models import Holiday, WeatherSeries
from .retry import RetryPolicy
from .transport import Transport


class BaseAPIClient:
    def __init__(
        self,
        config: ClientConfig,
        *,
        http_transport: Optional[httpx.BaseTransport] = None,
        policy: Optional[RetryPolicy] = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._transport = Transport(
            config, http_transport=http_transport, policy=policy, sleep=sleep, rng=rng
        )

    def get_json(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        response = self._transport.request("GET", path, params=params)
        try:
            return response.json()
        except ValueError as exc:
            raise ResponseValidationError(
                "Response was not valid JSON", url=str(response.request.url),
                status_code=response.status_code,
            ) from exc

    @property
    def health_log(self) -> list[dict[str, Any]]:
        return self._transport.health_log

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "BaseAPIClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class WeatherClient(BaseAPIClient):
    """Open-Meteo historical archive (daily rain and temperature)."""

    @classmethod
    def from_env(cls, **kwargs: Any) -> "WeatherClient":
        return cls(ClientConfig.from_env(OPEN_METEO_ARCHIVE_URL), **kwargs)

    def get_daily_weather(
        self,
        latitude: float,
        longitude: float,
        start: dt.date,
        end: dt.date,
        timezone: str = "Africa/Nairobi",
    ) -> WeatherSeries:
        payload = self.get_json(
            "/v1/archive",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": timezone,
            },
        )
        return WeatherSeries.from_open_meteo(payload)


class HolidayClient(BaseAPIClient):
    """Nager.Date public holidays."""

    @classmethod
    def from_env(cls, **kwargs: Any) -> "HolidayClient":
        return cls(ClientConfig.from_env(NAGER_DATE_URL), **kwargs)

    def get_public_holidays(self, year: int, country: str = "KE") -> list[Holiday]:
        payload = self.get_json(f"/api/v3/PublicHolidays/{year}/{country}")
        return Holiday.list_from_nager(payload)