"""Response models. Anything that does not fit raises ResponseValidationError, never silent bad data."""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from .errors import ResponseValidationError


class DailyWeather(BaseModel):
    date: dt.date
    temp_max_c: Optional[float] = None
    temp_min_c: Optional[float] = None
    precip_mm: Optional[float] = None


class WeatherSeries(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    days: list[DailyWeather]

    @classmethod
    def from_open_meteo(cls, payload: Any) -> "WeatherSeries":
        try:
            daily = payload["daily"]
            times = daily["time"]
            tmax = daily["temperature_2m_max"]
            tmin = daily["temperature_2m_min"]
            rain = daily["precipitation_sum"]
            if not (len(times) == len(tmax) == len(tmin) == len(rain)):
                raise ValueError("daily arrays have different lengths")
            days = [
                DailyWeather(date=t, temp_max_c=a, temp_min_c=b, precip_mm=r)
                for t, a, b, r in zip(times, tmax, tmin, rain)
            ]
            return cls(
                latitude=payload.get("latitude"),
                longitude=payload.get("longitude"),
                days=days,
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise ResponseValidationError(f"Unexpected weather payload: {exc}") from exc


class Holiday(BaseModel):
    date: dt.date
    name: str
    local_name: Optional[str] = None
    types: list[str] = []

    @classmethod
    def list_from_nager(cls, payload: Any) -> list["Holiday"]:
        try:
            return [
                cls(
                    date=item["date"],
                    name=item["name"],
                    local_name=item.get("localName"),
                    types=item.get("types") or [],
                )
                for item in payload
            ]
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise ResponseValidationError(f"Unexpected holiday payload: {exc}") from exc