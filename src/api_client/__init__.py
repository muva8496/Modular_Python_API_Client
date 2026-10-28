"""Resilient httpx API client: retries, exponential backoff, custom error mapping."""
from .client import BaseAPIClient, HolidayClient, WeatherClient
from .config import CITY_COORDS, ClientConfig
from .errors import (
    APIError,
    AuthenticationError,
    BadRequestError,
    ClientError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    ResponseValidationError,
    ServerError,
)
from .models import DailyWeather, Holiday, WeatherSeries
from .retry import RetryPolicy, compute_delay, execute_with_retry, parse_retry_after

__all__ = [
    "APIError", "AuthenticationError", "BadRequestError", "BaseAPIClient",
    "CITY_COORDS", "ClientConfig", "ClientError", "DailyWeather", "Holiday",
    "HolidayClient", "NetworkError", "NotFoundError", "RateLimitError",
    "ResponseValidationError", "RetryPolicy", "ServerError", "WeatherClient",
    "WeatherSeries", "compute_delay", "execute_with_retry", "parse_retry_after",
]
__version__ = "0.1.0"