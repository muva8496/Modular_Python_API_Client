import datetime as dt

import httpx
import pytest

from api_client import (
    AuthenticationError, ClientConfig, HolidayClient, NetworkError, NotFoundError,
    RateLimitError, ResponseValidationError, RetryPolicy, ServerError, WeatherClient,
)
from api_client.client import BaseAPIClient
from api_client.transport import Transport

CFG = ClientConfig(base_url="https://api.test", max_attempts=5, backoff_base_s=0.5, jitter=False)


def make(handler, sleeps=None, cfg=CFG, cls=BaseAPIClient):
    sleeps = sleeps if sleeps is not None else []
    return cls(cfg, http_transport=httpx.MockTransport(handler), sleep=sleeps.append), sleeps


def weather_payload(n=3):
    start = dt.date(2025, 1, 1)
    return {
        "latitude": -1.3, "longitude": 36.8,
        "daily": {
            "time": [(start + dt.timedelta(days=i)).isoformat() for i in range(n)],
            "temperature_2m_max": [24.0 + i for i in range(n)],
            "temperature_2m_min": [12.0] * n,
            "precipitation_sum": [0.0, 3.2, None][:n],
        },
    }


def test_fails_twice_then_succeeds():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        return httpx.Response(503) if state["n"] <= 2 else httpx.Response(200, json={"ok": True})

    client, sleeps = make(handler)
    assert client.get_json("/x") == {"ok": True}
    assert state["n"] == 3
    assert sleeps == [0.5, 1.0]
    log = client.health_log
    assert [r["attempt"] for r in log] == [1, 2, 3]
    assert [r["outcome"] for r in log] == ["retryable_error", "retryable_error", "success"]
    assert log[0]["next_delay_s"] == 0.5 and log[1]["next_delay_s"] == 1.0
    assert len({r["call_id"] for r in log}) == 1


def test_always_503_exhausts_attempts():
    client, sleeps = make(lambda r: httpx.Response(503))
    with pytest.raises(ServerError) as info:
        client.get_json("/x")
    assert info.value.attempts == 5 and info.value.status_code == 503
    assert len(sleeps) == 4
    assert len(client.health_log) == 5


def test_401_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401)

    client, sleeps = make(handler)
    with pytest.raises(AuthenticationError):
        client.get_json("/x")
    assert len(calls) == 1 and sleeps == []
    assert client.health_log[0]["outcome"] == "fatal_error"


def test_404_maps_to_not_found_without_retry():
    client, sleeps = make(lambda r: httpx.Response(404))
    with pytest.raises(NotFoundError):
        client.get_json("/missing")
    assert sleeps == []


def test_429_honours_retry_after():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "9"})
        return httpx.Response(200, json={"ok": 1})

    client, sleeps = make(handler)
    assert client.get_json("/x") == {"ok": 1}
    assert sleeps == [9.0]


def test_429_until_exhausted_raises_rate_limit_error():
    client, _ = make(lambda r: httpx.Response(429))
    with pytest.raises(RateLimitError):
        client.get_json("/x")


def test_timeouts_are_retried_then_mapped():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    client, sleeps = make(handler)
    with pytest.raises(NetworkError) as info:
        client.get_json("/x")
    assert info.value.attempts == 5 and len(sleeps) == 4


def test_connection_error_recovers():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=[1])

    client, sleeps = make(handler)
    assert client.get_json("/x") == [1]
    assert len(sleeps) == 1


def test_post_is_not_retried_by_default_but_can_opt_in():
    calls = []

    def handler(request):
        calls.append(request.method)
        return httpx.Response(503)

    t = Transport(CFG, http_transport=httpx.MockTransport(handler), sleep=lambda s: None)
    with pytest.raises(ServerError):
        t.request("POST", "/charge", json={"amount": 1})
    assert calls == ["POST"]

    calls.clear()
    with pytest.raises(ServerError):
        t.request("POST", "/charge", json={"amount": 1}, idempotent=True)
    assert len(calls) == 5


def test_malformed_json_raises_validation_error():
    client, _ = make(lambda r: httpx.Response(200, text="<html>not json</html>"))
    with pytest.raises(ResponseValidationError):
        client.get_json("/x")


def test_weather_client_parses_open_meteo_payload():
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=weather_payload())

    client, _ = make(handler, cls=WeatherClient)
    series = client.get_daily_weather(-1.29, 36.82, dt.date(2025, 1, 1), dt.date(2025, 1, 3))
    assert [d.precip_mm for d in series.days] == [0.0, 3.2, None]
    assert series.days[0].date == dt.date(2025, 1, 1)
    assert seen["params"]["start_date"] == "2025-01-01"
    assert "precipitation_sum" in seen["params"]["daily"]


@pytest.mark.parametrize("payload", [
    {},
    {"daily": {"time": ["2025-01-01"]}},
    {"daily": {"time": ["2025-01-01", "2025-01-02"], "temperature_2m_max": [1],
               "temperature_2m_min": [1], "precipitation_sum": [1]}},
    {"daily": {"time": ["not-a-date"], "temperature_2m_max": [1],
               "temperature_2m_min": [1], "precipitation_sum": [1]}},
])
def test_weather_client_rejects_bad_payloads(payload):
    client, _ = make(lambda r: httpx.Response(200, json=payload), cls=WeatherClient)
    with pytest.raises(ResponseValidationError):
        client.get_daily_weather(0, 0, dt.date(2025, 1, 1), dt.date(2025, 1, 2))


def test_holiday_client_parses_nager_payload():
    payload = [{"date": "2025-12-25", "localName": "Christmas Day", "name": "Christmas Day",
                "countryCode": "KE", "types": ["Public"]}]
    client, _ = make(lambda r: httpx.Response(200, json=payload), cls=HolidayClient)
    holidays = client.get_public_holidays(2025)
    assert holidays[0].date == dt.date(2025, 12, 25) and holidays[0].types == ["Public"]


def test_holiday_client_rejects_bad_payload():
    client, _ = make(lambda r: httpx.Response(200, json=[{"nope": 1}]), cls=HolidayClient)
    with pytest.raises(ResponseValidationError):
        client.get_public_holidays(2025)


def test_custom_retry_policy_is_respected():
    cfg = ClientConfig(base_url="https://api.test")
    client = BaseAPIClient(cfg, http_transport=httpx.MockTransport(lambda r: httpx.Response(500)),
                           policy=RetryPolicy(max_attempts=2, jitter=False), sleep=lambda s: None)
    with pytest.raises(ServerError) as info:
        client.get_json("/x")
    assert info.value.attempts == 2


def test_health_log_rows_have_expected_fields():
    client, _ = make(lambda r: httpx.Response(200, json={}))
    client.get_json("/x")
    row = client.health_log[0]
    assert set(row) == {"call_id", "timestamp_utc", "method", "url", "attempt", "outcome",
                        "status_code", "error_type", "latency_ms", "next_delay_s"}