import httpx
import pandas as pd

from api_client import HolidayClient
from tests.conftest import FAST_CFG, fake_open_meteo
from vellamo_pipeline import enrich


def run(tmp_path, xlsx, client, extra=None):
    args = ["--input", str(xlsx), "--data-dir", str(tmp_path / "out"), "--log-dir", str(tmp_path / "logs")]
    return enrich.main(args + (extra or []), weather_client=client)


def test_happy_path_enriches_every_row(tmp_path, tiny_sales_xlsx, make_weather_client):
    code = run(tmp_path, tiny_sales_xlsx, make_weather_client())
    out = pd.read_csv(tmp_path / "out" / "sales_enriched.csv")
    assert code == 0
    assert len(out) == 80 and out["precip_mm"].notna().all()
    assert (tmp_path / "out" / "weather_daily.csv").exists()
    health = pd.read_csv(tmp_path / "out" / "api_health_log.csv")
    assert health["call_id"].nunique() == 2  # one call per city, not per row


def test_flaky_api_recovers_through_retries(tmp_path, tiny_sales_xlsx, make_weather_client):
    seen = set()

    def flaky(request):
        key = request.url.params["latitude"]
        if key not in seen:
            seen.add(key)
            return httpx.Response(503)
        return fake_open_meteo(request)

    code = run(tmp_path, tiny_sales_xlsx, make_weather_client(flaky))
    health = pd.read_csv(tmp_path / "out" / "api_health_log.csv")
    assert code == 0
    assert (health["outcome"] == "retryable_error").sum() == 2
    assert (health["outcome"] == "success").sum() == 2


def test_one_dead_city_degrades_gracefully(tmp_path, tiny_sales_xlsx, make_weather_client):
    def eldoret_down(request):
        if abs(float(request.url.params["latitude"]) - 0.5143) < 1e-6:
            return httpx.Response(500)
        return fake_open_meteo(request)

    code = run(tmp_path, tiny_sales_xlsx, make_weather_client(eldoret_down))
    out = pd.read_csv(tmp_path / "out" / "sales_enriched.csv")
    assert code == 2
    assert out.loc[out.City == "Nairobi", "precip_mm"].notna().all()
    assert out.loc[out.City == "Eldoret", "precip_mm"].isna().all()
    assert len(out) == 80  # no sales rows lost


def test_second_run_uses_cache(tmp_path, tiny_sales_xlsx, make_weather_client):
    run(tmp_path, tiny_sales_xlsx, make_weather_client())
    calls = []

    def counting(request):
        calls.append(1)
        return fake_open_meteo(request)

    run(tmp_path, tiny_sales_xlsx, make_weather_client(counting))
    assert calls == []


def test_holidays_flag_adds_columns(tmp_path, tiny_sales_xlsx, make_weather_client, monkeypatch):
    payload = [{"date": "2025-01-01", "name": "New Year's Day", "localName": "New Year", "types": ["Public"]}]
    holiday_client = HolidayClient(FAST_CFG, http_transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json=payload)), sleep=lambda s: None)
    args = ["--input", str(tiny_sales_xlsx), "--data-dir", str(tmp_path / "out"),
            "--log-dir", str(tmp_path / "logs"), "--holidays"]
    code = enrich.main(args, weather_client=make_weather_client(), holiday_client=holiday_client)
    out = pd.read_csv(tmp_path / "out" / "sales_enriched.csv")
    assert code == 0
    assert out["is_public_holiday"].sum() == 4  # 2 cities x 2 rows on 1 Jan