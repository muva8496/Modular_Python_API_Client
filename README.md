# vellamo-api-client

A resilient `httpx` API client for Python, plus a pipeline that uses it to enrich fashion-retail sales data with real weather and logs every API call as a reliability dataset.

Third-party APIs are dependencies you do not control. This client keeps them from being fatal: it retries transient failures with exponential backoff, refuses to retry errors that cannot be fixed, maps every failure to a clear exception, and records how each vendor actually behaves.

## Verified live run

Run on 28 Oct 2026 against the Open-Meteo historical archive: one year of daily weather for 5 Kenyan cities, merged into 3,758 sales rows.

| Metric | Result |
|---|---|
| API calls | 5 (one per city, cached afterwards) |
| Attempts / retries | 5 / 0 |
| Failed calls | 0 (success rate 100%) |
| Latency p50 / p95 | 534 ms / 1,656 ms |

This is a single run. Nothing failed, so the retry path was not exercised live; it is covered by the test suite instead. Treat the latency figures as one sample, not a benchmark.

## Features

| Feature | Behaviour |
|---|---|
| Retries | Up to 5 attempts (configurable) on timeouts, connection errors, 408, 429 and 5xx |
| Exponential backoff | `min(30 s, 0.5 s x 2^(attempt-1))` with equal jitter (half fixed, half random) |
| `Retry-After` | Honoured for seconds or HTTP-date values, capped at 120 s |
| Error mapping | Every failure becomes a typed exception (see below) |
| Idempotency guard | POST and PATCH are never retried unless `idempotent=True` is passed |
| Timeouts | Explicit on every call (15 s read, 5 s connect by default) |
| Response validation | Pydantic models; a malformed 2xx raises `ResponseValidationError` instead of returning bad data |
| Health log | One row per HTTP attempt: status, latency, outcome, error type, next delay |
| Modular | Config, transport, retry, errors, models and endpoint clients are separate modules |

### Failure behaviour

| Situation | Behaviour |
|---|---|
| Timeout, connection error, 408 | Retry, then `NetworkError` |
| 429 | Retry (honours `Retry-After`), then `RateLimitError` |
| 500, 502, 503, 504 | Retry, then `ServerError` |
| 400, 422 | No retry, `BadRequestError` |
| 401, 403 | No retry, `AuthenticationError` |
| 404 | No retry, `NotFoundError` |
| 2xx with bad JSON or unexpected shape | No retry, `ResponseValidationError` |

## Quick start

```bash
conda create -n api-client python=3.12 -y
conda activate api-client
conda install -c conda-forge httpx pandas openpyxl pydantic pytest python-dotenv matplotlib -y
pip install -e .
python -m pytest
```

Optional: copy `.env.example` to `.env` to change timeouts, attempts or backoff.

### Use the client

```python
import datetime as dt
from api_client import WeatherClient, APIError

with WeatherClient.from_env() as client:
    try:
        series = client.get_daily_weather(-1.2921, 36.8219, dt.date(2025, 1, 1), dt.date(2025, 12, 31))
    except APIError as exc:
        print(type(exc).__name__, exc.status_code, "after", exc.attempts, "attempts")
    else:
        print(len(series.days), "days of weather")
    attempts = client.health_log  # list of dicts, one per HTTP attempt
```

A second endpoint client, `HolidayClient` (Kenyan public holidays via Nager.Date), shares the same core.

### Run the pipeline

Put your sales workbook in `data/raw/` (it is not included in this repository), then:

```bash
python scripts/enrich_sales.py                 # fetch weather, merge, write enriched CSV + API health log
python scripts/analyse_weather.py              # permutation tests on the real data
```

Flags: `--refresh` ignores the weather cache, `--holidays` adds public holidays. Exit code 2 means a source failed after all retries; the run still completes and affected rows are left without weather.

## Simulated data (positive control)

Real data this uniform can hide whether an analysis works at all. The generator builds a separate, clearly labelled **synthetic** workbook with effects of known size, so the analysis can be tested for recovery:

```bash
python scripts/simulate_sales.py --weather data/processed/weather_daily.csv
python scripts/enrich_sales.py --input data/processed/vellamo_sim_v2.xlsx --tag sim_v2
python scripts/analyse_weather.py --input data/processed/sales_enriched_sim_v2.csv \
    --truth data/processed/ground_truth_sim_v2.json
```

- Every assumption is listed in the `PARAMS` block at the top of `simulate.py`; all are unverified.
- The planted effects are written to `ground_truth_sim_v2.json` and the file carries a `data_source = SIMULATED_v2` column.
- Findings from the simulated file say nothing about the real market. The real workbook is only read, never modified.

## Analysis method

Permutation tests, used for screening rather than causal claims. Rain labels are shuffled within each city; payday labels within each month. The rain-volume test is adjusted for city, month, weekday and payday. "Rainy" means at least 1.0 mm of precipitation (an analyst choice).

## Project layout

```
src/api_client/          reusable library (no pandas)
  config.py              settings from env vars; city coordinates
  errors.py              exception hierarchy and status mapping
  retry.py               backoff, jitter, Retry-After, retry loop
  transport.py           httpx wrapper; logs every attempt
  client.py              BaseAPIClient, WeatherClient, HolidayClient
  models.py              pydantic response validation
src/vellamo_pipeline/    loader, enrichment, simulator, analysis, logging
scripts/                 thin command-line wrappers
tests/                   60 tests; no network needed (httpx.MockTransport)
```

## API health log

Every attempt is appended to `data/processed/api_health_log.csv`:

`run_id, source, call_id, timestamp_utc, method, url, attempt, outcome, status_code, error_type, latency_ms, next_delay_s`

Use it to measure each vendor: success rate, retries per call, latency percentiles and dominant error types.

## Limitations

- Only the Open-Meteo calls have been run live; `HolidayClient` is tested against fakes only.
- City coordinates are approximate city centres.
- One year of data across five cities gives small samples per city-day, so volume effects are noisy.
- The analysis is not causal.

## Privacy

`data/raw/`, `data/processed/`, `logs/` and `.env` are git-ignored. Sales, branch and staff data should not be committed.

## Author

Musa ([@muva8496](https://github.com/muva8496))"# Modular_Python_API_Client" 
