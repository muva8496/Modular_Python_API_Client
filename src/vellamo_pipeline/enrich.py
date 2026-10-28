"""Enrich Vellamo sales with real weather (and optionally Kenyan public holidays).

    python scripts/enrich_sales.py
    python scripts/enrich_sales.py --input data/processed/vellamo_sim_v2.xlsx --tag sim_v2
    python scripts/enrich_sales.py --holidays --refresh

Only one API call per city for the whole year (5 calls), cached to data/processed/.
A city that still fails after all retries is logged and skipped; the run continues.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd

from api_client import CITY_COORDS, APIError, HolidayClient, WeatherClient

from .loader import integrity_report, load_sales
from .logging_setup import setup_logging

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "raw" / "vellamo_fashion_sales_fy2025_with_lookups.xlsx"
DEFAULT_DATA_DIR = ROOT / "data" / "processed"
DEFAULT_LOG_DIR = ROOT / "logs"

logger = logging.getLogger("vellamo_pipeline.enrich")

WEATHER_COLUMNS = ["City", "Date", "precip_mm", "temp_max_c", "temp_min_c"]


def fetch_city_weather(client: WeatherClient, city: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    lat, lon = CITY_COORDS[city]
    series = client.get_daily_weather(lat, lon, start, end)
    df = pd.DataFrame([d.model_dump() for d in series.days])
    df = df.rename(columns={"date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df.insert(0, "City", city)
    return df[WEATHER_COLUMNS]


def fetch_all_weather(
    client: WeatherClient,
    cities: list[str],
    start: dt.date,
    end: dt.date,
    cache_dir: Path,
    refresh: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (weather for every city that worked, list of cities that failed)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for city in cities:
        if city not in CITY_COORDS:
            logger.error("unknown_city_no_coordinates", extra={"city": city})
            failed.append(city)
            continue
        cache = cache_dir / f"weather_{city}_{start}_{end}.csv"
        if cache.exists() and not refresh:
            logger.info("weather_cache_hit", extra={"city": city, "file": cache.name})
            frames.append(pd.read_csv(cache, parse_dates=["Date"]))
            continue
        try:
            df = fetch_city_weather(client, city, start, end)
        except APIError as exc:
            logger.error(
                "weather_fetch_failed",
                extra={"city": city, "error_type": type(exc).__name__,
                       "status_code": exc.status_code, "attempts": exc.attempts},
            )
            failed.append(city)
            continue
        df.to_csv(cache, index=False)
        frames.append(df)
        logger.info("weather_fetched", extra={"city": city, "rows": len(df)})
    weather = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=WEATHER_COLUMNS)
    return weather, failed


def merge_weather(sales: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.copy()
    weather["Date"] = pd.to_datetime(weather["Date"])
    return sales.merge(weather, on=["City", "Date"], how="left", validate="many_to_one")


def add_holidays(sales: pd.DataFrame, client: HolidayClient, years: list[int]) -> pd.DataFrame:
    rows = []
    for year in years:
        for h in client.get_public_holidays(year):
            rows.append({"Date": pd.Timestamp(h.date), "holiday_name": h.name})
    holidays = pd.DataFrame(rows, columns=["Date", "holiday_name"]).drop_duplicates("Date")
    out = sales.merge(holidays, on="Date", how="left")
    out["is_public_holiday"] = out["holiday_name"].notna()
    return out


def health_summary(records: list[dict]) -> dict:
    if not records:
        return {"calls": 0, "attempts": 0, "retries": 0, "failed_calls": 0}
    df = pd.DataFrame(records)
    per_call = df.groupby("call_id")["outcome"].apply(lambda s: (s == "success").any())
    return {
        "calls": int(df["call_id"].nunique()),
        "attempts": int(len(df)),
        "retries": int(len(df) - df["call_id"].nunique()),
        "failed_calls": int((~per_call).sum()),
        "success_rate": round(float(per_call.mean()), 4),
        "latency_ms_p50": round(float(df["latency_ms"].quantile(0.5)), 1),
        "latency_ms_p95": round(float(df["latency_ms"].quantile(0.95)), 1),
    }


def append_health_log(records: list[dict], path: Path, run_id: str, source: str) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    df.insert(0, "run_id", run_id)
    df.insert(1, "source", source)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main(
    argv: Optional[list[str]] = None,
    *,
    weather_client: Optional[WeatherClient] = None,
    holiday_client: Optional[HolidayClient] = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tag", default="", help="suffix for output files, e.g. sim_v2")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--refresh", action="store_true", help="ignore cached weather")
    parser.add_argument("--holidays", action="store_true", help="also add Kenyan public holidays")
    args = parser.parse_args(argv)

    setup_logging(args.log_dir)
    run_id = uuid.uuid4().hex[:8]
    suffix = f"_{args.tag}" if args.tag else ""

    sales = load_sales(args.input)
    logger.info("sales_loaded", extra={"run_id": run_id, "file": args.input.name, **integrity_report(sales)})
    start, end = sales["Date"].min().date(), sales["Date"].max().date()
    cities = sorted(sales["City"].unique())

    own_weather = weather_client is None
    weather_client = weather_client or WeatherClient.from_env()
    own_holiday = False
    try:
        weather, failed = fetch_all_weather(weather_client, cities, start, end, args.data_dir, args.refresh)
        weather.to_csv(args.data_dir / "weather_daily.csv", index=False)
        enriched = merge_weather(sales, weather)

        records = list(weather_client.health_log)
        if args.holidays:
            if holiday_client is None:
                holiday_client, own_holiday = HolidayClient.from_env(), True
            try:
                enriched = add_holidays(enriched, holiday_client, sorted({start.year, end.year}))
            except APIError as exc:
                logger.error("holiday_fetch_failed", extra={"error_type": type(exc).__name__})
                failed.append("holidays")
            records += holiday_client.health_log
    finally:
        if own_weather:
            weather_client.close()
        if own_holiday and holiday_client is not None:
            holiday_client.close()

    out_path = args.data_dir / f"sales_enriched{suffix}.csv"
    enriched.to_csv(out_path, index=False)
    append_health_log(records, args.data_dir / "api_health_log.csv", run_id, args.input.name)

    summary = health_summary(records)
    missing = int(enriched["precip_mm"].isna().sum()) if "precip_mm" in enriched else len(enriched)
    logger.info(
        "run_complete",
        extra={"run_id": run_id, "output": out_path.name, "rows": len(enriched),
               "rows_without_weather": missing, "failed_sources": failed, **summary},
    )
    print(f"\nEnriched {len(enriched):,} rows -> {out_path}")
    print(f"API health: {summary}")
    if failed:
        print(f"WARNING: failed sources (rows left without data): {failed}")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())