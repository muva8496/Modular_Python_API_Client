"""Generate a SIMULATED Vellamo workbook (v2) with documented Kenyan-market assumptions.

THIS IS SYNTHETIC DATA. Every effect below is an ASSUMPTION injected on purpose, with the
true value written to ground_truth_<tag>.json so the analysis can be tested for recovery.
Never present findings from this file as facts about the real market.

    python scripts/simulate_sales.py                       # no weather effect
    python scripts/simulate_sales.py --weather data/processed/weather_daily.csv

The real source workbook is only READ (for the product catalogue, price levels and
demographic mix). It is never modified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .loader import integrity_report, load_lookups, load_sales

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "raw" / "vellamo_fashion_sales_fy2025_with_lookups.xlsx"
DEFAULT_OUT = ROOT / "data" / "processed" / "vellamo_sim_v2.xlsx"
SEED = 2025
DATA_SOURCE_TAG = "SIMULATED_v2"

# ---------------------------------------------------------------------------------------
# ASSUMPTIONS (all unverified; calibrate against real data before using for decisions)
# ---------------------------------------------------------------------------------------
PARAMS: dict[str, Any] = {
    "seasonal_index": {1: 0.95, 2: 0.90, 3: 0.95, 4: 1.00, 5: 0.95, 6: 0.90,
                       7: 0.95, 8: 1.00, 9: 0.95, 10: 1.00, 11: 1.10, 12: 1.45},
    "day_of_week_multiplier": {0: 0.85, 1: 0.85, 2: 0.85, 3: 0.85, 4: 1.15, 5: 1.30, 6: 1.15},
    "payday_uplift": 0.20,                       # volume +20% on days 25..31 and 1..3
    "payday_days": {"from_day": 25, "to_day": 3},
    "branch_size_volume_weight": {"Flagship": 1.4, "Large": 1.2, "Medium": 1.0, "Small": 0.8},
    "price_tier_multiplier": {"Flagship": 1.10, "Large": 1.05, "Medium": 1.00, "Small": 0.95},
    "price_noise_sd": 0.05,
    "easter_weekend": {"start": "2025-04-18", "end": "2025-04-21", "volume_multiplier": 1.15},
    "festive_window": {"start": "2025-12-20", "end": "2025-12-26", "volume_multiplier": 1.20},
    "black_friday_window": {"start": "2025-11-24", "end": "2025-11-30",
                            "volume_multiplier": 1.25, "price_multiplier": 0.90},
    "valentines_window": {"start": "2025-02-12", "end": "2025-02-14", "category_multiplier": 1.8,
                          "categories": ["Women's Wear", "Jewelry", "Bags"]},
    # School terms: verify against the Ministry of Education calendar.
    "back_to_school_windows": [["2025-01-01", "2025-01-20"], ["2025-04-28", "2025-05-12"],
                               ["2025-08-25", "2025-09-08"]],
    "back_to_school_kids_wear_multiplier": 2.0,
    "channel_probs_by_city": {            # In-Store, Online, Social Media
        "Nairobi": [0.55, 0.28, 0.17], "Mombasa": [0.58, 0.25, 0.17],
        "Kisumu": [0.62, 0.22, 0.16], "Nakuru": [0.64, 0.21, 0.15],
        "Eldoret": [0.68, 0.19, 0.13],
    },
    "payment_probs_by_channel": {         # M-Pesa, Cash (incl. cash on delivery), Card
        "In-Store": [0.55, 0.25, 0.20], "Online": [0.70, 0.08, 0.22],
        "Social Media": [0.82, 0.10, 0.08],
    },
    # Weather effect: only injected when a weather file is supplied.
    "rain_threshold_mm": 1.0,
    "rain_instore_share_shift": -0.08,    # absolute shift; mass moves 75% Online / 25% Social
    "rain_volume_multiplier": 0.95,
}
CHANNELS = ["In-Store", "Online", "Social Media"]
PAYMENTS = ["M-Pesa", "Cash", "Card"]


def _in_window(dates: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))


def _round_to(x: np.ndarray, step: int) -> np.ndarray:
    return (np.round(x / step) * step).astype(int)


def build_catalogue(src: pd.DataFrame) -> dict[str, Any]:
    products = (
        src.groupby(["Category", "Product"])
        .agg(n=("sale_id", "size"), price=("Unit Price (KES)", "median"), cost=("Unit Cost (KES)", "median"))
        .reset_index()
    )
    cat_weights = src["Category"].value_counts(normalize=True)
    qty = src["Quantity"].value_counts(normalize=True).sort_index()
    demo: dict[str, dict[str, pd.Series]] = {}
    for cat, grp in src.groupby("Category"):
        demo[cat] = {
            "gender": grp["Customer Gender"].value_counts(normalize=True),
            "age": grp["Customer Age Group"].value_counts(normalize=True),
        }
    return {"products": products, "cat_weights": cat_weights, "qty": qty, "demo": demo}


def load_rain_flags(weather_csv: Optional[Path], dates: pd.DatetimeIndex, cities: list[str], threshold: float):
    """Return dict city -> boolean array over dates (all False without weather)."""
    flags = {c: np.zeros(len(dates), dtype=bool) for c in cities}
    if weather_csv is None:
        return flags, False
    w = pd.read_csv(weather_csv, parse_dates=["Date"])
    if w.duplicated(["City", "Date"]).any():
        raise ValueError("weather file has duplicate (City, Date) rows; expected one row per city per day")
    for city in cities:
        sub = w[w["City"] == city].set_index("Date")["precip_mm"]
        flags[city] = (sub.reindex(dates).fillna(0.0) >= threshold).to_numpy()
    return flags, True


def simulate(
    source: Path,
    weather_csv: Optional[Path] = None,
    seed: int = SEED,
    target_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    P = PARAMS
    rng = np.random.default_rng(seed)
    src = load_sales(source)
    branches, people = load_lookups(source)
    cat = build_catalogue(src)
    target = target_rows or len(src)

    dates = pd.date_range(src["Date"].min(), src["Date"].max(), freq="D")
    branch_names = branches["Branch"].tolist()
    branch_city = branches["City"].tolist()
    branch_size = branches["Branch Size"].tolist()
    cities = sorted(set(branch_city))
    rain_by_city, weather_used = load_rain_flags(weather_csv, dates, cities, P["rain_threshold_mm"])

    # ---- volume model -------------------------------------------------------------
    vol = np.array([P["seasonal_index"][d.month] * P["day_of_week_multiplier"][d.dayofweek] for d in dates])
    payday = np.array([(d.day >= P["payday_days"]["from_day"]) or (d.day <= P["payday_days"]["to_day"]) for d in dates])
    vol = vol * np.where(payday, 1 + P["payday_uplift"], 1.0)
    for key in ("easter_weekend", "festive_window", "black_friday_window"):
        w = P[key]
        vol = vol * np.where(_in_window(dates, w["start"], w["end"]), w["volume_multiplier"], 1.0)

    w_branch = np.array([P["branch_size_volume_weight"][s] for s in branch_size])
    rainy = np.stack([rain_by_city[c] for c in branch_city], axis=1)          # [days, branches]
    rain_mult = np.where(rainy, P["rain_volume_multiplier"], 1.0) if weather_used else np.ones_like(rainy, float)
    lam = vol[:, None] * w_branch[None, :] * rain_mult
    lam = lam * (target / lam.sum())
    counts = rng.poisson(lam)

    flat = np.repeat(np.arange(counts.size), counts.ravel())
    di, bi = flat // len(branch_names), flat % len(branch_names)
    n = len(flat)
    row_dates = dates[di]
    row_branch = np.array(branch_names)[bi]
    row_city = np.array(branch_city)[bi]
    row_size = np.array(branch_size)[bi]
    row_rainy = rainy[di, bi] if weather_used else np.zeros(n, dtype=bool)

    # ---- category -----------------------------------------------------------------
    cats = cat["cat_weights"].index.to_numpy()
    base_p = cat["cat_weights"].to_numpy()
    bts = np.zeros(len(dates), dtype=bool)
    for a, b in P["back_to_school_windows"]:
        bts |= _in_window(dates, a, b)
    val = _in_window(dates, P["valentines_window"]["start"], P["valentines_window"]["end"])
    row_cat = np.empty(n, dtype=object)
    for d_idx in np.unique(di):
        mask = di == d_idx
        p = base_p.copy()
        if bts[d_idx]:
            p = p * np.where(cats == "Kids Wear", P["back_to_school_kids_wear_multiplier"], 1.0)
        if val[d_idx]:
            p = p * np.where(np.isin(cats, P["valentines_window"]["categories"]),
                             P["valentines_window"]["category_multiplier"], 1.0)
        row_cat[mask] = rng.choice(cats, size=int(mask.sum()), p=p / p.sum())

    # ---- product, price, cost ------------------------------------------------------
    prod = cat["products"]
    row_product = np.empty(n, dtype=object)
    for c in cats:
        mask = row_cat == c
        sub = prod[prod["Category"] == c]
        row_product[mask] = rng.choice(sub["Product"].to_numpy(), size=int(mask.sum()),
                                       p=(sub["n"] / sub["n"].sum()).to_numpy())
    price_lookup = prod.set_index("Product")["price"]
    cost_lookup = prod.set_index("Product")["cost"]
    base_price = pd.Series(row_product).map(price_lookup).to_numpy(float)
    base_cost = pd.Series(row_product).map(cost_lookup).to_numpy(float)
    tier = pd.Series(row_size).map(P["price_tier_multiplier"]).to_numpy(float)
    bf = P["black_friday_window"]
    price_event = np.where(_in_window(row_dates, bf["start"], bf["end"]), bf["price_multiplier"], 1.0)
    noise = np.exp(rng.normal(0, P["price_noise_sd"], n))
    unit_price = np.maximum(_round_to(base_price * tier * price_event * noise, 50), 100)
    unit_cost = _round_to(base_cost * np.exp(rng.normal(0, 0.03, n)), 10)
    unit_cost = np.minimum(unit_cost, (unit_price * 0.9).astype(int))

    qty = rng.choice(cat["qty"].index.to_numpy(), size=n, p=cat["qty"].to_numpy())

    # ---- channel (with optional rain shift) and payment ----------------------------
    row_channel = np.empty(n, dtype=object)
    shift = P["rain_instore_share_shift"] if weather_used else 0.0
    for city in cities:
        for is_rain in (False, True):
            mask = (row_city == city) & (row_rainy == is_rain)
            if not mask.any():
                continue
            p = np.array(P["channel_probs_by_city"][city], dtype=float)
            if is_rain and weather_used:
                p = p + np.array([shift, -shift * 0.75, -shift * 0.25])
            row_channel[mask] = rng.choice(CHANNELS, size=int(mask.sum()), p=p / p.sum())
    row_payment = np.empty(n, dtype=object)
    for ch in CHANNELS:
        mask = row_channel == ch
        row_payment[mask] = rng.choice(PAYMENTS, size=int(mask.sum()), p=P["payment_probs_by_channel"][ch])

    # ---- demographics and staff ----------------------------------------------------
    row_gender = np.empty(n, dtype=object)
    row_age = np.empty(n, dtype=object)
    for c in cats:
        mask = row_cat == c
        g, a = cat["demo"][c]["gender"], cat["demo"][c]["age"]
        row_gender[mask] = rng.choice(g.index.to_numpy(), size=int(mask.sum()), p=g.to_numpy())
        row_age[mask] = rng.choice(a.index.to_numpy(), size=int(mask.sum()), p=a.to_numpy())
    row_staff = np.empty(n, dtype=object)
    all_staff = people["Salesperson"].to_numpy()
    for b in branch_names:
        mask = row_branch == b
        pool = people.loc[people["Home Branch"] == b, "Salesperson"].to_numpy()
        row_staff[mask] = rng.choice(pool if len(pool) else all_staff, size=int(mask.sum()))

    sales = pd.DataFrame({
        "..": 0, "Date": row_dates, "City": row_city, "Branch": row_branch, "Category": row_cat,
        "Product": row_product, "Quantity": qty, "Unit Price (KES)": unit_price,
        "Unit Cost (KES)": unit_cost, "Total Sales (KES)": qty * unit_price,
        "Total Cost (KES)": qty * unit_cost, "Profit (KES)": qty * (unit_price - unit_cost),
        "Sales Channel": row_channel, "Payment Method": row_payment, "Customer Gender": row_gender,
        "Customer Age Group": row_age, "Salesperson": row_staff, "data_source": DATA_SOURCE_TAG,
    }).sort_values(["Date", "Branch"], kind="stable").reset_index(drop=True)
    sales["Date"] = pd.to_datetime(sales["Date"]).dt.normalize()
    for col in ("Quantity", "Unit Price (KES)", "Unit Cost (KES)",
                "Total Sales (KES)", "Total Cost (KES)", "Profit (KES)"):
        sales[col] = sales[col].astype(int)
    sales[".."] = np.arange(100001, 100001 + len(sales))

    truth = {
        "data_source": DATA_SOURCE_TAG,
        "WARNING": "SYNTHETIC. These are injected assumptions, not measurements of the real market.",
        "seed": seed,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source_workbook": source.name,
        "rows": int(len(sales)),
        "weather_effect_injected": bool(weather_used),
        "injected": {
            "payday_volume_uplift": P["payday_uplift"],
            "rain_in_store_share_shift": P["rain_instore_share_shift"] if weather_used else 0.0,
            "rain_volume_change": (P["rain_volume_multiplier"] - 1) if weather_used else 0.0,
            "rain_threshold_mm": P["rain_threshold_mm"],
        },
        "params": P,
    }
    return sales, branches, people, truth


def write_workbook(path: Path, sales: pd.DataFrame, branches: pd.DataFrame, people: pd.DataFrame,
                   truth: dict[str, Any]) -> None:
    notes = pd.DataFrame({"README": [
        "SIMULATED DATA - not real Vellamo sales.",
        f"data_source = {truth['data_source']}; seed = {truth['seed']}; generated {truth['generated_at']}.",
        f"Built from the product catalogue and demographics of {truth['source_workbook']}.",
        "Injected effects and all assumptions are in ground_truth_*.json and docs/DATA_CARD_sim_v2.md.",
        "Do not report findings from this file as facts about the Kenyan market.",
    ]})
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        sales.to_excel(xw, sheet_name="Sales_Data", index=False)
        branches.to_excel(xw, sheet_name="Branches", index=False)
        people.to_excel(xw, sheet_name="Salespeople", index=False)
        notes.to_excel(xw, sheet_name="README_SIMULATED", index=False)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--weather", type=Path, default=None, help="weather_daily.csv from enrich_sales.py")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rows", type=int, default=None, help="target row count (default: same as source)")
    args = parser.parse_args(argv)

    sales, branches, people, truth = simulate(args.source, args.weather, args.seed, args.rows)
    write_workbook(args.out, sales, branches, people, truth)
    truth_path = args.out.with_name(f"ground_truth_{args.out.stem.replace('vellamo_', '')}.json")
    truth_path.write_text(json.dumps(truth, indent=2, default=str), encoding="utf-8")

    check = integrity_report(sales.rename(columns={"..": "sale_id"}))
    print(f"Simulated {len(sales):,} rows -> {args.out}")
    print(f"Ground truth -> {truth_path}")
    print(f"Weather effect injected: {truth['weather_effect_injected']}")
    print(f"Integrity: {check}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())