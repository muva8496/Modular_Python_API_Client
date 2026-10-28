"""Test whether weather and payday patterns show up in the enriched sales.

    python scripts/analyse_weather.py                                  # real workbook
    python scripts/analyse_weather.py --input data/processed/sales_enriched_sim_v2.csv \
        --truth data/processed/ground_truth_sim_v2.json                # recovery test

Method: permutation tests. Rain labels are shuffled WITHIN each city (so city differences
and seasonality are respected); payday labels are shuffled within each month. The p-value is
the share of shuffles that produce an effect at least as large as the observed one.

Rainy day = city-day with precip_mm >= threshold (an analyst choice; default 1.0 mm).
The volume test is calendar-adjusted (city x month x weekday x payday); the share and payday tests
are raw. None of this is causal. Treat as screening.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "sales_enriched.csv"
N_PERM = 2000


def city_day_table(df: pd.DataFrame, weather: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per (City, Date) over the full weather grid. Days with no sales count as 0 rows."""
    df = df.copy()
    df["in_store"] = (df["Sales Channel"] == "In-Store").astype(int)
    counts = (
        df.groupby(["City", "Date"])
        .agg(rows=("sale_id", "size"), in_store=("in_store", "sum"), sales_kes=("Total Sales (KES)", "sum"))
        .reset_index()
    )
    grid = weather.copy()
    grid["Date"] = pd.to_datetime(grid["Date"])
    grid = grid[grid["City"].isin(df["City"].unique())
                & grid["Date"].between(df["Date"].min(), df["Date"].max())]
    grid = grid.dropna(subset=["precip_mm"])
    daily = grid.merge(counts, on=["City", "Date"], how="left")
    for col in ("rows", "in_store", "sales_kes"):
        daily[col] = daily[col].fillna(0)
    daily["rainy"] = daily["precip_mm"] >= threshold
    return daily[["City", "Date", "rows", "in_store", "precip_mm", "sales_kes", "rainy"]]


def calendar_baseline(daily: pd.DataFrame, payday_from: int = 25, payday_to: int = 3, iters: int = 25) -> np.ndarray:
    """Expected rows per city-day from calendar effects only (city x month x weekday x payday).

    Multiplicative main-effects fit by iterative proportional fitting. It is computed WITHOUT
    looking at rain labels, so it can be reused unchanged inside the permutation test. Because
    rainy days are included in the fit, estimates are very slightly pulled toward zero.
    """
    d = daily["Date"]
    keys = [
        daily["City"].to_numpy(),
        d.dt.month.to_numpy(),
        d.dt.dayofweek.to_numpy(),
        ((d.dt.day >= payday_from) | (d.dt.day <= payday_to)).to_numpy(),
    ]
    rows = daily["rows"].to_numpy(float)
    base = np.full(len(rows), rows.mean())
    for _ in range(iters):
        for key in keys:
            for level in pd.unique(key):
                m = key == level
                total = base[m].sum()
                if total > 0:
                    base[m] *= rows[m].sum() / total
    return base


def _adjusted_ratio(rows: np.ndarray, base: np.ndarray, rainy: np.ndarray) -> float:
    r, d = rainy, ~rainy
    if r.sum() == 0 or d.sum() == 0 or base[r].sum() == 0 or base[d].sum() == 0:
        return np.nan
    return (rows[r].sum() / base[r].sum()) / (rows[d].sum() / base[d].sum()) - 1


def _pooled_share_diff(instore: np.ndarray, total: np.ndarray, rainy: np.ndarray) -> float:
    r, d = rainy, ~rainy
    if total[r].sum() == 0 or total[d].sum() == 0:
        return np.nan
    return instore[r].sum() / total[r].sum() - instore[d].sum() / total[d].sum()


def _mean_ratio(rows: np.ndarray, rainy: np.ndarray) -> float:
    if rainy.sum() == 0 or (~rainy).sum() == 0:
        return np.nan
    return rows[rainy].mean() / rows[~rainy].mean() - 1


def permute_within_groups(groups: list[np.ndarray], stat_fn, observed: float, rng, n_perm: int) -> float:
    """Two-sided permutation p-value; ``groups`` are index arrays shuffled independently."""
    exceed = 0
    for _ in range(n_perm):
        stat = stat_fn(groups, rng)
        if abs(stat) >= abs(observed) - 1e-12:
            exceed += 1
    return (exceed + 1) / (n_perm + 1)


def rain_tests(daily: pd.DataFrame, rng: np.random.Generator, n_perm: int) -> list[dict]:
    cities = daily["City"].to_numpy()
    rows = daily["rows"].to_numpy(float)
    instore = daily["in_store"].to_numpy(float)
    rainy = daily["rainy"].to_numpy(bool)
    groups = [np.where(cities == c)[0] for c in np.unique(cities)]

    def shuffled(groups_, rng_):
        perm = rainy.copy()
        for idx in groups_:
            perm[idx] = rng_.permutation(rainy[idx])
        return perm

    obs_share = _pooled_share_diff(instore, rows, rainy)
    base = calendar_baseline(daily)
    obs_vol = _adjusted_ratio(rows, base, rainy)
    p_share = permute_within_groups(groups, lambda g, r: _pooled_share_diff(instore, rows, shuffled(g, r)),
                                    obs_share, rng, n_perm)
    p_vol = permute_within_groups(groups, lambda g, r: _adjusted_ratio(rows, base, shuffled(g, r)),
                                  obs_vol, rng, n_perm)
    return [
        {"metric": "rain_in_store_share_shift", "estimate": obs_share, "p_value": p_share,
         "n_rainy_city_days": int(rainy.sum()), "n_dry_city_days": int((~rainy).sum()),
         "meaning": "In-Store share on rainy days minus dry days (absolute, pooled)"},
        {"metric": "rain_volume_change", "estimate": obs_vol, "p_value": p_vol,
         "n_rainy_city_days": int(rainy.sum()), "n_dry_city_days": int((~rainy).sum()),
         "meaning": "Rows per city-day, rainy vs dry, adjusted for city/month/weekday/payday (relative change)"},
    ]


def payday_test(df: pd.DataFrame, rng: np.random.Generator, n_perm: int,
                from_day: int = 25, to_day: int = 3) -> dict:
    full = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    daily = (df.groupby("Date").size().reindex(full, fill_value=0)
             .rename_axis("Date").rename("rows").reset_index())
    daily["month"] = daily["Date"].dt.month
    daily["payday"] = (daily["Date"].dt.day >= from_day) | (daily["Date"].dt.day <= to_day)
    rows = daily["rows"].to_numpy(float)
    flag = daily["payday"].to_numpy(bool)
    groups = [np.where(daily["month"].to_numpy() == m)[0] for m in sorted(daily["month"].unique())]
    obs = _mean_ratio(rows, flag)

    def stat(groups_, rng_):
        perm = flag.copy()
        for idx in groups_:
            perm[idx] = rng_.permutation(flag[idx])
        return _mean_ratio(rows, perm)

    p = permute_within_groups(groups, stat, obs, rng, n_perm)
    return {"metric": "payday_volume_uplift", "estimate": obs, "p_value": p,
            "n_rainy_city_days": int(flag.sum()), "n_dry_city_days": int((~flag).sum()),
            "meaning": f"Mean daily rows, day {from_day}-31 and 1-{to_day} vs other days (relative change)"}


def attach_truth(results: pd.DataFrame, truth_path: Optional[Path]) -> pd.DataFrame:
    results = results.copy()
    results["injected_truth"] = np.nan
    if truth_path and truth_path.exists():
        injected = json.loads(truth_path.read_text())["injected"]
        mapping = {
            "rain_in_store_share_shift": injected["rain_in_store_share_shift"],
            "rain_volume_change": injected["rain_volume_change"],
            "payday_volume_uplift": injected["payday_volume_uplift"],
        }
        results["injected_truth"] = results["metric"].map(mapping)
    return results


def plot_channel_share(daily: pd.DataFrame, out: Path) -> None:
    g = daily.groupby(["City", "rainy"]).agg(rows=("rows", "sum"), in_store=("in_store", "sum")).reset_index()
    g["share"] = g["in_store"] / g["rows"]
    piv = g.pivot(index="City", columns="rainy", values="share").rename(columns={False: "Dry", True: "Rainy"})
    ax = piv.plot(kind="bar", figsize=(7, 4), color=["#c9a227", "#2f5d50"], rot=0)
    ax.set_ylabel("In-Store share of sales rows")
    ax.set_title("In-Store share by city: dry vs rainy days")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--weather", type=Path, default=None,
                        help="weather_daily.csv (default: next to the input file)")
    parser.add_argument("--truth", type=Path, default=None, help="ground_truth_*.json (simulated data only)")
    parser.add_argument("--threshold", type=float, default=1.0, help="rainy day if precip_mm >= this")
    parser.add_argument("--perms", type=int, default=N_PERM)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path,)