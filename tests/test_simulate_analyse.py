import json

import numpy as np
import pandas as pd
import pytest

from vellamo_pipeline import analyse as A
from vellamo_pipeline import simulate as S
from vellamo_pipeline.loader import integrity_report


def as_ids(df):
    return df.rename(columns={"..": "sale_id"})


def test_simulation_is_clean_labelled_and_reproducible(tiny_sales_xlsx):
    a, *_rest, truth = S.simulate(tiny_sales_xlsx, seed=11)
    b, *_ = S.simulate(tiny_sales_xlsx, seed=11)
    c, *_ = S.simulate(tiny_sales_xlsx, seed=12)
    assert a.equals(b) and not a.equals(c)
    assert set(a["data_source"]) == {"SIMULATED_v2"}
    assert all(v == 0 for k, v in integrity_report(as_ids(a)).items() if k != "rows")
    assert truth["weather_effect_injected"] is False
    assert truth["injected"]["rain_in_store_share_shift"] == 0.0


def _weather_grid(days=20):
    dates = pd.date_range("2025-01-01", periods=days)
    return pd.DataFrame([(c, d, 5.0) for c in ["Nairobi", "Eldoret"] for d in dates],
                        columns=["City", "Date", "precip_mm"])


def test_simulation_with_weather_flags_the_injection(tiny_sales_xlsx, tmp_path):
    path = tmp_path / "w.csv"
    _weather_grid().to_csv(path, index=False)
    *_, truth = S.simulate(tiny_sales_xlsx, path, seed=3)
    assert truth["weather_effect_injected"] is True
    assert truth["injected"]["rain_in_store_share_shift"] == -0.08


def test_duplicate_weather_rows_are_rejected_clearly(tiny_sales_xlsx, tmp_path):
    path = tmp_path / "w.csv"
    grid = _weather_grid()
    pd.concat([grid, grid.head(3)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        S.simulate(tiny_sales_xlsx, path, seed=3)


def test_workbook_and_truth_files_are_written(tiny_sales_xlsx, tmp_path):
    out = tmp_path / "vellamo_sim_v2.xlsx"
    assert S.main(["--source", str(tiny_sales_xlsx), "--out", str(out)]) == 0
    assert out.exists() and (tmp_path / "ground_truth_sim_v2.json").exists()
    with pd.ExcelFile(out) as xl:
        sheets = xl.sheet_names
    assert sheets == ["Sales_Data", "Branches", "Salespeople", "README_SIMULATED"]
    assert "SYNTHETIC" in json.loads((tmp_path / "ground_truth_sim_v2.json").read_text())["WARNING"]


def test_city_day_table_keeps_days_with_zero_sales():
    sales = pd.DataFrame({
        "sale_id": [1, 2], "City": ["Nairobi", "Nairobi"],
        "Date": [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-04")],
        "Sales Channel": ["In-Store", "Online"], "Total Sales (KES)": [1000, 500],
    })
    weather = pd.DataFrame({"City": "Nairobi", "Date": pd.date_range("2025-01-01", periods=5),
                            "precip_mm": [0.0, 2.0, 0.0, 0.0, 0.0]})
    daily = A.city_day_table(sales, weather, 1.0)
    assert daily["Date"].dt.day.tolist() == [2, 3, 4]       # clipped to the sales date range
    assert daily["rows"].tolist() == [1, 0, 1]               # the empty day is kept as 0
    assert daily["rainy"].tolist() == [True, False, False]


def _synthetic_daily(effect, seed):
    rng = np.random.default_rng(seed)
    cities = ["A", "B", "C", "D", "E"]
    dates = pd.date_range("2025-01-01", periods=365)
    frame = pd.DataFrame([(c, d) for c in cities for d in dates], columns=["City", "Date"])
    frame["rainy"] = rng.random(len(frame)) < 0.3
    frame["rows"] = rng.poisson(3, len(frame)).astype(float)
    p_instore = np.where(frame["rainy"], 0.60 + effect, 0.60)
    frame["in_store"] = rng.binomial(frame["rows"].astype(int), p_instore).astype(float)
    frame["precip_mm"] = np.where(frame["rainy"], 5.0, 0.0)
    frame["sales_kes"] = frame["rows"] * 1000
    return frame


def test_rain_test_detects_a_real_share_effect():
    res = {r["metric"]: r for r in A.rain_tests(_synthetic_daily(-0.10, 1), np.random.default_rng(0), 500)}
    r = res["rain_in_store_share_shift"]
    assert -0.15 < r["estimate"] < -0.05 and r["p_value"] < 0.01


def test_rain_test_does_not_invent_an_effect_from_noise():
    res = {r["metric"]: r for r in A.rain_tests(_synthetic_daily(0.0, 5), np.random.default_rng(0), 500)}
    assert res["rain_in_store_share_shift"]["p_value"] > 0.05
    assert res["rain_volume_change"]["p_value"] > 0.05


def test_calendar_baseline_preserves_total_volume():
    daily = _synthetic_daily(0.0, 2)
    base = A.calendar_baseline(daily)
    assert abs(base.sum() - daily["rows"].sum()) / daily["rows"].sum() < 0.01


def test_attach_truth_maps_injected_values(tmp_path):
    truth = tmp_path / "t.json"
    truth.write_text(json.dumps({"injected": {"rain_in_store_share_shift": -0.08,
                                              "rain_volume_change": -0.05, "payday_volume_uplift": 0.2}}))
    res = pd.DataFrame({"metric": ["rain_in_store_share_shift", "payday_volume_uplift"], "estimate": [0, 0]})
    out = A.attach_truth(res, truth)
    assert out["injected_truth"].tolist() == [-0.08, 0.2]
    assert A.attach_truth(res, None)["injected_truth"].isna().all()