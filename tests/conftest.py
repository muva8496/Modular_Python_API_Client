import datetime as dt

import httpx
import numpy as np
import pandas as pd
import pytest

from api_client import ClientConfig, WeatherClient

FAST_CFG = ClientConfig(base_url="https://archive-api.open-meteo.test", max_attempts=4,
                        backoff_base_s=0.01, jitter=False)


def fake_open_meteo(request: httpx.Request) -> httpx.Response:
    """Deterministic fake of the Open-Meteo archive endpoint (seeded by latitude)."""
    q = request.url.params
    start, end = dt.date.fromisoformat(q["start_date"]), dt.date.fromisoformat(q["end_date"])
    n = (end - start).days + 1
    rng = np.random.default_rng(int(abs(float(q["latitude"])) * 1000))
    rain = np.where(rng.random(n) < 0.3, rng.gamma(2.0, 4.0, n), 0.0).round(1)
    return httpx.Response(200, json={
        "latitude": float(q["latitude"]), "longitude": float(q["longitude"]),
        "daily": {
            "time": [(start + dt.timedelta(days=i)).isoformat() for i in range(n)],
            "temperature_2m_max": (24 + rng.normal(0, 2, n)).round(1).tolist(),
            "temperature_2m_min": (13 + rng.normal(0, 2, n)).round(1).tolist(),
            "precipitation_sum": rain.tolist(),
        },
    })


@pytest.fixture
def make_weather_client():
    def _make(handler=fake_open_meteo):
        return WeatherClient(FAST_CFG, http_transport=httpx.MockTransport(handler), sleep=lambda s: None)
    return _make


@pytest.fixture
def tiny_sales_xlsx(tmp_path):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2025-01-01", "2025-01-20")
    rows = []
    sid = 100001
    for d in dates:
        for city, branch in [("Nairobi", "Vellamo CBD"), ("Eldoret", "Vellamo Rupa's Mall")]:
            for _ in range(2):
                qty, price, cost = int(rng.integers(1, 4)), 2000, 1100
                rows.append({"..": sid, "Date": d, "City": city, "Branch": branch,
                             "Category": "Shoes", "Product": "Sneakers", "Quantity": qty,
                             "Unit Price (KES)": price, "Unit Cost (KES)": cost,
                             "Total Sales (KES)": qty * price, "Total Cost (KES)": qty * cost,
                             "Profit (KES)": qty * (price - cost),
                             "Sales Channel": str(rng.choice(["In-Store", "Online", "Social Media"])),
                             "Payment Method": "M-Pesa", "Customer Gender": "Female",
                             "Customer Age Group": "25-34", "Salesperson": "Achieng M."})
                sid += 1
    path = tmp_path / "tiny.xlsx"
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="Sales_Data", index=False)
        pd.DataFrame({"Branch": ["Vellamo CBD", "Vellamo Rupa's Mall"], "City": ["Nairobi", "Eldoret"],
                      "Region": ["Central", "Rift Valley"], "Branch Manager": ["A", "B"],
                      "Year Opened": [2015, 2021], "Branch Size": ["Flagship", "Small"]}
                     ).to_excel(xw, sheet_name="Branches", index=False)
        pd.DataFrame({"Salesperson": ["Achieng M."], "Home Branch": ["Vellamo CBD"],
                      "Role": ["Sales Associate"], "Hire Year": [2020]}
                     ).to_excel(xw, sheet_name="Salespeople", index=False)
    return path