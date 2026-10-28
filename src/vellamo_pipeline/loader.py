"""Load the Vellamo workbook (real or simulated) into clean DataFrames."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_SALES_COLUMNS = [
    "sale_id", "Date", "City", "Branch", "Category", "Product", "Quantity",
    "Unit Price (KES)", "Unit Cost (KES)", "Total Sales (KES)", "Total Cost (KES)",
    "Profit (KES)", "Sales Channel", "Payment Method", "Customer Gender",
    "Customer Age Group", "Salesperson",
]


def load_sales(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sales_Data")
    df = df.rename(columns={"..": "sale_id"})
    missing = [c for c in REQUIRED_SALES_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Sales_Data is missing columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df


def load_lookups(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    branches = pd.read_excel(path, sheet_name="Branches")
    people = pd.read_excel(path, sheet_name="Salespeople")
    return branches, people


def integrity_report(df: pd.DataFrame) -> dict[str, int]:
    """Row-level arithmetic checks; zeros mean clean."""
    return {
        "rows": int(len(df)),
        "null_cells": int(df[REQUIRED_SALES_COLUMNS].isna().sum().sum()),
        "duplicate_ids": int(df["sale_id"].duplicated().sum()),
        "sales_mismatch": int((df["Quantity"] * df["Unit Price (KES)"] != df["Total Sales (KES)"]).sum()),
        "profit_mismatch": int((df["Total Sales (KES)"] - df["Total Cost (KES)"] != df["Profit (KES)"]).sum()),
    }