"""Thin CLI wrapper. Run: python scripts/simulate_sales.py --help"""
from vellamo_pipeline.simulate import main

if __name__ == "__main__":
    raise SystemExit(main())