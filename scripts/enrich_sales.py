"""Thin CLI wrapper. Run: python scripts/enrich_sales.py --help"""
from vellamo_pipeline.enrich import main

if __name__ == "__main__":
    raise SystemExit(main())