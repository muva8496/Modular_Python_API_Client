"""Thin CLI wrapper. Run: python scripts/analyse_weather.py --help"""
from vellamo_pipeline.analyse import main

if __name__ == "__main__":
    raise SystemExit(main())