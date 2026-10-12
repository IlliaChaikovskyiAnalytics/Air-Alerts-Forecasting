"""Entry point: python run_pipeline.py [--download] [--refresh-backtest]"""
import argparse

from airalerts import pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true", help="re-download the volunteer dataset first")
    ap.add_argument("--refresh-backtest", action="store_true", help="ignore the backtest cache")
    a = ap.parse_args()
    pipeline.run(download=a.download, refresh_backtest=a.refresh_backtest)
