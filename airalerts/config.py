"""Central configuration. Edit values here; every other module reads them from this file."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "reports"
VOLUNTEER_FILE = "volunteer_data_en.csv"
DATA_URL = ("https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/"
            "main/datasets/" + VOLUNTEER_FILE)

TZ = "Europe/Kyiv"

# Regions that are not forecast. Luhansk has a permanent siren since 2022-04-04 and Crimea since 2022-12-10;
# neither carries information as a time series (Crimea is not present in the source file at all).
EXCLUDED_REGIONS = ("Luhanska oblast",)

# ---------------- forecasting set-up ----------------
HORIZON = 7                    # days ahead
QUANTILES = (0.1, 0.5, 0.9)    # prediction interval 10-90 %

# ---------------- target definitions (all derived from the alert data itself) ----------------
HEAVY_COL = "alert_hours"      # "alert_hours" (time under alert) or "n_alerts"
HEAVY_Q = 0.90                 # a region's day is 'heavy' if HEAVY_COL >= this quantile of its last 365 days
HEAVY_MIN = {"alert_hours": 3.0, "n_alerts": 3}   # ...and at least this large (quiet regions)
HEAVY_LOOKBACK = 365
HEAVY_MIN_PERIODS = 180

# 'Massive strike' day (national, a proxy: the data contain no attack labels). A day is flagged when the total time
# under alert summed over REAR regions (far from the front) is >= the MASS_Q quantile of the previous 365 days AND at least
# MASS_MIN_BREADTH rear regions had an alert that day. See docs/target_definitions.md for the validation.
FRONTLINE = ("Donetska oblast", "Zaporizka oblast", "Khersonska oblast", "Kharkivska oblast", "Sumska oblast",
             "Mykolaivska oblast", "Dnipropetrovska oblast", "Chernihivska oblast", "Odeska oblast")
MASS_Q = 0.90
MASS_MIN_BREADTH = 10
MASS_LOOKBACK = 365
MASS_MIN_PERIODS = 180         # first days without enough history get no label

# ---------------- modelling ----------------
TRAIN_WINDOW_DAYS = 730        # training rows: targets within this many days before the forecast date
SEED = 42

# ---------------- backtest ----------------
BACKTEST_DAYS = 180            # forecast origins cover this many recent days
REFIT_EVERY_DAYS = 14          # models are re-fitted on data up to the refit date, then used until the next one
LATEST_WINDOW_DAYS = 28        # 'latest period' reported separately (targets in the last N days)
N_BOOT = 500                   # block-bootstrap draws for confidence intervals
