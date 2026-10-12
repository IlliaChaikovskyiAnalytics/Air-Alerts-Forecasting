"""Target definitions derived from the alert data: heavy regional day and national massive-strike day.

Labels for day t use only data up to day t (thresholds come from the 365 days BEFORE t), so a label never
depends on anything that happens after the day it describes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def heavy_day_labels(daily: pd.DataFrame) -> pd.Series:
    """1 if a region's HEAVY_COL on day t is >= its HEAVY_Q quantile over days t-365..t-1 (and >= HEAVY_MIN)."""
    col = daily[C.HEAVY_COL].unstack("region")
    thr = (col.shift(1).rolling(C.HEAVY_LOOKBACK, min_periods=C.HEAVY_MIN_PERIODS).quantile(C.HEAVY_Q))
    thr = thr.clip(lower=C.HEAVY_MIN[C.HEAVY_COL])
    lab = (col >= thr).astype(float).where(thr.notna())
    return lab.stack(future_stack=True).rename("heavy").swaplevel().sort_index()


def rear_regions(regions: list[str]) -> list[str]:
    return [r for r in regions if r not in C.FRONTLINE]


def national_series(daily: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    """Daily national aggregates over rear regions."""
    rear = rear_regions(regions)
    h = daily["alert_hours"].unstack("region")[rear]
    n = daily["n_alerts"].unstack("region")[rear]
    out = pd.DataFrame({"rear_hours": h.sum(axis=1), "rear_alerts": n.sum(axis=1),
                        "rear_breadth": (h > 0).sum(axis=1)})
    out["all_alerts"] = daily["n_alerts"].unstack("region").sum(axis=1)
    out["all_hours"] = daily["alert_hours"].unstack("region").sum(axis=1)
    return out


def massive_strike_labels(nat: pd.DataFrame) -> pd.Series:
    """1 if rear alert-hours >= trailing-365d quantile and breadth >= MASS_MIN_BREADTH."""
    thr = nat["rear_hours"].shift(1).rolling(C.MASS_LOOKBACK, min_periods=C.MASS_MIN_PERIODS).quantile(C.MASS_Q)
    lab = ((nat["rear_hours"] >= thr) & (nat["rear_breadth"] >= C.MASS_MIN_BREADTH)).astype(float)
    return lab.where(thr.notna()).rename("massive")
