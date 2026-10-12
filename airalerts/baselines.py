"""Transparent benchmarks. A model is only useful if it beats these at the same horizon.

Regional daily n_alerts / alert_hours (mean + 10/50/90 % quantiles):
    Flat 28d  : mean of the last 28 days
    EWMA 3d/7d: same with shorter half-lives;  Fast-up 7/28d: max(EWMA 7d, EWMA 28d)
    EWMA 14d  : exponentially weighted mean, half-life 14 days  (adaptive level, the 'regime tracker')
  Prediction band = empirical 10/50/90 % quantiles of the last 56 days, rescaled to the baseline's level.
Regional heavy-day probability:
    Clim 10%  : the long-run share implied by the definition of 'heavy'
    Recent 28d: share of heavy days among the last 28 days
National massive-strike probability: Clim, Recent 28d frequency, and for the 7-day window an independence formula.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

EPS = 0.01


def clip_p(p):
    return np.clip(p, EPS, 1 - EPS)


def level_forecast(D: pd.DataFrame, key: str, level_col: str) -> pd.DataFrame:
    """Level baseline for target `key` ('n' or 'h') using `level_col` as the mean; band from 56-day quantiles."""
    base56 = D[f"{key}_m56"].replace(0, np.nan)
    ratio = (D[level_col] / base56).fillna(1.0).clip(0.2, 5.0)
    out = pd.DataFrame(index=D.index)
    out[f"{key}_mean"] = D[level_col]
    for q in C.QUANTILES:
        qq = int(q * 100)
        out[f"{key}_q{qq}"] = D[f"{key}_q{qq}_56"] * ratio
    return out


def regional_baselines(D: pd.DataFrame) -> dict[str, pd.DataFrame]:
    res = {}
    for name, (nc, hc) in {"Flat 28d": ("n_m28", "h_m28"), "EWMA 14d": ("n_ewm14", "h_ewm14"),
                           "EWMA 3d": ("n_ewm3", "h_ewm3"), "EWMA 7d": ("n_ewm7", "h_ewm7")}.items():
        res[name] = pd.concat([level_forecast(D, "n", nc), level_forecast(D, "h", hc)], axis=1)
    # asymmetric tracker: reacts quickly to increases (fast level when above the slow one), decays slowly otherwise.
    # Motivated by the diagnosed bias (under-forecasting in rising regimes only).
    D = D.assign(n_up=np.maximum(D["n_ewm7"], D["n_ewm28"]), h_up=np.maximum(D["h_ewm7"], D["h_ewm28"]))
    res["Fast-up 7/28d"] = pd.concat([level_forecast(D, "n", "n_up"), level_forecast(D, "h", "h_up")], axis=1)
    return res


def heavy_baselines(D: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"Clim 10%": np.full(len(D), clip_p(1 - C.HEAVY_Q)),
            "Recent 28d": clip_p(D["hv_r28"].fillna(1 - C.HEAVY_Q).values)}


def massive_baselines(DN: pd.DataFrame, clim: float) -> dict[str, np.ndarray]:
    """`clim` = share of massive days in the TRAINING window (passed in, so no future information is used)."""
    return {"Clim": np.full(len(DN), clip_p(clim)),
            "Recent 28d": clip_p(DN["mass_r28"].values),
            "Recent 90d": clip_p(DN["mass_r90"].values)}


def any7_baselines(DN: pd.DataFrame) -> dict[str, np.ndarray]:
    """P(at least one massive strike in the next 7 days) from the recent daily frequency, assuming independence."""
    return {"Recent 28d (indep.)": clip_p(1 - (1 - clip_p(DN["mass_r28"].values)) ** C.HORIZON),
            "Recent 90d (indep.)": clip_p(1 - (1 - clip_p(DN["mass_r90"].values)) ** C.HORIZON)}
