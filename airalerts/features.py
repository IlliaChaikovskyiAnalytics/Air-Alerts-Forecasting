"""Feature engineering. Every feature in row (region, origin) uses data up to and including day `origin` only.

Layout of the modelling table ("direct multi-horizon"): one row per (region, origin day, horizon h) with the
features known at the origin, the horizon h and the weekday of the target day, and the targets observed on
origin + h. A single global model is trained on all regions at once (scaled targets, region id as a category).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import Panel
from . import targets as T

WINDOWS = (3, 7, 14, 28, 56, 90)
N_EPS, H_EPS = 0.25, 0.1           # smoothing in ratio features
SCALE_FLOOR_N, SCALE_FLOOR_H = 0.5, 0.3   # floor of the level used to scale targets (keeps tiny regions stable)


def _stack(wide: pd.DataFrame) -> pd.Series:
    s = wide.stack(future_stack=True)
    s.index = s.index.set_names(["date", "region"])
    return s.swaplevel().sort_index()


def _days_since(flag: pd.DataFrame, cap: int = 60) -> pd.DataFrame:
    """Days since the last 1 in each column (cap if never / long ago)."""
    pos = np.arange(len(flag))[:, None] * np.ones((1, flag.shape[1]))
    last = pd.DataFrame(np.where(flag.fillna(0).values == 1, pos, np.nan), index=flag.index, columns=flag.columns).ffill()
    since = (pd.DataFrame(pos, index=flag.index, columns=flag.columns) - last)
    return since.fillna(cap).clip(upper=cap)


def regional_features(panel: Panel, heavy: pd.Series) -> pd.DataFrame:
    """Region-level features indexed by (region, date)."""
    N = panel.daily["n_alerts"].unstack("region")
    H = panel.daily["alert_hours"].unstack("region")
    HV = heavy.unstack("region").reindex(N.index)
    f: dict[str, pd.DataFrame] = {}
    for w in WINDOWS:
        f[f"n_m{w}"] = N.rolling(w, min_periods=1).mean()
        f[f"h_m{w}"] = H.rolling(w, min_periods=1).mean()
    f["n_m365"] = N.rolling(365, min_periods=30).mean()
    f["h_m365"] = H.rolling(365, min_periods=30).mean()
    f["n_ewm14"] = N.ewm(halflife=14, adjust=False).mean()
    f["h_ewm14"] = H.ewm(halflife=14, adjust=False).mean()
    f["n_r7_28"] = f["n_m7"] / (f["n_m28"] + N_EPS)
    f["h_r7_28"] = f["h_m7"] / (f["h_m28"] + H_EPS)
    f["n_r28_90"] = f["n_m28"] / (f["n_m90"] + N_EPS)
    f["h_r28_90"] = f["h_m28"] / (f["h_m90"] + H_EPS)
    f["n_r28_365"] = f["n_m28"] / (f["n_m365"] + N_EPS)
    f["n_cv28"] = N.rolling(28, min_periods=7).std() / (f["n_m28"] + N_EPS)
    f["h_cv28"] = H.rolling(28, min_periods=7).std() / (f["h_m28"] + H_EPS)
    f["zero28"] = (N == 0).rolling(28, min_periods=1).mean()
    dur = H.rolling(28, min_periods=7).sum() / N.rolling(28, min_periods=7).sum().replace(0, np.nan)
    f["dur28"] = (dur * 60).fillna(30.0).clip(upper=600)
    f["n_last"], f["h_last"] = N, H
    f["hv_r7"] = HV.rolling(7, min_periods=1).mean()
    f["hv_r28"] = HV.rolling(28, min_periods=1).mean()
    f["since_heavy"] = _days_since(HV)
    # empirical 56-day quantiles (used by the baselines to build prediction intervals)
    for q in C.QUANTILES:
        qq = int(q * 100)
        f[f"n_q{qq}_56"] = N.rolling(56, min_periods=14).quantile(q)
        f[f"h_q{qq}_56"] = H.rolling(56, min_periods=14).quantile(q)
    df = pd.concat({k: _stack(v) for k, v in f.items()}, axis=1)
    df["region_id"] = df.index.get_level_values("region").map({r: i for i, r in enumerate(panel.regions)}).astype(int)
    return df


def national_features(nat: pd.DataFrame, massive: pd.Series) -> pd.DataFrame:
    """National features indexed by date (same for every region)."""
    f = pd.DataFrame(index=nat.index)
    for w in (7, 28, 90):
        f[f"rear_h_m{w}"] = nat["rear_hours"].rolling(w, min_periods=1).mean()
        f[f"all_n_m{w}"] = nat["all_alerts"].rolling(w, min_periods=1).mean()
    f["rear_h_r7_28"] = f["rear_h_m7"] / (f["rear_h_m28"] + 1.0)
    f["all_n_r7_28"] = f["all_n_m7"] / (f["all_n_m28"] + 1.0)
    f["rear_breadth_m7"] = nat["rear_breadth"].rolling(7, min_periods=1).mean()
    f["rear_h_last"] = nat["rear_hours"]
    mass = massive.reindex(nat.index)
    for w in (7, 14, 28, 60, 90):
        f[f"mass_r{w}"] = mass.rolling(w, min_periods=1).mean()
    f["mass_last"] = mass.fillna(0)
    f["since_massive"] = _days_since(mass.to_frame("m"))["m"]
    f["mass_any_prev7"] = mass.rolling(7, min_periods=1).max()
    return f


def make_direct(panel: Panel, heavy: pd.Series, massive: pd.Series, reg_feat: pd.DataFrame,
                nat_feat: pd.DataFrame, horizons: range = range(1, C.HORIZON + 1)) -> pd.DataFrame:
    """Rows (region, origin, h) with features at origin and targets at origin + h."""
    N = panel.daily["n_alerts"].unstack("region")
    H = panel.daily["alert_hours"].unstack("region")
    HV = heavy.unstack("region").reindex(N.index)
    parts = []
    first = N.index.min() + pd.Timedelta(days=90)           # need some history for the rolling features
    for h in horizons:
        d = reg_feat.copy()
        d["h"] = h
        d["y_n"] = _stack(N.shift(-h)).reindex(d.index)
        d["y_h"] = _stack(H.shift(-h)).reindex(d.index)
        d["y_heavy"] = _stack(HV.shift(-h)).reindex(d.index)
        d["origin"] = d.index.get_level_values("date")
        d["target_date"] = d["origin"] + pd.Timedelta(days=h)
        d["dow"] = d["target_date"].dt.dayofweek
        parts.append(d)
    D = pd.concat(parts, ignore_index=False)
    D = D.reset_index().rename(columns={"date": "origin_idx"}).drop(columns="origin_idx")
    D = D[D["origin"] >= first].reset_index(drop=True)
    nf = nat_feat.reindex(D["origin"]).reset_index(drop=True)
    D = pd.concat([D, nf], axis=1)
    D["scale_n"] = np.maximum(D["n_m28"], SCALE_FLOOR_N)
    D["scale_h"] = np.maximum(D["h_m28"], SCALE_FLOOR_H)
    for c in D.select_dtypes("float64").columns:
        D[c] = D[c].astype("float32")
    return D


def make_direct_national(nat: pd.DataFrame, massive: pd.Series, nat_feat: pd.DataFrame,
                         horizons: range = range(1, C.HORIZON + 1)) -> pd.DataFrame:
    """Rows (origin, h): national features at origin, massive-strike flag on origin + h, and whether any
    massive strike happens in origin+1 .. origin+HORIZON ('y_any')."""
    m = massive.reindex(nat.index)
    parts = []
    for h in horizons:
        d = nat_feat.copy()
        d["h"] = h
        d["origin"] = d.index
        d["target_date"] = d.index + pd.Timedelta(days=h)
        d["dow"] = d["target_date"].dt.dayofweek
        d["y_mass"] = m.shift(-h)
        parts.append(d)
    D = pd.concat(parts, ignore_index=True)
    fut = pd.concat([m.shift(-h) for h in range(1, C.HORIZON + 1)], axis=1)
    any7 = fut.max(axis=1).where(fut.notna().all(axis=1))
    D["y_any"] = D["origin"].map(any7)
    D = D[D["origin"] >= nat.index.min() + pd.Timedelta(days=90)].reset_index(drop=True)
    return D


def build_all(panel: Panel):
    """Convenience: labels + features + both modelling tables."""
    heavy = T.heavy_day_labels(panel.daily)
    nat = T.national_series(panel.daily, panel.regions)
    massive = T.massive_strike_labels(nat)
    reg_feat = regional_features(panel, heavy)
    nat_feat = national_features(nat, massive)
    D = make_direct(panel, heavy, massive, reg_feat, nat_feat)
    DN = make_direct_national(nat, massive, nat_feat)
    return dict(heavy=heavy, nat=nat, massive=massive, reg_feat=reg_feat, nat_feat=nat_feat, D=D, DN=DN)
