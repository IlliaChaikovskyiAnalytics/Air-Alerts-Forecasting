"""Rolling-origin backtest.

Models are re-fitted every REFIT_EVERY_DAYS on data that ends at the refit date T (training rows are those whose TARGET day
is <= T and within TRAIN_WINDOW_DAYS), and are then used for every forecast origin from T to the next refit. Nothing observed
after T enters a fit; this is asserted in the loop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import baselines as B
from . import config as C
from . import models as M

ID_COLS = ["region", "origin", "h", "target_date", "y_n", "y_h", "y_heavy", "scale_n", "scale_h"]
CONT = [f"{k}_{s}" for k in "nh" for s in ("mean", "q10", "q50", "q90")]


@dataclass
class BacktestResult:
    reg: pd.DataFrame        # long: model + id cols + CONT predictions (continuous targets)
    heavy: pd.DataFrame      # long: model + id cols + p
    nat_day: pd.DataFrame    # long: model, origin, h, target_date, y_mass, p
    nat_any: pd.DataFrame    # long: model, origin, y_any, p
    refit_dates: list


def candidate_regional(rf: M.RegionalForecaster, te: pd.DataFrame):
    """All candidate models for the rows `te`: returns (continuous dict, heavy dict)."""
    base = B.regional_baselines(te)
    g = rf.predict(te)
    cont = {**base, "GBM": g[CONT]}
    cont["Blend"] = (g[CONT] + base["EWMA 14d"][CONT]) / 2.0
    heavy = B.heavy_baselines(te)
    heavy["GBM"] = g["p_heavy"].values
    heavy["Blend"] = B.clip_p((g["p_heavy"].values + heavy["Recent 28d"]) / 2.0)
    return cont, heavy


def run_backtest(D: pd.DataFrame, DN: pd.DataFrame, last_day: pd.Timestamp, verbose: bool = True) -> BacktestResult:
    day = pd.Timedelta(days=1)
    first_origin = last_day - pd.Timedelta(days=C.BACKTEST_DAYS)
    refits = list(pd.date_range(first_origin, last_day - day, freq=f"{C.REFIT_EVERY_DAYS}D"))
    reg_parts, hv_parts, nd_parts, na_parts = [], [], [], []
    for i, T in enumerate(refits, 1):
        t0 = time.time()
        lo = T - pd.Timedelta(days=C.TRAIN_WINDOW_DAYS)
        tr = D[(D["target_date"] <= T) & (D["target_date"] > lo)]
        assert tr["target_date"].max() <= T, "leakage: a training target lies after the refit date"
        assert (tr["origin"] < T).all(), "leakage: a training origin is not before the refit date"
        te = D[(D["origin"] >= T) & (D["origin"] < T + pd.Timedelta(days=C.REFIT_EVERY_DAYS)) & (D["target_date"] <= last_day)]
        rf = M.RegionalForecaster().fit(tr)
        cont, heavy = candidate_regional(rf, te)
        ids = te[ID_COLS].reset_index(drop=True)
        for name, p in cont.items():
            reg_parts.append(pd.concat([ids, p.reset_index(drop=True)], axis=1).assign(model=name))
        for name, p in heavy.items():
            hv_parts.append(ids.assign(p=np.asarray(p), model=name))

        trN = DN[(DN["target_date"] <= T) & (DN["target_date"] > lo)]
        assert trN["target_date"].max() <= T
        nf = M.NationalForecaster().fit(trN)
        clim = trN["y_mass"].mean()
        teN = DN[(DN["origin"] >= T) & (DN["origin"] < T + pd.Timedelta(days=C.REFIT_EVERY_DAYS)) & (DN["target_date"] <= last_day)]
        day_p = {**B.massive_baselines(teN, clim), **nf.predict_day(teN)}
        for name, p in day_p.items():
            nd_parts.append(teN[["origin", "h", "target_date", "y_mass"]].assign(p=p, model=name).reset_index(drop=True))
        teA = DN[(DN["origin"] >= T) & (DN["origin"] < T + pd.Timedelta(days=C.REFIT_EVERY_DAYS))].drop_duplicates("origin")
        teA = teA[teA["origin"] + pd.Timedelta(days=C.HORIZON) <= last_day]
        if len(teA):
            any_p = {**B.any7_baselines(teA), **nf.predict_any(teA)}
            for name, p in any_p.items():
                na_parts.append(teA[["origin", "y_any"]].assign(p=p, model=name).reset_index(drop=True))
        if verbose:
            print(f"  refit {i:2d}/{len(refits)} at {T.date()}  train rows {len(tr):,}  ({time.time() - t0:.0f}s)", flush=True)

    return BacktestResult(reg=pd.concat(reg_parts, ignore_index=True), heavy=pd.concat(hv_parts, ignore_index=True),
                          nat_day=pd.concat(nd_parts, ignore_index=True),
                          nat_any=pd.concat(na_parts, ignore_index=True) if na_parts else pd.DataFrame(),
                          refit_dates=refits)
