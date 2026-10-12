"""Production forecast: fit on everything observed up to the last full day and predict the next HORIZON days."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import baselines as B
from . import config as C
from . import models as M
from .backtest import candidate_regional
from .data import Panel


def make_forecast(D: pd.DataFrame, DN: pd.DataFrame, panel: Panel, selection: dict[str, str]) -> dict[str, pd.DataFrame]:
    """selection: {'n', 'h', 'heavy', 'mass_day', 'any7'} -> name of the model to publish for each target."""
    T = panel.last_day
    lo = T - pd.Timedelta(days=C.TRAIN_WINDOW_DAYS)
    tr = D[(D["target_date"] <= T) & (D["target_date"] > lo)]
    assert tr["target_date"].max() <= T
    te = D[D["origin"] == T].sort_values(["region", "h"]).reset_index(drop=True)
    assert len(te) == len(panel.regions) * C.HORIZON, "origin rows missing"
    rf = M.RegionalForecaster().fit(tr)
    cont, heavy = candidate_regional(rf, te)

    out = te[["region", "h", "target_date"]].rename(columns={"target_date": "date"}).copy()
    out["weekday"] = out["date"].dt.strftime("%a")
    for key, label in (("n", "alerts"), ("h", "hours")):
        c = cont[selection[key]]
        out[f"{label}_expected"] = c[f"{key}_mean"].values
        for q in C.QUANTILES:
            out[f"{label}_p{int(q * 100)}"] = c[f"{key}_q{int(q * 100)}"].values
    out["avg_alert_minutes"] = np.where(out["alerts_expected"] > 0.05, out["hours_expected"] / out["alerts_expected"] * 60, np.nan)
    out["p_heavy_day"] = np.asarray(heavy[selection["heavy"]])
    out["model_alerts"], out["model_hours"], out["model_heavy"] = selection["n"], selection["h"], selection["heavy"]
    # all candidates for transparency
    cand = []
    for name, c in cont.items():
        cand.append(te[["region", "h", "target_date"]].assign(model=name, alerts=c["n_mean"].values, hours=c["h_mean"].values,
                                                               p_heavy=np.asarray(heavy.get(name, heavy["GBM"])) if name in heavy else np.nan))

    # ---- national: massive-strike probabilities ----
    trN = DN[(DN["target_date"] <= T) & (DN["target_date"] > lo)]
    nf = M.NationalForecaster().fit(trN)
    clim = trN["y_mass"].mean()
    teN = DN[DN["origin"] == T].sort_values("h").reset_index(drop=True)
    day_p = {**B.massive_baselines(teN, clim), **nf.predict_day(teN)}
    teA = DN[DN["origin"] == T].drop_duplicates("origin")
    any_p = {**B.any7_baselines(teA), **nf.predict_any(teA)}
    nat = teN[["h", "target_date"]].rename(columns={"target_date": "date"}).copy()
    nat["weekday"] = nat["date"].dt.strftime("%a")
    nat["p_massive_day"] = day_p[selection["mass_day"]]
    for k, v in day_p.items():
        nat[f"p[{k}]"] = v
    any_summary = pd.DataFrame({"model": list(any_p), "p_any_massive_next7": [float(v[0]) for v in any_p.values()]})
    any_summary["selected"] = any_summary["model"] == selection["any7"]
    return {"regional": out, "national_days": nat, "national_any7": any_summary, "candidates": pd.concat(cand, ignore_index=True)}
