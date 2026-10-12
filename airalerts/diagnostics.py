"""Statistical diagnostics of the backtest: Ljung-Box (residual autocorrelation) and Diebold-Mariano (forecast comparison).

Both are implemented with numpy/scipy only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import metrics as MT

# fixed reference model per target (chosen a priori: the simplest adaptive-level baseline), not from the results
DM_REF = {"alerts": "Flat 28d", "hours": "Flat 28d", "heavy": "Recent 28d", "massive_day": "Clim",
          "massive_any7": "Recent 28d (indep.)"}


def ljung_box(x: np.ndarray, lags: int) -> tuple[float, float]:
    """Q = n(n+2) sum_{k=1..m} r_k^2/(n-k); p-value from chi2(m)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n <= lags + 2 or np.allclose(x, x.mean()):
        return np.nan, np.nan
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    r = np.array([(xc[k:] * xc[:-k]).sum() / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * (r ** 2 / (n - np.arange(1, lags + 1))).sum()
    return float(q), float(stats.chi2.sf(q, lags))


def ljung_box_table(bt, sel: dict, lags: int = 7, horizons=(1, 7)) -> pd.DataFrame:
    """Per region: Ljung-Box on the standardised forecast error of the selected model (h-step errors ordered by target day).
    Reports the share of regions with p < 0.05 (about 5% is expected under no autocorrelation)."""
    rows = []
    for key, name in (("n", "alerts"), ("h", "hours")):
        df = MT.add_reg_losses(bt.reg, key)
        df = df[df["model"] == sel[key]]
        df = df.assign(z=df["err"] / df[f"scale_{key}"])
        for h in horizons:
            d = df[df["h"] == h].sort_values("target_date")
            ps = []
            for _, g in d.groupby("region"):
                ps.append(ljung_box(g["z"].to_numpy(), lags)[1])
            ps = np.array([p for p in ps if not np.isnan(p)])
            nat = d.groupby("target_date")["z"].mean()
            q_nat, p_nat = ljung_box(nat.to_numpy(), lags)
            rows.append(dict(target=name, model=sel[key], h=h, lags=lags, regions=len(ps),
                             share_p_lt_05=float((ps < 0.05).mean()) if len(ps) else np.nan,
                             median_p=float(np.median(ps)) if len(ps) else np.nan,
                             mean_resid_z=float(d["z"].mean()), national_Q=q_nat, national_p=p_nat))
    return pd.DataFrame(rows)


def dm_test(d: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano on a loss differential series d_t = L_ref - L_model (positive: model better).
    Bartlett-kernel HAC variance with lag h-1, Harvey-Leybourne-Newbold small-sample correction, Student-t(T-1)."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    T = len(d)
    if T < 10 or np.allclose(d, 0):
        return np.nan, np.nan
    dc = d - d.mean()
    lag = max(h - 1, 0)
    g0 = (dc ** 2).sum() / T
    var = g0 + 2 * sum((1 - k / (lag + 1)) * (dc[k:] * dc[:-k]).sum() / T for k in range(1, lag + 1))
    if var <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var / T)
    dm *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return float(dm), float(2 * stats.t.sf(abs(dm), T - 1))


def _dm_rows(df, loss_col, target, scopes):
    ref = DM_REF[target]
    rows = []
    for model in [m for m in df["model"].unique() if m != ref]:
        for label, h, sub in scopes(df):
            a = sub[sub["model"] == ref].groupby("target_date")[loss_col].mean()
            b = sub[sub["model"] == model].groupby("target_date")[loss_col].mean()
            d = (a - b).dropna()
            stat, p = dm_test(d.to_numpy(), h)
            rows.append(dict(target=target, model=model, ref=ref, scope=label, T=len(d),
                             rel_gain=float(d.mean() / a.loc[d.index].mean()) if len(d) else np.nan,
                             DM=stat, p_value=p))
    return rows


def dm_table(bt) -> pd.DataFrame:
    """Rows: every candidate vs the fixed reference; scopes h=1, h=7, and all horizons pooled (HAC lag 6)."""
    def scopes_h(df):
        out = [(f"h={h}", h, df[df["h"] == h]) for h in (1, 7)]
        return out + [("all h", C.HORIZON, df)]
    rows = []
    for key, name in (("n", "alerts"), ("h", "hours")):
        rows += _dm_rows(MT.add_reg_losses(bt.reg, key), "pin_s", name, scopes_h)
    rows += _dm_rows(MT.add_prob_losses(bt.heavy, "y_heavy"), "brier", "heavy", scopes_h)
    nd = MT.add_prob_losses(bt.nat_day, "y_mass")
    rows += _dm_rows(nd, "brier", "massive_day", scopes_h)
    a7 = bt.nat_any.copy()
    a7["target_date"] = a7["origin"]
    a7 = MT.add_prob_losses(a7, "y_any")
    rows += _dm_rows(a7, "brier", "massive_any7", lambda df: [("all origins", C.HORIZON, df)])
    return pd.DataFrame(rows)


def regime_bias_table(bt, sel: dict) -> pd.DataFrame:
    """Mean standardised error (forecast - actual) of the selected model at h=1, split into thirds by the recent trend
    of the target: mean of the previous 7 days / mean of the 28 days before that - 1 (known at forecast time).
    Negative = under-forecasting."""
    rows = []
    for key, name in (("n", "alerts"), ("h", "hours")):
        d = MT.add_reg_losses(bt.reg, key)
        d = d[(d["model"] == sel[key]) & (d["h"] == 1)]
        y = d.pivot(index="target_date", columns="region", values=f"y_{key}").sort_index()
        z = (d["err"] / d[f"scale_{key}"]).rename("z").to_frame().assign(target_date=d["target_date"], region=d["region"])
        z = z.pivot(index="target_date", columns="region", values="z").sort_index()
        with np.errstate(divide="ignore", invalid="ignore"):
            chg = y.shift(1).rolling(7).mean() / y.shift(8).rolling(28).mean() - 1
        s = pd.DataFrame({"trend": chg.stack(), "z": z.stack()}).replace([np.inf, -np.inf], np.nan).dropna()
        s["regime"] = pd.qcut(s["trend"], 3, labels=["falling", "stable", "rising"])
        g = s.groupby("regime", observed=True)["z"].agg(["mean", "count"]).reset_index()
        g.insert(0, "model", sel[key]); g.insert(0, "target", name)
        rows.append(g.rename(columns={"mean": "mean_resid_z", "count": "n"}))
    return pd.concat(rows, ignore_index=True)
