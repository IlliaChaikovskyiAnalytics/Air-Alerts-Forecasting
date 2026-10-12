"""Scoring rules and uncertainty. Continuous targets are scored on region-normalised errors (divided by the region's own
trailing 28-day level) so that busy frontline regions do not dominate the average; raw MAE is also reported."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


# ------------------------------------------------------------------ continuous targets
def add_reg_losses(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per-row losses for target `key` ('n' = alerts/day, 'h' = hours/day)."""
    df = df[df[f"y_{key}"].notna()].copy()
    y, sc = df[f"y_{key}"], df[f"scale_{key}"]
    df["ae"] = (df[f"{key}_mean"] - y).abs()
    df["ae_s"] = df["ae"] / sc
    df["err"] = df[f"{key}_mean"] - y
    df["se_s"] = (df["err"] / sc) ** 2
    df["pin_s"] = sum(pinball(y, df[f"{key}_q{int(q * 100)}"], q) for q in C.QUANTILES) / len(C.QUANTILES) / sc
    df["cover80"] = ((y >= df[f"{key}_q10"]) & (y <= df[f"{key}_q90"])).astype(float)
    df["width_s"] = (df[f"{key}_q90"] - df[f"{key}_q10"]) / sc
    return df


def summarise_reg(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    g = df.groupby(["model"] + (by or []), observed=True)
    out = g.agg(MAE=("ae", "mean"), sMAE=("ae_s", "mean"), sRMSE=("se_s", lambda s: np.sqrt(s.mean())),
                bias=("err", "mean"), sPinball=("pin_s", "mean"), cover80=("cover80", "mean"),
                width_s=("width_s", "mean"), n=("ae", "size"))
    return out.reset_index()


# ------------------------------------------------------------------ probabilities
def add_prob_losses(df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    df = df[df[ycol].notna()].copy()
    df["brier"] = (df["p"] - df[ycol]) ** 2
    df["logloss"] = -(df[ycol] * np.log(df["p"]) + (1 - df[ycol]) * np.log(1 - df["p"]))
    return df


def auc_table(df: pd.DataFrame, ycol: str, by: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(["model"] + (by or []), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        a = roc_auc_score(g[ycol], g["p"]) if g[ycol].nunique() == 2 else np.nan
        rows.append({**dict(zip(["model"] + (by or []), key)), "AUC": a})
    return pd.DataFrame(rows)


def summarise_prob(df: pd.DataFrame, ycol: str, by: list[str] | None = None) -> pd.DataFrame:
    g = df.groupby(["model"] + (by or []), observed=True)
    out = g.agg(Brier=("brier", "mean"), LogLoss=("logloss", "mean"), base_rate=(ycol, "mean"), n=("brier", "size")).reset_index()
    return out.merge(auc_table(df, ycol, by), on=["model"] + (by or []))


# ------------------------------------------------------------------ selection and bootstrap
def pick_best(df: pd.DataFrame, loss_col: str, cutoff: pd.Timestamp, date_col: str = "target_date") -> str:
    """Model with the lowest mean loss on rows whose target day is <= cutoff (the selection window)."""
    sel = df[df[date_col] <= cutoff]
    return sel.groupby("model", observed=True)[loss_col].mean().idxmin()


def boot_skill(df: pd.DataFrame, model: str, ref: str, loss_col: str, by: str | None, week_of: str = "origin",
               n_boot: int = C.N_BOOT, seed: int = C.SEED):
    """Skill = 1 - loss_model / loss_ref (pooled), optionally per level of `by`; block bootstrap over origin weeks.
    Returns DataFrame [by, skill, lo90, hi90]."""
    rng = np.random.default_rng(seed)
    a = df[df["model"] == model]
    b = df[df["model"] == ref]
    keys = ["origin", "h"] + (["region"] if "region" in df.columns else [])
    keys = [k for k in keys if k in a.columns]
    m = a[keys + [loss_col] + ([by] if by and by not in keys else [])].merge(
        b[keys + [loss_col]], on=keys, suffixes=("_m", "_r"))
    m["week"] = ((m[week_of] - m[week_of].min()).dt.days // 7).astype(int)
    grp = m[by] if by else pd.Series(0, index=m.index)
    levels = sorted(grp.unique())
    gi = grp.map({v: i for i, v in enumerate(levels)}).values
    nw = m["week"].max() + 1
    Sm, Sr = np.zeros((nw, len(levels))), np.zeros((nw, len(levels)))
    np.add.at(Sm, (m["week"].values, gi), m[f"{loss_col}_m"].values)
    np.add.at(Sr, (m["week"].values, gi), m[f"{loss_col}_r"].values)
    point = 1 - Sm.sum(0) / Sr.sum(0)
    W = np.stack([np.bincount(rng.integers(0, nw, nw), minlength=nw) for _ in range(n_boot)]).astype(float)
    boot = 1 - (W @ Sm) / (W @ Sr)
    return pd.DataFrame({by or "all": levels, "skill": point, "lo90": np.percentile(boot, 5, axis=0),
                         "hi90": np.percentile(boot, 95, axis=0)})


def reliability(df: pd.DataFrame, ycol: str, model: str, bins: int = 8) -> pd.DataFrame:
    g = df[df["model"] == model]
    q = pd.qcut(g["p"], bins, duplicates="drop")
    return g.groupby(q, observed=True).agg(p=("p", "mean"), freq=(ycol, "mean"), n=(ycol, "size")).reset_index(drop=True)
