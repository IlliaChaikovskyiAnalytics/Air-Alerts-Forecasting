"""Figures. One restrained palette; colour follows the model, not its rank."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config as C
from . import metrics as MT

SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
COL = {"GBM": "#2a78d6", "Blend": "#6250d6", "EWMA 14d": "#eb6834", "Flat 28d": "#898781", "Recent 28d": "#eb6834",
       "Recent 90d": "#eda100", "Clim 10%": "#898781", "Clim": "#898781", "Logistic": "#1baf7a", "Hazard GBM": "#e87ba4",
       "Recent 28d (indep.)": "#eb6834", "Recent 90d (indep.)": "#eda100"}
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blues", RAMP)
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "legend.frameon": False,
})


def _short(r: str) -> str:
    return r.replace(" oblast", "").replace("Kyiv City", "Kyiv (city)")


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _color(m):
    return COL.get(m, MUTED)


# ------------------------------------------------------------------ forecast
def forecast_heatmaps(fc: pd.DataFrame, out: Path) -> Path:
    order = fc.groupby("region")["hours_expected"].mean().sort_values(ascending=False).index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax, col, title, fmt, vmax in ((axes[0], "hours_expected", "Expected hours under alert", "{:.1f}", None),
                                      (axes[1], "p_heavy_day", "Probability of a 'heavy' day", "{:.0%}", 1.0)):
        M = fc.pivot(index="region", columns="date", values=col).loc[order]
        im = ax.imshow(M.values, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax)
        ax.set_yticks(range(len(order)), [_short(r) for r in order])
        ax.set_xticks(range(M.shape[1]), [f"{d:%a}\n{d:%d.%m}" for d in M.columns], fontsize=8)
        thr = (np.nanmax(M.values) if vmax is None else vmax) * 0.6
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7, color="white" if v > thr else INK)
        ax.set_title(title)
        ax.grid(False)
        plt.colorbar(im, ax=ax, fraction=0.04)
    return _save(fig, out / "forecast_regions_heatmap.png")


def forecast_national(nat: pd.DataFrame, any7: pd.DataFrame, nat_hist: pd.DataFrame, massive: pd.Series, out: Path) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2, 1.4]})
    hist = massive.dropna().tail(45)
    ax[0].bar(hist.index, hist.values * 0.08, width=0.8, color=MUTED, alpha=0.6, label="massive-strike day (observed)")
    ax[0].bar(nat["date"], nat["p_massive_day"], width=0.8, color=_color("GBM"), label="forecast probability")
    ax[0].set_title("National massive-strike proxy: daily probability")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].tick_params(axis="x", labelrotation=30)
    sel = any7[any7["selected"]].iloc[0]
    ax[1].barh(any7["model"], any7["p_any_massive_next7"], color=[_color(m) for m in any7["model"]])
    for i, v in enumerate(any7["p_any_massive_next7"]):
        ax[1].text(v + 0.01, i, f"{v:.0%}", va="center")
    ax[1].set_xlim(0, 1)
    ax[1].set_title(f"P(at least one massive strike in the next {C.HORIZON} days)\nselected: {sel['model']}")
    return _save(fig, out / "forecast_national.png")


# ------------------------------------------------------------------ backtest diagnostics
def skill_by_horizon(bt, sel: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    specs = [("n", "Alerts per day", "pin_s", "Flat 28d"), ("h", "Hours under alert per day", "pin_s", "Flat 28d")]
    for ax, (key, title, loss, ref) in zip(axes[:2], specs):
        df = MT.add_reg_losses(bt.reg, key)
        for model in dict.fromkeys(["GBM", "Blend", "EWMA 14d"]):
            if model == ref:
                continue
            s = MT.boot_skill(df, model, ref, loss, "h")
            ax.plot(s["h"], s["skill"], color=_color(model), lw=2, marker="o", ms=4, label=model)
            if model == sel[key]:
                ax.fill_between(s["h"], s["lo90"], s["hi90"], color=_color(model), alpha=0.15, lw=0)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(f"{title}\nskill vs {ref} (normalised pinball loss)")
        ax.set_xlabel("days ahead")
        ax.legend(fontsize=8)
    df = MT.add_prob_losses(bt.heavy, "y_heavy")
    for model in ("GBM", "Blend"):
        s = MT.boot_skill(df, model, "Recent 28d", "brier", "h")
        axes[2].plot(s["h"], s["skill"], color=_color(model), lw=2, marker="o", ms=4, label=model)
        if model == sel["heavy"]:
            axes[2].fill_between(s["h"], s["lo90"], s["hi90"], color=_color(model), alpha=0.15, lw=0)
    axes[2].axhline(0, color=INK, lw=0.8)
    axes[2].set_title("Heavy day: Brier skill vs 'Recent 28d' frequency")
    axes[2].set_xlabel("days ahead")
    axes[2].legend(fontsize=8)
    return _save(fig, out / "backtest_skill_by_horizon.png")


def skill_by_region(bt, sel: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    items = [("n", "Alerts per day", MT.add_reg_losses(bt.reg, "n"), "pin_s", "Flat 28d"),
             ("h", "Hours under alert per day", MT.add_reg_losses(bt.reg, "h"), "pin_s", "Flat 28d"),
             ("heavy", "Heavy-day probability", MT.add_prob_losses(bt.heavy, "y_heavy"), "brier", "Recent 28d")]
    order = None
    for ax, (key, title, df, loss, ref) in zip(axes, items):
        s = MT.boot_skill(df, sel[key], ref, loss, "region")
        if order is None:
            order = s.sort_values("skill")["region"].tolist()
        s = s.set_index("region").loc[order].reset_index()
        y = np.arange(len(s))
        ax.barh(y, s["skill"], color=[_color(sel[key]) if v > 0 else MUTED for v in s["skill"]], height=0.7)
        ax.errorbar(s["skill"], y, xerr=[s["skill"] - s["lo90"], s["hi90"] - s["skill"]], fmt="none", ecolor=INK, lw=0.8)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(y, [_short(r) for r in s["region"]])
        ax.set_title(f"{title}\n{sel[key]} vs {ref}")
        ax.set_xlabel("skill (>0 = better than the baseline)")
    return _save(fig, out / "backtest_skill_by_region.png")


def reliability_plots(bt, sel: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (title, df, ycol, models) in zip(axes, [
            ("Heavy day (regions)", MT.add_prob_losses(bt.heavy, "y_heavy"), "y_heavy", [sel["heavy"], "Recent 28d"]),
            ("Massive-strike day (national)", MT.add_prob_losses(bt.nat_day, "y_mass"), "y_mass", [sel["mass_day"], "Recent 28d"])]):
        ax.plot([0, 1], [0, 1], color=MUTED, ls=":")
        for m in dict.fromkeys(models):
            r = MT.reliability(df, ycol, m, bins=8)
            ax.plot(r["p"], r["freq"], marker="o", color=_color(m), lw=1.8, label=m)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(f"{title}: reliability")
        ax.legend(fontsize=8)
    return _save(fig, out / "backtest_reliability.png")


def latest_window(bt, sel: dict, panel_daily: pd.DataFrame, last_day: pd.Timestamp, out: Path) -> Path:
    """National totals: actual vs predicted (h=1 and h=7) over the latest window."""
    start = last_day - pd.Timedelta(days=C.LATEST_WINDOW_DAYS)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, (key, title, ycol) in zip(axes, [("n", "Alerts per day, all 24 regions", "y_n"), ("h", "Hours under alert per day, sum over regions", "y_h")]):
        df = bt.reg[(bt.reg["model"] == sel[key]) & (bt.reg["target_date"] > start)]
        act = panel_daily[f"{'n_alerts' if key == 'n' else 'alert_hours'}"].unstack("region").sum(axis=1).loc[start + pd.Timedelta(days=1):last_day]
        ax.plot(act.index, act.values, color=INK, lw=2, marker="o", ms=3, label="Actual")
        for h, ls in ((1, "-"), (7, "--")):
            p = df[df["h"] == h].groupby("target_date")[f"{key}_mean"].sum()
            ax.plot(p.index, p.values, color=_color(sel[key]), lw=1.8, ls=ls, label=f"{sel[key]}, h={h}")
        fl = bt.reg[(bt.reg["model"] == "Flat 28d") & (bt.reg["target_date"] > start) & (bt.reg["h"] == 7)].groupby("target_date")[f"{key}_mean"].sum()
        ax.plot(fl.index, fl.values, color=_color("Flat 28d"), lw=1.4, ls=":", label="Flat 28d, h=7")
        ax.set_title(f"{title}: latest {C.LATEST_WINDOW_DAYS} days")
        ax.legend(ncol=4, fontsize=8, loc="upper left")
    return _save(fig, out / "backtest_latest_window.png")


def massive_backtest(bt, sel: dict, out: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    d = bt.nat_day
    for m, ls in ((sel["mass_day"], "-"), ("Recent 28d", "--")):
        g = d[(d["model"] == m) & (d["h"] == 1)].sort_values("target_date")
        axes[0].plot(g["target_date"], g["p"], color=_color(m), lw=1.6, ls=ls, label=f"{m} (h=1)")
    g = d[(d["model"] == sel["mass_day"]) & (d["h"] == 1)].sort_values("target_date")
    axes[0].scatter(g["target_date"], g["y_mass"] * 1.0, s=14, color=INK, label="Observed (0/1)", zorder=3)
    axes[0].set_ylim(-0.05, 1.3)
    axes[0].set_title("Massive-strike day: next-day probability vs outcome")
    axes[0].legend(ncol=3, fontsize=8, loc="upper left")
    a = bt.nat_any
    if len(a):
        for m, ls in ((sel["any7"], "-"), ("Recent 28d (indep.)", "--")):
            g = a[a["model"] == m].sort_values("origin")
            axes[1].plot(g["origin"], g["p"], color=_color(m), lw=1.6, ls=ls, label=m)
        g = a[a["model"] == sel["any7"]].sort_values("origin")
        axes[1].scatter(g["origin"], g["y_any"] * 1.0, s=14, color=INK, label="Observed: any massive day in next 7", zorder=3)
        axes[1].set_ylim(-0.05, 1.3)
        axes[1].legend(ncol=3, fontsize=8, loc="upper left")
    axes[1].set_title(f"P(at least one massive strike in the next {C.HORIZON} days) vs outcome")
    return _save(fig, out / "backtest_massive_strike.png")
