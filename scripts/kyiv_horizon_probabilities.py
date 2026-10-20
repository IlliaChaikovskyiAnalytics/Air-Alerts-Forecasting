"""Air-raid alerts in Kyiv: at which horizon does a probabilistic forecast make sense? (PyCharm version)

Forecast, for every day 1..MAX_H days ahead, the probability of
    - a day with NO alerts (n = 0)
    - a busy day   (n >= 3)
    - a heavy day  (n >= 5)
and measure how much skill the models have over simple baselines AT EACH HORIZON.

References (known at the forecast origin):
    Clim 365d  : event frequency over the last 365 days
    Recent 28d : event frequency over the last 28 days  (the "current regime" baseline)
Models: compact Logistic regression (recent/long-run event frequency, weekday, horizon) and a Gradient boosting
classifier on many features (one pooled model with the horizon as a feature).

Metrics: Brier score, Brier Skill Score (BSS = 1 - Brier_model / Brier_reference), ROC-AUC, reliability;
block-bootstrap (by week of origin) 90% intervals. Backtest: models are re-fitted every REFIT_DAYS days on
data that ends at the refit date, then used for all origins until the next refit.

Requires:  pip install pandas numpy scipy matplotlib scikit-learn
Needs data/raw/volunteer_data_en.csv (created by the EDA script) next to this file. Press Run.
Outputs: ./reports/kyiv_city/horizon/
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import urllib.request
import warnings
from pathlib import Path

# ============================ CONFIG (edit me) ============================
REGION = "Kyiv City"
MAX_H = 30               # longest horizon (days) to study
TEST_DAYS = 365          # backtest covers targets in the last TEST_DAYS days
REFIT_DAYS = 14          # re-fit models every N days during the backtest
TRAIN_WINDOW_DAYS = 730  # training uses only targets in this many days before the refit date
N_BOOT = 500             # bootstrap draws for the confidence intervals
FORECAST_DAYS = 14       # horizon of the final forecast table / plot
SHOW_PLOTS = True        # True -> show all figures at the end (one blocking call); they are always saved
SEED = 42
# ==========================================================================

import matplotlib

if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parent.parent   # repo root
RAW = ROOT / "data" / "raw"
TZ = "Europe/Kyiv"
BASE_URL = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/"

# events defined on the daily number of alerts
EVENTS = {
    "zero": ("No alerts (n = 0)", lambda n: n == 0),
    "ge3": ("Busy day (n >= 3)", lambda n: n >= 3),
    "ge5": ("Heavy day (n >= 5)", lambda n: n >= 5),
}
MODELS = ["Clim 365d", "Recent 28d", "Logistic", "GBM"]
PCOL = {"Clim 365d": "p_clim", "Recent 28d": "p_recent", "Logistic": "p_logit", "GBM": "p_gbm"}
EPS = 0.01


# ============================ data ============================
def download():
    RAW.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(BASE_URL + "volunteer_data_en.csv", RAW / "volunteer_data_en.csv")


def merge_overlaps(df):
    out, cur_s, cur_e = [], None, None
    for s, e in df[["started_at", "finished_at"]].itertuples(index=False):
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_s is not None:
        out.append((cur_s, cur_e))
    return pd.DataFrame(out, columns=["started_at", "finished_at"])


def load_daily(region):
    """Daily series in Kyiv local days: n_alerts, alert_hours. Returns (daily, last_full_day)."""
    raw = pd.read_csv(RAW / "volunteer_data_en.csv", parse_dates=["started_at", "finished_at"])
    end = raw["finished_at"].max().ceil("h")
    df = raw[raw["region"] == region][["started_at", "finished_at"]].sort_values("started_at")
    df = merge_overlaps(df[df["finished_at"] > df["started_at"]].reset_index(drop=True))
    idx = pd.date_range(df["started_at"].min().floor("h"), end, freq="h", tz="UTC", inclusive="left")
    frac = np.zeros(len(idx))
    t0 = idx[0]
    for s, e in df.itertuples(index=False):
        i0, i1 = int((s - t0) / pd.Timedelta(hours=1)), int((e - t0) / pd.Timedelta(hours=1))
        for i in range(max(i0, 0), min(i1, len(idx) - 1) + 1):
            lo, hi = max(s, idx[i]), min(e, idx[i] + pd.Timedelta(hours=1))
            if hi > lo:
                frac[i] += (hi - lo).total_seconds() / 3600
    h = pd.DataFrame({"alert_frac": np.clip(frac, 0, 1)}, index=idx)
    h["starts"] = df["started_at"].dt.floor("h").value_counts().reindex(idx, fill_value=0).values
    h["date"] = h.index.tz_convert(TZ).tz_localize(None).normalize()
    end_loc = end.tz_convert(TZ)
    last_day = end_loc.tz_localize(None).normalize()
    if not (end_loc.hour == 0 and end_loc.minute == 0):
        last_day -= pd.Timedelta(days=1)
    d = (h.groupby("date").agg(alert_hours=("alert_frac", "sum"), n_alerts=("starts", "sum"))
         .asfreq("D", fill_value=0).loc[:last_day])
    return d[["n_alerts", "alert_hours"]], last_day


# ============================ features (known at the origin only) ============================
WINDOWS = (3, 7, 14, 28, 56, 90, 365)


def make_features(d):
    F = pd.DataFrame(index=d.index)
    for w in WINDOWS:
        F[f"n_m{w}"] = d.n_alerts.rolling(w, min_periods=1).mean()
        F[f"h_m{w}"] = d.alert_hours.rolling(w, min_periods=1).mean()
    F["n_std28"] = d.n_alerts.rolling(28, min_periods=7).std().fillna(0)
    for k, (_, f) in EVENTS.items():
        F[f"r7_{k}"] = f(d.n_alerts).astype(float).rolling(7, min_periods=1).mean()
        F[f"r28_{k}"] = f(d.n_alerts).astype(float).rolling(28, min_periods=1).mean()
    F["n_mom"] = F["n_m7"] / (F["n_m56"] + 0.1)
    F["n_last"] = d.n_alerts
    F["h_last"] = d.alert_hours
    F["since_zero"] = _days_since(d.n_alerts == 0)
    return F


def _days_since(flag):
    out, c = [], 365
    for v in flag.values:
        c = 0 if v else min(c + 1, 365)
        out.append(c)
    return pd.Series(out, index=flag.index)


def make_direct(d, F, H):
    parts = []
    for h in range(1, H + 1):
        X = F.copy()
        X["origin"] = F.index
        X["h"] = h
        X["target_date"] = F.index + pd.Timedelta(days=h)
        X["dow"] = X["target_date"].dt.dayofweek
        X["n_alerts"] = d.n_alerts.reindex(X["target_date"]).values
        parts.append(X)
    D = pd.concat(parts, ignore_index=True)
    for k, (_, f) in EVENTS.items():
        D[f"y_{k}"] = np.where(D["n_alerts"].isna(), np.nan, f(D["n_alerts"]).astype(float))
        e = f(d.n_alerts).astype(float)
        D[f"clim_{k}"] = e.rolling(365, min_periods=60).mean().reindex(D["origin"]).values
        D[f"recent_{k}"] = e.rolling(28, min_periods=14).mean().reindex(D["origin"]).values
    return D


FEATS = list(make_features(pd.DataFrame({"n_alerts": [0.0] * 5, "alert_hours": [0.0] * 5},
                                        index=pd.date_range("2022-01-01", periods=5))).columns) + ["h", "dow"]


def logit_matrix(X, k, H):
    """Compact logistic model: the event's recent / long-run frequency (as log-odds), weekday and horizon."""
    lg = lambda v: np.log(np.clip(v, EPS, 1 - EPS) / (1 - np.clip(v, EPS, 1 - EPS)))
    A = np.column_stack([lg(X[f"recent_{k}"].values), lg(X[f"r7_{k}"].values), lg(X[f"clim_{k}"].fillna(0.3).values),
                         X["h"].values / H])
    dow = pd.get_dummies(X["dow"]).reindex(columns=range(7), fill_value=0).astype(float).values
    return np.hstack([A, dow])


def clip(p):
    return np.clip(p, EPS, 1 - EPS)


def fit_predict(tr, te, k, H):
    y = tr[f"y_{k}"].values
    sc = StandardScaler().fit(logit_matrix(tr, k, H))
    lg = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(logit_matrix(tr, k, H)), y)
    gb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=80,
                                        l2_regularization=1.0, random_state=SEED).fit(tr[FEATS], y)
    return (clip(lg.predict_proba(sc.transform(logit_matrix(te, k, H)))[:, 1]),
            clip(gb.predict_proba(te[FEATS])[:, 1]))


# ============================ backtest ============================
def run_backtest(D, last_day, H):
    first_test = last_day - pd.Timedelta(days=TEST_DAYS)
    refits = list(pd.date_range(first_test, last_day, freq=f"{REFIT_DAYS}D"))
    recs = []
    for i, T in enumerate(refits, 1):
        lo = T - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
        tr = D[(D["target_date"] <= T) & (D["target_date"] > lo)].dropna(subset=["n_alerts"])
        assert tr["target_date"].max() <= T, "leakage: training target after refit date"
        te = D[(D["origin"] >= T) & (D["origin"] < T + pd.Timedelta(days=REFIT_DAYS)) & (D["target_date"] <= last_day)]
        if te.empty:
            continue
        for k in EVENTS:
            p_lg, p_gb = fit_predict(tr, te, k, H)
            recs.append(pd.DataFrame({
                "event": k, "origin": te["origin"].values, "h": te["h"].values, "date": te["target_date"].values,
                "y": te[f"y_{k}"].values, "p_clim": clip(te[f"clim_{k}"].values),
                "p_recent": clip(te[f"recent_{k}"].values), "p_logit": p_lg, "p_gbm": p_gb}))
        print(f"  refit {i:2d}/{len(refits)} at {T.date()} done", flush=True)
    R = pd.concat(recs, ignore_index=True).dropna(subset=["y", "p_clim", "p_recent"])
    return R


# ============================ metrics ============================
def brier(y, p):
    return (p - y) ** 2


def by_horizon(R, k, ref, mod):
    """Per-horizon Brier skill of `mod` vs `ref` with a block bootstrap by origin week."""
    g = R[R["event"] == k].copy()
    g["w"] = ((g["origin"] - g["origin"].min()).dt.days // 7).astype(int)
    nw, H = g["w"].max() + 1, g["h"].max()
    Sm, Sr, C = np.zeros((nw, H)), np.zeros((nw, H)), np.zeros((nw, H))
    np.add.at(Sm, (g["w"].values, g["h"].values - 1), brier(g["y"].values, g[PCOL[mod]].values))
    np.add.at(Sr, (g["w"].values, g["h"].values - 1), brier(g["y"].values, g[PCOL[ref]].values))
    np.add.at(C, (g["w"].values, g["h"].values - 1), 1)
    point = 1 - Sm.sum(0) / Sr.sum(0)
    W = np.stack([np.bincount(rng.integers(0, nw, nw), minlength=nw) for _ in range(N_BOOT)]).astype(float)
    boot = 1 - (W @ Sm) / (W @ Sr)
    return point, np.percentile(boot, 5, axis=0), np.percentile(boot, 95, axis=0)


def auc_by_h(R, k, mod):
    g = R[R["event"] == k]
    out = []
    for h in range(1, MAX_H + 1):
        s = g[g["h"] == h]
        out.append(roc_auc_score(s["y"], s[PCOL[mod]]) if s["y"].nunique() == 2 else np.nan)
    return np.array(out)


def useful_horizon(lo):
    """largest h such that the lower CI bound is > 0 for all horizons 1..h."""
    h = 0
    for v in lo:
        if v > 0:
            h += 1
        else:
            break
    return h


# ============================ plotting ============================
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
COL = {"GBM": "#2a78d6", "Logistic": "#1baf7a", "Recent 28d": "#eb6834", "Clim 365d": "#898781"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "legend.frameon": False,
})


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print("  saved", path.relative_to(ROOT))


def plot_bss_clim(S, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    hs = np.arange(1, MAX_H + 1)
    for ax, (k, (title, _)) in zip(axes, EVENTS.items()):
        for mod in ("Recent 28d", "Logistic", "GBM"):
            ax.plot(hs, S[(k, mod, "Clim 365d")][0], color=COL[mod], lw=2.2 if mod == "GBM" else 1.5, label=mod)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(f"{title}\nskill vs 365-day climatology")
        ax.set_xlabel("days ahead")
        ax.set_ylabel("Brier Skill Score")
        ax.legend(fontsize=8)
    save(fig, out / "01_skill_vs_climatology.png")
    return fig


def plot_bss_recent(S, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    hs = np.arange(1, MAX_H + 1)
    for ax, (k, (title, _)) in zip(axes, EVENTS.items()):
        for mod in ("Logistic", "GBM"):
            p, lo, hi = S[(k, mod, "Recent 28d")]
            if mod == "GBM":
                ax.fill_between(hs, lo, hi, color=COL[mod], alpha=0.18, lw=0, label="GBM 90% CI")
            ax.plot(hs, p, color=COL[mod], lw=2.2 if mod == "GBM" else 1.5, label=mod)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(f"{title}\nskill beyond the 'recent 28 days' baseline")
        ax.set_xlabel("days ahead")
        ax.set_ylabel("Brier Skill Score")
        ax.legend(fontsize=8)
    save(fig, out / "02_skill_beyond_recent_level.png")
    return fig


def plot_auc(A, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    hs = np.arange(1, MAX_H + 1)
    for ax, (k, (title, _)) in zip(axes, EVENTS.items()):
        for mod in ("Recent 28d", "Logistic", "GBM"):
            ax.plot(hs, A[(k, mod)], color=COL[mod], lw=2.2 if mod == "GBM" else 1.5, label=mod)
        ax.axhline(0.5, color=INK, lw=0.8)
        ax.set_ylim(0.4, 1)
        ax.set_title(f"{title}\nROC-AUC by horizon")
        ax.set_xlabel("days ahead")
        ax.legend(fontsize=8)
    save(fig, out / "03_auc_by_horizon.png")
    return fig


def plot_reliability(R, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (k, (title, _)) in zip(axes, EVENTS.items()):
        ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1)
        for lab, lo, hi, col in (("h = 1-3", 1, 3, COL["GBM"]), ("h = 15-30", 15, 30, "#6250d6")):
            g = R[(R["event"] == k) & (R["h"] >= lo) & (R["h"] <= hi)]
            q = pd.qcut(g["p_gbm"], 6, duplicates="drop")
            t = g.groupby(q, observed=True).agg(p=("p_gbm", "mean"), f=("y", "mean"), n=("y", "size"))
            ax.plot(t["p"], t["f"], marker="o", color=col, lw=1.8, label=f"GBM, {lab}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(f"{title}\nreliability")
        ax.legend(fontsize=8)
    save(fig, out / "04_reliability.png")
    return fig


def plot_next_day(R, out):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, (k, (title, _)) in zip(axes, EVENTS.items()):
        g = R[(R["event"] == k) & (R["h"] == 1)].sort_values("date").tail(120)
        dt = pd.to_datetime(g["date"])
        ax.plot(dt, g["p_recent"], color=COL["Recent 28d"], lw=1.4, ls="--", label="Recent 28d")
        ax.plot(dt, g["p_gbm"], color=COL["GBM"], lw=2, label="GBM")
        ax.scatter(dt, g["y"], s=14, color=INK, label="Actual (0/1)", zorder=3)
        ax.set_ylim(-0.05, 1.3)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(f"{title}: probability for the NEXT day (h = 1), last 120 days")
        ax.legend(ncol=3, loc="upper left", fontsize=8)
    save(fig, out / "05_next_day_probabilities.png")
    return fig


def plot_final(Fc, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (k, (title, _)) in zip(axes, EVENTS.items()):
        g = Fc[Fc["event"] == k].sort_values("h")
        x = np.arange(len(g))
        ax.bar(x - 0.2, g["p_recent"], width=0.4, color=COL["Recent 28d"], alpha=0.7, label="Recent 28d")
        ax.bar(x + 0.2, g["p_gbm"], width=0.4, color=COL["GBM"], label="GBM")
        ax.set_xticks(x, [d.strftime("%d.%m") for d in pd.to_datetime(g["date"])], rotation=60, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("probability")
    fig.suptitle(f"Forecast for the next {FORECAST_DAYS} days", x=0.01, ha="left", fontweight="bold")
    save(fig, out / "06_forecast_next_days.png")
    return fig


# ============================ main ============================
def run():
    if not (RAW / "volunteer_data_en.csv").exists():
        try:
            download()
        except Exception as ex:
            sys.exit(f"No data and download failed ({ex}). Put volunteer_data_en.csv into {RAW}")
    out = ROOT / "reports" / slug(REGION) / "horizon"
    out.mkdir(parents=True, exist_ok=True)

    d, last_day = load_daily(REGION)
    print(f"=== {REGION}: {d.index.min().date()} .. {last_day.date()} ({len(d)} days) ===")
    print(f"backtest: targets in the last {TEST_DAYS} days, refit every {REFIT_DAYS} days, horizons 1..{MAX_H}\n")
    print("event base rates (share of days):")
    for k, (title, f) in EVENTS.items():
        e = f(d.n_alerts)
        print(f"  {title:20s} last 365d: {e.iloc[-365:].mean():5.1%} | last 28d: {e.iloc[-28:].mean():5.1%} "
              f"| last 7d: {e.iloc[-7:].mean():5.1%}")

    F = make_features(d)
    D = make_direct(d, F, max(MAX_H, FORECAST_DAYS))
    D = D[D["origin"] >= d.index.min() + pd.Timedelta(days=90)]
    D = D[D["h"] <= MAX_H]

    print("\nbacktest:")
    R = run_backtest(D, last_day, MAX_H)
    R.to_csv(out / "backtest_probabilities.csv", index=False)

    # ---------------- skill by horizon ----------------
    S, A = {}, {}
    for k in EVENTS:
        for mod in ("Recent 28d", "Logistic", "GBM"):
            S[(k, mod, "Clim 365d")] = by_horizon(R, k, "Clim 365d", mod)
        for mod in ("Logistic", "GBM"):
            S[(k, mod, "Recent 28d")] = by_horizon(R, k, "Recent 28d", mod)
        for mod in ("Recent 28d", "Logistic", "GBM"):
            A[(k, mod)] = auc_by_h(R, k, mod)

    show_h = [1, 2, 3, 5, 7, 10, 14, 21, 30]
    show_h = [h for h in show_h if h <= MAX_H]
    for k, (title, _) in EVENTS.items():
        print(f"\n=== {title} ===")
        rows = []
        for h in show_h:
            i = h - 1
            gp, glo, ghi = S[(k, "GBM", "Clim 365d")]
            bp, blo, bhi = S[(k, "GBM", "Recent 28d")]
            rows.append({"h": h, "Brier GBM": R[(R.event == k) & (R.h == h)].eval("(p_gbm-y)**2").mean(),
                         "BSS vs clim365": gp[i], "BSS vs recent28": bp[i],
                         "  90% CI": f"[{blo[i]:+.3f}, {bhi[i]:+.3f}]",
                         "AUC GBM": A[(k, "GBM")][i], "AUC recent28": A[(k, "Recent 28d")][i]})
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        u_clim = useful_horizon(S[(k, "GBM", "Clim 365d")][1])
        u_rec = useful_horizon(S[(k, "GBM", "Recent 28d")][1])
        print(f"  -> GBM beats 365d climatology (90% CI > 0) up to h = {u_clim} days;"
              f" beats the 'recent 28 days' baseline up to h = {u_rec} days")

    # ---------------- latest period ----------------
    print("\n=== LATEST 56 target days only (Brier, lower is better) ===")
    L = R[R["date"] > last_day - pd.Timedelta(days=56)]
    rows = []
    for k, (title, _) in EVENTS.items():
        for hs, lab in (((1,), "h=1"), ((7,), "h=7"), (tuple(range(1, 8)), "h=1..7"), (tuple(range(8, 31)), "h=8..30")):
            g = L[(L["event"] == k) & (L["h"].isin(hs))]
            if len(g):
                rows.append({"event": title, "horizon": lab, "n": len(g),
                             **{m: brier(g["y"], g[PCOL[m]]).mean() for m in MODELS}})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---------------- final forecast ----------------
    print(f"\n=== FORECAST for {FORECAST_DAYS} days after {last_day.date()} (models trained on data up to {last_day.date()}) ===")
    Dfull = make_direct(d, F, max(MAX_H, FORECAST_DAYS))
    Dfull = Dfull[Dfull["origin"] >= d.index.min() + pd.Timedelta(days=90)]
    lo = last_day - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
    tr = Dfull[(Dfull["target_date"] <= last_day) & (Dfull["target_date"] > lo) & (Dfull["h"] <= MAX_H)].dropna(subset=["n_alerts"])
    te = Dfull[(Dfull["origin"] == last_day) & (Dfull["h"] <= FORECAST_DAYS)].sort_values("h")
    recs = []
    for k in EVENTS:
        p_lg, p_gb = fit_predict(tr, te, k, MAX_H)
        recs.append(pd.DataFrame({"event": k, "h": te["h"].values, "date": te["target_date"].values,
                                  "p_clim": clip(te[f"clim_{k}"].values), "p_recent": clip(te[f"recent_{k}"].values),
                                  "p_logit": p_lg, "p_gbm": p_gb}))
    Fc = pd.concat(recs, ignore_index=True)
    Fc.to_csv(out / "forecast_next_days.csv", index=False)
    cols = {}
    for k, (title, _) in EVENTS.items():
        g = Fc[Fc["event"] == k].set_index("date")
        cols[f"{title} | Recent 28d"] = g["p_recent"]
        cols[f"{title} | GBM"] = g["p_gbm"]
    tab = pd.DataFrame(cols)
    tab.insert(0, "weekday", pd.to_datetime(tab.index).strftime("%a"))
    tab.index = pd.to_datetime(tab.index).strftime("%Y-%m-%d")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("probabilities (Recent 28d = event frequency over the last 28 days; GBM = gradient boosting):")
    print(tab.round(2).to_string())
    print("\nexpected number of days with no alerts in the window: GBM %.1f | recent 28d %.1f | clim 365d %.1f" % (
        Fc[Fc.event == "zero"]["p_gbm"].sum(), Fc[Fc.event == "zero"]["p_recent"].sum(), Fc[Fc.event == "zero"]["p_clim"].sum()))

    print("\nfigures:")
    figs = [plot_bss_clim(S, out), plot_bss_recent(S, out), plot_auc(A, out), plot_reliability(R, out),
            plot_next_day(R, out), plot_final(Fc, out)]
    if SHOW_PLOTS:
        plt.show()
    else:
        for f in figs:
            plt.close(f)


if __name__ == "__main__":
    buf = io.StringIO()

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)

        def flush(self):
            sys.__stdout__.flush()

    with contextlib.redirect_stdout(Tee()):
        run()
    (ROOT / "reports" / slug(REGION) / "horizon" / "horizon_report.txt").write_text(buf.getvalue(), encoding="utf-8")
