"""Air-raid alerts in Kyiv: 30-day-ahead forecast of DAILY activity (PyCharm version).

What is forecast (one value per day for the next HORIZON days):
  1. n_alerts    - number of alerts that started that day
  2. alert_hours - total time under alert that day (hours)
  derived: expected total over the window and mean alert duration (alert_hours / n_alerts).

Models, compared in a rolling-origin backtest (nothing from the future is used at any origin):
  baselines : Flat 28d mean | Weekday-adjusted 28d mean | Mean 365d
  statistical: Poisson GLM (+ NegBin / Gamma prediction intervals)
  ML        : Gradient boosting (direct multi-horizon, Poisson loss + quantile models for 10/50/90%)

Requires:  pip install pandas numpy scipy matplotlib scikit-learn
Needs data/raw/volunteer_data_en.csv (created by the EDA script / download below), next to this file.
Edit the CONFIG block and press Run. Outputs: ./reports/kyiv_city/forecast/
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
HORIZON = 30              # forecast days ahead
N_ORIGINS = 26            # number of backtest forecast origins
ORIGIN_STEP = 14          # days between backtest origins (last origin is placed so its window ends on the last full day)
TRAIN_WINDOW_DAYS = 730   # only target days within this many days before the origin are used for training
SHOW_PLOTS = True         # True -> show all figures at the very end (one blocking call); they are always saved
SEED = 42
# ==========================================================================

import matplotlib

if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent   # repo root
RAW = ROOT / "data" / "raw"
TZ = "Europe/Kyiv"
BASE_URL = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/"
QS = (0.1, 0.5, 0.9)
TARGETS = {"n_alerts": "Alerts per day", "alert_hours": "Hours under alert per day"}
MODELS = ["Flat 28d", "Weekday 28d", "Mean 365d", "Poisson GLM", "GBM"]


# ============================ data ============================
def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in ("volunteer_data_en.csv",):
        print("downloading", name, flush=True)
        urllib.request.urlretrieve(BASE_URL + name, RAW / name)


def merge_overlaps(df: pd.DataFrame) -> pd.DataFrame:
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


def load_daily(region: str):
    """Daily series (Kyiv local days): n_alerts, alert_hours. Returns (daily, last_full_day)."""
    raw = pd.read_csv(RAW / "volunteer_data_en.csv", parse_dates=["started_at", "finished_at"])
    end = raw["finished_at"].max().ceil("h")  # end of data coverage (UTC)
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


# ============================ features (known at the forecast origin only) ============================
WINDOWS = (3, 7, 14, 28, 56, 90, 365)


def make_features(d: pd.DataFrame) -> pd.DataFrame:
    """Row at date t uses data up to and including day t only."""
    F = pd.DataFrame(index=d.index)
    for w in WINDOWS:
        F[f"n_m{w}"] = d.n_alerts.rolling(w, min_periods=1).mean()
        F[f"h_m{w}"] = d.alert_hours.rolling(w, min_periods=1).mean()
    F["n_std28"] = d.n_alerts.rolling(28, min_periods=7).std().fillna(0)
    F["zero28"] = (d.n_alerts == 0).rolling(28, min_periods=1).mean()
    dur = d.alert_hours.rolling(28, min_periods=7).sum() / d.n_alerts.rolling(28, min_periods=7).sum().replace(0, np.nan)
    F["dur28"] = (dur * 60).ffill().fillna(30)
    F["n_mom"] = F["n_m7"] / (F["n_m56"] + 0.1)
    F["h_mom"] = F["h_m7"] / (F["h_m56"] + 0.1)
    F["n_last"] = d.n_alerts
    F["h_last"] = d.alert_hours
    return F


def make_direct(d: pd.DataFrame, F: pd.DataFrame, H: int) -> pd.DataFrame:
    """One row per (origin, horizon h): features at origin, target = value on origin + h."""
    parts = []
    for h in range(1, H + 1):
        X = F.copy()
        X["origin"] = F.index
        X["h"] = h
        X["target_date"] = F.index + pd.Timedelta(days=h)
        X["dow"] = X["target_date"].dt.dayofweek
        X["n_alerts"] = d.n_alerts.reindex(X["target_date"]).values
        X["alert_hours"] = d.alert_hours.reindex(X["target_date"]).values
        parts.append(X)
    return pd.concat(parts, ignore_index=True)


FEATS = [c for c in make_features(pd.DataFrame({"n_alerts": [0.0] * 5, "alert_hours": [0.0] * 5},
                                               index=pd.date_range("2022-01-01", periods=5))).columns] + ["h", "dow"]


# ============================ models ============================
def glm_matrix(X: pd.DataFrame) -> np.ndarray:
    A = X[FEATS].copy()
    for c in A.columns:
        if c.startswith(("n_m", "h_m", "n_std", "n_last", "h_last", "dur")):
            A[c] = np.log1p(A[c])
    A["h"] = A["h"] / HORIZON
    dow = pd.get_dummies(A.pop("dow")).reindex(columns=range(7), fill_value=0).astype(float)
    return np.hstack([A.values.astype(float), dow.values])


def fit_predict_glm(Xtr, ytr, Xte, target):
    sc = StandardScaler().fit(glm_matrix(Xtr))
    m = PoissonRegressor(alpha=1.0, max_iter=500).fit(sc.transform(glm_matrix(Xtr)), ytr)
    mu_tr = np.maximum(m.predict(sc.transform(glm_matrix(Xtr))), 1e-3)
    mu = np.maximum(m.predict(sc.transform(glm_matrix(Xte))), 1e-3)
    out = {"mean": mu}
    if target == "n_alerts":  # NegBin intervals, dispersion by method of moments
        alpha = max(np.sum((ytr - mu_tr) ** 2 - mu_tr) / np.sum(mu_tr ** 2), 0.01)
        n = 1 / alpha
        for q in QS:
            out[f"q{int(q * 100)}"] = stats.nbinom.ppf(q, n, n / (n + mu))
    else:  # Gamma intervals
        phi = max(np.mean(((ytr - mu_tr) / mu_tr) ** 2), 0.05)
        k = 1 / phi
        for q in QS:
            out[f"q{int(q * 100)}"] = stats.gamma.ppf(q, k, scale=mu / k)
    return out


def fit_predict_gbm(Xtr, ytr, Xte, target):
    kw = dict(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=60,
              l2_regularization=1.0, random_state=SEED)
    out = {"mean": HistGradientBoostingRegressor(loss="poisson", **kw).fit(Xtr[FEATS], ytr).predict(Xte[FEATS])}
    qs = np.column_stack([HistGradientBoostingRegressor(loss="quantile", quantile=q, **kw)
                          .fit(Xtr[FEATS], ytr).predict(Xte[FEATS]) for q in QS])
    qs = np.maximum(np.sort(qs, axis=1), 0)  # no crossing, no negatives
    for j, q in enumerate(QS):
        out[f"q{int(q * 100)}"] = qs[:, j]
    out["mean"] = np.maximum(out["mean"], 0)
    return out


def baselines(d: pd.DataFrame, T: pd.Timestamp, te: pd.DataFrame, target: str):
    """Three simple benchmarks built from history up to T."""
    hist = d[target].loc[:T]
    dow = te["dow"].values
    res = {}
    # Flat 28d: last-28-day mean; band = empirical quantiles of the last 56 days
    band = hist.iloc[-56:].quantile(list(QS)).values
    res["Flat 28d"] = {"mean": np.full(len(te), hist.iloc[-28:].mean()),
                       **{f"q{int(q * 100)}": np.full(len(te), band[j]) for j, q in enumerate(QS)}}
    # Weekday 28d: same level times a weekday factor learnt on the last 91 days
    h91 = hist.iloc[-91:]
    f = (h91.groupby(h91.index.dayofweek).mean() / max(h91.mean(), 1e-9)).reindex(range(7)).fillna(1.0).values
    mu = hist.iloc[-28:].mean() * f[dow]
    res["Weekday 28d"] = {"mean": mu, **{f"q{int(q * 100)}": band[j] * f[dow] for j, q in enumerate(QS)}}
    # Mean 365d: climatology
    h365 = hist.iloc[-365:]
    b365 = h365.quantile(list(QS)).values
    res["Mean 365d"] = {"mean": np.full(len(te), h365.mean()),
                        **{f"q{int(q * 100)}": np.full(len(te), b365[j]) for j, q in enumerate(QS)}}
    return res


def predict_origin(d, D, T, with_truth=True):
    """All models for one origin T. Training uses only target days <= T (no leakage)."""
    lo = T - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
    te = D[D["origin"] == T].sort_values("h")
    recs = []
    for target in TARGETS:
        tr = D[(D["target_date"] <= T) & (D["target_date"] > lo)].dropna(subset=[target])
        assert tr["target_date"].max() <= T, "leakage: training target after origin"
        assert (tr["origin"] + pd.to_timedelta(tr["h"], unit="D") <= T).all()
        preds = baselines(d, T, te, target)
        preds["Poisson GLM"] = fit_predict_glm(tr, tr[target].values, te, target)
        preds["GBM"] = fit_predict_gbm(tr, tr[target].values, te, target)
        for model, p in preds.items():
            r = pd.DataFrame({"origin": T, "h": te["h"].values, "date": te["target_date"].values,
                              "target": target, "model": model,
                              "y": te[target].values if with_truth else np.nan,
                              "mean": p["mean"], "q10": p["q10"], "q50": p["q50"], "q90": p["q90"]})
            recs.append(r)
    return pd.concat(recs, ignore_index=True)


# ============================ metrics ============================
def pinball(y, q, tau):
    diff = y - q
    return np.maximum(tau * diff, (tau - 1) * diff)


def metrics_table(R: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), g in R.groupby(["target", "model"]):
        err = g["mean"] - g["y"]
        pb = np.mean([pinball(g["y"], g[f"q{int(q * 100)}"], q).mean() for q in QS])
        rows.append({"target": target, "model": model, "MAE": err.abs().mean(),
                     "RMSE": np.sqrt((err ** 2).mean()), "bias": err.mean(), "pinball": pb,
                     "cover80": ((g["y"] >= g["q10"]) & (g["y"] <= g["q90"])).mean(),
                     "width80": (g["q90"] - g["q10"]).mean()})
    M = pd.DataFrame(rows)
    M["model"] = pd.Categorical(M["model"], MODELS, ordered=True)
    return M.sort_values(["target", "model"]).reset_index(drop=True)


def totals_table(R: pd.DataFrame) -> pd.DataFrame:
    """Per origin and model: predicted vs actual 30-day totals and mean alert duration (min)."""
    g = R.groupby(["model", "origin", "target"]).agg(y=("y", "sum"), mean=("mean", "sum")).unstack("target")
    T = pd.DataFrame({
        "act_alerts": g[("y", "n_alerts")], "pred_alerts": g[("mean", "n_alerts")],
        "act_hours": g[("y", "alert_hours")], "pred_hours": g[("mean", "alert_hours")]})
    T["act_dur"] = T["act_hours"] / T["act_alerts"].replace(0, np.nan) * 60
    T["pred_dur"] = T["pred_hours"] / T["pred_alerts"].replace(0, np.nan) * 60
    return T.reset_index()


# ============================ plotting ============================
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
COL = {"GBM": "#2a78d6", "Flat 28d": "#eb6834", "Poisson GLM": "#1baf7a",
       "Weekday 28d": "#6250d6", "Mean 365d": "#898781"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left", "legend.frameon": False,
})


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print("  saved", path.relative_to(ROOT))


def plot_latest_window(d, R, out, T):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5))
    for ax, (target, title) in zip(axes, TARGETS.items()):
        ctx = d[target].loc[T - pd.Timedelta(days=45): T + pd.Timedelta(days=HORIZON)]
        g = R[(R["origin"] == T) & (R["target"] == target)]
        gb = g[g["model"] == "GBM"].sort_values("h")
        fl = g[g["model"] == "Flat 28d"].sort_values("h")
        dates = pd.to_datetime(gb["date"])
        ax.fill_between(dates, gb["q10"], gb["q90"], color=COL["GBM"], alpha=0.18, lw=0, label="GBM 10-90%")
        ax.plot(dates, gb["q50"], color=COL["GBM"], lw=2, label="GBM median")
        ax.plot(dates, fl["mean"], color=COL["Flat 28d"], lw=2, ls="--", label="Flat 28d mean")
        ax.plot(ctx.index, ctx.values, color=INK, lw=1.2, marker="o", ms=3, label="Actual")
        ax.axvline(T, color=MUTED, lw=1)
        ax.set_title(f"{title}: forecast made on {T.date()} for the next {HORIZON} days (latest period)")
        ax.legend(ncol=4, loc="upper left")
    save(fig, out / "01_latest_window.png")
    return fig


def plot_comparison(M, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for j, (target, title) in enumerate(TARGETS.items()):
        m = M[M["target"] == target]
        for i, metric in enumerate(["MAE", "pinball"]):
            ax = axes[i, j]
            bars = ax.bar(m["model"].astype(str), m[metric], color=[COL[x] for x in m["model"]], width=0.7)
            for b, v in zip(bars, m[metric]):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
            ax.set_title(f"{title}: {metric} (lower is better)")
            ax.tick_params(axis="x", labelrotation=15)
            ax.set_ylim(0, m[metric].max() * 1.15)
    save(fig, out / "02_model_comparison.png")
    return fig


def plot_by_horizon(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (target, title) in zip(axes, TARGETS.items()):
        g = R[R["target"] == target].assign(ae=lambda x: (x["mean"] - x["y"]).abs())
        for model in MODELS:
            s = g[g["model"] == model].groupby("h")["ae"].mean().rolling(3, center=True, min_periods=1).mean()
            ax.plot(s.index, s.values, color=COL[model], lw=2 if model in ("GBM", "Flat 28d") else 1.2, label=model)
        ax.set_title(f"{title}: MAE by horizon")
        ax.set_xlabel("days ahead")
        ax.legend(fontsize=8)
    save(fig, out / "03_mae_by_horizon.png")
    return fig


def plot_totals(Tt, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (act, pred, title) in zip(axes, [("act_alerts", "pred_alerts", f"Total alerts in the next {HORIZON} days"),
                                             ("act_dur", "pred_dur", "Mean alert duration (min)")]):
        a = Tt[Tt["model"] == "GBM"].sort_values("origin")
        ax.plot(a["origin"], a[act], color=INK, lw=2.2, marker="o", ms=4, label="Actual")
        for model in ("GBM", "Poisson GLM", "Flat 28d"):
            b = Tt[Tt["model"] == model].sort_values("origin")
            ax.plot(b["origin"], b[pred], color=COL[model], lw=1.6, label=model)
        ax.set_title(title + " by forecast origin")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelrotation=20)
    save(fig, out / "04_totals_backtest.png")
    return fig


def plot_calibration(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    nominal = np.array(QS)
    for ax, (target, title) in zip(axes, TARGETS.items()):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
        for model in MODELS:
            g = R[(R["target"] == target) & (R["model"] == model)]
            emp = [(g["y"] <= g[f"q{int(q * 100)}"]).mean() for q in QS]
            ax.plot(nominal, emp, color=COL[model], marker="o", lw=1.6, label=model)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("nominal quantile")
        ax.set_ylabel("share of actuals below it")
        ax.set_title(f"{title}: quantile calibration")
        ax.legend(fontsize=8)
    save(fig, out / "05_calibration.png")
    return fig


def plot_final(d, F, best, out, T):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5))
    for ax, (target, title) in zip(axes, TARGETS.items()):
        hist = d[target].loc[T - pd.Timedelta(days=120): T]
        sel = lambda m: F[(F["target"] == target) & (F["model"] == m)].sort_values("h")
        b = best[target]
        g = sel(b)
        dates = pd.to_datetime(g["date"])
        ax.fill_between(dates, g["q10"], g["q90"], color=COL[b], alpha=0.18, lw=0, label=f"{b}: 10-90% band (best on backtest)")
        ax.plot(dates, g["mean"], color=COL[b], lw=2, label=f"{b}: mean")
        for other in ("GBM", "Poisson GLM"):  # show the ML / statistical model next to the best one
            if other != b:
                ax.plot(pd.to_datetime(sel(other)["date"]), sel(other)["mean"], color=COL[other], lw=1.4, label=f"{other}: mean")
        ax.plot(hist.index, hist.values, color=INK, lw=1.2, label="History")
        ax.axvline(T, color=MUTED, lw=1)
        ax.set_title(f"{title}: forecast {dates.min().date()} - {dates.max().date()}")
        ax.legend(ncol=5, loc="upper left", fontsize=8)
    save(fig, out / "06_forecast_next_30d.png")
    return fig


# ============================ main ============================
def run():
    if not (RAW / "volunteer_data_en.csv").exists():
        try:
            download()
        except Exception as ex:
            sys.exit(f"No data found and download failed ({ex}). Put volunteer_data_en.csv into {RAW}")
    out = ROOT / "reports" / slug(REGION) / "forecast"
    out.mkdir(parents=True, exist_ok=True)

    d, last_day = load_daily(REGION)
    print(f"=== {REGION}: daily series {d.index.min().date()} .. {last_day.date()} ({len(d)} days) ===")
    print(f"horizon={HORIZON} days | backtest origins={N_ORIGINS} every {ORIGIN_STEP} days | train window={TRAIN_WINDOW_DAYS} days")

    F = make_features(d)
    D = make_direct(d, F, HORIZON)
    D = D[D["origin"] >= d.index.min() + pd.Timedelta(days=90)]

    # last origin: its 30-day window ends exactly on the last full day -> evaluates the latest period
    last_origin = last_day - pd.Timedelta(days=HORIZON)
    origins = [last_origin - pd.Timedelta(days=ORIGIN_STEP * k) for k in range(N_ORIGINS)][::-1]
    print(f"backtest origins: {origins[0].date()} .. {origins[-1].date()}  (latest test window: "
          f"{(last_origin + pd.Timedelta(days=1)).date()} .. {last_day.date()})\n")

    recs = []
    for i, T in enumerate(origins, 1):
        recs.append(predict_origin(d, D, T))
        print(f"  origin {i:2d}/{len(origins)} {T.date()} done", flush=True)
    R = pd.concat(recs, ignore_index=True)
    R.to_csv(out / "backtest_predictions.csv", index=False)

    # ---------------- metrics ----------------
    M = metrics_table(R)
    M.to_csv(out / "metrics_all_origins.csv", index=False)
    Mlast = metrics_table(R[R["origin"] == last_origin])
    Mlast.to_csv(out / "metrics_latest_window.csv", index=False)
    pd.set_option("display.width", 200)
    fmt = lambda t: t.round(3).to_string(index=False)
    print("\n=== METRICS over ALL backtest origins (daily forecasts, horizon 1-%d) ===" % HORIZON)
    print(fmt(M))
    print(f"\n=== METRICS on the LATEST window only (origin {last_origin.date()}) ===")
    print(fmt(Mlast))

    print("\n=== SKILL vs 'Flat 28d' (1 - MAE_model / MAE_flat; >0 means better than the naive level) ===")
    for target in TARGETS:
        m = M[M["target"] == target].set_index("model")
        print(f"  {target}: " + ", ".join(f"{k}: {1 - m.loc[k, 'MAE'] / m.loc['Flat 28d', 'MAE']:+.1%}"
                                          for k in MODELS if k != "Flat 28d"))

    Tt = totals_table(R)
    Tt.to_csv(out / "totals_backtest.csv", index=False)
    print(f"\n=== {HORIZON}-DAY TOTALS: mean absolute % error across origins ===")
    rows = []
    for model, g in Tt.groupby("model"):
        rows.append({"model": model,
                     "alerts_MAPE": (g["pred_alerts"] / g["act_alerts"] - 1).abs().mean(),
                     "hours_MAPE": (g["pred_hours"] / g["act_hours"] - 1).abs().mean(),
                     "duration_MAPE": (g["pred_dur"] / g["act_dur"] - 1).abs().mean()})
    tm = pd.DataFrame(rows).set_index("model").reindex(MODELS)
    print((tm * 100).round(1).astype(str).add("%").to_string())

    # ---------------- final forecast ----------------
    best = {t: M[(M["target"] == t)].sort_values("pinball").iloc[0]["model"] for t in TARGETS}
    print("\nbest model per target by pinball loss on the backtest:", {k: str(v) for k, v in best.items()})
    final_origin = last_day
    Fr = predict_origin(d, D, final_origin, with_truth=False)
    Fr.to_csv(out / "forecast_next_30d.csv", index=False)

    fut_dates = pd.to_datetime(Fr[Fr["model"] == "GBM"].sort_values("h")["date"].drop_duplicates())
    print(f"\n=== FORECAST {fut_dates.min().date()} .. {fut_dates.max().date()} (origin {final_origin.date()}) ===")
    for target in TARGETS:
        g = Fr[(Fr["target"] == target) & (Fr["model"] == best[target])]
        a = Tt[Tt["model"] == best[target]]
        col = "alerts" if target == "n_alerts" else "hours"
        rel = (a[f"act_{col}"] / a[f"pred_{col}"] - 1).quantile([0.1, 0.9]).values  # empirical error of totals
        tot = g["mean"].sum()
        print(f"  {target:11s} model={best[target]}: expected total = {tot:.0f}  (80% range from backtest errors: "
              f"{tot * (1 + rel[0]):.0f} .. {tot * (1 + rel[1]):.0f}) | per day: {tot / HORIZON:.2f}")
    tn = Fr[(Fr["target"] == "n_alerts") & (Fr["model"] == best["n_alerts"])]["mean"].sum()
    th = Fr[(Fr["target"] == "alert_hours") & (Fr["model"] == best["alert_hours"])]["mean"].sum()
    print(f"  expected mean alert duration: {th / tn * 60:.1f} min  "
          f"(last 28 days actual: {d.alert_hours.iloc[-28:].sum() / d.n_alerts.iloc[-28:].sum() * 60:.1f} min)")
    print("  last 28 days actual: %.2f alerts/day, %.2f hours/day" % (d.n_alerts.iloc[-28:].mean(), d.alert_hours.iloc[-28:].mean()))

    # ---------------- figures ----------------
    print("\nfigures:")
    figs = [plot_latest_window(d, R, out, last_origin), plot_comparison(M, out), plot_by_horizon(R, out),
            plot_totals(Tt, out), plot_calibration(R, out), plot_final(d, Fr, best, out, final_origin)]
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
    (ROOT / "reports" / slug(REGION) / "forecast" / "forecast_report.txt").write_text(buf.getvalue(), encoding="utf-8")
