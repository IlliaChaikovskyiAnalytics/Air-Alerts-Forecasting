"""Air-raid alerts: exploratory time-series analysis for ONE region (default: Kyiv City).

PyCharm version: edit the CONFIG block below and press Run (Shift+F10). No command-line arguments needed.
    pip install pandas numpy scipy matplotlib statsmodels   # statsmodels is optional

Raw data is UTC; calendar features use Europe/Kyiv local time. The last incomplete day is dropped.
Outputs: ./reports/<region>/ (figures + eda_report.txt) and ./data/<region>_{hourly,daily}.csv
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import urllib.request
from pathlib import Path

# ============================ CONFIG (edit me) ============================
REGION = "Kyiv City"      # e.g. "Kharkivska oblast", "Lvivska oblast"; see list printed by LIST_REGIONS
SOURCE = "volunteer"      # "volunteer" (oblast level, up to date) or "official" (raion level, may lag)
HOLDOUT_WEEKS = 4         # length of the candidate holdout window used in the regime check
DOWNLOAD = False          # True -> re-download the latest CSVs before running (also happens if files are missing)
SHOW_PLOTS = True         # True -> show figures in PyCharm (SciView / window) as well as saving them
LIST_REGIONS = False      # True -> only print available region names and exit
# ==========================================================================

import matplotlib

if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

try:  # optional, nicer tests
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller, kpss
    HAVE_SM = True
except Exception:
    HAVE_SM = False

ROOT = Path(__file__).resolve().parent.parent   # repo root
RAW = ROOT / "data" / "raw"
TZ = "Europe/Kyiv"
BASE_URL = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/"


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in ("volunteer_data_en.csv", "official_data_en.csv"):
        print("downloading", name, flush=True)
        urllib.request.urlretrieve(BASE_URL + name, RAW / name)


# ============================ data ============================


def load_events(region: str = "Kyiv City", source: str = "volunteer") -> pd.DataFrame:
    """Return alert events for one region, intervals merged (no overlaps).

    columns: started_at, finished_at (UTC, tz-aware), naive (bool), duration_min
    """
    if source == "volunteer":
        df = pd.read_csv(RAW / "volunteer_data_en.csv", parse_dates=["started_at", "finished_at"])
        df = df[df["region"] == region].copy()
    elif source == "official":
        df = pd.read_csv(RAW / "official_data_en.csv", parse_dates=["started_at", "finished_at"])
        df = df[df["oblast"] == region].copy()
        df["naive"] = False
    else:
        raise ValueError(source)

    df = df[["started_at", "finished_at", "naive"]].sort_values("started_at").reset_index(drop=True)
    df = df[df["finished_at"] > df["started_at"]]
    return merge_overlaps(df)


def dataset_end(source: str = "volunteer") -> pd.Timestamp:
    """Last moment covered by the source file (UTC). Needed so that 'no alert'
    at the tail of the series is not confused with 'no data'."""
    f = {"volunteer": "volunteer_data_en.csv", "official": "official_data_en.csv"}[source]
    df = pd.read_csv(RAW / f, usecols=["finished_at"], parse_dates=["finished_at"])
    return df["finished_at"].max().ceil("h")


def last_full_local_day(end_utc: pd.Timestamp) -> pd.Timestamp:
    """Last calendar day (Kyiv time, tz-naive) that is fully covered by the data."""
    end_loc = end_utc.tz_convert(TZ)
    day = end_loc.tz_localize(None).normalize()
    return day if end_loc.hour == 0 and end_loc.minute == 0 else day - pd.Timedelta(days=1)


def merge_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping/touching intervals (official data has raion-level rows)."""
    out, cur_s, cur_e, cur_n = [], None, None, False
    for s, e, n in df[["started_at", "finished_at", "naive"]].itertuples(index=False):
        if cur_s is None:
            cur_s, cur_e, cur_n = s, e, n
        elif s <= cur_e:
            cur_e = max(cur_e, e)
            cur_n = cur_n or n
        else:
            out.append((cur_s, cur_e, cur_n))
            cur_s, cur_e, cur_n = s, e, n
    if cur_s is not None:
        out.append((cur_s, cur_e, cur_n))
    res = pd.DataFrame(out, columns=["started_at", "finished_at", "naive"])
    res["duration_min"] = (res["finished_at"] - res["started_at"]).dt.total_seconds() / 60
    return res


def hourly_series(events: pd.DataFrame, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Hourly UTC grid with:
    - alert_frac: fraction of the hour under alert (0..1)
    - active:     1 if any alert overlapped the hour
    - starts:     number of alerts that began in the hour
    """
    start = events["started_at"].min().floor("h")
    end = (end or events["finished_at"].max()).ceil("h")
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    frac = np.zeros(len(idx))
    t0 = idx[0]
    for s, e in events[["started_at", "finished_at"]].itertuples(index=False):
        i0 = int((s - t0) / pd.Timedelta(hours=1))
        i1 = int((e - t0) / pd.Timedelta(hours=1))
        for i in range(max(i0, 0), min(i1, len(idx) - 1) + 1):
            lo = max(s, idx[i])
            hi = min(e, idx[i] + pd.Timedelta(hours=1))
            if hi > lo:
                frac[i] += (hi - lo).total_seconds() / 3600
    out = pd.DataFrame({"alert_frac": np.clip(frac, 0, 1)}, index=idx)
    out["active"] = (out["alert_frac"] > 0).astype(int)
    starts = events["started_at"].dt.floor("h").value_counts()
    out["starts"] = starts.reindex(idx, fill_value=0).values
    return out


def add_local_calendar(df: pd.DataFrame) -> pd.DataFrame:
    loc = df.index.tz_convert(TZ)
    df = df.copy()
    df["local_hour"] = loc.hour
    df["dow"] = loc.dayofweek
    df["date"] = loc.tz_localize(None).normalize()
    df["month"] = loc.tz_localize(None).to_period("M")
    return df


# ============================ analysis ============================
# ---- plotting style (restrained: one blue, one orange accent, sequential blue ramp) ----
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blues", RAMP)
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left",
})
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ------------------------------------------------------------------ statistics helpers
def adf_manual(y, maxlag=14):
    """Fallback ADF (constant, lag chosen by AIC); asymptotic 5% critical value -2.86."""
    y = np.asarray(y, float)
    dy = np.diff(y)
    best = None
    for k in range(maxlag + 1):
        n = len(dy) - k
        X = np.column_stack([y[k:-1], np.ones(n)] + [dy[k - j - 1: len(dy) - j - 1] for j in range(k)])
        Y = dy[k:]
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        aic = n * np.log(res @ res / n) + 2 * X.shape[1]
        if best is None or aic < best[0]:
            s2 = res @ res / (n - X.shape[1])
            best = (aic, beta[0] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0]), k)
    return best[1], best[2]


def adf_test(y):
    if HAVE_SM:
        stat, p, lags, *_ = adfuller(np.asarray(y, float), autolag="AIC")
        return stat, p, lags
    stat, lags = adf_manual(y)
    return stat, np.nan, lags


def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    d = x @ x
    return np.array([x[: len(x) - k] @ x[k:] / d for k in range(nlags + 1)])


def slug(region: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", region.lower()).strip("_")


def style_axes(ax):
    ax.tick_params(length=0)


def save_fig(fig, path, dpi=140):
    """Save the figure and (optionally) show it in PyCharm."""
    fig.savefig(path, dpi=dpi)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------ main analysis
def run(region: str, source: str = "volunteer", holdout_weeks: int = 4):
    out_dir = ROOT / "reports" / slug(region)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)

    end = dataset_end(source)
    last_day = last_full_local_day(end)

    ev = load_events(region, source)
    h = add_local_calendar(hourly_series(ev, end=end))
    h = h[h["date"] <= last_day]

    loc = ev["started_at"].dt.tz_convert(TZ)
    e = ev.assign(hour=loc.dt.hour, dow=loc.dt.dayofweek, year=loc.dt.year,
                  date=loc.dt.tz_localize(None).dt.normalize(),
                  month=loc.dt.tz_localize(None).dt.to_period("M"))
    e = e[e["date"] <= last_day]

    daily = (h.groupby("date").agg(alert_hours=("alert_frac", "sum"), n_alerts=("starts", "sum"))
             .asfreq("D", fill_value=0))

    # ---------------------------------------------------------------- overview
    print(f"=== {region} | source={source} ===")
    print(f"data end (UTC): {end} | last full local day: {last_day.date()}")
    print(f"period: {h.index.min()} -> {h.index.max()} | hours: {len(h)}")
    print(f"alerts: {len(e)} | 'naive' (no end message): {int(e.naive.sum())} ({e.naive.mean():.1%})")
    print(f"share of hours with any alert: {h.active.mean():.3f} | share of time under alert: {h.alert_frac.mean():.3f}")
    print("\nper year:")
    for y, g in h.groupby(h.index.tz_convert(TZ).year):
        ey = e[e.year == y]
        print(f"  {y}: alerts={len(ey):5d}  P(active hour)={g.active.mean():.3f}  "
              f"time_under_alert={g.alert_frac.mean():.3f}  median_dur={ey.duration_min.median():.1f} min")

    # ---------------------------------------------------------------- Fig 1: overview
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    ax[0].plot(daily.index, daily.alert_hours, color=BLUE, lw=0.6, alpha=0.45)
    ax[0].plot(daily.index, daily.alert_hours.rolling(28, center=True).mean(), color=BLUE, lw=2)
    ax[0].set_title("Hours under alert per day (thin: daily, thick: 28-day mean)")
    ax[1].plot(daily.index, daily.n_alerts, color=ORANGE, lw=0.6, alpha=0.45)
    ax[1].plot(daily.index, daily.n_alerts.rolling(28, center=True).mean(), color=ORANGE, lw=2)
    ax[1].set_title("Number of alerts per day")
    md = e.groupby("month").duration_min.median()
    ax[2].step(md.index.to_timestamp(), md.values, where="post", color=BLUE, lw=2)
    ax[2].set_title("Median alert duration by month (min)")
    for a in ax:
        style_axes(a)
    fig.tight_layout()
    save_fig(fig, fig_dir / "01_overview.png", 140)

    # ---------------------------------------------------------------- Fig 2: hour / weekday
    by_year = pd.crosstab(e.hour, e.year).reindex(index=range(24), fill_value=0)
    by_year = by_year / by_year.sum()
    recent = e[e.date > last_day - pd.Timedelta(days=180)]
    hw = pd.crosstab(recent.hour, recent.dow).reindex(index=range(24), columns=range(7), fill_value=0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    im = ax[0].imshow(by_year.values, aspect="auto", cmap=CMAP, origin="lower")
    ax[0].set_xticks(range(by_year.shape[1]), by_year.columns)
    ax[0].set_yticks(range(0, 24, 3))
    ax[0].set_ylabel("local hour")
    ax[0].set_title("Share of each year's alert starts by hour")
    plt.colorbar(im, ax=ax[0], fraction=0.046)
    im = ax[1].imshow(hw.values, aspect="auto", cmap=CMAP, origin="lower")
    ax[1].set_xticks(range(7), DOW)
    ax[1].set_yticks(range(0, 24, 3))
    ax[1].set_title("Alert starts, last 180 days: hour x weekday")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    for a in ax:
        a.grid(False)
    fig.tight_layout()
    save_fig(fig, fig_dir / "02_hour_weekday.png", 140)

    # ---------------------------------------------------------------- Fig 3: duration & gaps
    evs = ev.sort_values("started_at").reset_index(drop=True)
    gap_h = ((evs.started_at.shift(-1) - evs.finished_at).dt.total_seconds() / 3600).dropna()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.clip(evs.duration_min, 0, 240), bins=48, color=BLUE)
    ax[0].set_title("Alert duration (min, clipped at 240)")
    ax[1].hist(np.clip(gap_h, 0, 48), bins=48, color=ORANGE)
    ax[1].set_title("Gap between alerts (h, clipped at 48)")
    fig.tight_layout()
    save_fig(fig, fig_dir / "03_duration_gap.png", 140)

    # ---------------------------------------------------------------- Fig 4: ACF
    a_h = acf(h.active.values, 24 * 14)
    a_d = acf(daily.alert_hours.values, 30)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(range(len(a_h)), a_h, color=BLUE, width=1.0)
    ax[0].set_title("ACF of hourly 'active' flag (lag in hours, 14 days)")
    ax[1].bar(range(len(a_d)), a_d, color=ORANGE, width=0.8)
    ax[1].set_title("ACF of daily alert-hours (lag in days)")
    fig.tight_layout()
    save_fig(fig, fig_dir / "04_acf.png", 140)

    # ---------------------------------------------------------------- Fig 5: monthly + recent
    mon = daily.alert_hours.groupby(daily.index.to_period("M")).mean()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1.2]})
    ax[0].bar(mon.index.to_timestamp(), mon.values, width=24, color=BLUE)
    ax[0].set_title("Mean alert-hours per day, by month")
    z = daily.iloc[-120:]
    ax[1].bar(z.index, z.alert_hours, width=0.9, color=BLUE)
    ax[1].set_title("Last 120 days: alert-hours per day")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, fig_dir / "05_monthly_recent.png", 140)

    # ---------------------------------------------------------------- STL (needs statsmodels)
    if HAVE_SM and len(daily) > 400:
        res = STL(daily.alert_hours.iloc[-730:], period=7, robust=True).fit()
        fs = max(0, 1 - res.resid.var() / (res.seasonal + res.resid).var())
        ft = max(0, 1 - res.resid.var() / (res.trend + res.resid).var())
        print(f"\n=== STL (weekly, last 730 days) === strength of weekly seasonality Fs={fs:.2f}, trend Ft={ft:.2f}")
        fig = res.plot()
        fig.set_size_inches(11, 7)
        save_fig(fig, fig_dir / "06_stl_weekly.png", 120)

    # ---------------------------------------------------------------- profiles
    ly = h[h.index >= h.index.max() - pd.Timedelta(days=365)]
    prof = ly.groupby("local_hour").active.mean()
    st = e[e.date > last_day - pd.Timedelta(days=365)].hour.value_counts(normalize=True).sort_index()
    print("\n=== HOURLY PROFILE, last 365 days (local hour) ===")
    print(pd.DataFrame({"P_active": prof.round(3), "start_share": st.reindex(prof.index, fill_value=0).round(3)}).T.to_string())
    day = ly[(ly.local_hour >= 6) & (ly.local_hour < 22)].active.mean()
    night = ly[(ly.local_hour < 6) | (ly.local_hour >= 22)].active.mean()
    print(f"P(active) day(06-22) / night(22-06) = {day / night:.2f}")

    dd = daily[daily.index > last_day - pd.Timedelta(days=365)]
    dw = dd.groupby(dd.index.dayofweek).alert_hours.mean()
    print("\n=== DAY OF WEEK, last 365 days: mean alert-hours/day ===")
    print({DOW[k]: round(v, 2) for k, v in dw.items()})
    H, p = stats.kruskal(*[g.alert_hours.values for _, g in dd.groupby(dd.index.dayofweek)])
    print(f"Kruskal-Wallis across weekdays: H={H:.2f}, p={p:.3f}")

    # ---------------------------------------------------------------- persistence
    a = h.active.values
    print("\n=== PERSISTENCE ===")
    print(f"P(active t+1 | active t)={a[1:][a[:-1] == 1].mean():.3f}  "
          f"P(active t+1 | idle t)={a[1:][a[:-1] == 0].mean():.4f}  unconditional={a.mean():.3f}")
    a_long = acf(a, 168)
    print("ACF hourly 'active' at lags", {k: round(float(a_long[k]), 3) for k in (1, 2, 3, 6, 12, 24, 168)})
    ad = acf(daily.alert_hours.values, 14)
    print("ACF daily alert-hours at lags", {k: round(float(ad[k]), 3) for k in (1, 2, 7, 14)})

    # ---------------------------------------------------------------- stationarity & trend
    print("\n=== STATIONARITY / TREND ===" + ("" if HAVE_SM else "  (statsmodels missing -> manual ADF only)"))
    series = {"daily alert-hours (all)": daily.alert_hours,
              "daily alert-hours (last 365d)": daily.alert_hours.iloc[-365:],
              "daily n_alerts (last 365d)": daily.n_alerts.iloc[-365:]}
    for name, s in series.items():
        t, p, k = adf_test(s.values)
        verdict = (p < 0.05) if not np.isnan(p) else (t < -2.86)
        line = f"  ADF {name}: stat={t:.2f}" + ("" if np.isnan(p) else f", p={p:.3f}") + f", lags={k} -> "
        print(line + ("stationary (reject unit root)" if verdict else "cannot reject unit root"))
        if HAVE_SM:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InterpolationWarning)
                ks, kp, *_ = kpss(s.values, regression="c", nlags="auto")
            print(f"     KPSS: stat={ks:.2f}, p={kp:.3f} -> " + ("stationary" if kp > 0.05 else "non-stationary (reject H0)"))
    for name, s in {"last 365d": daily.alert_hours.iloc[-365:], "last 90d": daily.alert_hours.iloc[-90:]}.items():
        tau, p = stats.kendalltau(np.arange(len(s)), s.values)
        print(f"  Kendall trend of daily alert-hours, {name}: tau={tau:.3f}, p={p:.4f}")

    # ---------------------------------------------------------------- counts, duration, gaps
    x = daily.n_alerts.iloc[-365:]
    print("\n=== DAILY COUNT DISPERSION (last 365 days) ===")
    print(f"mean={x.mean():.2f} var={x.var():.2f} var/mean={x.var() / max(x.mean(), 1e-9):.2f} "
          f"zero-days={(x == 0).mean():.1%}")
    e1 = e[e.date > last_day - pd.Timedelta(days=365)]
    print("\n=== DURATION (last 365 days, minutes) ===")
    print(e1.duration_min.describe(percentiles=[.1, .5, .9, .99]).round(1).to_dict())
    print(f"share of alerts > 2h: {(e1.duration_min > 120).mean():.3f}")
    evs["gap_h"] = (evs.started_at.shift(-1) - evs.finished_at).dt.total_seconds() / 3600
    g1 = evs[evs.started_at > end - pd.Timedelta(days=365)].gap_h.dropna()
    print("gap to next alert (h) quantiles:", g1.quantile([.1, .25, .5, .75, .9]).round(2).to_dict())
    print(f"next alert within 1h: {(g1 < 1).mean():.1%}; within 3h: {(g1 < 3).mean():.1%}")

    # ---------------------------------------------------------------- regime check for holdout
    n = holdout_weeks * 7
    print(f"\n=== REGIME CHECK: last {holdout_weeks} weeks (candidate holdout) vs the {holdout_weeks} weeks before ===")
    cur, prev = daily.iloc[-n:], daily.iloc[-2 * n:-n]
    for lab, s in (("previous", prev), ("last", cur)):
        print(f"  {lab:9s} {s.index.min().date()}..{s.index.max().date()}  alerts/day={s.n_alerts.mean():.2f}  "
              f"alert-hours/day={s.alert_hours.mean():.2f}  zero-days={(s.n_alerts == 0).mean():.1%}")
    u, pu = stats.mannwhitneyu(cur.n_alerts, prev.n_alerts)
    print(f"  Mann-Whitney on daily n_alerts: U={u:.0f}, p={pu:.4g}")
    print("  weekly totals (last 12 weeks):")
    print(daily.iloc[-84:].resample("W").agg({"alert_hours": "mean", "n_alerts": "sum"}).round(2).to_string())

    # ---------------------------------------------------------------- cross-check with official source
    if source == "volunteer":
        try:
            off = load_events(region, "official")
            o_end = dataset_end("official")
            lo = off.started_at.min().floor("h")
            hi = min(o_end, end) - pd.Timedelta(days=7)  # official feed can lag; stay clear of the tail
            idx = pd.date_range(lo, hi, freq="h", tz="UTC", inclusive="left")
            hv = hourly_series(ev[ev.started_at < hi], end=hi).active.reindex(idx, fill_value=0)
            ho = hourly_series(off[off.started_at < hi], end=hi).active.reindex(idx, fill_value=0)
            po = (hv == ho).mean()
            pe = hv.mean() * ho.mean() + (1 - hv.mean()) * (1 - ho.mean())
            print("\n=== VOLUNTEER vs OFFICIAL (overlap period) ===")
            print(f"official data ends {o_end}; compared up to {hi}")
            print(f"hourly agreement={po:.3f}, Cohen's kappa={(po - pe) / (1 - pe):.3f}; "
                  f"alerts: volunteer={int((ev.started_at < hi).sum())}, official={int((off.started_at < hi).sum())}")
            print("note: official oblast rows are raion-level since Dec 2025 -> merged to 'any raion alert'.")
        except Exception as ex:  # keep EDA robust
            print("\n(official cross-check skipped:", ex, ")")

    daily.to_csv(ROOT / "data" / f"{slug(region)}_daily.csv")
    h.to_csv(ROOT / "data" / f"{slug(region)}_hourly.csv")
    print(f"\nfigures -> {fig_dir}\nseries  -> data/{slug(region)}_{{hourly,daily}}.csv")


def main():
    if DOWNLOAD or not (RAW / "volunteer_data_en.csv").exists():
        try:
            download()
        except Exception as ex:
            sys.exit(f"Could not download data ({ex}).\nClone https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset "
                     f"and copy datasets/volunteer_data_en.csv and official_data_en.csv into {RAW}")
    if LIST_REGIONS:
        v = pd.read_csv(RAW / "volunteer_data_en.csv", usecols=["region"])
        print("\n".join(sorted(v.region.unique())))
        return

    buf = io.StringIO()

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)

        def flush(self):
            sys.__stdout__.flush()

    with contextlib.redirect_stdout(Tee()):
        run(REGION, SOURCE, HOLDOUT_WEEKS)
    rep = ROOT / "reports" / slug(REGION) / "eda_report.txt"
    rep.write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
