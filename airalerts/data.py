"""Download, cleaning and the region x day panel.

Raw source: volunteer dataset (oblast level). All timestamps are UTC; days are Europe/Kyiv local days.
Everything is vectorised: building the full 24-region panel takes a few seconds.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


def download() -> None:
    """Fetch the latest volunteer CSV (updated daily upstream)."""
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(C.DATA_URL, C.RAW_DIR / C.VOLUNTEER_FILE)


def load_raw() -> pd.DataFrame:
    path = C.RAW_DIR / C.VOLUNTEER_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python run_pipeline.py --download` "
                                f"or copy {C.VOLUNTEER_FILE} into {C.RAW_DIR}")
    df = pd.read_csv(path, parse_dates=["started_at", "finished_at"])
    return df[["region", "started_at", "finished_at", "naive"]]


def merge_intervals(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge overlapping / touching intervals. Inputs are float seconds; output sorted, disjoint."""
    order = np.argsort(starts, kind="stable")
    s, e = starts[order], ends[order]
    keep = e > s
    s, e = s[keep], e[keep]
    if len(s) == 0:
        return s, e
    run_end = np.maximum.accumulate(e)
    new_group = np.concatenate([[True], s[1:] > run_end[:-1]])
    gid = np.cumsum(new_group) - 1
    first = np.flatnonzero(new_group)
    return s[first], np.maximum.reduceat(e, first)


def covered_seconds(starts: np.ndarray, ends: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Seconds covered by the (disjoint, sorted) intervals inside each [bounds[i], bounds[i+1])."""
    if len(starts) == 0:
        return np.zeros(len(bounds) - 1)
    dur = ends - starts
    before = np.concatenate([[0.0], np.cumsum(dur)[:-1]])  # covered time before interval j
    j = np.searchsorted(starts, bounds, side="right") - 1
    cum = np.where(j >= 0, before[np.maximum(j, 0)] + np.clip(bounds - starts[np.maximum(j, 0)], 0, dur[np.maximum(j, 0)]), 0.0)
    return np.diff(cum)


@dataclass
class Panel:
    daily: pd.DataFrame          # index (region, date); columns n_alerts, alert_hours
    active_hourly: pd.DataFrame  # index hourly UTC, columns regions, bool: any alert in that hour
    data_end: pd.Timestamp       # end of data coverage (UTC)
    last_day: pd.Timestamp       # last fully covered local day (tz-naive)
    regions: list[str]
    naive_share: pd.Series       # per region share of alerts without an end message

    @property
    def first_day(self) -> pd.Timestamp:
        return self.daily.index.get_level_values("date").min()


def build_panel(exclude: tuple[str, ...] = C.EXCLUDED_REGIONS) -> Panel:
    raw = load_raw()
    data_end = raw["finished_at"].max().ceil("h")
    regions = sorted(r for r in raw["region"].unique() if r not in exclude)

    t0 = raw["started_at"].min().floor("h")
    grid = pd.date_range(t0, data_end, freq="h", tz="UTC", inclusive="left")
    bounds = np.arange(len(grid) + 1) * 3600.0
    t0v = t0.to_datetime64()

    local_date = grid.tz_convert(C.TZ).tz_localize(None).normalize()
    day_codes, days = pd.factorize(local_date)
    n_days = len(days)

    n_alerts = np.zeros((n_days, len(regions)))
    alert_h = np.zeros((n_days, len(regions)))
    active = np.zeros((len(grid), len(regions)), dtype=bool)
    naive_share = {}

    for k, region in enumerate(regions):
        g = raw[raw["region"] == region]
        s = (g["started_at"].values.astype("datetime64[s]") - t0v.astype("datetime64[s]")).astype(float)
        e = (g["finished_at"].values.astype("datetime64[s]") - t0v.astype("datetime64[s]")).astype(float)
        naive_share[region] = float(g["naive"].mean())
        ms, me = merge_intervals(s, e)
        frac = covered_seconds(ms, me, bounds) / 3600.0
        active[:, k] = frac > 0
        alert_h[:, k] = np.bincount(day_codes, weights=frac, minlength=n_days)
        start_hour = np.floor(ms / 3600.0).astype(int)  # merged alerts = what a resident experiences
        n_alerts[:, k] = np.bincount(day_codes[np.clip(start_hour, 0, len(grid) - 1)], minlength=n_days)

    end_loc = data_end.tz_convert(C.TZ)
    last_day = end_loc.tz_localize(None).normalize()
    if not (end_loc.hour == 0 and end_loc.minute == 0):
        last_day -= pd.Timedelta(days=1)

    idx = pd.DatetimeIndex(days, name="date")
    daily = pd.concat({r: pd.DataFrame({"n_alerts": n_alerts[:, k], "alert_hours": alert_h[:, k]}, index=idx)
                       for k, r in enumerate(regions)}, names=["region", "date"])
    daily = daily[daily.index.get_level_values("date") <= last_day]
    ah = pd.DataFrame(active, index=grid, columns=regions)
    return Panel(daily=daily, active_hourly=ah, data_end=data_end, last_day=last_day, regions=regions,
                 naive_share=pd.Series(naive_share))
