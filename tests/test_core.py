import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airalerts import config as C
from airalerts import data, targets


def test_merge_intervals_union():
    s = np.array([0, 5, 20, 8]); e = np.array([10, 7, 30, 12])
    ms, me = data.merge_intervals(s, e)
    assert list(zip(ms, me)) == [(0, 12), (20, 30)]


def test_heavy_labels_ignore_future():
    """Changing data after day t must not change the label of day t."""
    idx = pd.MultiIndex.from_product([["A"], pd.date_range("2024-01-01", periods=500)], names=["region", "date"])
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"alert_hours": rng.gamma(2, 2, 500), "n_alerts": rng.poisson(3, 500)}, index=idx)
    full = targets.heavy_day_labels(d)
    cut = d.copy(); cut.iloc[400:] = 99.0
    alt = targets.heavy_day_labels(cut)
    a = full.iloc[:400].dropna(); b = alt.iloc[:400].dropna()
    assert a.equals(b)


def test_massive_labels_ignore_future():
    rng = np.random.default_rng(1)
    nat = pd.DataFrame({"rear_hours": rng.gamma(3, 5, 500), "rear_breadth": rng.integers(0, 16, 500)},
                       index=pd.date_range("2024-01-01", periods=500))
    full = targets.massive_strike_labels(nat)
    cut = nat.copy(); cut.iloc[300:] = [999.0, 15]
    assert full.iloc[:300].dropna().equals(targets.massive_strike_labels(cut).iloc[:300].dropna())


def test_panel_matches_bruteforce():
    if not (C.RAW_DIR / C.VOLUNTEER_FILE).exists():
        return
    P = data.build_panel()
    raw = data.load_raw()
    reg = P.regions[0]
    day = P.daily.loc[reg].index[-40]
    r = raw[raw["region"] == reg]
    st = r["started_at"].dt.tz_convert(C.TZ) if r["started_at"].dt.tz is not None else r["started_at"].dt.tz_localize("UTC").dt.tz_convert(C.TZ)
    n = int((st.dt.normalize().dt.tz_localize(None) == pd.Timestamp(day)).sum())
    assert n == int(P.daily.loc[(reg, day), "n_alerts"])


def test_excluded_region_absent():
    if (C.RAW_DIR / C.VOLUNTEER_FILE).exists():
        P = data.build_panel()
        assert "Luhanska oblast" not in P.regions
