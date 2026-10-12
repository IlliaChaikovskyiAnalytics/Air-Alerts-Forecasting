"""RESULTS.md generator. Wording is neutral and driven by the computed numbers (no hand-written conclusions)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as C
from . import metrics as MT


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df is None or df.empty:
        return "_n/a_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)) + "\n"


def _skill_line(df, model, ref, loss, label):
    try:
        s = MT.boot_skill(df, model, ref, loss, None, n_boot=C.N_BOOT, seed=C.SEED).iloc[0]
        return f"- {label}: `{model}` vs `{ref}` skill = {s['skill']:+.1%} (90% block-bootstrap CI {s['lo90']:+.1%} .. {s['hi90']:+.1%})\n"
    except Exception as e:  # report must never fail the pipeline
        return f"- {label}: skill not computed ({e})\n"


def write_report(panel, A, bt, sel, tables, fc, path: Path) -> None:
    L = []
    L.append("# Air-raid alerts: 7-day forecast and backtest\n")
    L.append(f"Data: {len(panel.regions)} oblasts (Luhanska excluded; Crimea is not in the source), last full day "
             f"**{panel.last_day.date()}**, source coverage ends {panel.data_end}. Forecast origin = end of {panel.last_day.date()}; "
             f"horizon {C.HORIZON} days. Calendar in {C.TZ}.\n")
    L.append("## Definitions\n- alerts/day = number of alert intervals started that local day; alert-hours/day = total covered time (union of intervals).\n"
             f"- **Heavy region-day**: alert-hours >= trailing {C.HEAVY_LOOKBACK}-day q{int(C.HEAVY_Q*100)} of that region "
             f"(computed from past days only) and >= {C.HEAVY_MIN['alert_hours']} h.\n"
             f"- **Massive-strike day (proxy)**: summed alert-hours of rear regions >= trailing q{int(C.MASS_Q*100)} and alerts in >= {C.MASS_MIN_BREADTH} rear regions. "
             "No ground-truth labels exist: heavy and massive are data-derived proxies.\n")
    L.append("## Selected models (chosen on data ending 28 days before the last day)\n" + "".join(f"- {k}: `{v}`\n" for k, v in sel.items()))

    r = fc["regional"]
    day1 = r[r["h"] == 1].sort_values("p_heavy_day", ascending=False)
    L.append("\n## Forecast, day 1 (regions by P(heavy day))\n" + md_table(
        day1[["region", "date", "alerts_expected", "hours_expected", "p_heavy_day"]].head(10)))
    L.append("\n## Massive strike\n" + md_table(fc["national_days"][["date", "weekday", "p_massive_day"]])
             + "\nP(at least one massive-strike day in the next 7 days):\n" + md_table(fc["national_any7"]))

    L.append("\n## Backtest (rolling origin, strict no-leakage)\n"
             f"{C.BACKTEST_DAYS} days, refit every {C.REFIT_EVERY_DAYS} days; 'latest' = last {C.LATEST_WINDOW_DAYS} target days.\n")
    for key, title in (("alerts", "Alerts per day (sMAE/sPinball are region-normalised)"), ("hours", "Alert-hours per day"),
                       ("heavy", "Heavy day (probability)"), ("massive_day", "Massive-strike day (probability)"),
                       ("massive_any7", "Any massive strike within 7 days")):
        L.append(f"\n### {title}\nAll origins:\n" + md_table(tables[f"{key}_all"]) + "\nLatest window:\n" + md_table(tables[f"{key}_latest"]))

    L.append("\n## Skill vs simple baselines (all origins)\n")
    d = MT.add_reg_losses(bt.reg, "n")
    L.append(_skill_line(d, "GBM", "EWMA 14d", "pin_s", "alerts, GBM"))
    d = MT.add_reg_losses(bt.reg, "h")
    L.append(_skill_line(d, "GBM", "Flat 28d", "pin_s", "hours, GBM"))
    d = MT.add_prob_losses(bt.heavy, "y_heavy")
    L.append(_skill_line(d, "Blend", "Clim 10%", "brier", "heavy day, Blend"))
    L.append(_skill_line(d, "GBM", "Clim 10%", "brier", "heavy day, GBM"))

    L.append("\n## Diagnostics\n### Ljung-Box: autocorrelation of forecast errors (selected models)\n"
             "Standardised errors (forecast - actual, divided by the region's level scale), per region, 7 lags. "
             "Under no autocorrelation about 5% of regions would have p < 0.05; `mean_resid_z` < 0 means under-forecasting. "
             "`national_*` is the same test on the cross-region mean error.\n" + md_table(tables["ljung_box"], 4)
             + "\n### Diebold-Mariano: each candidate vs a fixed reference model\n"
             "Loss differential = loss(reference) - loss(model) per target day (mean over regions), HAC variance with lag h-1, "
             "Harvey-Leybourne-Newbold correction. `rel_gain` > 0 and small p: the model is significantly better than the reference. "
             "References are fixed a priori (Flat 28d, Recent 28d, Clim, Recent 28d independence). Many comparisons are shown; "
             "do not read single p-values near 0.05 as findings.\n" + md_table(tables["diebold_mariano"], 4))
    L.append("\n## Caveats\n- Most skill comes from the current regime level; ML adds little where the table shows no gain.\n"
             "- Massive-strike labels are a proxy; probabilities with AUC near 0.5 carry no discriminative information.\n"
             f"- The latest window has only {C.LATEST_WINDOW_DAYS} origins; any-7-day results there are statistically unstable.\n"
             "- Alerts are a reaction to attacks, not the attacks themselves; no text/media signals are used.\n")
    Path(path).write_text("\n".join(L), encoding="utf-8")
