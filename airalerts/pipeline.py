"""End-to-end pipeline: data -> features -> backtest -> model selection -> forecast -> figures -> report."""
from __future__ import annotations

import hashlib
import json
import pickle
import time
import warnings

import pandas as pd

from . import config as C
from . import data, diagnostics, evaluation, features, plots, report
from .backtest import run_backtest
from .forecast import make_forecast


def _signature(panel) -> str:
    keys = {k: getattr(C, k) for k in dir(C) if k.isupper() and not k.startswith("_") and k not in ("ROOT", "RAW_DIR", "OUT_DIR")}
    keys["last_day"] = str(panel.last_day)
    keys["rows"] = len(panel.daily)
    return hashlib.md5(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()


def run(download: bool = False, refresh_backtest: bool = False, verbose: bool = True) -> dict:
    warnings.filterwarnings("ignore")
    out = C.OUT_DIR
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "cache").mkdir(exist_ok=True)
    t0 = time.time()

    if download:
        print("downloading the latest data ...", flush=True)
        data.download()
    panel = data.build_panel()
    A = features.build_all(panel)
    print(f"data: {len(panel.regions)} regions, {panel.first_day.date()} .. {panel.last_day.date()} "
          f"(source coverage ends {panel.data_end})", flush=True)

    cache = out / "cache" / "backtest.pkl"
    sig = _signature(panel)
    bt = None
    if cache.exists() and not refresh_backtest:
        stored_sig, bt = pickle.load(open(cache, "rb"))
        if stored_sig != sig:
            bt = None
            print("cached backtest is stale (data or config changed) -> re-running", flush=True)
    if bt is None:
        print(f"backtest: {C.BACKTEST_DAYS} days, refit every {C.REFIT_EVERY_DAYS} days", flush=True)
        bt = run_backtest(A["D"], A["DN"], panel.last_day, verbose=verbose)
        pickle.dump((sig, bt), open(cache, "wb"))
    else:
        print("backtest: loaded from cache (use --refresh-backtest to recompute)", flush=True)

    sel = evaluation.select_models(bt, panel.last_day)
    tables = evaluation.evaluation_tables(bt, panel.last_day)
    tables["ljung_box"] = diagnostics.ljung_box_table(bt, sel)
    tables["diebold_mariano"] = diagnostics.dm_table(bt)
    for k, v in tables.items():
        v.to_csv(out / "tables" / (f"eval_{k}.csv" if k not in ("ljung_box", "diebold_mariano") else f"diag_{k}.csv"), index=False)
    print("selected models:", sel, flush=True)

    print("fitting the production models and forecasting ...", flush=True)
    fc = make_forecast(A["D"], A["DN"], panel, sel)
    fc["regional"].to_csv(out / "tables" / "forecast_regions_next7d.csv", index=False)
    fc["national_days"].to_csv(out / "tables" / "forecast_national_days.csv", index=False)
    fc["national_any7"].to_csv(out / "tables" / "forecast_national_any7.csv", index=False)
    fc["candidates"].to_csv(out / "tables" / "forecast_all_candidate_models.csv", index=False)
    json.dump(sel, open(out / "tables" / "selected_models.json", "w"), indent=2)

    print("figures ...", flush=True)
    figs = out / "figures"
    plots.forecast_heatmaps(fc["regional"], figs)
    plots.forecast_national(fc["national_days"], fc["national_any7"], A["nat"], A["massive"], figs)
    plots.skill_by_horizon(bt, sel, figs)
    plots.skill_by_region(bt, sel, figs)
    plots.reliability_plots(bt, sel, figs)
    plots.latest_window(bt, sel, panel.daily, panel.last_day, figs)
    plots.massive_backtest(bt, sel, figs)

    report.write_report(panel, A, bt, sel, tables, fc, out / "RESULTS.md")
    print(f"done in {time.time() - t0:.0f}s -> {out}", flush=True)
    return dict(panel=panel, A=A, bt=bt, sel=sel, tables=tables, fc=fc)
