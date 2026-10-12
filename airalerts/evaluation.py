"""Model selection and evaluation tables.

Protocol (no information from the evaluation window is used for selection):
  * the model published for each target is the one with the lowest mean loss on the SELECTION window =
    backtest rows whose target day is <= last_day - LATEST_WINDOW_DAYS;
  * the LATEST window (the last LATEST_WINDOW_DAYS days) is then reported separately, next to the full backtest.
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from . import metrics as MT


def _any7(bt):
    a = bt.nat_any.copy()
    a["target_date"] = a["origin"] + pd.Timedelta(days=C.HORIZON)   # the window is complete on this day
    return a


def select_models(bt, last_day: pd.Timestamp) -> dict[str, str]:
    cut = last_day - pd.Timedelta(days=C.LATEST_WINDOW_DAYS)
    return {
        "n": MT.pick_best(MT.add_reg_losses(bt.reg, "n"), "pin_s", cut),
        "h": MT.pick_best(MT.add_reg_losses(bt.reg, "h"), "pin_s", cut),
        # a constant climatology is kept as a reference only: publishing the same probability for every region is not a forecast
        "heavy": MT.pick_best(MT.add_prob_losses(bt.heavy, "y_heavy").query("model != 'Clim 10%'"), "brier", cut),
        "mass_day": MT.pick_best(MT.add_prob_losses(bt.nat_day, "y_mass"), "brier", cut),
        "any7": MT.pick_best(MT.add_prob_losses(_any7(bt), "y_any"), "brier", cut),
    }


def evaluation_tables(bt, last_day: pd.Timestamp) -> dict[str, pd.DataFrame]:
    cut = last_day - pd.Timedelta(days=C.LATEST_WINDOW_DAYS)
    T: dict[str, pd.DataFrame] = {}
    for key, name in (("n", "alerts"), ("h", "hours")):
        df = MT.add_reg_losses(bt.reg, key)
        T[f"{name}_all"] = MT.summarise_reg(df)
        T[f"{name}_latest"] = MT.summarise_reg(df[df["target_date"] > cut])
        T[f"{name}_by_horizon"] = MT.summarise_reg(df, ["h"])
        T[f"{name}_by_region"] = MT.summarise_reg(df, ["region"])
    for name, df, ycol in (("heavy", MT.add_prob_losses(bt.heavy, "y_heavy"), "y_heavy"),
                           ("massive_day", MT.add_prob_losses(bt.nat_day, "y_mass"), "y_mass"),
                           ("massive_any7", MT.add_prob_losses(_any7(bt), "y_any"), "y_any")):
        T[f"{name}_all"] = MT.summarise_prob(df, ycol)
        T[f"{name}_latest"] = MT.summarise_prob(df[df["target_date"] > cut], ycol)
        T[f"{name}_by_horizon"] = MT.summarise_prob(df, ycol, ["h"]) if "h" in df.columns else pd.DataFrame()
    return T
