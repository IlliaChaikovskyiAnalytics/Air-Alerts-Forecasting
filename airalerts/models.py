"""Global (pooled) models.

RegionalForecaster : one set of gradient-boosting models for ALL regions.
    - daily n_alerts and alert_hours are divided by the region's trailing 28-day level (known at the origin), so regions with
      very different activity share one model; predictions are multiplied back. Region id and weekday are categorical.
    - mean (Poisson loss) + 10/50/90 % quantile models per target
    - P(heavy day) classifier
NationalForecaster : P(massive-strike day) for h = 1..7 and P(at least one massive strike in the next 7 days).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import config as C
from .baselines import clip_p

REG_NON_FEATURES = {"region", "origin", "target_date", "y_n", "y_h", "y_heavy", "scale_n", "scale_h"}
NAT_NON_FEATURES = {"origin", "target_date", "y_mass", "y_any"}
# the 56-day quantile columns only exist for the baselines
DROP_FROM_MODEL = {c for c in [f"{k}_q{int(q * 100)}_56" for k in "nh" for q in C.QUANTILES]}
DROP_FROM_MODEL |= {f"{k}_ewm{hl}" for k in "nh" for hl in (3, 7, 28)}   # baseline-only level trackers


def reg_feature_cols(D: pd.DataFrame) -> list[str]:
    return [c for c in D.columns if c not in REG_NON_FEATURES and c not in DROP_FROM_MODEL]


def nat_feature_cols(DN: pd.DataFrame) -> list[str]:
    return [c for c in DN.columns if c not in NAT_NON_FEATURES]


def _gb_kwargs(**extra):
    kw = dict(max_iter=120, learning_rate=0.06, max_leaf_nodes=24, min_samples_leaf=120, l2_regularization=1.0,
              random_state=C.SEED)
    kw.update(extra)
    return kw


class RegionalForecaster:
    def __init__(self):
        self.mean, self.quant, self.heavy = {}, {}, None
        self.cols: list[str] = []
        self.cat = ["region_id", "dow"]

    def fit(self, tr: pd.DataFrame) -> "RegionalForecaster":
        self.cols = reg_feature_cols(tr)
        X = tr[self.cols]
        cat = [c for c in self.cat if c in self.cols]
        for key, ycol, scol in (("n", "y_n", "scale_n"), ("h", "y_h", "scale_h")):
            m = tr[ycol].notna().values
            y = (tr.loc[m, ycol] / tr.loc[m, scol]).astype(float)
            self.mean[key] = HGBR(loss="poisson", categorical_features=cat, **_gb_kwargs()).fit(X[m], y)
            self.quant[key] = {q: HGBR(loss="quantile", quantile=q, categorical_features=cat, **_gb_kwargs()).fit(X[m], y)
                               for q in C.QUANTILES}
        m = tr["y_heavy"].notna().values
        self.heavy = HGBC(categorical_features=cat, **_gb_kwargs()).fit(X[m], tr.loc[m, "y_heavy"].astype(int))
        return self

    def predict(self, te: pd.DataFrame) -> pd.DataFrame:
        X = te[self.cols]
        out = pd.DataFrame(index=te.index)
        for key, scol in (("n", "scale_n"), ("h", "scale_h")):
            sc = te[scol].values.astype(float)
            cap = 24.0 if key == "h" else np.inf   # a day has 24 hours
            out[f"{key}_mean"] = np.minimum(np.maximum(self.mean[key].predict(X), 0) * sc, cap)
            qs = np.column_stack([self.quant[key][q].predict(X) for q in C.QUANTILES])
            qs = np.minimum(np.maximum(np.sort(qs, axis=1), 0) * sc[:, None], cap)
            for j, q in enumerate(C.QUANTILES):
                out[f"{key}_q{int(q * 100)}"] = qs[:, j]
        out["p_heavy"] = clip_p(self.heavy.predict_proba(X)[:, 1])
        return out


class NationalForecaster:
    """Compact, strongly regularised models: only ~140 massive-strike days exist in the training window."""

    def __init__(self):
        self.cols: list[str] = []

    def _logit_matrix(self, D: pd.DataFrame, with_h: bool) -> np.ndarray:
        A = D[self.cols].copy()
        if with_h:
            A["h"] = D["h"].values / C.HORIZON
            dow = pd.get_dummies(D["dow"]).reindex(columns=range(7), fill_value=0).astype(float)
            return np.hstack([A.values.astype(float), dow.values])
        return A.values.astype(float)

    def fit(self, trN: pd.DataFrame) -> "NationalForecaster":
        self.cols = nat_feature_cols(trN)
        self.cols = [c for c in self.cols if c not in ("h", "dow")]
        # per-day model
        m = trN["y_mass"].notna().values
        d = trN[m]
        self.sc_day = StandardScaler().fit(self._logit_matrix(d, True))
        self.lr_day = LogisticRegression(C=0.05, max_iter=3000).fit(self.sc_day.transform(self._logit_matrix(d, True)),
                                                                      d["y_mass"].astype(int))
        small = dict(max_iter=80, learning_rate=0.05, max_depth=3, min_samples_leaf=150, l2_regularization=2.0,
                     random_state=C.SEED)
        self.gb_day = HGBC(categorical_features=["dow"], **small).fit(d[self.cols + ["h", "dow"]], d["y_mass"].astype(int))
        self.hz_day = HGBC(categorical_features=["dow"], **small).fit(d[["since_massive", "h", "dow"]], d["y_mass"].astype(int))
        # 7-day window model (one row per origin)
        w = trN.drop_duplicates("origin")
        w = w[w["y_any"].notna()]
        self.sc_any = StandardScaler().fit(self._logit_matrix(w, False))
        self.lr_any = LogisticRegression(C=0.05, max_iter=3000).fit(self.sc_any.transform(self._logit_matrix(w, False)),
                                                                      w["y_any"].astype(int))
        self.gb_any = HGBC(**small).fit(w[self.cols], w["y_any"].astype(int))
        return self

    def predict_day(self, te: pd.DataFrame) -> dict[str, np.ndarray]:
        return {"Logistic": clip_p(self.lr_day.predict_proba(self.sc_day.transform(self._logit_matrix(te, True)))[:, 1]),
                "GBM": clip_p(self.gb_day.predict_proba(te[self.cols + ["h", "dow"]])[:, 1]),
                "Hazard GBM": clip_p(self.hz_day.predict_proba(te[["since_massive", "h", "dow"]])[:, 1])}

    def predict_any(self, te: pd.DataFrame) -> dict[str, np.ndarray]:
        return {"Logistic": clip_p(self.lr_any.predict_proba(self.sc_any.transform(self._logit_matrix(te, False)))[:, 1]),
                "GBM": clip_p(self.gb_any.predict_proba(te[self.cols])[:, 1])}
