# Air-raid alerts: 7-day forecast and backtest

Data: 24 oblasts (Luhanska excluded; Crimea is not in the source), last full day **2026-10-11**, source coverage ends 2026-10-12 05:00:00+00:00. Forecast origin = end of 2026-10-11; horizon 7 days. Calendar in Europe/Kyiv.

## Definitions
- alerts/day = number of alert intervals started that local day; alert-hours/day = total covered time (union of intervals).
- **Heavy region-day**: alert-hours >= trailing 365-day q90 of that region (computed from past days only) and >= 3.0 h.
- **Massive-strike day (proxy)**: summed alert-hours of rear regions >= trailing q90 and alerts in >= 10 rear regions. No ground-truth labels exist: heavy and massive are data-derived proxies.

## Selected models (chosen on data ending 28 days before the last day)
- n: `EWMA 14d`
- h: `Flat 28d`
- heavy: `Blend`
- mass_day: `Clim`
- any7: `GBM`


## Forecast, day 1 (regions by P(heavy day))
| region | date | alerts_expected | hours_expected | p_heavy_day |
|---|---|---|---|---|
| Kyivska oblast | 2026-10-12 00:00:00 | 12.319 | 3.933 | 0.413 |
| Cherkaska oblast | 2026-10-12 00:00:00 | 4.668 | 3.744 | 0.355 |
| Mykolaivska oblast | 2026-10-12 00:00:00 | 13.985 | 5.014 | 0.326 |
| Kyiv City | 2026-10-12 00:00:00 | 4.512 | 2.342 | 0.268 |
| Kirovohradska oblast | 2026-10-12 00:00:00 | 6.911 | 4.230 | 0.262 |
| Khmelnytska oblast | 2026-10-12 00:00:00 | 2.266 | 1.238 | 0.210 |
| Zhytomyrska oblast | 2026-10-12 00:00:00 | 5.031 | 2.626 | 0.196 |
| Poltavska oblast | 2026-10-12 00:00:00 | 6.971 | 4.039 | 0.171 |
| Rivnenska oblast | 2026-10-12 00:00:00 | 3.038 | 1.434 | 0.146 |
| Ternopilska oblast | 2026-10-12 00:00:00 | 0.956 | 3.380 | 0.144 |


## Massive strike
| date | weekday | p_massive_day |
|---|---|---|
| 2026-10-12 00:00:00 | Mon | 0.095 |
| 2026-10-13 00:00:00 | Tue | 0.095 |
| 2026-10-14 00:00:00 | Wed | 0.095 |
| 2026-10-15 00:00:00 | Thu | 0.095 |
| 2026-10-16 00:00:00 | Fri | 0.095 |
| 2026-10-17 00:00:00 | Sat | 0.095 |
| 2026-10-18 00:00:00 | Sun | 0.095 |

P(at least one massive-strike day in the next 7 days):
| model | p_any_massive_next7 | selected |
|---|---|---|
| Recent 28d (indep.) | 0.815 | False |
| Recent 90d (indep.) | 0.746 | False |
| Logistic | 0.396 | False |
| GBM | 0.333 | True |


## Backtest (rolling origin, strict no-leakage)
180 days, refit every 14 days; 'latest' = last 28 target days.


### Alerts per day (sMAE/sPinball are region-normalised)
All origins:
| model | MAE | sMAE | sRMSE | bias | sPinball | cover80 | width_s | n |
|---|---|---|---|---|---|---|---|---|
| Blend | 1.742 | 0.599 | 1.007 | -0.288 | 0.185 | 0.836 | 1.616 | 29736 |
| EWMA 14d | 1.715 | 0.594 | 0.995 | -0.270 | 0.183 | 0.795 | 1.482 | 29736 |
| Flat 28d | 1.719 | 0.593 | 1.002 | -0.220 | 0.186 | 0.797 | 1.471 | 29736 |
| GBM | 1.829 | 0.617 | 1.044 | -0.305 | 0.193 | 0.837 | 1.750 | 29736 |

Latest window:
| model | MAE | sMAE | sRMSE | bias | sPinball | cover80 | width_s | n |
|---|---|---|---|---|---|---|---|---|
| Blend | 1.942 | 0.508 | 0.945 | -0.361 | 0.171 | 0.851 | 1.525 | 4704 |
| EWMA 14d | 1.971 | 0.512 | 0.950 | -0.614 | 0.174 | 0.804 | 1.366 | 4704 |
| Flat 28d | 1.910 | 0.501 | 0.936 | -0.402 | 0.171 | 0.822 | 1.477 | 4704 |
| GBM | 2.009 | 0.528 | 0.963 | -0.109 | 0.175 | 0.842 | 1.684 | 4704 |


### Alert-hours per day
All origins:
| model | MAE | sMAE | sRMSE | bias | sPinball | cover80 | width_s | n |
|---|---|---|---|---|---|---|---|---|
| Blend | 1.371 | 0.918 | 3.925 | -0.123 | 0.325 | 0.856 | 2.071 | 29736 |
| EWMA 14d | 1.403 | 0.931 | 3.918 | -0.050 | 0.320 | 0.815 | 1.882 | 29736 |
| Flat 28d | 1.431 | 0.895 | 3.911 | -0.031 | 0.320 | 0.807 | 1.672 | 29736 |
| GBM | 1.357 | 0.917 | 3.945 | -0.196 | 0.337 | 0.867 | 2.260 | 29736 |

Latest window:
| model | MAE | sMAE | sRMSE | bias | sPinball | cover80 | width_s | n |
|---|---|---|---|---|---|---|---|---|
| Blend | 1.331 | 0.554 | 1.060 | -0.161 | 0.192 | 0.873 | 1.882 | 4704 |
| EWMA 14d | 1.358 | 0.558 | 1.060 | -0.112 | 0.192 | 0.791 | 1.523 | 4704 |
| Flat 28d | 1.374 | 0.555 | 1.052 | -0.018 | 0.193 | 0.794 | 1.559 | 4704 |
| GBM | 1.315 | 0.556 | 1.076 | -0.210 | 0.201 | 0.873 | 2.241 | 4704 |


### Heavy day (probability)
All origins:
| model | Brier | LogLoss | base_rate | n | AUC |
|---|---|---|---|---|---|
| Blend | 0.093 | 0.330 | 0.106 | 29736 | 0.654 |
| Clim 10% | 0.095 | 0.339 | 0.106 | 29736 | 0.500 |
| GBM | 0.094 | 0.336 | 0.106 | 29736 | 0.640 |
| Recent 28d | 0.095 | 0.340 | 0.106 | 29736 | 0.649 |

Latest window:
| model | Brier | LogLoss | base_rate | n | AUC |
|---|---|---|---|---|---|
| Blend | 0.125 | 0.409 | 0.161 | 4704 | 0.713 |
| Clim 10% | 0.139 | 0.458 | 0.161 | 4704 | 0.500 |
| GBM | 0.128 | 0.419 | 0.161 | 4704 | 0.701 |
| Recent 28d | 0.127 | 0.420 | 0.161 | 4704 | 0.701 |


### Massive-strike day (probability)
All origins:
| model | Brier | LogLoss | base_rate | n | AUC |
|---|---|---|---|---|---|
| Clim | 0.114 | 0.394 | 0.129 | 1239 | 0.487 |
| GBM | 0.117 | 0.435 | 0.129 | 1239 | 0.497 |
| Hazard GBM | 0.118 | 0.419 | 0.129 | 1239 | 0.394 |
| Logistic | 0.117 | 0.398 | 0.129 | 1239 | 0.565 |
| Recent 28d | 0.116 | 0.406 | 0.129 | 1239 | 0.560 |
| Recent 90d | 0.116 | 0.407 | 0.129 | 1239 | 0.400 |

Latest window:
| model | Brier | LogLoss | base_rate | n | AUC |
|---|---|---|---|---|---|
| Clim | 0.184 | 0.590 | 0.214 | 196 | 0.485 |
| GBM | 0.184 | 0.642 | 0.214 | 196 | 0.572 |
| Hazard GBM | 0.188 | 0.617 | 0.214 | 196 | 0.369 |
| Logistic | 0.179 | 0.574 | 0.214 | 196 | 0.739 |
| Recent 28d | 0.178 | 0.547 | 0.214 | 196 | 0.396 |
| Recent 90d | 0.178 | 0.558 | 0.214 | 196 | 0.382 |


### Any massive strike within 7 days
All origins:
| model | Brier | LogLoss | base_rate | n | AUC |
|---|---|---|---|---|---|
| GBM | 0.253 | 0.706 | 0.454 | 174 | 0.579 |
| Logistic | 0.289 | 0.801 | 0.454 | 174 | 0.500 |
| Recent 28d (indep.) | 0.300 | 0.829 | 0.454 | 174 | 0.551 |
| Recent 90d (indep.) | 0.292 | 0.781 | 0.454 | 174 | 0.330 |

Latest window:
| model | Brier | LogLoss | base_rate | n | AUC |
|---|---|---|---|---|---|
| GBM | 0.381 | 1.010 | 0.571 | 28 | 0.443 |
| Logistic | 0.303 | 0.831 | 0.571 | 28 | 0.698 |
| Recent 28d (indep.) | 0.319 | 0.902 | 0.571 | 28 | 0.417 |
| Recent 90d (indep.) | 0.261 | 0.718 | 0.571 | 28 | 0.398 |


## Skill vs simple baselines (all origins)

- alerts, GBM: `GBM` vs `EWMA 14d` skill = -5.3% (90% block-bootstrap CI -7.2% .. -3.8%)

- hours, GBM: `GBM` vs `Flat 28d` skill = -5.5% (90% block-bootstrap CI -7.9% .. -3.6%)

- heavy day, Blend: `Blend` vs `Clim 10%` skill = +2.2% (90% block-bootstrap CI +0.3% .. +4.1%)

- heavy day, GBM: `GBM` vs `Clim 10%` skill = +1.5% (90% block-bootstrap CI -0.3% .. +3.3%)


## Diagnostics
### Ljung-Box: autocorrelation of forecast errors (selected models)
Standardised errors (forecast - actual, divided by the region's level scale), per region, 7 lags. Under no autocorrelation about 5% of regions would have p < 0.05; `mean_resid_z` < 0 means under-forecasting. `national_*` is the same test on the cross-region mean error.
| target | model | h | lags | regions | share_p_lt_05 | median_p | mean_resid_z | national_Q | national_p |
|---|---|---|---|---|---|---|---|---|---|
| alerts | EWMA 14d | 1 | 7 | 24 | 0.5000 | 0.0510 | -0.0754 | 25.1098 | 0.0007 |
| alerts | EWMA 14d | 7 | 7 | 24 | 0.8750 | 0.0001 | -0.1112 | 48.0297 | 0.0000 |
| hours | Flat 28d | 1 | 7 | 24 | 0.5833 | 0.0292 | -0.1632 | 32.7590 | 0.0000 |
| hours | Flat 28d | 7 | 7 | 24 | 0.7083 | 0.0003 | -0.2839 | 80.3353 | 0.0000 |

### Diebold-Mariano: each candidate vs a fixed reference model
Loss differential = loss(reference) - loss(model) per target day (mean over regions), HAC variance with lag h-1, Harvey-Leybourne-Newbold correction. `rel_gain` > 0 and small p: the model is significantly better than the reference. References are fixed a priori (Flat 28d, Recent 28d, Clim, Recent 28d independence). Many comparisons are shown; do not read single p-values near 0.05 as findings.
| target | model | ref | scope | T | rel_gain | DM | p_value |
|---|---|---|---|---|---|---|---|
| alerts | EWMA 14d | Flat 28d | h=1 | 180 | 0.0163 | 3.6665 | 0.0003 |
| alerts | EWMA 14d | Flat 28d | h=7 | 174 | 0.0115 | 2.0907 | 0.0380 |
| alerts | EWMA 14d | Flat 28d | all h | 180 | 0.0133 | 2.3261 | 0.0211 |
| alerts | GBM | Flat 28d | h=1 | 180 | -0.0454 | -5.0028 | 0.0000 |
| alerts | GBM | Flat 28d | h=7 | 174 | -0.0419 | -3.6259 | 0.0004 |
| alerts | GBM | Flat 28d | all h | 180 | -0.0406 | -4.2320 | 0.0000 |
| alerts | Blend | Flat 28d | h=1 | 180 | 0.0001 | 0.0108 | 0.9914 |
| alerts | Blend | Flat 28d | h=7 | 174 | -0.0007 | -0.0962 | 0.9234 |
| alerts | Blend | Flat 28d | all h | 180 | 0.0007 | 0.1156 | 0.9081 |
| hours | EWMA 14d | Flat 28d | h=1 | 180 | -0.0009 | -0.1451 | 0.8848 |
| hours | EWMA 14d | Flat 28d | h=7 | 174 | -0.0006 | -0.0784 | 0.9376 |
| hours | EWMA 14d | Flat 28d | all h | 180 | -0.0012 | -0.2012 | 0.8408 |
| hours | GBM | Flat 28d | h=1 | 180 | -0.0697 | -6.5053 | 0.0000 |
| hours | GBM | Flat 28d | h=7 | 174 | -0.0506 | -4.2719 | 0.0000 |
| hours | GBM | Flat 28d | all h | 180 | -0.0565 | -4.5228 | 0.0000 |
| hours | Blend | Flat 28d | h=1 | 180 | -0.0179 | -2.3795 | 0.0184 |
| hours | Blend | Flat 28d | h=7 | 174 | -0.0120 | -1.2656 | 0.2074 |
| hours | Blend | Flat 28d | all h | 180 | -0.0159 | -1.8312 | 0.0687 |
| heavy | Clim 10% | Recent 28d | h=1 | 180 | -0.0258 | -1.7336 | 0.0847 |
| heavy | Clim 10% | Recent 28d | h=7 | 174 | 0.0125 | 0.7312 | 0.4657 |
| heavy | Clim 10% | Recent 28d | all h | 180 | -0.0006 | -0.0318 | 0.9746 |
| heavy | GBM | Recent 28d | h=1 | 180 | 0.0016 | 0.1678 | 0.8669 |
| heavy | GBM | Recent 28d | h=7 | 174 | 0.0258 | 2.4740 | 0.0143 |
| heavy | GBM | Recent 28d | all h | 180 | 0.0118 | 0.9762 | 0.3303 |
| heavy | Blend | Recent 28d | h=1 | 180 | 0.0158 | 3.2746 | 0.0013 |
| heavy | Blend | Recent 28d | h=7 | 174 | 0.0278 | 4.9619 | 0.0000 |
| heavy | Blend | Recent 28d | all h | 180 | 0.0206 | 3.3103 | 0.0011 |
| massive_day | Recent 28d | Clim | h=1 | 180 | 0.0244 | 0.6103 | 0.5425 |
| massive_day | Recent 28d | Clim | h=7 | 174 | -0.0295 | -0.5715 | 0.5684 |
| massive_day | Recent 28d | Clim | all h | 180 | -0.0147 | -0.3096 | 0.7572 |
| massive_day | Recent 90d | Clim | h=1 | 180 | -0.0087 | -0.6119 | 0.5414 |
| massive_day | Recent 90d | Clim | h=7 | 174 | -0.0177 | -1.1523 | 0.2508 |
| massive_day | Recent 90d | Clim | all h | 180 | -0.0169 | -1.0455 | 0.2972 |
| massive_day | Logistic | Clim | h=1 | 180 | 0.0375 | 0.7656 | 0.4449 |
| massive_day | Logistic | Clim | h=7 | 174 | -0.0628 | -1.1361 | 0.2575 |
| massive_day | Logistic | Clim | all h | 180 | -0.0223 | -0.5240 | 0.6009 |
| massive_day | GBM | Clim | h=1 | 180 | -0.0075 | -0.2325 | 0.8164 |
| massive_day | GBM | Clim | h=7 | 174 | -0.0529 | -2.0385 | 0.0430 |
| massive_day | GBM | Clim | all h | 180 | -0.0237 | -0.7920 | 0.4294 |
| massive_day | Hazard GBM | Clim | h=1 | 180 | -0.0098 | -0.7019 | 0.4837 |
| massive_day | Hazard GBM | Clim | h=7 | 174 | -0.0333 | -2.8389 | 0.0051 |
| massive_day | Hazard GBM | Clim | all h | 180 | -0.0332 | -3.4700 | 0.0007 |
| massive_any7 | Recent 90d (indep.) | Recent 28d (indep.) | all origins | 174 | 0.0246 | 0.2234 | 0.8235 |
| massive_any7 | Logistic | Recent 28d (indep.) | all origins | 174 | 0.0342 | 0.1909 | 0.8489 |
| massive_any7 | GBM | Recent 28d (indep.) | all origins | 174 | 0.1554 | 0.7727 | 0.4408 |


## Caveats
- Most skill comes from the current regime level; ML adds little where the table shows no gain.
- Massive-strike labels are a proxy; probabilities with AUC near 0.5 carry no discriminative information.
- The latest window has only 28 origins; any-7-day results there are statistically unstable.
- Alerts are a reaction to attacks, not the attacks themselves; no text/media signals are used.
