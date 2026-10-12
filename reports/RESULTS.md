# Air-raid alerts: 7-day forecast and backtest

Data: 24 oblasts (Luhanska excluded; Crimea is not in the source), last full day **2026-10-11**, source coverage ends 2026-10-12 05:00:00+00:00. Forecast origin = end of 2026-10-11; horizon 7 days. Calendar in Europe/Kyiv.

## Definitions
- alerts/day = number of alert intervals started that local day; alert-hours/day = total covered time (union of intervals).
- **Heavy region-day**: alert-hours >= trailing 365-day q90 of that region (computed from past days only) and >= 3.0 h.
- **Massive-strike day (proxy)**: summed alert-hours of rear regions >= trailing q90 and alerts in >= 10 rear regions. No ground-truth labels exist: heavy and massive are data-derived proxies.

## Selected models (chosen on data ending 28 days before the last day)
- n: `Fast-up 7/28d`
- h: `EWMA 7d`
- heavy: `Blend`
- mass_day: `Clim`
- any7: `GBM`


## Forecast, day 1 (regions by P(heavy day))
| region | date | alerts_expected | hours_expected | p_heavy_day |
|---|---|---|---|---|
| Kyivska oblast | 2026-10-12 00:00:00 | 13.514 | 4.003 | 0.449 |
| Cherkaska oblast | 2026-10-12 00:00:00 | 4.892 | 3.636 | 0.355 |
| Mykolaivska oblast | 2026-10-12 00:00:00 | 15.017 | 5.004 | 0.316 |
| Kirovohradska oblast | 2026-10-12 00:00:00 | 7.298 | 4.386 | 0.295 |
| Kyiv City | 2026-10-12 00:00:00 | 4.762 | 2.671 | 0.262 |
| Zhytomyrska oblast | 2026-10-12 00:00:00 | 5.706 | 2.717 | 0.226 |
| Khmelnytska oblast | 2026-10-12 00:00:00 | 3.111 | 1.801 | 0.202 |
| Poltavska oblast | 2026-10-12 00:00:00 | 6.914 | 4.136 | 0.172 |
| Ternopilska oblast | 2026-10-12 00:00:00 | 1.494 | 1.823 | 0.158 |
| Rivnenska oblast | 2026-10-12 00:00:00 | 3.972 | 1.637 | 0.148 |


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
| Blend | 1.731 | 0.598 | 1.007 | -0.292 | 0.186 | 0.838 | 1.633 | 29736 |
| EWMA 14d | 1.715 | 0.594 | 0.995 | -0.270 | 0.183 | 0.795 | 1.482 | 29736 |
| EWMA 3d | 1.734 | 0.587 | 1.010 | -0.119 | 0.190 | 0.795 | 1.531 | 29736 |
| EWMA 7d | 1.697 | 0.583 | 0.993 | -0.179 | 0.184 | 0.802 | 1.493 | 29736 |
| Fast-up 7/28d | 1.717 | 0.608 | 0.985 | -0.058 | 0.181 | 0.815 | 1.653 | 29736 |
| Flat 28d | 1.719 | 0.593 | 1.002 | -0.220 | 0.186 | 0.797 | 1.471 | 29736 |
| GBM | 1.796 | 0.614 | 1.042 | -0.315 | 0.194 | 0.842 | 1.784 | 29736 |

Latest window:
| model | MAE | sMAE | sRMSE | bias | sPinball | cover80 | width_s | n |
|---|---|---|---|---|---|---|---|---|
| Blend | 1.930 | 0.505 | 0.943 | -0.368 | 0.172 | 0.849 | 1.516 | 4704 |
| EWMA 14d | 1.971 | 0.512 | 0.950 | -0.614 | 0.174 | 0.804 | 1.366 | 4704 |
| EWMA 3d | 1.914 | 0.502 | 0.919 | -0.322 | 0.173 | 0.817 | 1.586 | 4704 |
| EWMA 7d | 1.880 | 0.494 | 0.922 | -0.389 | 0.169 | 0.823 | 1.515 | 4704 |
| Fast-up 7/28d | 1.918 | 0.505 | 0.925 | -0.262 | 0.170 | 0.828 | 1.543 | 4704 |
| Flat 28d | 1.910 | 0.501 | 0.936 | -0.402 | 0.171 | 0.822 | 1.477 | 4704 |
| GBM | 1.976 | 0.522 | 0.958 | -0.122 | 0.176 | 0.837 | 1.665 | 4704 |


### Alert-hours per day
All origins:
| model | MAE | sMAE | sRMSE | bias | sPinball | cover80 | width_s | n |
|---|---|---|---|---|---|---|---|---|
| Blend | 1.377 | 0.922 | 3.923 | -0.115 | 0.323 | 0.859 | 2.064 | 29736 |
| EWMA 14d | 1.403 | 0.931 | 3.918 | -0.050 | 0.320 | 0.815 | 1.882 | 29736 |
| EWMA 3d | 1.443 | 0.890 | 3.911 | -0.010 | 0.321 | 0.801 | 1.696 | 29736 |
| EWMA 7d | 1.414 | 0.890 | 3.909 | -0.024 | 0.317 | 0.812 | 1.714 | 29736 |
| Fast-up 7/28d | 1.475 | 1.020 | 3.922 | 0.144 | 0.324 | 0.831 | 2.297 | 29736 |
| Flat 28d | 1.431 | 0.895 | 3.911 | -0.031 | 0.320 | 0.807 | 1.672 | 29736 |
| GBM | 1.370 | 0.924 | 3.941 | -0.180 | 0.335 | 0.871 | 2.246 | 29736 |

Latest window:
| model | MAE | sMAE | sRMSE | bias | sPinball | cover80 | width_s | n |
|---|---|---|---|---|---|---|---|---|
| Blend | 1.339 | 0.558 | 1.062 | -0.140 | 0.188 | 0.875 | 1.778 | 4704 |
| EWMA 14d | 1.358 | 0.558 | 1.060 | -0.112 | 0.192 | 0.791 | 1.523 | 4704 |
| EWMA 3d | 1.407 | 0.572 | 1.078 | -0.052 | 0.196 | 0.796 | 1.644 | 4704 |
| EWMA 7d | 1.367 | 0.556 | 1.053 | -0.045 | 0.192 | 0.804 | 1.613 | 4704 |
| Fast-up 7/28d | 1.391 | 0.580 | 1.055 | 0.055 | 0.190 | 0.814 | 1.687 | 4704 |
| Flat 28d | 1.374 | 0.555 | 1.052 | -0.018 | 0.193 | 0.794 | 1.559 | 4704 |
| GBM | 1.336 | 0.567 | 1.083 | -0.169 | 0.194 | 0.882 | 2.033 | 4704 |


### Heavy day (probability)
All origins:
| model | Brier | LogLoss | base_rate | n | AUC |
|---|---|---|---|---|---|
| Blend | 0.093 | 0.330 | 0.106 | 29736 | 0.652 |
| Clim 10% | 0.095 | 0.339 | 0.106 | 29736 | 0.500 |
| GBM | 0.094 | 0.337 | 0.106 | 29736 | 0.638 |
| Recent 28d | 0.095 | 0.340 | 0.106 | 29736 | 0.649 |

Latest window:
| model | Brier | LogLoss | base_rate | n | AUC |
|---|---|---|---|---|---|
| Blend | 0.125 | 0.408 | 0.161 | 4704 | 0.714 |
| Clim 10% | 0.139 | 0.458 | 0.161 | 4704 | 0.500 |
| GBM | 0.127 | 0.417 | 0.161 | 4704 | 0.704 |
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

- alerts, GBM: `GBM` vs `EWMA 14d` skill = -6.1% (90% block-bootstrap CI -7.9% .. -4.5%)

- hours, GBM: `GBM` vs `Flat 28d` skill = -4.6% (90% block-bootstrap CI -7.0% .. -2.5%)

- heavy day, Blend: `Blend` vs `Clim 10%` skill = +2.1% (90% block-bootstrap CI +0.1% .. +4.0%)

- heavy day, GBM: `GBM` vs `Clim 10%` skill = +1.3% (90% block-bootstrap CI -0.5% .. +3.2%)


## Diagnostics
### Ljung-Box: autocorrelation of forecast errors (selected models)
Standardised errors (forecast - actual, divided by the region's level scale), per region, 7 lags. Under no autocorrelation about 5% of regions would have p < 0.05; `mean_resid_z` < 0 means under-forecasting. `national_*` is the same test on the cross-region mean error.
| target | model | h | lags | regions | share_p_lt_05 | median_p | mean_resid_z | national_Q | national_p |
|---|---|---|---|---|---|---|---|---|---|
| alerts | Fast-up 7/28d | 1 | 7 | 24 | 0.5417 | 0.0372 | -0.0021 | 23.2029 | 0.0016 |
| alerts | Fast-up 7/28d | 7 | 7 | 24 | 0.8750 | 0.0000 | -0.0381 | 44.8167 | 0.0000 |
| hours | EWMA 7d | 1 | 7 | 24 | 0.3333 | 0.1128 | -0.1409 | 26.8737 | 0.0004 |
| hours | EWMA 7d | 7 | 7 | 24 | 0.7917 | 0.0012 | -0.2599 | 81.7066 | 0.0000 |

### Bias by regime (h=1)
Mean standardised error by the recent trend of the target (previous 7 days vs the 28 before). Negative = under-forecasting; a bias concentrated in 'rising' means the level estimate lags behind regime changes.
| target | model | regime | mean_resid_z | n |
|---|---|---|---|---|
| alerts | Fast-up 7/28d | falling | 0.062 | 1146 |
| alerts | Fast-up 7/28d | stable | -0.006 | 1150 |
| alerts | Fast-up 7/28d | rising | -0.050 | 1140 |
| hours | EWMA 7d | falling | -0.077 | 1146 |
| hours | EWMA 7d | stable | -0.031 | 1145 |
| hours | EWMA 7d | rising | -0.165 | 1145 |

### Diebold-Mariano: each candidate vs a fixed reference model
Loss differential = loss(reference) - loss(model) per target day (mean over regions), HAC variance with lag h-1, Harvey-Leybourne-Newbold correction. `rel_gain` > 0 and small p: the model is significantly better than the reference. References are fixed a priori (Flat 28d, Recent 28d, Clim, Recent 28d independence). Many comparisons are shown; do not read single p-values near 0.05 as findings.
| target | model | ref | scope | T | rel_gain | DM | p_value |
|---|---|---|---|---|---|---|---|
| alerts | EWMA 14d | Flat 28d | h=1 | 180 | 0.0163 | 3.6665 | 0.0003 |
| alerts | EWMA 14d | Flat 28d | h=7 | 174 | 0.0115 | 2.0907 | 0.0380 |
| alerts | EWMA 14d | Flat 28d | all h | 180 | 0.0133 | 2.3261 | 0.0211 |
| alerts | EWMA 3d | Flat 28d | h=1 | 180 | 0.0067 | 0.4678 | 0.6405 |
| alerts | EWMA 3d | Flat 28d | h=7 | 174 | -0.0356 | -1.6919 | 0.0925 |
| alerts | EWMA 3d | Flat 28d | all h | 180 | -0.0252 | -1.4281 | 0.1550 |
| alerts | EWMA 7d | Flat 28d | h=1 | 180 | 0.0219 | 3.0079 | 0.0030 |
| alerts | EWMA 7d | Flat 28d | h=7 | 174 | 0.0036 | 0.3802 | 0.7043 |
| alerts | EWMA 7d | Flat 28d | all h | 180 | 0.0093 | 1.0331 | 0.3029 |
| alerts | Fast-up 7/28d | Flat 28d | h=1 | 180 | 0.0275 | 3.2884 | 0.0012 |
| alerts | Fast-up 7/28d | Flat 28d | h=7 | 174 | 0.0251 | 2.4197 | 0.0166 |
| alerts | Fast-up 7/28d | Flat 28d | all h | 180 | 0.0249 | 2.3644 | 0.0191 |
| alerts | GBM | Flat 28d | h=1 | 180 | -0.0545 | -5.5760 | 0.0000 |
| alerts | GBM | Flat 28d | h=7 | 174 | -0.0430 | -3.8983 | 0.0001 |
| alerts | GBM | Flat 28d | all h | 180 | -0.0476 | -4.6273 | 0.0000 |
| alerts | Blend | Flat 28d | h=1 | 180 | -0.0039 | -0.5871 | 0.5579 |
| alerts | Blend | Flat 28d | h=7 | 174 | -0.0007 | -0.0885 | 0.9296 |
| alerts | Blend | Flat 28d | all h | 180 | -0.0023 | -0.3502 | 0.7266 |
| hours | EWMA 14d | Flat 28d | h=1 | 180 | -0.0009 | -0.1451 | 0.8848 |
| hours | EWMA 14d | Flat 28d | h=7 | 174 | -0.0006 | -0.0784 | 0.9376 |
| hours | EWMA 14d | Flat 28d | all h | 180 | -0.0012 | -0.2012 | 0.8408 |
| hours | EWMA 3d | Flat 28d | h=1 | 180 | 0.0162 | 1.3434 | 0.1809 |
| hours | EWMA 3d | Flat 28d | h=7 | 174 | -0.0134 | -1.3453 | 0.1803 |
| hours | EWMA 3d | Flat 28d | all h | 180 | -0.0070 | -0.8112 | 0.4183 |
| hours | EWMA 7d | Flat 28d | h=1 | 180 | 0.0192 | 2.9157 | 0.0040 |
| hours | EWMA 7d | Flat 28d | h=7 | 174 | 0.0040 | 0.6190 | 0.5368 |
| hours | EWMA 7d | Flat 28d | all h | 180 | 0.0064 | 1.3278 | 0.1859 |
| hours | Fast-up 7/28d | Flat 28d | h=1 | 180 | -0.0155 | -1.5292 | 0.1280 |
| hours | Fast-up 7/28d | Flat 28d | h=7 | 174 | -0.0150 | -1.1165 | 0.2657 |
| hours | Fast-up 7/28d | Flat 28d | all h | 180 | -0.0128 | -1.1444 | 0.2540 |
| hours | GBM | Flat 28d | h=1 | 180 | -0.0609 | -5.8677 | 0.0000 |
| hours | GBM | Flat 28d | h=7 | 174 | -0.0403 | -3.1773 | 0.0018 |
| hours | GBM | Flat 28d | all h | 180 | -0.0492 | -3.5969 | 0.0004 |
| hours | Blend | Flat 28d | h=1 | 180 | -0.0136 | -1.6994 | 0.0910 |
| hours | Blend | Flat 28d | h=7 | 174 | -0.0063 | -0.5974 | 0.5510 |
| hours | Blend | Flat 28d | all h | 180 | -0.0119 | -1.2521 | 0.2122 |
| heavy | Clim 10% | Recent 28d | h=1 | 180 | -0.0258 | -1.7336 | 0.0847 |
| heavy | Clim 10% | Recent 28d | h=7 | 174 | 0.0125 | 0.7312 | 0.4657 |
| heavy | Clim 10% | Recent 28d | all h | 180 | -0.0006 | -0.0318 | 0.9746 |
| heavy | GBM | Recent 28d | h=1 | 180 | 0.0025 | 0.2759 | 0.7830 |
| heavy | GBM | Recent 28d | h=7 | 174 | 0.0241 | 2.4738 | 0.0143 |
| heavy | GBM | Recent 28d | all h | 180 | 0.0101 | 0.9106 | 0.3637 |
| heavy | Blend | Recent 28d | h=1 | 180 | 0.0152 | 3.2562 | 0.0014 |
| heavy | Blend | Recent 28d | h=7 | 174 | 0.0258 | 4.8889 | 0.0000 |
| heavy | Blend | Recent 28d | all h | 180 | 0.0186 | 3.1942 | 0.0017 |
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
- Forecast errors are autocorrelated (Ljung-Box) and biased downwards, almost entirely in periods when alert intensity is rising: the level estimate lags behind regime changes. Nominal 80% intervals should therefore be read as optimistic.
- Most skill comes from the current regime level; ML adds little where the table shows no gain.
- Massive-strike labels are a proxy; probabilities with AUC near 0.5 carry no discriminative information.
- The latest window has only 28 origins; any-7-day results there are statistically unstable.
- Alerts are a reaction to attacks, not the attacks themselves; no text/media signals are used.
