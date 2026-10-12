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


## Caveats
- Most skill comes from the current regime level; ML adds little where the table shows no gain.
- Massive-strike labels are a proxy; probabilities with AUC near 0.5 carry no discriminative information.
- The latest window has only 28 origins; any-7-day results there are statistically unstable.
- Alerts are a reaction to attacks, not the attacks themselves; no text/media signals are used.
