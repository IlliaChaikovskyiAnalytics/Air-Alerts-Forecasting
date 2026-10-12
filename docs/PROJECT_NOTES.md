# KSE AI Agentic School, stage 2: air-raid alerts time series. Project notes (state at 2026-10-12)

## Decisions so far
- Data: Vadimkin/ukrainian-air-raid-sirens-dataset. Main source = volunteer (oblast level, current to 2026-10-12 04:59 UTC).
  official = raion level since Dec 2025, stopped 2026-09-07, gap 30.08-01.09 -> cross-check only.
- Scope: Kyiv City first, then replicate for all oblasts. Calendar features in Europe/Kyiv local time; last incomplete day dropped.
- Evaluation: rolling-origin backtest, latest period = last 30 days (12.09-11.10.2026); no future data in training (asserted in code).
- Environment limits: sandbox has no PyPI/alerts.in.ua access; user runs scripts in PyCharm (Python 3.15 venv). statsmodels/lightgbm untested.

## Scripts (pycharm_scripts/, single-file, config block on top, one Run)
1. Air Alerts.py          EDA for one region (REGION constant). 5-6 figures + eda_report.txt + hourly/daily CSV.
2. Air Alerts Forecast.py 30-day daily forecast (n_alerts, alert_hours); baselines + Poisson GLM + GBM; 26 origins; needs scikit-learn.
3. Air Alerts Horizon.py  P(day with 0 / >=3 / >=5 alerts) for h=1..30; skill vs baselines; bootstrap CI; 14-day forecast.

## Findings (Kyiv)
- Persistence: P(active t+1|active t)=0.53 vs 0.06 after idle hour; ACF hourly lag1 0.47; next alert <1h after end in 10%, <3h in 29%.
- Weak diurnal pattern (day/night 0.83) that changed by year; weak weekday effect (Kruskal p=0.037); var/mean of daily counts 1.57 (NegBin, not Poisson).
- Regime shift ~17-23 Aug 2026: 2.2 -> 4.5 alerts/day, zero-days 12% -> 0-4%. Last 28 days: 0 days without alerts.
- 30-day daily forecast: no model beats Flat-28d meaningfully (GBM +0.5% MAE on counts, -1.1% on hours); 30-day total MAPE 26-29%.
  Forecast 12.10-10.11: ~137 alerts (80%: 94-239), ~70 h, mean duration ~30.7 min.
- Probabilities by horizon: P(zero-day) unpredictable at all h (AUC 0.5-0.58). P(n>=3), P(n>=5): "recent 28d frequency" beats GBM and logistic at all h
  (BSS vs recent28 -0.06..-0.12); recent28 itself has ~20% skill vs clim365 at h=1, ~11-13% at h=30 -> signal = current regime level.
- volunteer vs official agree on 97.6% of hours (kappa 0.88). 8% of Kyiv alerts are 'naive' (no end message, +30 min assumed).

## Open issues / caveats
- Features lack "time since last heavy day" (renewal/hazard hypothesis not tested yet) and any cross-region or text signal.
- Dec 2025 raion-level change may be a measurement break for other oblasts (check before the multi-region step).
- Plot display in PyCharm, scikit-learn on Python 3.15, statsmodels branch (KPSS/STL) not verified.
- No text/media data yet. Deadline and report format unknown.
