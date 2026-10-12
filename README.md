# Air-raid alerts in Ukraine: 7-day forecast by oblast

Forecasts, for 24 oblasts (incl. Kyiv City; Luhanska excluded), for the next 7 days:
expected alerts/day and alert-hours/day (with 10/50/90% quantiles), probability of a **heavy day**
per region, and probability of a **massive-strike day** (nationally) and of at least one in 7 days.

Data: [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), `volunteer_data_en.csv` (oblast level).

## Run
```
pip install -r requirements.txt
python run_pipeline.py --download     # fetch the latest data, backtest (~5 min), forecast, figures
python run_pipeline.py                # reuse the cached backtest if data/config are unchanged
python -m pytest tests
```
Outputs: `reports/RESULTS.md`, `reports/tables/*.csv`, `reports/figures/*.png`. Settings: `airalerts/config.py`.

## Method
Direct multi-horizon forecasts; one pooled gradient-boosting model over all regions (Poisson + quantile losses on level-scaled targets)
compared against simple adaptive-level baselines (flat 28-day, EWMA, recent frequency, climatology) and blends.
Rolling-origin backtest over the last 180 days with refits every 14 days; training rows are restricted to targets known at the refit date.
Model for each target is chosen on a window ending 28 days before the last day; the latest 28 days are reported separately.

## Honest summary
See `reports/RESULTS.md` for the numbers. Skill comes mostly from the current regime level; ML gives little or no gain over simple
baselines for counts and hours, a small gain for heavy-day probability, and none for the massive-strike proxy. Labels are proxies, no text/media data used.
Details of definitions: `docs/target_definitions.md`.
