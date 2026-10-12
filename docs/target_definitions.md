# Target definitions

All targets are computed per local calendar day (Europe/Kyiv) from the oblast-level volunteer dataset.

| target | definition |
|---|---|
| alerts/day | number of alert intervals started on the day |
| alert-hours/day | length of the union of alert intervals on the day (hours) |
| heavy region-day | alert-hours >= q90 of the same region over the previous 365 days (min 180 days history) and >= 3 h |
| massive-strike day (proxy) | summed alert-hours over the 15 rear oblasts >= q90 of the previous 365 days and alerts in >= 10 rear oblasts |
| any massive strike in 7 days | at least one massive-strike day among the 7 forecast days |

Frontline (excluded from the "rear" aggregate): Donetska, Zaporizka, Khersonska, Kharkivska, Sumska, Mykolaivska, Dnipropetrovska, Chernihivska, Odeska.
Luhanska is excluded from the whole project; Crimea is not present in the source data.

**Important:** "heavy" and "massive strike" are statistical proxies for unusual alert load, not verified events.
No official list of massive strikes was used. Thresholds use only past days, so a label never depends on its future.
