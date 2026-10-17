# 013 data

| File | What | Source | Licence | In the repo? |
|---|---|---|---|---|
| `weather_era5.csv` | daily minimum temperature and rain at 8 coffee regions, 1990– | ERA5 via the [Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api) | CC BY 4.0 (Open-Meteo; ERA5: Copernicus) | yes |
| `forecast_gfs.csv` | daily minimum temperature forecast 1–7 days ahead, 2021– | NOAA GFS via the [Open-Meteo previous-runs API](https://open-meteo.com/en/docs/previous-runs-api) | CC BY 4.0 | yes |
| `forecast_ecmwf.csv` | the same from ECMWF IFS, 2024– | ECMWF IFS via Open-Meteo | CC BY 4.0 | yes |
| `kc_front.csv`, `kc_contracts.csv` | ICE arabica futures, front month and live contracts | Yahoo Finance | not redistributable | **no**: rebuilt by the script |

Rebuild everything, or just the prices, from any machine:

```
pip install yfinance pandas numpy
python fetch_data.py            # all of it, a few minutes
python fetch_data.py prices     # only the Yahoo files
```

Yahoo lists only contracts still trading, so a later download of `kc_contracts.csv` starts later than the one used here (January 2024).
