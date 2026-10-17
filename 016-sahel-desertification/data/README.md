# 016 data

| File | What | Source | Licence | In the repo? |
|---|---|---|---|---|
| `rain_gourma.csv`, `rain_fakara.csv` | daily rain, mean of 9 points around each site, 1950–2025 | ERA5 via the [Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api) | CC BY 4.0 | yes |
| `ndvi_gourma.csv`, `ndvi_fakara.csv` | MODIS MOD13Q1 NDVI, July–October 16-day composites, median over 5 × 5 km, 2000– | [ORNL DAAC MODIS subset service](https://modis.ornl.gov/data/modis_webservice.html) | NASA: no restrictions | yes |
| `s2_gourma.npz`, `s2_fakara.npz` | Sentinel-2 NDVI over 6 × 6 km, vegetation mask and patch sizes | Copernicus Sentinel-2 L2A via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) | contains modified Copernicus Sentinel data 2023 (free and open) | yes |

Rebuild from any machine, no accounts needed:

```
pip install numpy pandas rasterio
python fetch_sites.py          # rain in seconds; NDVI is slow, about 40 s per year per site
python sentinel_patches.py     # a minute
```
