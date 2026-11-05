# 016 data

| File | What | Source | Licence | In the repo? |
|---|---|---|---|---|
| `rain_gourma.csv`, `rain_fakara.csv` | daily rain, mean of 9 points around each site, 1950–2025 | ERA5 via the [Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api) | CC BY 4.0 | yes |
| `ndvi_gourma.csv`, `ndvi_fakara.csv` | MODIS MOD13Q1 NDVI, July–October 16-day composites, median over 5 × 5 km, 2000– | [ORNL DAAC MODIS subset service](https://modis.ornl.gov/data/modis_webservice.html) | NASA: no restrictions | yes |
| `sahel_grid.csv` | August-September MODIS NDVI and June-October CHIRPS rain, 297 sites, 2001-2024 | Earth Engine (MODIS/061/MOD13Q1, UCSB-CHG/CHIRPS/DAILY) | NASA: no restrictions; CHIRPS public domain | yes |
| `sahel_static.csv` | cropland share, population 2000 and 2020, topsoil sand and organic carbon, livestock units per km² | Earth Engine (ESA/WorldCover/v200, CIESIN/GPWv411, OpenLandMap); FAO GLW4 | CC BY 4.0 | yes |
| `glw4/5_{Ct,Sh,Gt}_2015_Da.tif` | cattle, sheep and goats per pixel in 2015, 5 arc-minutes | FAO Gridded Livestock of the World on [Harvard Dataverse](https://dataverse.harvard.edu/dataverse/glw_4) (LHBICE, VZOYHM, YYG6ET) | CC BY 4.0 | no, about 30 MB |
| `s2_gourma.npz`, `s2_fakara.npz` | Sentinel-2 NDVI over 6 × 6 km, vegetation mask and patch sizes | Copernicus Sentinel-2 L2A via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) | contains modified Copernicus Sentinel data 2023 (free and open) | yes |

Rebuild from any machine, no accounts needed:

```
pip install numpy pandas rasterio
python fetch_sites.py          # rain in seconds; NDVI is slow, about 40 s per year per site
python sentinel_patches.py     # a minute
```
