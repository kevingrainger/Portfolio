# Data: what is in the repo, and how to get the rest

Open data that is small enough is committed next to the code. Two kinds of data are not:
- **Market prices from Yahoo Finance and similar sources,** whose terms don't allow redistribution. Each project's script downloads them again.
- **Anything over GitHub's 100 MB file limit,** which is attached to a GitHub Release instead.

Every project's `data/README.md` has the details.

| Project | In the repo | Not in the repo, and how to get it |
|---|---|---|
| 001 Coral | NOAA Coral Reef Watch DHW, Scott–Ashmore 2015–16; Global Coral Bleaching Database, Scott Reef 2016 (`data/`) | `Coral temps.xlsx`, the old hand-drawn curve: no longer used |
| 002 Olive | — (simulation only) | — |
| 003 Po Valley | static 1 km fields and station list (`data/estimated/`) | real ERA5-Land, E-OBS, MODIS, terrain and station inputs: `python download_data.py --ee-project YOUR_PROJECT` (needs a free [Copernicus CDS key](https://cds.climate.copernicus.eu/how-to-api) and an Earth Engine sign-in); the 1.6 GB training cache is rebuilt by `run_pinn.py` |
| 004 corrlib | — | stock closes (Yahoo): re-downloaded by `corrlib_showcase.ipynb` |
| 005 Seismic | — | SCSN labelled waveforms (HDF5, about 6 GB): Ross, Meier & Hauksson (2018), via the SCEDC, [doi:10.7909/C3WD3xH1](https://doi.org/10.7909/C3WD3xH1) |
| 006 Options | — | softs futures (Yahoo, re-downloaded by `options_03_asian_softs.ipynb`); S&P 500 option chain for 4 January 2023 and SET50 history (Investing.com), which can't be redistributed: put them in `data/` to re-run |
| 007 Italian power | — (simulated data) | real GME, TTF, load and SEAS5 data: still to be built |
| 008 Spanish solar | the pipeline, recipe and test fixtures; AIFS and IFS solar forecasts and ERA5 daily solar energy at the 25 sites (`aifs/data/`, Open-Meteo, CC BY 4.0) | **the ERA5 Iberia dataset for 2023 (Anemoi Zarr, 279 MB): GitHub Release `data-v1`, file `iberia-era5-2023.zarr.zip` (232 MB), checksum below**; or rebuild it with `python download_era5.py 2023-01 2023-12` then `anemoi-datasets create recipes/iberia-era5.yaml data/iberia-era5.zarr` (free CDS key) |
| 012 Percolation | — (generated networks) | — |
| 013 Coffee | ERA5, GFS and ECMWF temperatures at 8 regions (`data/`) | coffee futures (Yahoo): `python fetch_data.py prices` |
| 014 Copper | PortWatch ports, BCRP copper exports, smelter hot spots | copper futures (Yahoo): fetched by `analyse_ports.py` |
| 015 Hedging | — | six commodity futures (Yahoo): fetched by `backtest.py` |
| 016 Sahel | ERA5 rain, MODIS NDVI, Sentinel-2 patches at Gourma and Fakara (`data/`) | — |

## The 008 dataset from the Release

```
curl -L -o iberia-era5-2023.zarr.zip https://github.com/kevingrainger/Portfolio/releases/download/data-v1/iberia-era5-2023.zarr.zip
sha256sum iberia-era5-2023.zarr.zip      # should match the line below
unzip iberia-era5-2023.zarr.zip -d 008-spanish-solar-portfolio/iberia_datasets/data/
```

`cb418f205c286e88ee02da9f1743dbdfe1fe542adf3c3a0303ba3e512acbe885  iberia-era5-2023.zarr.zip`

Contains modified Copernicus Climate Change Service information (2023). ERA5: Hersbach et al. (2020), QJRMS 146, 1999.
