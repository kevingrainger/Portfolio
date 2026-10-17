# Next steps (handoff, 17 October 2026)

Where this portfolio stands, and how to pick it up on any machine. Read this first. Everything listed as done is committed and pushed, and the site (kevingrainger.github.io) is republished to match. Data sources and download commands for every project are in [`DATA.md`](DATA.md).

## Completion

| Project | Done | What is left |
|---|:-:|---|
| 001 Coral reef | 97% | optional: larval supply, so a dead reef can recover |
| 002 Olive grove | 98% | optional: plan item X8 (annual reset on/off) |
| **003 Po Valley 1 km** | **80%** | **the real-data run: see below** |
| 004 corrlib | 98% | – |
| 005 Seismic | 100% | – |
| 006 Option pricer | 97% | optional: calibrate SET50 to TFEX option quotes |
| 007 Italian power | 85% | SEAS5 seasonal forecasts, regime model (N4), regime and fan figures |
| 008 Spanish solar | 55% | CERRA at 5.5 km; prices and cannibalisation (needs an ENTSO-E key: Energy-Charts' Spanish prices are private-use only); strategy arms A1–A4; hedging rules H1–H3; the map |
| 012 Ecology for bankers | 95% | optional: graph neural network appendix |
| 013 Coffee | 90% | hedge application (plan step 6); a curve test needs a frost after 2024 |
| 014 Copper from space | 95% | Earth Engine version for near-live data (`smelter_heat.py`) |
| 015 Hedging | 95% | minimum-variance delta needs implied volatilities (not free) |
| 016 Sahel | 92% | degradation rising with land use over time; CHIRPS rainfall; inference of f, m, d from patch sizes |
| 009–011 Undergraduate | 100% | – |

## 003: resume the real-data run (the one task in flight)

The code for the real data is pushed and was checked end to end with a smoke run. The downloads are git-ignored and live in `003-po-valley-downscaling/data/real/` on the original machine (OneDrive). On a new machine, fetch them again first:

```
cd 003-po-valley-downscaling
pip install cdsapi xarray netCDF4 requests rasterio
python download_data.py --only stations era5 eobs     # needs a Copernicus CDS key in ~/.cdsapirc; about 1 hour (CDS queue)
python -c "import download_data as dd; from tmax1km.grid import Grid; dd.modis_lst_pc(Grid())"   # MODIS from Planetary Computer, no account; about 1 hour, resumable
```

Then run the pipeline on the real data:

```
rm -rf data/cache/real_* data/cache/run_real        # clear any cache built before MODIS was complete
python run_pipeline.py --real --force               # about 1.5-2 hours; results go to results_real/
```

- The terrain and land-cover fields stay at the estimated tier (`data/estimated/`) unless `python download_data.py --only static --ee-project ...` is run with an Earth Engine account.
- After the run, compare `results_real/` with the simulated `results/`, add a "Real data" section to the 003 README (held-out RMSE for B0–B3 and M1–M3), and update the site entry in `kevingrainger.github.io/build_index.py`.
- A preliminary run was stopped part-way on 17 October, so there are no real-data results yet.

## Decisions waiting on Kevin

1. Where the private planning notes (`_planning/`, not in this repo) should live: a private GitHub repo, or an encrypted archive.
2. An ENTSO-E transparency platform key (free registration) for 008's prices.
3. An Earth Engine sign-in, for 003's terrain and land cover and 014's near-live smelter series.
4. Whether IMF PortWatch is the ship data source intended for 014.
5. Uploading the 008 ERA5 dataset (`release-assets/iberia-era5-2023.zarr.zip`, 232 MB, on the original machine) as GitHub Release `data-v1`; `DATA.md` already links to it.

## Conventions

- Commits are in Kevin's name with no co-author line.
- Write-ups read like a Quanta Magazine article: a simple question, the equations, credited sources, and few figures.
- Never pair orange with blue, and keep labels off lines. `figstyle.PALETTE` begins blue then orange, so new figures pick colours explicitly.
- Free, openly available data only; no paid services or heavy installs.
- Say so plainly when a result is negative or unclear.
- Push the Portfolio repo first, then rebuild the site:

  ```
  python build_index.py --publish https://raw.githubusercontent.com/kevingrainger/Portfolio/main/
  ```
