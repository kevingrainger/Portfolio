# iberia_datasets

ML-ready weather datasets for the Iberian Peninsula, built with ECMWF's
[`anemoi-datasets`](https://anemoi.readthedocs.io/projects/datasets/) — the data layer of
the Spanish solar portfolio project (`../spanish_solar_portfolio_proposal.md`, §2b).

```
Copernicus CDS ──download_era5.py──▶ monthly GRIB ──anemoi-datasets create──▶ Zarr ──iberia.check──▶ PASS / FAIL
                   (Slurm array)                    (recipes/iberia-era5.yaml)        (quality checks)
```

## What is in it

| Path | |
|---|---|
| `recipes/iberia-era5.yaml` | Anemoi recipe: ERA5 single levels over Iberia (35.5–44.5 °N, 10 °W–4.5 °E), hourly, 0.25°: 2 m temperature, total cloud cover, surface solar radiation downwards, 100 m wind, plus computed forcings (position, time of day and year, insolation) |
| `download_era5.py` | stage 1 — one GRIB per month from the CDS, restartable, Slurm-array friendly |
| `iberia/quality.py` | data-quality checks: time axis, missing values, physical ranges, solar-radiation units, timing and night-time darkness |
| `iberia/solar.py` | ssrd → irradiance (units and mid-hour timing), vectorised solar position, clear-sky model and clear-sky index |
| `iberia/synthetic.py` | an ERA5-lookalike GRIB generator, so the whole pipeline is testable without downloads |
| `iberia/check.py` | `python -m iberia.check DATASET.zarr` — runs every check, exits non-zero on failure |
| `tests/` | end-to-end and fault-injection tests |
| `slurm/` | download array job and build job for a cluster |
| `Dockerfile` | the pipeline in a container; runs the tests by default |
| `../.github/workflows/ci.yml` | CI: corrlib tests, pipeline tests, Docker build and test |

## The traps the checks exist for

ERA5 stores **surface solar radiation downwards** (`ssrd`) as energy **accumulated over the
hour ending at the timestamp**, in J m⁻². Two consequences break solar analyses silently:

- **units** — divide by 3,600 for the mean power in W m⁻². Data stored already divided
  looks plausible and is wrong by a factor of 3,600.
- **timing** — that mean belongs to the *middle* of the hour, 30 minutes before the
  timestamp. Compare it with the sun *at* the timestamp and every afternoon looks
  physically impossible (more sunlight at the surface than arrives at the top of the
  atmosphere).

The tests include the naive version of the timing check and show it raising thousands of
false alarms on correct data — then show the real check passing. Every other check is
tested the same way: break a copy of the data on purpose (drop an hour, convert to
Celsius, pre-divide ssrd, shift it by an hour) and confirm the check catches it.

## Testing without data

```bash
python -m pytest tests -q
```

writes a 48-hour synthetic ERA5-like GRIB file — same grid, same parameters, ssrd encoded
as a one-hour accumulation — runs the real `anemoi-datasets create` on it, and runs every
quality check on the resulting Zarr. About 15 seconds, no network, no credentials. This is
what CI runs on every push.

## Building the real dataset

**1. A Copernicus CDS account** (free) at <https://cds.climate.copernicus.eu>: accept the
ERA5 licence on the dataset page, then put your API key in `~/.cdsapirc` as described on
the CDS "how to API" page. The key stays on your machine; nothing in this repository
contains or needs it.

**2. Download and build**

```bash
python download_era5.py 2022-01 2023-12
anemoi-datasets create recipes/iberia-era5.yaml data/iberia-era5.zarr
python -m iberia.check data/iberia-era5.zarr
```

or on a cluster:

```bash
jid=$(sbatch --parsable slurm/download_era5.sbatch)
sbatch --dependency=afterok:${jid} slurm/build_dataset.sbatch
```

## Installing

```bash
pip install -r requirements.txt
```

**On Windows**, two workarounds were needed (Linux and the Docker image need neither):

- `anemoi-transform` depends on `healpy`, which has no Windows build. It is only used for
  HEALPix grids, so the stack was installed without it (`pip install --no-deps` from a
  resolved list with `healpy` removed); regular lat-lon datasets work.
- `anemoi-datasets` prints emoji in its progress messages, which the Windows console
  encoding cannot show. Set `PYTHONIOENCODING=utf-8`.

## Next

- `recipes/iberia-cerra.yaml` — the same variables from CERRA, the 5.5 km Copernicus
  regional reanalysis, for the resolution comparison in the proposal (§3.6)
- site extraction and the clear-sky-index field for the correlation analysis
- AIFS forecasts run from this dataset with `anemoi-inference` (proposal §4.2b)
