#!/usr/bin/env python3
#-------- download_data.py -------------------------------------------------------------
#-----------------------------------------------------------------------------
# The first tier of the fallback rule: the real data. Run this on a machine with
#   - a Copernicus CDS key in ~/.cdsapirc            (ERA5-Land, E-OBS)
#   - an authenticated Earth Engine account         (GLO-30 DEM, WorldCover, MODIS LST)
#   - open internet                                  (Meteostat, GHCN-Daily)
#
#   pip install cdsapi xarray netCDF4 earthengine-api requests
#   python download_data.py --ee-project YOUR_EARTH_ENGINE_PROJECT
#   python download_data.py --ee-project ... --only stations era5      (any subset)
#
# Everything is regridded here and written to data/real/ in exactly the layout the
# simulation uses, so the rest of the project does not know the difference. Each part is
# independent and skipped if its output exists; what cannot be fetched is left to the
# fallback tiers and shows up as such in the provenance table.
#
# NOTE: this script was written in an environment that could not reach any of these
# services, so unlike the rest of the project it has not been run end to end. The
# regridding helpers are shared with the tested code; the request syntax is the part to
# check first if something fails.

import argparse
import gzip
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tmax1km.grid import BBOX, CLASSES, EPSG, Grid
from tmax1km.simulation import sample_coarse, summer_days

REAL = os.path.join(HERE, "data", "real")
RAW = os.path.join(REAL, "raw")
YEARS = (2021, 2022, 2023)
DAYS = summer_days(YEARS)
AREA = [BBOX["lat1"] + 0.1, BBOX["lon0"] - 0.1, BBOX["lat0"] - 0.1, BBOX["lon1"] + 0.1]      # N, W, S, E


def have(name):
    return os.path.exists(os.path.join(REAL, name))


#-------- stations: Meteostat, then GHCN-Daily ----------------------------------------------
def meteostat_daily(station):
    """Daily Tmax for one station, from the Meteostat bulk files."""
    import requests
    r = requests.get(f"https://bulk.meteostat.net/v2/daily/{station}.csv.gz", timeout=60)
    r.raise_for_status()
    cols = ["date", "tavg", "tmin", "tmax", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"]
    df = pd.read_csv(io.BytesIO(gzip.decompress(r.content)), names=cols, parse_dates=["date"]).set_index("date")
    return df.tmax.reindex(DAYS)


def ghcn_daily(grid):
    """Fallback: every GHCN-Daily station in the domain, straight from NOAA."""
    import requests
    base = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"
    txt = requests.get(base + "ghcnd-stations.txt", timeout=120).text
    rows = [(l[:11], float(l[12:20]), float(l[21:30]), float(l[31:37]), l[41:71].strip()) for l in txt.splitlines()]
    st = pd.DataFrame(rows, columns=["id", "latitude", "longitude", "elevation", "name"])
    st = st[st.latitude.between(BBOX["lat0"], BBOX["lat1"]) & st.longitude.between(BBOX["lon0"], BBOX["lon1"]) & (st.elevation > -900)]
    series = {}
    for sid in st.id:
        url = f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{sid}.csv"
        try:
            df = pd.read_csv(url, usecols=lambda c: c in ("DATE", "TMAX"), parse_dates=["DATE"]).set_index("DATE")
            if "TMAX" in df:
                series[sid] = (df.TMAX / 10.0).reindex(DAYS)                 # tenths of a degree
        except Exception as err:
            print("  GHCN", sid, "skipped:", err)
    return st[st.id.isin(series)].assign(country=""), pd.DataFrame(series)


def stations(grid):
    st = pd.read_csv(os.path.join(HERE, "data", "estimated", "stations.csv"), dtype={"id": str})
    series = {}
    for sid in st.id:
        try:
            series[sid] = meteostat_daily(sid)
        except Exception as err:
            print("  Meteostat", sid, "skipped:", err)
    tm = pd.DataFrame(series, index=DAYS)
    good = tm.columns[tm.notna().mean() >= 0.8] if len(tm.columns) else []
    source = "Meteostat daily"
    if len(good) < 10:
        print("fewer than 10 Meteostat stations with 80% coverage: falling back to GHCN-Daily")
        st, tm = ghcn_daily(grid)
        good = tm.columns[tm.notna().mean() >= 0.8]
        source = "GHCN-Daily"
    st[st.id.isin(good)].to_csv(os.path.join(REAL, "stations.csv"), index=False)
    tm[list(good)].to_csv(os.path.join(REAL, "station_tmax.csv"), float_format="%.2f")
    print(f"stations: {len(good)} with >= 80% coverage ({source})")


#-------- ERA5-Land ------------------------------------------------------------------------------
def era5land(grid):
    """Hourly 2 m temperature, skin temperature and 10 m wind, reduced to daily fields:
    Tmax over the day, skin temperature near the Aqua overpass, mean 12-15 UTC wind."""
    import cdsapi
    import xarray as xr
    client = cdsapi.Client()
    parts = []
    for year in YEARS:
        for month in (6, 7, 8):
            target = os.path.join(RAW, f"era5land_{year}{month:02d}.nc")
            if not os.path.exists(target):
                client.retrieve("reanalysis-era5-land", dict(
                    variable=["2m_temperature", "skin_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"],
                    year=str(year), month=f"{month:02d}", day=[f"{d:02d}" for d in range(1, 32)],
                    time=[f"{h:02d}:00" for h in range(24)], area=AREA, data_format="netcdf", download_format="unarchived"), target)
            parts.append(xr.open_dataset(target))
    ds = xr.concat(parts, "valid_time" if "valid_time" in parts[0].dims else "time")
    t = "valid_time" if "valid_time" in ds.dims else "time"
    ds = ds.sortby("latitude").sortby(t)
    daily = lambda a, how: getattr(a.groupby(a[t].dt.floor("D")), how)()
    at = lambda hours: ds.sel({t: ds[t].dt.hour.isin(hours)})
    tmax = daily(ds.t2m - 273.15, "max")
    skin = daily(at([12, 13]).skt - 273.15, "mean")              # Aqua passes at about 13:30 local solar time
    u = daily(at([12, 13, 14, 15]).u10, "mean")
    v = daily(at([12, 13, 14, 15]).v10, "mean")
    pick = lambda a: a.sel({a.dims[0]: DAYS.values}).values.astype(np.float32)
    lat, lon = ds.latitude.values, ds.longitude.values

    # model orography: ERA5-Land does not serve it through the CDS, so use the mean of the
    # 1 km elevation within each 0.1 degree box (recorded as an estimate)
    static = np.load(os.path.join(REAL if have("static_1km.npz") else os.path.join(HERE, "data", "estimated"), "static_1km.npz"))
    oro = sample_coarse(static["elevation_km"].astype(float), grid, lat, lon, 4.0) * 1000
    np.savez_compressed(os.path.join(REAL, "era5land.npz"), lat=lat, lon=lon, tmax=pick(tmax), skin=pick(skin), u=pick(u), v=pick(v),
                        orography_m=oro, orography_source="mean of 1 km elevation per grid box")
    print("ERA5-Land:", pick(tmax).shape)


#-------- Earth Engine: DEM, land cover, MODIS LST ---------------------------------------------
def ee_grid(grid):
    top = (grid.y0 + grid.ny * grid.dx) * 1000
    return dict(dimensions=dict(width=grid.nx, height=grid.ny), crsCode=f"EPSG:{EPSG}",
                affineTransform=dict(scaleX=grid.dx * 1000, shearX=0, translateX=grid.x0 * 1000,
                                     shearY=0, scaleY=-grid.dx * 1000, translateY=top))


def ee_pixels(image, grid):
    """An Earth Engine image sampled on the project grid, as (bands, ny, nx) with row 0 in the south."""
    import ee
    arr = ee.data.computePixels(dict(expression=image, fileFormat="NUMPY_NDARRAY", grid=ee_grid(grid)))
    return np.stack([arr[name].astype(np.float32) for name in arr.dtype.names])[:, ::-1, :]


def ee_mean_on_grid(image, grid, source_projection):
    """Average a fine image over each 1 km cell (rather than sampling its centre)."""
    import ee
    t = ee_grid(grid)["affineTransform"]
    transform = [t["scaleX"], 0, t["translateX"], 0, t["scaleY"], t["translateY"]]
    return image.setDefaultProjection(source_projection).reduceResolution(ee.Reducer.mean(), bestEffort=True, maxPixels=65535) \
        .reproject(crs=f"EPSG:{EPSG}", crsTransform=transform)


def static_fields(grid):
    import ee
    dem_col = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
    dem = ee_mean_on_grid(dem_col.mosaic(), grid, dem_col.first().projection())
    elev = np.nan_to_num(ee_pixels(dem.unmask(0), grid)[0], nan=0.0) / 1000.0

    # ESA WorldCover 2021 (v200) codes -> the six classes
    codes = {"built-up": [50], "cropland": [40], "tree cover": [10, 95], "grass/shrub": [20, 30, 90, 100], "water": [80], "other": [60, 70]}
    wc_col = ee.ImageCollection("ESA/WorldCover/v200")
    wc = wc_col.mosaic().unmask(80)                                         # unmapped open sea counts as water
    bands = [ee_mean_on_grid(wc.remap(codes[c], [1] * len(codes[c]), 0).rename(f"c{i}").toFloat(), grid, wc_col.first().projection())
             for i, c in enumerate(CLASSES)]
    frac = np.clip(ee_pixels(ee.Image.cat(bands), grid), 0, 1)
    frac = frac / np.maximum(frac.sum(0), 1e-6)
    sea = np.load(os.path.join(HERE, "data", "estimated", "static_1km.npz"))["sea"]     # coastline from Natural Earth
    np.savez_compressed(os.path.join(REAL, "static_1km.npz"), elevation_km=elev.astype(np.float32), fractions=frac.astype(np.float32), sea=sea)
    print("static fields: elevation up to", int(elev.max() * 1000), "m")


def modis_lst(grid):
    """MYD11A1 daytime LST in degrees C on the 1 km grid, NaN where cloudy or poor quality.
    Kept: mandatory QA 'good', or 'other quality' with an LST error of at most 2 K."""
    import ee
    out = np.lib.format.open_memmap(os.path.join(REAL, "lst_1km.npy"), "w+", np.float32, (len(DAYS),) + grid.shape)
    out[:] = np.nan

    def clean(img):
        qc = img.select("QC_Day")
        good = qc.bitwiseAnd(3).eq(0).Or(qc.bitwiseAnd(3).eq(1).And(qc.rightShift(6).bitwiseAnd(3).lte(1)))
        return img.select("LST_Day_1km").multiply(0.02).subtract(273.15).updateMask(good).unmask(-9999).toFloat()

    col = ee.ImageCollection("MODIS/061/MYD11A1")
    for i, day in enumerate(DAYS):
        imgs = col.filterDate(day.strftime("%Y-%m-%d"), (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        if imgs.size().getInfo() == 0:
            continue
        a = ee_pixels(clean(imgs.first()), grid)[0]
        out[i] = np.where(a < -1000, np.nan, a)
        if i % 30 == 0:
            print("  MODIS", day.date(), f"{np.isfinite(out[i]).mean():.0%} clear")
    out.flush()


#-------- E-OBS --------------------------------------------------------------------------------
def eobs(grid, version):
    import cdsapi
    import xarray as xr
    target = os.path.join(RAW, "eobs.zip")
    if not os.path.exists(target):
        cdsapi.Client().retrieve("insitu-gridded-observations-europe", dict(
            product_type="ensemble_mean", variable=["maximum_temperature", "land_surface_elevation"],
            grid_resolution="0_1deg", period="2011_2023", version=[version]), target)
    with zipfile.ZipFile(target) as z:
        z.extractall(RAW)
        names = z.namelist()
    box = dict(latitude=slice(AREA[2], AREA[0]), longitude=slice(AREA[1], AREA[3]))
    tx = xr.open_dataset(os.path.join(RAW, next(n for n in names if n.startswith("tx_")))).tx.sel(**box)
    el = xr.open_dataset(os.path.join(RAW, next(n for n in names if n.startswith("elev_")))).elevation.sel(**box)
    np.savez_compressed(os.path.join(REAL, "eobs.npz"), lat=tx.latitude.values, lon=tx.longitude.values,
                        tx=tx.sel(time=DAYS.values).values.astype(np.float32), elevation_m=np.nan_to_num(el.values, nan=0.0))
    print("E-OBS:", tx.sel(time=DAYS.values).shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ee-project", default=os.environ.get("EE_PROJECT"))
    ap.add_argument("--eobs-version", default="30_0e")
    ap.add_argument("--only", nargs="*", default=["static", "stations", "era5", "modis", "eobs"])
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    grid = Grid()
    needs_ee = {"static", "modis"} & set(args.only)
    if needs_ee:
        import ee
        try:
            ee.Initialize(project=args.ee_project)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=args.ee_project)
    steps = [("static", "static_1km.npz", lambda: static_fields(grid)), ("stations", "station_tmax.csv", lambda: stations(grid)),
             ("era5", "era5land.npz", lambda: era5land(grid)), ("modis", "lst_1km.npy", lambda: modis_lst(grid)),
             ("eobs", "eobs.npz", lambda: eobs(grid, args.eobs_version))]
    for name, output, run in steps:
        if name not in args.only:
            continue
        if have(output):
            print(name, "already there, skipped")
            continue
        try:
            run()
        except Exception as err:
            print(f"{name} FAILED and is left to the fallback tiers: {type(err).__name__}: {err}")
            partial = os.path.join(REAL, output)
            if os.path.exists(partial):
                os.remove(partial)
    print("done. Delete data/cache/ and re-run run_pipeline.py --force to rebuild everything on the real data.")
