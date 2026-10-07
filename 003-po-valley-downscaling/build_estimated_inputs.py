#!/usr/bin/env python3
#-------- build_estimated_inputs.py -------------------------------------------------
#-----------------------------------------------------------------------------
# The middle tier of the fallback rule: real download -> LOOKUP / ESTIMATE -> synthetic.
#
# When the real rasters cannot be downloaded, this builds stand-ins from small open
# files that can: real station locations, real coastline, lake and city outlines, and
# a real but coarse elevation model. The outputs are committed in data/estimated/ so
# the project runs with no downloads at all; download_data.py replaces them with the
# real thing.
#
#   python build_estimated_inputs.py RAW_DIR
#
# RAW_DIR must hold
#   stations.db                          github.com/meteostat/weather-stations (CC BY 4.0)
#   ne_10m_land.shp, ne_10m_lakes.shp,
#   ne_10m_urban_areas.shp               github.com/nvkelso/natural-earth-vector (public domain)
# and pvlib must be installed: its bundled Altitude.h5 is a 5 arc-minute global
# elevation grid in 28 m steps, derived from the Mapzen terrain tiles (SRTM, GMTED2010, ETOPO1).

import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import shapefile
import shapely
from scipy import ndimage
from shapely.geometry import box, shape

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmax1km.grid import BBOX, CLASSES, Grid, block_mean, lonlat_to_xy, slope

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "estimated")
SUB = 4                                          # polygons are rasterised on a 250 m sub-grid


def polygons(path, pad=0.5):
    """Polygons of a Natural Earth layer that touch the domain, as one geometry in UTM km."""
    window = box(BBOX["lon0"] - pad, BBOX["lat0"] - pad, BBOX["lon1"] + pad, BBOX["lat1"] + pad)
    parts = []
    for s in shapefile.Reader(path).shapes():
        b = s.bbox
        if b[2] < window.bounds[0] or b[0] > window.bounds[2] or b[3] < window.bounds[1] or b[1] > window.bounds[3]:
            continue
        geom = shape(s.__geo_interface__).buffer(0).intersection(window)
        if not geom.is_empty:
            parts.append(geom)
    merged = shapely.unary_union(parts)
    return shapely.transform(merged, lambda c: np.column_stack(lonlat_to_xy(c[:, 0], c[:, 1])))


def fraction_inside(geom, grid):
    """Share of each 1 km cell covered by a geometry."""
    fine = Grid(grid.x0, grid.y0, grid.nx * SUB, grid.ny * SUB, grid.dx / SUB)
    X, Y = np.meshgrid(fine.x, fine.y)
    inside = shapely.contains_xy(geom, X.ravel(), Y.ravel()).reshape(fine.shape)
    return block_mean(inside.astype(float), SUB)


def fractal(shape, rng, beta=2.4, lo_km=1.5, hi_km=12.0, dx=1.0):
    """Random field with a power-law spectrum between two wavelengths, unit variance."""
    ky = np.fft.fftfreq(shape[0], dx)[:, None]
    kx = np.fft.rfftfreq(shape[1], dx)[None, :]
    k = np.hypot(kx, ky)
    amp = np.where((k > 1 / hi_km) & (k < 1 / lo_km), np.maximum(k, 1e-9) ** (-beta / 2), 0.0)
    f = np.fft.irfft2(amp * np.exp(2j * np.pi * rng.random(amp.shape)), s=shape)
    return f / f.std()


def elevation(grid, land):
    """Real relief at 5 arc-minutes, interpolated to 1 km, plus synthetic sub-grid roughness.

    The coarse grid knows where the Alps and Apennines are but not their valleys, so a
    random rough field scaled by the local relief is added. Without it every cell would
    sit at its ERA5-Land grid-box height and the lapse-rate step would have nothing to do."""
    import h5py
    import pvlib
    with h5py.File(os.path.join(os.path.dirname(pvlib.__file__), "data", "Altitude.h5"), "r") as f:
        raw = f["Altitude"][:]                                           # rows from 90N, columns from 180W
    coarse = np.where(raw == 255, 0.0, raw.astype(float) * 28.0 - 450.0)   # 28 m steps; 255 marks the sea
    lon, lat = grid.lonlat()
    rows = (90.0 - lat) * 12.0 - 0.5
    cols = (lon + 180.0) * 12.0 - 0.5
    smooth = ndimage.map_coordinates(coarse, [rows, cols], order=3, mode="nearest")
    smooth = ndimage.gaussian_filter(smooth, 2.0)
    relief = np.sqrt(np.maximum(ndimage.gaussian_filter(smooth ** 2, 12) - ndimage.gaussian_filter(smooth, 12) ** 2, 0.0))
    rough = fractal(grid.shape, np.random.default_rng(32632), dx=grid.dx)
    z = smooth + 0.55 * relief * rough
    z = np.where(land > 0.5, np.maximum(z, 1.0), 0.0)
    crop = coarse[int((90 - BBOX["lat1"]) * 12) - 2:int((90 - BBOX["lat0"]) * 12) + 3,
                  int((BBOX["lon0"] + 180) * 12) - 2:int((BBOX["lon1"] + 180) * 12) + 3]
    return z / 1000.0, crop


def land_cover(grid, elev_km, land, lakes, urban):
    """Six land-cover fractions per cell. Water and built-up come from real outlines;
    the vegetation classes are assigned from elevation and slope, the way the real
    valley is zoned: crops on the plain, forest on the slopes, pasture and rock above."""
    rng = np.random.default_rng(2021)
    water = np.clip(1.0 - land + lakes, 0, 1)
    # the outlines enclose whole urban regions; roughly 45% of that ground is sealed
    built = 0.45 * ndimage.gaussian_filter(np.clip(urban, 0, 1), 0.8) * (1 - water)
    # small towns the 1:10m outlines miss: a sparse scatter over the plain
    plain = (elev_km < 0.3) & (water < 0.5)
    towns = ndimage.gaussian_filter((rng.random(grid.shape) < 0.004) * plain * 1.0, 1.0) * 6.0
    built = np.clip(built + np.clip(towns, 0, 0.6) * (1 - water), 0, 1 - water)

    sl = slope(elev_km, grid.dx)
    wobble = ndimage.gaussian_filter(rng.normal(size=grid.shape), 6) * 6
    sig = lambda v, mid, width: 1.0 / (1.0 + np.exp(-(v - mid) / width))
    hill = sig(elev_km, 0.28, 0.06) * 0.85 + sig(sl, 4.0, 1.5) * 0.15           # 0 on the plain, 1 on slopes
    alpine = sig(elev_km, 1.9 + 0.02 * wobble, 0.12)
    nival = sig(elev_km, 2.6 + 0.02 * wobble, 0.10)
    free = 1 - water - built
    crop = free * (1 - hill) * (0.88 + 0.01 * wobble).clip(0.6, 0.97)
    tree = free * (1 - hill) - crop + free * hill * (1 - alpine) * 0.85
    grass = free * hill * (1 - alpine) * 0.15 + free * hill * alpine * (1 - nival)
    other = free * hill * alpine * nival
    f = np.stack([built, crop, tree, grass, water, other]).clip(0, 1)
    return (f / f.sum(0)).astype(np.float32)


OBSERVED = ("isd_lite", "metar", "ghcnd", "climat", "dwd_poi")          # the inventory also lists forecast feeds


def stations(grid, db):
    """Real Meteostat stations inside the domain whose observed temperature record covers
    June 2021 to August 2023. Model-forecast feeds in the inventory do not count."""
    con = sqlite3.connect(db)
    q = f"""select s.id, n.name, s.country, s.latitude, s.longitude, s.elevation,
                   min(i.start) as first_day, max(i.end) as last_day, max(i.completeness) as completeness
            from stations s
            join inventory i on i.station = s.id and i.parameter in ('tmax', 'temp')
                 and i.provider in {OBSERVED} and i.start <= '2021-06-01' and i.end >= '2023-08-31'
            left join names n on n.station = s.id and n.language = 'en'
            where s.latitude between ? and ? and s.longitude between ? and ?
            group by s.id"""
    df = pd.read_sql(q, con, params=(BBOX["lat0"], BBOX["lat1"], BBOX["lon0"], BBOX["lon1"]))
    df = df[df.elevation.notna()]
    iy, ix, inside = grid.cell(df.longitude, df.latitude)
    edge = (ix < 2) | (ix > grid.nx - 3) | (iy < 2) | (iy > grid.ny - 3)
    return df[inside & ~edge].sort_values("id").reset_index(drop=True)


if __name__ == "__main__":
    raw = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    grid = Grid()
    land = fraction_inside(polygons(os.path.join(raw, "ne_10m_land.shp")), grid)
    lakes = fraction_inside(polygons(os.path.join(raw, "ne_10m_lakes.shp")), grid)
    urban = fraction_inside(polygons(os.path.join(raw, "ne_10m_urban_areas.shp")), grid)
    elev, coarse = elevation(grid, land - lakes)
    frac = land_cover(grid, elev, land, lakes, urban)
    np.savez_compressed(os.path.join(OUT, "static_1km.npz"), elevation_km=elev.astype(np.float32),
                        fractions=frac, sea=(land < 0.5), dem_5arcmin_m=coarse.astype(np.int16))
    st = stations(grid, os.path.join(raw, "stations.db"))
    st.to_csv(os.path.join(OUT, "stations.csv"), index=False)
    print(f"grid {grid.nx} x {grid.ny}, {len(st)} stations, elevation {elev.min()*1000:.0f}-{elev.max()*1000:.0f} m")
    print("mean land-cover fractions:", dict(zip(CLASSES, frac.mean((1, 2)).round(3))))
