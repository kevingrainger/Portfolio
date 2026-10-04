#!/usr/bin/env python3
#-------- solar_sites.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# From the Iberian Anemoi dataset to the correlation structure of a solar fleet.
#
#   dataset (ssrd, J m^-2 per hour)  ->  irradiance at mid-hour (W m^-2)
#   ->  clear-sky index k = GHI / GHI_clear, daytime only
#   ->  Gaussian-rank correlation between sites, by season (corrlib)
#   ->  correlation against distance, fitted with rho(d) = exp(-(d/L)^p)
#   ->  N_eff, the effective number of independent sites, raw and cleaned
#
# rho(d) = exp(-(d/L)^p) is a valid (positive-definite) correlation model for any
# 0 < p <= 2: p = 1 is the exponential, p = 2 the Gaussian. Fitting p as well as L
# lets the data say which shape it has, instead of assuming one.

import os
import sys

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "iberia_datasets"))
sys.path.insert(0, os.path.join(HERE, "..", "003-corrlib-correlation-toolbox"))

from iberia.solar import ssrd_to_irradiance, clear_sky_ghi, _solar_elevation   # noqa: E402

DATASET = os.path.join(HERE, "iberia_datasets", "data", "iberia-era5.zarr")

# Approximate locations in the main Spanish solar regions (city coordinates; the
# nearest 0.25-degree grid point is used, so precision beyond ~10 km is irrelevant)
SITES = {
    "Huelva": (37.26, -6.95), "Sevilla": (37.39, -5.98), "Cordoba": (37.88, -4.78),
    "Jaen": (37.77, -3.79), "Granada": (37.18, -3.60), "Almeria": (36.84, -2.46),
    "Malaga": (36.72, -4.42), "Badajoz": (38.88, -6.97), "Merida": (38.92, -6.34),
    "Caceres": (39.47, -6.37), "Ciudad Real": (38.99, -3.93), "Puertollano": (38.69, -4.11),
    "Toledo": (39.86, -4.03), "Albacete": (38.99, -1.86), "Cuenca": (40.07, -2.14),
    "Murcia": (37.99, -1.13), "Alicante": (38.35, -0.48), "Valencia": (39.47, -0.38),
    "Madrid": (40.42, -3.70), "Guadalajara": (40.63, -3.17), "Salamanca": (40.97, -5.66),
    "Valladolid": (41.65, -4.72), "Zaragoza": (41.65, -0.89), "Teruel": (40.35, -1.11),
    "Lleida": (41.62, 0.62),
}

SEASONS = {"winter (DJF)": (12, 1, 2), "spring (MAM)": (3, 4, 5),
           "summer (JJA)": (6, 7, 8), "autumn (SON)": (9, 10, 11)}


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def load(path=DATASET, sites=SITES):
    """Clear-sky index at each site, every daytime hour. Returns k (sites x hours, NaN
    where the sun is too low), the mid-hour times, site names, coordinates (grid
    points actually used), and the whole-grid arrays for maps."""
    from anemoi.datasets import open_dataset
    ds = open_dataset(path)
    lats, lons = np.asarray(ds.latitudes), np.asarray(ds.longitudes)
    names = list(sites)
    idx = np.array([np.argmin(haversine_km(la, lo, lats, lons)) for la, lo in sites.values()])

    v = ds.variables.index("ssrd")
    ssrd = np.asarray(ds[:, v, 0, :])                   # hours x grid points
    irr, mid = ssrd_to_irradiance(ssrd.T, np.asarray(ds.dates))
    k_grid = clear_sky_index_grid(irr, mid, lats, lons)
    return dict(k_sites=k_grid[idx], k_grid=k_grid, times=mid, names=names,
                lat=lats[idx], lon=lons[idx], grid_lat=lats, grid_lon=lons,
                is_synthetic=_placeholder_count() > 0)


def clear_sky_index_grid(irr, mid_times, lats, lons, min_elevation=10.0):
    """k for every grid point and hour; NaN when the sun is below min_elevation."""
    k = np.full(irr.shape, np.nan)
    import pandas as pd
    for j, t in enumerate(pd.to_datetime(mid_times)):
        elev = _solar_elevation(lats, lons, t.to_pydatetime())
        up = elev > min_elevation
        if not up.any():
            continue
        clear = clear_sky_ghi(lats[up], lons[up], t.to_pydatetime())
        k[up, j] = irr[up, j] / np.maximum(clear, 1.0)
    return k


def _placeholder_count():
    import glob
    return len(glob.glob(os.path.join(HERE, "iberia_datasets", "data", "era5", "*.synthetic")))


# The clear-sky index is meant to divide the sun out. On real data it does not quite:
# a simple clear-sky model misjudges the atmosphere differently at different sun
# angles, and summer afternoons cloud over more than mornings. Every site inherits the
# same daily shape - which shows up as correlation between sites that have nothing to
# do with shared weather. So each site's typical value for that hour of that month is
# subtracted first, and what is left - the weather - is what gets correlated.
def deseasonalise(k, times):
    t = np.asarray(times, dtype="datetime64[h]")
    month = t.astype("datetime64[M]").astype(int) % 12
    hour = (t - t.astype("datetime64[D]")).astype(int)
    anomaly = np.full(k.shape, np.nan)
    for m in range(12):
        for h in range(24):
            cols = (month == m) & (hour == h)
            if cols.any():
                block = k[:, cols]
                with np.errstate(all="ignore"):
                    typical = np.nanmedian(block, axis=1, keepdims=True)
                anomaly[:, cols] = block - typical
    return anomaly


def season_mask(times, months):
    m = np.asarray(times, dtype="datetime64[M]").astype(int) % 12 + 1
    return np.isin(m, months)


def correlation(k, measure="gaussian_rank"):
    """Gaussian-rank correlation over the hours where every site has daylight."""
    from corrlib import correlation_measures as cm
    ok = np.all(np.isfinite(k), axis=0)
    X = k[:, ok]
    return (cm.gaussian_rank(X) if measure == "gaussian_rank" else cm.pearson(X)), X


def stretched_exponential(d, L, p):
    return np.exp(-(d / L) ** p)


def fit_correlation_length(C, lat, lon):
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    iu = np.triu_indices(len(lat), k=1)
    d, r = D[iu], C[iu]
    (L, p), cov = curve_fit(stretched_exponential, d, r, p0=(200.0, 1.0), bounds=([1.0, 0.2], [5000.0, 2.0]))
    return L, p, np.sqrt(np.diag(cov)), d, r
