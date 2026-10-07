#-------- placeholder.py ---------------------------------------------------------------
#-----------------------------------------------------------------------------
# The last tier of the fallback rule: synthetic daily fields, in the shape of the real
# products, for when ERA5-Land, MODIS, E-OBS and the station records cannot be reached.
#
# A hidden "true" 1 km temperature field is built each day and every product is then
# derived from it the way the real one relates to the real atmosphere: ERA5-Land sees
# it blurred to 0.1 degrees with its own errors, MODIS sees the skin of the surface
# through gaps in the cloud, stations read it at a point.
#
# The truth is deliberately NOT a solution of the model's PDE. Heat from each land-cover
# class is spread with a Gaussian kernel and displaced downwind, the lapse rate changes
# from day to day, and valley floors run warm. The model has to cope with physics it
# does not contain, as it would on real data. Nothing here is an observation.

import numpy as np
import pandas as pd
from scipy import ndimage

from .grid import CLASSES, GAMMA

# what each land-cover class adds to afternoon air temperature on a clear day (K), and
# how much hotter than the air its surface runs (K)
AIR_EFFECT = dict(zip(CLASSES, [2.6, 0.6, -1.2, -0.4, -3.2, -0.5]))
SKIN_EXCESS = dict(zip(CLASSES, [9.5, 6.0, 1.5, 5.0, -5.0, 8.5]))
TRUE = dict(mean_lapse=6.9, lapse_sd=0.6, drift_hours=1.2, tpi_effect=0.6, station_noise=0.25,
            era5_bias=-0.7, missing=0.07)
HEATWAVES = [("2021-08-12", 4.0), ("2022-07-22", 5.5), ("2023-08-23", 4.5)]     # peak day, size in K


EOBS_NETWORK = 30                # stations in the stand-in E-OBS network
EOBS_BANDWIDTH_KM = 35.0         # width of the kernel that interpolates between them


def _normalised(p):
    return p / p.sum()


def summer_days(years=(2021, 2022, 2023)):
    return pd.DatetimeIndex(np.concatenate([pd.date_range(f"{y}-06-01", f"{y}-08-31") for y in years]))


def _ar1(rng, n, rho, sd):
    x = np.zeros(n)
    e = rng.normal(0, sd * np.sqrt(1 - rho ** 2), n)
    x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def smooth_field(shape, length_km, rng, dx=1.0):
    """Random field with unit variance and a given correlation length, made on a coarse
    grid and interpolated up so that long length scales stay cheap."""
    k = max(1, int(length_km / dx / 4))
    small = rng.normal(size=(shape[0] // k + 3, shape[1] // k + 3))
    small = ndimage.gaussian_filter(small, length_km / dx / k, mode="wrap")
    f = ndimage.zoom(small, k, order=3)[:shape[0], :shape[1]]
    return (f - f.mean()) / f.std()


def coarse_grid(pad=0.1, step=0.1):
    """The 0.1 degree latitude-longitude grid of ERA5-Land and E-OBS over the domain."""
    from .grid import BBOX
    lat = np.round(np.arange(BBOX["lat0"] - pad, BBOX["lat1"] + pad + 1e-9, step), 3)
    lon = np.round(np.arange(BBOX["lon0"] - pad, BBOX["lon1"] + pad + 1e-9, step), 3)
    return lat, lon


def sample_coarse(field, grid, lat, lon, blur_km):
    """What a 0.1 degree product sees of a 1 km field: a local average at each node.
    Nodes outside the 1 km grid take the nearest value inside it."""
    from .grid import lonlat_to_xy
    LON, LAT = np.meshgrid(lon, lat)
    x, y = lonlat_to_xy(LON, LAT)
    cols = np.clip((x - grid.x0) / grid.dx - 0.5, 0, grid.nx - 1)
    rows = np.clip((y - grid.y0) / grid.dx - 0.5, 0, grid.ny - 1)
    blurred = ndimage.gaussian_filter(field, blur_km / grid.dx, mode="nearest")
    return ndimage.map_coordinates(blurred, [rows, cols], order=1, mode="nearest")


def generate(grid, elev_km, frac, sea, features, stations, days=None, seed=7, keep_truth=None):
    """Synthetic products for every day. Returns a dict shaped like the real downloads:

      era5   0.1 degree Tmax, skin temperature and 12-15 UTC wind, plus model orography
      lst    1 km daytime land surface temperature with NaN under cloud and over the sea
      eobs   0.1 degree gridded-station Tmax and its elevation
      tmax   station daily maxima (days x stations), NaN where a reading is missing

    keep_truth, if given, is an array (days, ny, nx) that receives the hidden true Tmax."""
    rng = np.random.default_rng(seed)
    days = summer_days() if days is None else days
    n = len(days)
    doy = days.dayofyear.values
    lat, lon = coarse_grid()
    land_coarse = sample_coarse(1.0 - sea.astype(float), grid, lat, lon, 4.0) > 0.5

    # day-to-day drivers, each summer its own run of weather
    base = 30.5 + 1.5 * np.sin(np.pi * (doy - 152) / 92)
    sun = np.zeros(n); lapse = np.zeros(n); tilt = np.zeros((n, 2)); era_bias = np.zeros(n)
    for year in np.unique(days.year):
        m = days.year == year
        base[m] += _ar1(rng, m.sum(), 0.8, 1.6)
        sun[m] = np.clip(0.85 + _ar1(rng, m.sum(), 0.5, 0.22), 0.35, 1.15)
        lapse[m] = np.clip(TRUE["mean_lapse"] + _ar1(rng, m.sum(), 0.7, TRUE["lapse_sd"]), 5.2, 8.4)
        tilt[m, 0] = _ar1(rng, m.sum(), 0.7, 1.0); tilt[m, 1] = _ar1(rng, m.sum(), 0.7, 1.0)
        era_bias[m] = TRUE["era5_bias"] + _ar1(rng, m.sum(), 0.5, 0.4)
    for peak, size in HEATWAVES:
        base += size * np.exp(-0.5 * ((days - pd.Timestamp(peak)).days.values / 3.0) ** 2)
    # wind: a weak easterly basin flow most days, a few strong north-westerly foehn days
    speed = 7.0 * np.exp(rng.normal(0, 0.5, n))                       # km/h
    heading = np.radians(180 + rng.normal(0, 50, n))                  # direction the air moves towards
    foehn = rng.random(n) < 0.06
    speed[foehn] = rng.uniform(22, 32, foehn.sum())
    heading[foehn] = np.radians(-45 + rng.normal(0, 15, foehn.sum()))

    air = sum(AIR_EFFECT[c] * frac[i] for i, c in enumerate(CLASSES))
    skin = sum(SKIN_EXCESS[c] * frac[i] for i, c in enumerate(CLASSES))
    tpi_term = TRUE["tpi_effect"] * np.clip(-features["tpi"] / 0.3, -1, 1)
    coast_term = -1.2 * np.exp(-features["coast"] / 25.0)
    gx = (grid.x[None, :] - grid.x.mean()) / 200.0
    gy = (grid.y[:, None] - grid.y.mean()) / 200.0
    z_model = sample_coarse(elev_km, grid, lat, lon, 4.0)             # km
    z_eobs = sample_coarse(elev_km, grid, lat, lon, 6.0)
    relief = (elev_km - elev_km.min()) / (elev_km.max() - elev_km.min())

    # the stand-in E-OBS network: sparse, mostly low-lying, and not the validation stations
    land = np.flatnonzero((~sea & grid.interior()).ravel())
    pick = rng.choice(land, EOBS_NETWORK, replace=False, p=_normalised(np.exp(-elev_km.ravel()[land] / 0.8)))
    net_iy, net_ix = np.unravel_index(pick, grid.shape)
    from .grid import lonlat_to_xy
    cx, cy = lonlat_to_xy(*np.meshgrid(lon, lat))
    dist2 = (cx.ravel()[:, None] - grid.x[net_ix][None, :]) ** 2 + (cy.ravel()[:, None] - grid.y[net_iy][None, :]) ** 2
    net_weights = np.exp(-0.5 * dist2 / EOBS_BANDWIDTH_KM ** 2)
    net_weights /= net_weights.sum(1, keepdims=True)

    era = dict(lat=lat, lon=lon, orography_m=z_model * 1000, land=land_coarse,
               tmax=np.full((n,) + z_model.shape, np.nan, np.float32), skin=np.full((n,) + z_model.shape, np.nan, np.float32),
               u=np.zeros((n,) + z_model.shape, np.float32), v=np.zeros((n,) + z_model.shape, np.float32))
    eobs = dict(lat=lat, lon=lon, elevation_m=z_eobs * 1000, tx=np.zeros((n,) + z_model.shape, np.float32))
    lst = np.full((n,) + grid.shape, np.nan, np.float32)
    tmax = np.full((n, len(stations)), np.nan)
    synoptic = smooth_field(grid.shape, 120, rng)
    era_err = smooth_field(grid.shape, 70, rng)

    for d in range(n):
        new = d == 0 or days[d].year != days[d - 1].year
        synoptic = smooth_field(grid.shape, 120, rng) if new else 0.6 * synoptic + 0.8 * smooth_field(grid.shape, 120, rng)
        era_err = smooth_field(grid.shape, 70, rng) if new else 0.5 * era_err + 0.87 * smooth_field(grid.shape, 70, rng)
        u = speed[d] * np.cos(heading[d]) * (1 + 0.3 * smooth_field(grid.shape, 80, rng))
        v = speed[d] * np.sin(heading[d]) * (1 + 0.3 * smooth_field(grid.shape, 80, rng))
        large = base[d] + tilt[d, 0] * gx + tilt[d, 1] * gy + 0.8 * synoptic

        # heat from the surface, spread out and carried downwind
        local = ndimage.gaussian_filter(sun[d] * air, 2.5 + 0.2 * speed[d], mode="nearest")
        drift = TRUE["drift_hours"] * speed[d] / grid.dx
        local = ndimage.shift(local, (drift * np.sin(heading[d]), drift * np.cos(heading[d])), order=1, mode="nearest")
        theta0 = large + local + sun[d] * coast_term + tpi_term + 0.35 * smooth_field(grid.shape, 6, rng)
        truth = theta0 - lapse[d] * elev_km                          # the temperature a thermometer would read

        if keep_truth is not None:
            keep_truth[d] = truth
        obs = theta0[stations.iy, stations.ix] - lapse[d] * stations.z_km.values + rng.normal(0, TRUE["station_noise"], len(stations))
        obs[rng.random(len(stations)) < TRUE["missing"]] = np.nan
        tmax[d] = obs

        coarse_view = large + 0.3 * local + sun[d] * coast_term + era_bias[d] + 0.6 * era_err
        era["tmax"][d] = np.where(land_coarse, sample_coarse(coarse_view, grid, lat, lon, 4.0) - lapse[d] * z_model, np.nan)
        surface = truth + sun[d] * skin + 0.8 * smooth_field(grid.shape, 3, rng)
        era["skin"][d] = np.where(land_coarse, sample_coarse(surface + 1.0 * era_err, grid, lat, lon, 4.0), np.nan)
        era["u"][d] = sample_coarse(u, grid, lat, lon, 4.0) / 3.6     # stored in m/s like the real product
        era["v"][d] = sample_coarse(v, grid, lat, lon, 4.0) / 3.6
        # E-OBS is itself an interpolation of stations, so it is made the same way: read the
        # truth at its own sparse network, then smooth between those readings
        reading = theta0[net_iy, net_ix] + (GAMMA - lapse[d]) * elev_km[net_iy, net_ix] + rng.normal(0, TRUE["station_noise"], EOBS_NETWORK)
        eobs["tx"][d] = (net_weights @ reading).reshape(z_eobs.shape) - GAMMA * z_eobs

        cloud_share = np.clip(0.08 + 1.1 * (1.15 - sun[d]), 0.05, 0.97)
        cloud = smooth_field(grid.shape, 40, rng) + 0.8 * relief
        clear = (cloud < np.quantile(cloud, 1 - cloud_share)) & ~sea
        lst[d] = np.where(clear, surface, np.nan)

    drivers = pd.DataFrame(dict(base=base, sun=sun, lapse=lapse, speed_kmh=speed, heading_deg=np.degrees(heading) % 360,
                                foehn=foehn), index=days)
    return dict(days=days, era5=era, lst=lst, eobs=eobs, tmax=tmax, drivers=drivers)
