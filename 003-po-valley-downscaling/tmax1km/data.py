#-------- data.py ----------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Loading, with the fallback rule applied dataset by dataset:
#
#     real download  ->  lookup / estimate  ->  synthetic
#
# download_data.py writes real products into data/real/. Whatever is missing there is
# taken from data/estimated/ (real station locations, coastline and city outlines, a
# coarse real elevation model) and, failing that, from placeholder.py. Every choice is
# recorded in a provenance table that the notebook and README print, so a synthetic
# number can never pass for an observation.
#
# Daily fields are then built the same way whatever their origin (Step 3):
#   theta_E   ERA5-Land Tmax reduced to sea level with ERA5-Land's own orography
#   theta_s   MODIS land surface temperature reduced to sea level, cloud gaps filled
#             with ERA5-Land skin temperature plus the cell's mean clear-sky offset
#             for that month
#   u, v      mean 12-15 UTC wind, km/h
#   y         station Tmax reduced to sea level with the station's own elevation

import os

import numpy as np
import pandas as pd
from scipy import ndimage

from . import placeholder
from .grid import GAMMA, Grid, lonlat_to_xy, static_features

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(HERE, "data", "real")
ESTIMATED = os.path.join(HERE, "data", "estimated")
CACHE = os.path.join(HERE, "data", "cache")
DAILY = ["theta_E", "theta_s", "u", "v", "theta_eobs", "truth"]


def _first(name):
    """Path and tier of the best available copy of a file."""
    for tier, folder in (("real", REAL), ("estimated", ESTIMATED)):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p, tier
    return None, "synthetic"


def to_grid(coarse, lat, lon, grid):
    """Bilinear interpolation of a 0.1 degree field onto the 1 km grid. Missing nodes
    (ERA5-Land has none over the sea) are first filled from their nearest neighbour."""
    coarse = np.asarray(coarse, float)
    if np.isnan(coarse).any():
        idx = ndimage.distance_transform_edt(np.isnan(coarse), return_distances=False, return_indices=True)
        coarse = coarse[tuple(idx)]
    glon, glat = grid.lonlat()
    rows = (glat - lat[0]) / (lat[1] - lat[0])
    cols = (glon - lon[0]) / (lon[1] - lon[0])
    return ndimage.map_coordinates(coarse, [rows, cols], order=1, mode="nearest")


def fill_gaps(theta_lst, theta_skin, months):
    """Fill cloud gaps in land surface temperature (arrays are days x ny x nx).
    Under cloud, use ERA5-Land skin temperature plus the mean amount by which that cell's
    clear-sky MODIS reading exceeded it in the same month. Cells never seen clear in a
    month borrow the offset of the nearest cell that was."""
    out = np.empty_like(theta_skin)
    for m in np.unique(months):
        sel = np.flatnonzero(months == m)
        diff = theta_lst[sel] - theta_skin[sel]
        seen = np.isfinite(diff)
        count = seen.sum(0)
        offset = np.where(count > 0, np.nansum(np.where(seen, diff, 0.0), 0) / np.maximum(count, 1), np.nan)
        if np.isnan(offset).all():
            offset[:] = 0.0
        elif np.isnan(offset).any():
            unseen = np.isnan(offset)
            idx = ndimage.distance_transform_edt(unseen, return_distances=False, return_indices=True)
            nearest = offset[tuple(idx)]
            offset = np.where(unseen, ndimage.gaussian_filter(nearest, 8.0, mode="nearest"), offset)
        out[sel] = np.where(seen, theta_lst[sel], theta_skin[sel] + offset)
    return out


class Dataset:
    """Static fields, stations and daily fields on the 1 km grid, with their provenance."""

    def __init__(self, rebuild=False, verbose=True):
        self.grid = Grid()
        rows = []

        path, tier = _first("static_1km.npz")
        s = np.load(path)
        self.elev = s["elevation_km"].astype(float)
        self.frac = s["fractions"].astype(float)
        self.sea = s["sea"]
        rows.append(("Elevation (1 km)", tier, "Copernicus GLO-30 via Earth Engine" if tier == "real" else
                     "5 arc-minute Mapzen/SRTM grid (pvlib) interpolated, plus synthetic sub-grid roughness"))
        rows.append(("Land cover fractions", tier, "ESA WorldCover 2021" if tier == "real" else
                     "Natural Earth coastline, lakes and urban outlines; vegetation zoned by elevation"))
        self.features = static_features(self.grid, self.elev, self.frac, self.sea)

        path, tier_st = _first("stations.csv")
        st = pd.read_csv(path, dtype={"id": str})
        st["iy"], st["ix"], inside = self.grid.cell(st.longitude, st.latitude)
        st = st[inside].reset_index(drop=True)
        st["z_km"] = st.elevation / 1000.0
        st["x"], st["y"] = lonlat_to_xy(st.longitude, st.latitude)
        rows.append(("Station locations and elevations", "real", "Meteostat station list (CC BY 4.0)"))

        real_daily = all(os.path.exists(os.path.join(REAL, f)) for f in ("era5land.npz", "lst_1km.npy", "station_tmax.csv"))
        self.placeholder = not real_daily
        if real_daily:
            raw = self._read_real(st)
            st = raw.pop("stations")
            rows += [("Station daily Tmax", "real", "Meteostat daily"), ("ERA5-Land Tmax, skin temperature, wind", "real", "Copernicus CDS"),
                     ("MODIS land surface temperature", "real", "MYD11A1 via Earth Engine"),
                     ("E-OBS daily tx", "real" if raw["eobs"] is not None else "missing", "Copernicus CDS")]
        else:
            rows += [("Station daily Tmax", "synthetic", "placeholder.py"), ("ERA5-Land Tmax, skin temperature, wind", "synthetic", "placeholder.py"),
                     ("MODIS land surface temperature", "synthetic", "placeholder.py"), ("E-OBS daily tx", "synthetic", "placeholder.py")]
            raw = None
        self.stations = self._merge_shared_cells(st)
        self.cells = self.grid.flat(self.stations.iy, self.stations.ix)
        self.provenance = pd.DataFrame(rows, columns=["dataset", "tier", "source"])
        self._build(raw, rebuild, verbose)

    #-------- stations sharing a cell are averaged --------------------------------------
    def _merge_shared_cells(self, st):
        st = st.copy()
        st["cell"] = self.grid.flat(st.iy, st.ix)
        self._members = [g.index.values for _, g in st.groupby("cell", sort=False)]
        merged = st.groupby("cell", sort=False).agg(
            id=("id", "first"), name=("name", "first"), latitude=("latitude", "mean"), longitude=("longitude", "mean"),
            elevation=("elevation", "mean"), iy=("iy", "first"), ix=("ix", "first"), z_km=("z_km", "mean"),
            x=("x", "mean"), y=("y", "mean")).reset_index(drop=True)
        return merged

    def _read_real(self, st):
        era = dict(np.load(os.path.join(REAL, "era5land.npz")))
        tm = pd.read_csv(os.path.join(REAL, "station_tmax.csv"), index_col=0, parse_dates=True)
        days = tm.index
        covered = tm.notna().mean() >= 0.8                                    # keep stations with >= 80% coverage
        st = st[st.id.isin(covered.index[covered])].reset_index(drop=True)
        eobs_path = os.path.join(REAL, "eobs.npz")
        return dict(days=days, era5=era, lst=np.load(os.path.join(REAL, "lst_1km.npy"), mmap_mode="r"),
                    eobs=dict(np.load(eobs_path)) if os.path.exists(eobs_path) else None,
                    tmax=tm[st.id].values, stations=st, drivers=None)

    #-------- Step 3: daily fields ------------------------------------------------------------
    def _build(self, raw, rebuild, verbose):
        os.makedirs(CACHE, exist_ok=True)
        tag = "real" if not self.placeholder else "placeholder"
        meta = os.path.join(CACHE, f"{tag}_meta.npz")
        paths = {k: os.path.join(CACHE, f"{tag}_{k}.npy") for k in DAILY}
        if rebuild or not os.path.exists(meta):
            g = self.grid
            if raw is None:
                if verbose:
                    print("building placeholder daily fields (about two minutes, cached afterwards)")
                days = placeholder.summer_days()
                truth = np.lib.format.open_memmap(paths["truth"], "w+", np.float32, (len(days),) + g.shape)
                raw = placeholder.generate(g, self.elev, self.frac, self.sea, self.features, self.stations, days, keep_truth=truth)
                truth.flush()
                merged_tmax = raw["tmax"]
            else:
                days = raw["days"]
                merged_tmax = np.column_stack([np.nanmean(raw["tmax"][:, m], axis=1) for m in self._members])
            era, n = raw["era5"], len(days)
            z_E = era["orography_m"] / 1000.0
            out = {k: np.lib.format.open_memmap(paths[k], "w+", np.float32, (n,) + g.shape) for k in DAILY if k != "truth"}
            skin = np.empty((n,) + g.shape, np.float32)
            for d in range(n):
                out["theta_E"][d] = to_grid(era["tmax"][d] + GAMMA * z_E, era["lat"], era["lon"], g)
                skin[d] = to_grid(era["skin"][d] + GAMMA * z_E, era["lat"], era["lon"], g)
                out["u"][d] = to_grid(era["u"][d], era["lat"], era["lon"], g) * 3.6
                out["v"][d] = to_grid(era["v"][d], era["lat"], era["lon"], g) * 3.6
                if raw["eobs"] is not None:
                    e = raw["eobs"]
                    out["theta_eobs"][d] = to_grid(e["tx"][d] + GAMMA * e["elevation_m"] / 1000.0, e["lat"], e["lon"], g)
            months = days.year.values * 100 + days.month.values
            theta_lst = np.asarray(raw["lst"], np.float32) + (GAMMA * self.elev).astype(np.float32)
            out["theta_s"][:] = fill_gaps(theta_lst, skin, months)
            cloud = 1.0 - np.isfinite(theta_lst[:, ~self.sea]).mean(1)
            for a in out.values():
                a.flush()
            y = merged_tmax + GAMMA * self.stations.z_km.values[None, :]
            np.savez(meta, days=days.values.astype("datetime64[D]"), y=y, cloud=cloud, has_eobs=raw["eobs"] is not None)
            if raw.get("drivers") is not None:
                raw["drivers"].to_csv(os.path.join(CACHE, "placeholder_drivers.csv"))
        m = np.load(meta)
        self.days = pd.DatetimeIndex(m["days"])
        self.y = m["y"]                                   # days x stations, sea-level-reduced, NaN where missing
        self.cloud = m["cloud"]                           # share of land under cloud in the MODIS overpass
        self.has_eobs = bool(m["has_eobs"])
        self._maps = {k: np.load(p, mmap_mode="r") for k, p in paths.items() if os.path.exists(p)}
        dpath = os.path.join(CACHE, "placeholder_drivers.csv")
        self.drivers = pd.read_csv(dpath, index_col=0, parse_dates=True) if self.placeholder and os.path.exists(dpath) else None

    def day(self, d):
        """The fields the PDE needs on day d (index into self.days)."""
        return {k: np.asarray(self._maps[k][d], float) for k in ("theta_E", "theta_s", "u", "v")}

    def field(self, name, d):
        return np.asarray(self._maps[name][d], float)

    @property
    def tmax(self):
        """Station Tmax as measured (degrees C at the station's own elevation)."""
        return self.y - GAMMA * self.stations.z_km.values[None, :]

    def index(self, date):
        return int(np.flatnonzero(self.days == pd.Timestamp(date))[0])

    def years(self, *years):
        return np.flatnonzero(np.isin(self.days.year, years))
