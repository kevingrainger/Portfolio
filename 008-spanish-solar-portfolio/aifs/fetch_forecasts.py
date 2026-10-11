#!/usr/bin/env python3
#-------- fetch_forecasts.py ------------------------------------------------------------
#-----------------------------------------------------------------------------
# One-off download of past AIFS and IFS forecasts of solar radiation and cloud cover,
# kept only at the 25 Spanish solar sites. Run it once, on your own computer.
#
# Where the data comes from: ECMWF open data, mirrored on AWS (bucket ecmwf-forecasts,
# no account needed, CC BY 4.0). The forecasts are stored as global GRIB files, so
# no server will cut out a region for you. What this script does instead:
#   1. asks only for the fields it needs (two parameters, ten lead times); the
#      downloader uses each file's index to fetch just those fields, not the whole file
#   2. reads the global field, keeps the 25 site values (and an Iberia box if asked)
#   3. deletes the GRIB file
# So the download is a few MB per field, and what stays on disk is tiny.
#
#   pip install ecmwf-opendata xarray cfgrib
#   python aifs/fetch_forecasts.py 2025-06-01 2025-06-02          # try two days first
#   python aifs/fetch_forecasts.py 2025-06-01 2025-08-31
#
# Output: aifs/forecasts/{model}_{YYYYMMDD}.npz with ssrd (J m-2, accumulated since
# the start of the run) and tcc (0-1) at each site, for lead times 24, 48, ... 240 h.
# Daily solar energy for day n is ssrd(24 n) - ssrd(24 (n - 1)).
# Each run is checked for before downloading, so the script can be stopped and
# restarted without fetching anything twice.

import datetime
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from solar_sites import SITES                                    # noqa: E402

MODELS = ["aifs-single", "ifs"]
PARAMS = ["ssrd", "tcc"]
STEPS = list(range(24, 241, 24))                                 # 1 to 10 days ahead
OUT = os.path.join(HERE, "forecasts")
TMP = os.path.join(HERE, "_tmp.grib2")


def site_values(path, param):
    """Values of one parameter at every site and lead time, from a downloaded GRIB file."""
    import xarray as xr
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=dict(indexpath=""))
    da = ds[list(ds.data_vars)[0]]
    lon = da.longitude.values
    out = []
    for lat, lo in SITES.values():
        lo = lo % 360 if lon.max() > 180 else lo                  # grids run 0-360 or -180-180
        out.append(da.sel(latitude=lat, longitude=lo, method="nearest").values)
    ds.close()
    return np.array(out).T, da.step.values                         # (steps, sites)


def fetch_day(client, model, date):
    path = os.path.join(OUT, f"{model}_{date:%Y%m%d}.npz")
    if os.path.exists(path):
        return 0
    got, nbytes = {}, 0
    for p in PARAMS:
        client.retrieve(date=date, time=0, step=STEPS, param=p, type="fc", target=TMP)
        nbytes += os.path.getsize(TMP)
        got[p], steps = site_values(TMP, p)
        os.remove(TMP)
    np.savez(path, sites=np.array(list(SITES)), lead_hours=np.array(STEPS), **got)
    return nbytes


def main():
    from ecmwf.opendata import Client
    start, end = (datetime.date.fromisoformat(a) for a in sys.argv[1:3])
    os.makedirs(OUT, exist_ok=True)
    clients = {m: Client(source="aws", model=m) for m in MODELS}
    total, d = 0, start
    while d <= end:
        for m in MODELS:
            t0 = time.time()
            try:
                n = fetch_day(clients[m], m, d)
            except Exception as e:                                # a missing run is skipped, not fatal
                print(f"{m} {d}: not available ({e})")
                continue
            total += n
            if n:
                print(f"{m} {d}: {n / 1e6:.1f} MB downloaded in {time.time() - t0:.0f}s (total {total / 1e9:.2f} GB)")
        d += datetime.timedelta(days=1)


if __name__ == "__main__":
    main()
