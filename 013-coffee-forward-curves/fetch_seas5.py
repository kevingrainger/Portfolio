#!/usr/bin/env python3
#-------- fetch_seas5.py ------------------------------------------------------------
#-----------------------------------------------------------------------------
# ECMWF SEAS5 seasonal forecasts (Copernicus CDS, "seasonal-monthly-single-levels",
# system 51) of monthly mean rain and 2 m temperature over the two coffee origins,
# issued every month from 1993, for lead months 1 to 3. Hindcasts (1993-2016) have 25
# members, real-time forecasts (2017 on) 51; only the ensemble mean is kept.
#
#   python fetch_seas5.py           # resumable: one GRIB per origin and year
#
# Needs a CDS account (~/.cdsapirc) with the SEAS5 licence accepted.
# Output: data/seas5/<origin>_<year>.grib, data/seas5_monthly.csv

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "seas5")

# boxes (N, W, S, E) over the producing areas used for the ERA5 weather
BOXES = {"brazil": [-18.5, -47.5, -22.5, -44.0],      # Sul de Minas, Cerrado, Mogiana
         "vietnam": [14.0, 107.5, 11.0, 109.0]}       # Dak Lak, Lam Dong


def fetch(origin, year, client):
    path = os.path.join(OUT, f"{origin}_{year}.grib")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    months = [f"{m:02d}" for m in range(1, 13)] if year < int(time.strftime("%Y")) else \
        [f"{m:02d}" for m in range(1, int(time.strftime("%m")) + 1)]
    client.retrieve("seasonal-monthly-single-levels", {
        "originating_centre": "ecmwf", "system": "51",
        "variable": ["total_precipitation", "2m_temperature"],
        "product_type": ["monthly_mean"], "year": [str(year)], "month": months,
        "leadtime_month": ["1", "2", "3"], "data_format": "grib", "area": BOXES[origin]}, path)
    return path


def to_table():
    import xarray as xr
    rows = []
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".grib"):
            continue
        origin = f.split("_")[0]
        for var, short in [("tprate", "rain"), ("t2m", "t2m")]:
            try:
                ds = xr.open_dataset(os.path.join(OUT, f), engine="cfgrib",
                                     backend_kwargs={"filter_by_keys": {"shortName": short if short == "t2m" else "tprate"},
                                                     "indexpath": ""})
            except Exception:
                continue
            v = ds[list(ds.data_vars)[0]].mean(dim=[d for d in ds.dims if d in ("number", "latitude", "longitude")])
            df = v.to_dataframe(name="value").reset_index()
            df["origin"], df["variable"] = origin, short
            rows.append(df[["time", "step", "value", "origin", "variable"]])
    out = pd.concat(rows)
    out.to_csv(os.path.join(HERE, "data", "seas5_monthly.csv"), index=False)
    return out


def main():
    import cdsapi
    os.makedirs(OUT, exist_ok=True)
    c = cdsapi.Client(quiet=True)
    years = range(1993, int(time.strftime("%Y")) + 1)
    for origin in BOXES:
        for y in years:
            for attempt in range(3):
                try:
                    fetch(origin, y, c)
                    print(origin, y, "ok", flush=True)
                    break
                except Exception as e:
                    print(origin, y, "retry", e, flush=True)
                    time.sleep(30)
    if "--table" in sys.argv or True:
        print(to_table().groupby(["origin", "variable"]).size())


if __name__ == "__main__":
    main()
