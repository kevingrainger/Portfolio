#!/usr/bin/env python3
#-------- fetch_seas5_italy.py -------------------------------------------------------
# ECMWF SEAS5 seasonal forecasts of monthly mean 2 m temperature over northern Italy
# (the Italy North bidding zone), issued every month from 1993, lead months 1-3, from the
# Copernicus CDS ("seasonal-monthly-single-levels", system 51). Batched by period: the
# CDS queue, not the size, is the slow part. 2026 is fetched month by month so far.
#
#   python fetch_seas5_italy.py
#
# Output: data/seas5/italy_<years>.grib

import os
import time

import cdsapi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "seas5")
BOX = [46.5, 7.0, 44.0, 13.0]                     # N, W, S, E: Po valley and the northern cities


def get(c, name, years, months):
    path = os.path.join(OUT, f"italy_{name}.grib")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    for attempt in range(3):
        try:
            c.retrieve("seasonal-monthly-single-levels", {
                "originating_centre": "ecmwf", "system": "51", "variable": ["2m_temperature"],
                "product_type": ["monthly_mean"], "year": [str(y) for y in years],
                "month": [f"{m:02d}" for m in months], "leadtime_month": ["1", "2", "3"],
                "data_format": "grib", "area": BOX}, path)
            print(name, "ok", flush=True)
            return
        except Exception as e:
            print(name, "retry", e, flush=True)
            time.sleep(30)


def main():
    os.makedirs(OUT, exist_ok=True)
    c = cdsapi.Client(quiet=True)
    get(c, "1993-2016", range(1993, 2017), range(1, 13))
    for y in range(2017, 2026):                    # year by year: mixed forecast years failed in one request
        get(c, str(y), [y], range(1, 13))
    get(c, "2026", [2026], range(1, int(time.strftime("%m")) + 1))


if __name__ == "__main__":
    main()
