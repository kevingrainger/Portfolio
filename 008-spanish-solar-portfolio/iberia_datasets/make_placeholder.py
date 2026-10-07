#!/usr/bin/env python3
#-------- make_placeholder.py ------------------------------------------------------
#-----------------------------------------------------------------------------
# Synthetic stand-ins for the ERA5 downloads, so everything downstream can be built
# and tested before the Copernicus API key is set up.
#
# Writes data/era5/iberia-era5-YYYY-MM.grib - exactly the files download_era5.py
# writes, encoded exactly as real ERA5 is - plus an iberia-era5-YYYY-MM.synthetic
# marker beside each. download_era5.py replaces any month that has a marker, and
# `python -m iberia.check` warns while any remain. So the real data drops in by
# running the downloader; nothing else changes.
#
# The weather in the placeholder has planted structure with known values (see
# iberia/synthetic.py): anything computed from it is a test of the method, not a
# finding about Spain.
#
#   python make_placeholder.py 2023-01 2023-12

import argparse
import datetime as dt
import os

from download_era5 import months_between, OUT_DIR
from iberia.synthetic import write_synthetic_era5


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first")
    parser.add_argument("last")
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    state = None
    for i, (year, month) in enumerate(months_between(args.first, args.last)):
        target = os.path.join(OUT_DIR, f"iberia-era5-{year:04d}-{month:02d}.grib")
        marker = target.replace(".grib", ".synthetic")
        if os.path.exists(target) and not os.path.exists(marker):
            print(f"real data present, leaving it alone: {target}")
            state = None
            continue
        start = dt.datetime(year, month, 1)
        end = dt.datetime(year + (month == 12), month % 12 + 1, 1)
        hours = int((end - start).total_seconds() // 3600)
        state = write_synthetic_era5(target, start, hours, seed=args.seed + i, state=state)
        with open(marker, "w") as f:
            f.write("synthetic placeholder written by make_placeholder.py - replace with download_era5.py\n")
        print(f"placeholder: {target}")


if __name__ == "__main__":
    main()
