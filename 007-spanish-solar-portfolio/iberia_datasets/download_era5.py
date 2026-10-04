#!/usr/bin/env python3
#-------- download_era5.py ------------------------------------------------------
#-----------------------------------------------------------------------------
# Stage 1 of the pipeline: fetch ERA5 single-level fields for Iberia from the
# Copernicus Climate Data Store, one GRIB file per month, into data/era5/.
#
# One file per month because that is the natural unit of a CDS request, and it
# makes the download embarrassingly parallel: on a cluster, one Slurm array task
# per month (slurm/download_era5.sbatch).
#
# Needs a CDS account with the ERA5 licence accepted, and ~/.cdsapirc holding the
# API key - see README. Files already present are skipped, so an interrupted run
# can simply be restarted.
#
#   python download_era5.py 2023-06                 one month
#   python download_era5.py 2022-01 2023-12         a range
#   python download_era5.py --index 7 2022-01 2023-12   the 7th month of a range (Slurm arrays)

import argparse
import calendar
import os
import sys

from iberia.synthetic import AREA

VARIABLES = [
    "2m_temperature",
    "total_cloud_cover",
    "surface_solar_radiation_downwards",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "era5")


def months_between(first, last):
    y, m = map(int, first.split("-"))
    y2, m2 = map(int, last.split("-"))
    while (y, m) <= (y2, m2):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def download_month(year, month, out_dir=OUT_DIR):
    import cdsapi

    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, f"iberia-era5-{year:04d}-{month:02d}.grib")
    marker = target.replace(".grib", ".synthetic")
    if os.path.exists(target) and os.path.getsize(target) > 0 and not os.path.exists(marker):
        print(f"exists, skipping: {target}")
        return target
    if os.path.exists(marker):
        print(f"replacing synthetic placeholder: {target}")

    days = calendar.monthrange(year, month)[1]
    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, days + 1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": list(AREA),                     # north, west, south, east
        "data_format": "grib",
        "download_format": "unarchived",
    }
    partial = target + ".part"                  # never leave a half-written file under the real name
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", request, partial)
    os.replace(partial, target)
    if os.path.exists(marker):
        os.remove(marker)                       # real data now - no longer a placeholder
    print(f"downloaded: {target}")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first", help="first month, YYYY-MM")
    parser.add_argument("last", nargs="?", help="last month, YYYY-MM (default: same as first)")
    parser.add_argument("--index", type=int, help="download only the Nth month of the range (0-based)")
    args = parser.parse_args(argv)

    months = list(months_between(args.first, args.last or args.first))
    if args.index is not None:
        if not 0 <= args.index < len(months):
            sys.exit(f"index {args.index} outside the {len(months)} months in the range")
        months = [months[args.index]]
    for year, month in months:
        download_month(year, month)


if __name__ == "__main__":
    main()
