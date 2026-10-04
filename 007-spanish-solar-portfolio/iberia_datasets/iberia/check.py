#-------- iberia/check.py ---------------------------------------------------------
# Run every quality check on a built dataset and exit non-zero if any fail, so a
# Slurm job or CI step stops on bad data instead of passing it downstream.
#
#   python -m iberia.check data/iberia-era5.zarr

import sys

from . import quality as q


def main(path):
    import glob
    import os
    results = q.run_all(*q.from_anemoi(path))
    print(q.report(results))
    era5_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "era5")
    placeholders = sorted(glob.glob(os.path.join(era5_dir, "*.synthetic")))
    if placeholders:
        print(f"\nWARNING: {len(placeholders)} month(s) of the source data are SYNTHETIC placeholders "
              f"(e.g. {os.path.basename(placeholders[0])}). Results from this dataset test the "
              f"method only. Replace with: python download_era5.py")
    return 0 if all(ok for ok, _ in results.values()) else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m iberia.check DATASET.zarr")
    sys.exit(main(sys.argv[1]))
