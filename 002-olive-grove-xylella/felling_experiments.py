#!/usr/bin/env python3
#-------- felling_experiments.py ----------------------------------------------------
#-----------------------------------------------------------------------------
# The simulations behind Part 2, saved to results/ so the notebook only has to read
# them. About 15 minutes on six cores.
#
#   unmanaged      16 runs of an unmanaged 180 m grove, transmission logs kept - the
#                  data the kernel and the prediction are measured from
#   felling grid   felling radius x detection speed x 6 seeds, 5 years each

import os
import pickle
import time

import numpy as np

from olive_model import run_ensemble

GROVE = dict(width=180, height=180, sampling_rate=0.001, time_steps_per_day=30)
YEARS = 5
RADII = [0, 5, 10, 15, 20, 30, 50]                     # metres
DETECTION = {"symptoms (~2 years)": None, "lab test after 1 year": 365, "lab test after 4 months": 120}
SEEDS = range(6)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    t = time.time()
    unmanaged = run_ensemble(365 * YEARS, range(100, 116), **GROVE)
    with open("results/unmanaged_big_grove.pkl", "wb") as f:
        pickle.dump(unmanaged, f)
    print(f"unmanaged done in {time.time() - t:.0f}s", flush=True)

    grid = {}
    for label, delay in DETECTION.items():
        for r in RADII:
            runs = run_ensemble(365 * YEARS, SEEDS, felling_radius=r, detection_delay=delay, **GROVE)
            grid[(label, r)] = [dict(ever_infected=x["ever_infected"], felled=x["F"][-1],
                                     healthy=x["S"][-1], S=x["S"], F=x["F"], day=x["day"]) for x in runs]
            print(f"{label:24s} r={r:3d} m: healthy {np.median([g['healthy'] for g in grid[(label, r)]]):.0%}"
                  f"  ({time.time() - t:.0f}s)", flush=True)
    with open("results/felling_grid.pkl", "wb") as f:
        pickle.dump(dict(grid=grid, radii=RADII, detection=DETECTION, years=YEARS), f)
    print("done")
