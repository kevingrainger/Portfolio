#!/usr/bin/env python3
#-------- simulate_bank.py ------------------------------------------------------------
#-----------------------------------------------------------------------------
# A bank of simulated bleaching events for simulation-based inference.
#
# Four parameters are drawn from broad priors; for each draw the reef is run, free
# (no nudge), through the same 151-day Scott Reef temperature record as the
# calibration, and the bleached fraction is summarised in 10-day blocks.
#
#   theta = (J, s, T0, beta)
#     J     coupling_strength  how strongly a patch follows its neighbours
#     s     temp_sensitivity   heat stress per degree above the threshold (the field)
#     T0    bleaching_thresh   temperature at which stress begins (deg C)
#     beta  inverse noise      how deterministic each Metropolis decision is
#
#   python inference/simulate_bank.py N_SIMS [N_WORKERS]     -> inference/bank.npz

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import coral_model as cm

NAMES = ["J", "s", "T0", "beta"]
LABELS = [r"coupling $J$", r"heat sensitivity $s$", r"threshold $T_0$ (°C)", r"inverse noise $\beta$"]
LOW = np.array([0.0, 0.2, 29.5, 0.05])
HIGH = np.array([1.5, 2.0, 31.0, 0.40])
CALIBRATED = np.array([0.5, 0.8, 30.0, 0.15])     # the hand-set values of the original model
BLOCK = 10                                         # days per summary block
GRID = 100


def record():
    r = pd.read_excel(os.path.join(os.path.dirname(HERE), "Coral temps.xlsx"))
    return r["Temp"].values, r["Smooth"].values


def summarise(bleached):
    """Bleached fraction averaged over consecutive 10-day blocks (15 numbers)."""
    n = len(bleached) // BLOCK * BLOCK
    return bleached[:n].reshape(-1, BLOCK).mean(1)


def simulate(theta, temps, seed):
    J, s, T0, beta = theta
    _, bleached, _ = cm.run_record(temps, seed=seed, grid=GRID, coupling_strength=J,
                                   temp_sensitivity=s, bleaching_thresh=T0, beta=beta)
    return summarise(bleached)


def _work(args):
    i, theta = args
    temps, _ = record()
    return i, simulate(theta, temps, seed=10_000 + i)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    rng = np.random.default_rng(2026)
    thetas = LOW + (HIGH - LOW) * rng.random((n, 4))
    out = np.zeros((n, 15))
    t0 = time.time()
    with Pool(workers) as pool:
        for k, (i, x) in enumerate(pool.imap_unordered(_work, enumerate(thetas), chunksize=20)):
            out[i] = x
            if (k + 1) % 500 == 0:
                print(f"{k + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
                np.savez(os.path.join(HERE, "bank_partial.npz"), theta=thetas, x=out, done=k + 1)
    np.savez(os.path.join(HERE, "bank.npz"), theta=thetas, x=out)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
