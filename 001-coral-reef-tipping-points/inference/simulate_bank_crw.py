#!/usr/bin/env python3
#-------- simulate_bank_crw.py ---------------------------------------------------------
#-----------------------------------------------------------------------------
# Simulation bank for the CRW-driven reef (coral_crw.py), fitted to the real April 2016
# Scott Reef surveys.
#
# Forcing: NOAA Coral Reef Watch SST for the Scott-Ashmore virtual station, 1 October
# 2015 to 16 April 2016 (the middle of the AIMS survey week). Each simulated reef is
# "surveyed" the way the real one was: 65 sites, each scored into the bleaching
# categories the database uses (< 30%, 30-60%, 60-90%, 91-100% of living coral).
#
#   theta = (mu_D, sigma_D, s, beta)
#     mu_D     mean bleaching tolerance of a colony (DHW)
#     sigma_D  spread of tolerance between colonies (DHW)
#     s        how sharply stress beyond tolerance drives bleaching
#     beta     inverse noise of the Metropolis decisions
#   coupling J is held at 0.5: one survey cannot also separate neighbour effects
#
#   python inference/simulate_bank_crw.py N_SIMS [N_WORKERS]   -> inference/bank_crw.npz

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from coral_crw import CRWReef

NAMES = ["mu_D", "sigma_D", "s", "beta"]
LABELS = [r"mean tolerance $\mu_D$ (DHW)", r"tolerance spread $\sigma_D$ (DHW)", r"stress sensitivity $s$", r"inverse noise $\beta$"]
LOW = np.array([2.0, 0.25, 0.1, 0.05])
HIGH = np.array([16.0, 5.0, 1.5, 0.5])
N_SITES = 65
BINS = [-1, 30, 60, 90.5, 101]
SURVEY_DAY = "2016-04-16"


def forcing():
    c = pd.read_csv(os.path.join(ROOT, "data", "crw_scott_ashmore_2015-2016.csv"), comment="#", parse_dates=["date"])
    return c.date.values, c.sst_90th_hs.values


def observed():
    d = pd.read_csv(os.path.join(ROOT, "data", "gcbd_scott_reef_2016.csv"))
    counts = np.zeros(4)
    for lo in d.bleached_min:
        counts[{0: 0, 30: 1, 60: 2, 91: 3}[int(lo)]] += 1
    return counts


def simulate(theta, seed, keep=False):
    mu, sd, s, b = theta
    dates, T = forcing()
    day = int(np.flatnonzero(dates == np.datetime64(SURVEY_DAY))[0])
    r = CRWReef(mu_D=mu, sigma_D=sd, s=s, beta=b, seed=seed)
    r.mark_start()
    path, counts = [], None
    for d in range(len(T) if keep else day + 1):           # with keep, carry on to the end of July
        r.step(T[d])
        if keep:
            path.append((r.bleached_of_living(), r.cover()))
        if d == day:
            sites = r.survey(N_SITES, np.random.default_rng(seed + 1))
            counts = np.histogram(sites * 100, bins=BINS)[0].astype(float)
    return (counts, np.array(path), r) if keep else counts


def _work(args):
    i, theta = args
    return i, simulate(theta, seed=20_000 + i)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    rng = np.random.default_rng(2016)
    thetas = LOW + (HIGH - LOW) * rng.random((n, 4))
    path = os.path.join(HERE, "bank_crw.npz")
    out, done = np.zeros((n, 4)), np.zeros(n, bool)
    if os.path.exists(path):                                # resume after an interruption
        z = np.load(path)
        out[:len(z["x"])], done[:len(z["done"])] = z["x"], z["done"]
    t0 = time.time()
    todo = [(i, thetas[i]) for i in range(n) if not done[i]]
    with Pool(workers) as pool:
        for k, (i, x) in enumerate(pool.imap_unordered(_work, todo, chunksize=10)):
            out[i], done[i] = x, True
            if (k + 1) % 250 == 0 or k + 1 == len(todo):
                np.savez(path, theta=thetas, x=out, done=done)
                print(f"{done.sum()}/{n}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
