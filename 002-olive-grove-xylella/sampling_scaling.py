#!/usr/bin/env python3
#-------- sampling_scaling.py -------------------------------------------------------
#-----------------------------------------------------------------------------
# Letting each simulated spittlebug stand for many.
#
# mlmc.py showed the grove model hasn't converged in the number of spittlebugs simulated:
# the share of trees infected after a year rises from 1% with 405 spittlebugs to 49% with
# 12,960. A sparse run is not a cheap version of a dense one, it's a slower epidemic.
#
# The fix tried here: a run that simulates k times fewer spittlebugs than a reference
# density treats each one as a group of k. Its chance of infecting a tree on a visit, and
# of picking the bacterium up from a tree, become the chance that at least one of the k
# would: p -> 1 - (1 - p)^k. The group still moves as one, so this is an approximation.
#
# Test: with the reference set to 12,960 spittlebugs, do runs with 405 to 6,480 scaled
# spittlebugs reproduce the dense run's share infected after a year? 40 seeds per level.
#
#   python sampling_scaling.py
#
# Output: results/sampling_scaling.json, figures/fig10_sampling_scaling.png

import json
import os
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from olive_model import OliveGrove

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
BASE, REF_LEVEL, DAYS, SEEDS = 0.0005, 5, 365, 40


def run(args):
    level, seed, scaled = args
    rate = BASE * 2 ** level
    k = 2 ** (REF_LEVEL - level) if scaled else 1
    # the two per-contact chances are set to 0.5 at the reference density (the paper's 1.0
    # would leave nothing for the scaling to act on), and raised for the groups
    g = OliveGrove(seed=seed, sampling_rate=rate)
    g.tree_susceptibility = 1 - (1 - P_TREE) ** k
    g.vector_susceptibility = 1 - (1 - P_VEC) ** k
    g.run(DAYS, initial_infected=1)
    return level, seed, scaled, float(g.ever_infected()), g.n_vectors


P_TREE, P_VEC = 0.5, 0.5


def main():
    jobs = [(l, 2000 + s, sc) for sc in (False, True) for l in range(REF_LEVEL + 1) for s in range(SEEDS)
            if not (sc and l == REF_LEVEL)]
    jobs.sort(key=lambda j: -j[0])
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        res = pool.map(run, jobs)
    out = {"reference_spittlebugs": int(810000 * BASE * 2 ** REF_LEVEL), "days": DAYS, "p_tree": P_TREE, "p_vector": P_VEC}
    for sc in (False, True):
        rows = {}
        for l in range(REF_LEVEL + 1):
            q = np.array([r[3] for r in res if r[0] == l and r[2] == sc])
            if not len(q):
                q = np.array([r[3] for r in res if r[0] == l and not r[2]])          # the reference itself
            rows[l] = dict(spittlebugs=int(810000 * BASE * 2 ** l), mean=float(q.mean()), se=float(q.std(ddof=1) / np.sqrt(len(q))),
                           share_outbreaks_over_10pct=float((q > 0.1).mean()))
        out["scaled" if sc else "unscaled"] = rows
    json.dump(out, open(os.path.join(HERE, "results", "sampling_scaling.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for key, c, ls, lab in [("unscaled", NAVY, "-", "each simulated spittlebug is one spittlebug"),
                            ("scaled", RED, "-", f"each stands for a group, scaled to {out['reference_spittlebugs']:,}")]:
        r = out[key]; x = [r[l]["spittlebugs"] for l in r]; m = np.array([100 * r[l]["mean"] for l in r]); s = np.array([100 * r[l]["se"] for l in r])
        ax.fill_between(x, m - 1.645 * s, m + 1.645 * s, color=c, alpha=0.15, lw=0)
        ax.plot(x, m, "-o", color=c, lw=1.7, ms=5, label=lab)
    ref = out["unscaled"][REF_LEVEL]
    ax.axhline(100 * ref["mean"], color=GREY, lw=0.9, ls="--")
    ax.text(420, 100 * ref["mean"] + 1, f"dense run, {out['reference_spittlebugs']:,} spittlebugs", color=GREY, fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("spittlebugs simulated"); ax.set_ylabel("trees infected after a year (%)")
    ax.set_title("Share infected after a year against spittlebugs simulated", loc="left"); ax.legend(frameon=False, fontsize=9, loc="center left")
    fig.savefig(os.path.join(HERE, "figures", "fig10_sampling_scaling.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
