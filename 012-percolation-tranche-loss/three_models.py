#!/usr/bin/env python3
#-------- three_models.py -----------------------------------------------------------
#-----------------------------------------------------------------------------
# One trigger, three ways of modelling how firms fail together. The trigger is low water
# on the Rhine (the Kaub gauge at a given level for a month); each firm is simply disrupted
# or not for that month, and the loss is the share of German industrial output in
# disrupted firms.
#
#   independent   each firm fails on its own, with the same probability it has in the
#                 network model (so the average is identical by construction)
#   correlated    a one-factor Gaussian copula, the standard credit-risk way of making
#                 failures move together: firm i fails when sqrt(rho) Z + sqrt(1 - rho) e_i
#                 falls below N^-1(p_i), with one common shock Z per month (rho = 0.2, a
#                 typical asset correlation); same probabilities, same average
#   percolation   the supply network: barge-exposed firms fail first, and failure passes
#                 along each supplier link with probability T (the calibrated economy)
#
# All three have the same chance of each firm failing, so they agree on the expected loss
# and differ only in how the failures cluster: the tail.
#
#   python three_models.py
#
# Output: results/three_models.json, figures/fig8_three_models.png

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rhine as R                                          # noqa: E402
from simulate_rhine import ECON, load_calib, pick_three  # noqa: E402

RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
RHO, RUNS = 0.2, 2000


def losses(kaub, T, sev, rng):
    perc = np.empty(RUNS); masks = np.zeros(ECON.N)
    for r in range(RUNS):
        m = ECON.disrupted(kaub, T, rng, sev)
        perc[r] = ECON.output_loss(m); masks += m
    p = np.clip(masks / RUNS, 1e-9, 1 - 1e-9)
    ind = np.array([ECON.output_loss(rng.random(ECON.N) < p) for _ in range(RUNS)])
    thr = norm.ppf(p)
    cor = np.empty(RUNS)
    for r in range(RUNS):
        z = rng.standard_normal()
        cor[r] = ECON.output_loss(np.sqrt(RHO) * z + np.sqrt(1 - RHO) * rng.standard_normal(ECON.N) < thr)
    return dict(independent=ind, correlated=cor, percolation=perc)


def summary(x):
    return dict(mean=float(x.mean()), p95=float(np.quantile(x, 0.95)), p99=float(np.quantile(x, 0.99)),
                p_above_5pct=float((x > 0.05).mean()), p_above_20pct=float((x > 0.20).mean()))


def main():
    pick = pick_three(load_calib())[1]                     # the middle economy, ~55% of the way to its threshold
    sev, T = pick["severity"], pick["T_star"]
    rng = np.random.default_rng(0)
    out = dict(severity=sev, T=T, rho=RHO, runs=RUNS, levels={})
    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, (kaub, T_use, title) in zip(axes, [(55.0, T, "Kaub at 55 cm, network as calibrated"),
                                               (25.0, 1.25 * ECON.threshold(), "Kaub at 25 cm, road and rail strained")]):
        L = losses(kaub, T_use, sev, rng)
        out["levels"][f"{int(kaub)}cm"] = {k: summary(v) for k, v in L.items()}
        bins = np.linspace(0, max(np.quantile(L["percolation"], 0.995), 0.05), 50)
        for k, c in [("independent", LIGHT), ("correlated", BLUE), ("percolation", RED)]:
            ax.hist(L[k], bins=bins, color=c, alpha=0.55 if k != "percolation" else 0.65, density=True,
                    label=f"{k}: 99th pct {100 * np.quantile(L[k], 0.99):.1f}%")
        ax.axvline(L["percolation"].mean(), color=NAVY, lw=1, ls="--")
        ax.set_yscale("log"); ax.set_xlabel("share of German industrial output disrupted")
        ax.set_title(title, loc="left"); ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("density (log scale)")
    fig.savefig(os.path.join(FIG, "fig8_three_models.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    json.dump(out, open(os.path.join(RES, "three_models.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
