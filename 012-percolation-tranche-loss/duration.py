#!/usr/bin/env python3
#-------- duration.py ---------------------------------------------------------------
#-----------------------------------------------------------------------------
# How long disruption lasts. A low-water episode of four months with the Kaub gauge at
# 55 cm, run through the three models of three_models.py. Each month, new failures arrive
# as in that model; a firm already disrupted stays down the next month with probability
# 0.5, the time it takes to rebuild stocks (an assumption, not a fitted rate). In the
# correlated model the common shock carries over from month to month (an AR(1) with
# coefficient 0.5), since the same low water persists.
#
# Outputs per model: German output-months lost over the episode (mean and worst 1%), and
# the share of firms out for two months or more.
#
#   python duration.py
#
# Output: results/duration.json, figures/fig9_duration.png

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
from simulate_rhine import ECON, load_calib, pick_three   # noqa: E402

RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
MONTHS, STAY, RHO, PHI, EPISODES, KAUB = 4, 0.5, 0.2, 0.5, 600, 55.0


def main():
    pick = pick_three(load_calib())[1]
    sev, T = pick["severity"], pick["T_star"]
    rng = np.random.default_rng(0)
    p = np.zeros(ECON.N)
    for _ in range(600):
        p += ECON.disrupted(KAUB, T, rng, sev)
    p = np.clip(p / 600, 1e-9, 1 - 1e-9); thr = norm.ppf(p)
    g, w = ECON.germany, ECON.output
    res = {}
    for model in ["independent", "correlated", "percolation"]:
        lost, long_out = np.empty(EPISODES), np.empty(EPISODES)
        for e in range(EPISODES):
            down = np.zeros(ECON.N, bool); months_down = np.zeros(ECON.N, int); z = 0.0; total = 0.0
            for m in range(MONTHS):
                if model == "independent":
                    new = rng.random(ECON.N) < p
                elif model == "correlated":
                    z = PHI * z + np.sqrt(1 - PHI ** 2) * rng.standard_normal()
                    new = np.sqrt(RHO) * z + np.sqrt(1 - RHO) * rng.standard_normal(ECON.N) < thr
                else:
                    new = ECON.disrupted(KAUB, T, rng, sev)
                down = new | (down & (rng.random(ECON.N) < STAY))
                months_down += down
                total += w[down & g].sum() / w[g].sum()
            lost[e] = total
            long_out[e] = (months_down[g] >= 2).mean()
        res[model] = dict(mean_output_months=float(lost.mean()), p99_output_months=float(np.quantile(lost, 0.99)),
                          share_out_2_months=float(long_out.mean()), dist=lost.tolist())
    out = {k: {a: b for a, b in v.items() if a != "dist"} for k, v in res.items()}
    out.update(months=MONTHS, stay=STAY, kaub_cm=KAUB)
    json.dump(out, open(os.path.join(RES, "duration.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 4.4))
    bins = np.linspace(0, max(np.quantile(res[k]["dist"], 0.995) for k in res) * 100, 45)
    for k, c in [("independent", LIGHT), ("correlated", BLUE), ("percolation", RED)]:
        ax.hist(np.array(res[k]["dist"]) * 100, bins=bins, color=c, alpha=0.6, density=True,
                label=f"{k}: mean {100 * res[k]['mean_output_months']:.1f}, worst 1% {100 * res[k]['p99_output_months']:.1f}")
    ax.set_yscale("log"); ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("German industrial output lost over a four-month low-water episode (% of one month's output)")
    ax.set_title("Four months of low water, three models", loc="left")
    fig.savefig(os.path.join(FIG, "fig9_duration.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
