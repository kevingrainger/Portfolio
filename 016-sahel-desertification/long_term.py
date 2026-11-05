#!/usr/bin/env python3
#-------- long_term.py --------------------------------------------------------------
#-----------------------------------------------------------------------------
# Does the lattice track long-term change, not only tipping points? At each site the
# model's vegetation cover (CHIRPS-driven, mean of 8 seeds, degradation rate 0.2) and
# MODIS peak greenness are both indexed to their 2000-2004 mean, and their trends over
# 2000-2025 compared (Theil-Sen slope, robust to single bad years), with a bootstrap of
# years for the uncertainty.
#
#   python long_term.py
#
# Output: results/long_term.json, figures/fig8_long_term.png

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import theilslopes

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
rng = np.random.default_rng(0)


def main():
    m = pd.read_csv(os.path.join(HERE, "results", "sites_chirps.csv"))
    m = m[m.d == 0.2]
    n = pd.read_csv(os.path.join(HERE, "results", "ndvi_annual.csv"))
    out = {}
    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, site in zip(axes, ["gourma", "fakara"]):
        mod = m[m.site == site].set_index("year").cover.loc[2000:2025]
        sat = n[n.site == site].set_index("year").ndvi_peak.loc[2000:2025]
        mi, si = 100 * mod / mod.loc[2000:2004].mean(), 100 * sat / sat.loc[2000:2004].mean()
        res = {}
        for name, s in [("model", mi), ("MODIS", si)]:
            slope = theilslopes(s.values, s.index.values)[0] * 10
            boot = [theilslopes(*(lambda i: (s.values[i], s.index.values[i]))(rng.integers(0, len(s), len(s))))[0] * 10
                    for _ in range(500)]
            res[name] = dict(change_per_decade_pct=float(slope), ci90=[float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))])
        out[site] = res
        ax.plot(mi.index, mi, color=NAVY, lw=1.8, label=f"model cover ({res['model']['change_per_decade_pct']:+.1f}% a decade)")
        ax.plot(si.index, si, color=RED, lw=1.4, marker="o", ms=3, label=f"MODIS peak greenness ({res['MODIS']['change_per_decade_pct']:+.1f}% a decade)")
        ax.axhline(100, color="0.8", lw=0.8)
        ax.set_title(site.capitalize(), loc="left"); ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.set_xlabel("year")
    axes[0].set_ylabel("index, 2000-2004 mean = 100")
    fig.savefig(os.path.join(HERE, "figures", "fig8_long_term.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    json.dump(out, open(os.path.join(HERE, "results", "long_term.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
