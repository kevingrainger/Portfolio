#!/usr/bin/env python3
#-------- make_ranking_figure.py ------------------------------------------------------
#-----------------------------------------------------------------------------
# Each hedge's 2010 outcome against its expected shortfall seen on 3 May, for the
# producer (revenue) and the miller (cost). Drawn from results/wheat_2010.json
# (run wheat_2010.py first).
#
# Output: figures/fig9_hindsight_vs_may.png

import json
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figstyle                                                  # noqa: E402

o = json.load(open(os.path.join(HERE, "results", "wheat_2010.json")))
b = o["buyer"]
NAVY, RED = "#0b2545", "#c0392b"

figstyle.apply()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(wspace=0.3))
panels = [(axes[0], o, "cvar5", "Producer revenue", "worst-5% average expected on 3 May, € per tonne",
           "revenue in 2010, € per tonne", NAVY),
          (axes[1], b, "cvar95", "Miller cost", "worst-5% average expected on 3 May, € per tonne",
           "cost in 2010, € per tonne", RED)]
for ax, d, key, title, xl, yl, c in panels:
    for name, real in d["realised"].items():
        x = d["ex_ante"][name][key]
        ax.plot(x, real, "o", color=c, ms=7, mec="white", mew=1)
        ax.annotate(name, (x, real), xytext=(6, -11) if name in ("call spread", "layered") else (6, 3), textcoords="offset points", fontsize=8.5)
    ax.set_xlabel(xl, fontsize=9.5); ax.set_ylabel(yl)
    ax.set_title(title, loc="left")
    ax.margins(x=0.18, y=0.1)
figstyle.save(fig, os.path.join(HERE, "figures", "fig9_hindsight_vs_may"), formats=("png",))
