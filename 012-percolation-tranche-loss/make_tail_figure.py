#!/usr/bin/env python3
#-------- make_tail_figure.py --------------------------------------------------------
#-----------------------------------------------------------------------------
# Mean, 95th and 99th percentile of the monthly output lost under the three models,
# drawn from results/three_models.json (run three_models.py first).
#
# Output: figures/fig10_three_model_tails.png

import json
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figstyle                                                  # noqa: E402

d = json.load(open(os.path.join(HERE, "results", "three_models.json")))["levels"]
MODELS = [("independent", "independent"), ("percolation", "supply network"), ("correlated", "one-factor correlated")]
STATS = [("mean", "average", "#8a8a8a", "o"), ("p95", "worst 5%", "#1f6fb2", "s"), ("p99", "worst 1%", "#c0392b", "D")]

figstyle.apply()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
for ax, (lvl, title) in zip(axes, [("55cm", "Kaub at 55 cm, network as calibrated"), ("25cm", "Kaub at 25 cm, road and rail strained")]):
    for j, (m, lab) in enumerate(MODELS):
        vals = [100 * d[lvl][m][k] for k, *_ in STATS]
        ax.plot([min(vals), max(vals)], [j, j], color="0.8", lw=1.5, zorder=1)
        for (k, sl, c, mk), v in zip(STATS, vals):
            ax.plot(v, j, mk, color=c, ms=7, mec="white", mew=0.8, zorder=2, label=sl if j == 0 else None)
    ax.set_yticks(range(len(MODELS)), [lab for _, lab in MODELS])
    ax.set_ylim(-0.6, len(MODELS) - 0.4); ax.invert_yaxis()
    ax.set_xlabel("German industrial output lost in the month (%)")
    ax.set_title(title, loc="left", fontsize=10)
axes[0].legend(frameon=False, fontsize=8.5, loc="center right")
figstyle.save(fig, os.path.join(HERE, "figures", "fig10_three_model_tails"), formats=("png",))
