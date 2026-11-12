#-------- make_stage3_figure.py ----------------------------------------------------------
# The forecast test of run_real.py stage 3, drawn from results/real_stage3.csv: the gain in
# volatility forecast loss (QLIKE) over plain GARCH for three ways of supplying each
# month's extreme-day count, 2023 onwards, with the Diebold-Mariano t-statistic.
#
#   python make_stage3_figure.py

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figstyle

if __name__ == "__main__":
    figstyle.apply()
    d = pd.read_csv(os.path.join(HERE, "results", "real_stage3.csv")).set_index("weather")
    labels = {"perfect": "known in advance", "persistence": "last month's count", "climatology": "calendar average"}
    order = ["perfect", "persistence", "climatology"]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    y = np.arange(len(order))[::-1]
    for yi, k in zip(y, order):
        g, t = d.loc[k, "gain_pct"], d.loc[k, "dm_t"]
        ax.hlines(yi, 0, g, color="0.8", lw=2)
        ax.plot(g, yi, "o", color="#7b1e3c", ms=9)
        ax.text(g + 0.03, yi, f"{g:.2f}%  (DM t = {t:.2f})", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([labels[k] for k in order])
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("lower volatility forecast loss than plain GARCH (QLIKE, %)")
    ax.set_title("Volatility forecast gain from the extreme-day count, 2023-2026")
    fig.tight_layout()
    figstyle.save(fig, os.path.join(HERE, "figures", "fig13_forecast_gain"))
