#-------- make_pick_errors.py ------------------------------------------------------------
# Distribution of pick errors for each method on the 3,000 noisy traces with the arrival
# moved (the same seeds as the notebook). Needs the SCSN file (not in the repository).
#
#   python make_pick_errors.py

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import figstyle
import seismic as s

if __name__ == "__main__":
    figstyle.apply()
    os.chdir(HERE)
    X, Y, *_ = s.load_scsn(n=3000, snr_quantile=(0.0, 0.2), seed=0)
    X_shift, arrivals = s.random_shift(X, seed=0)
    board = s.scoreboard(X_shift, Y, arrivals)
    methods = list(board)
    bins = np.linspace(-1.5, 1.5, 61)
    fig, axes = plt.subplots(len(methods), 1, figsize=(8, 5.2), sharex=True)
    for ax, m, c in zip(axes, methods, figstyle.PALETTE):
        e = board[m]["errors"]
        missed = np.mean(np.isnan(e))
        ax.hist(np.clip(e[~np.isnan(e)], -1.5, 1.5), bins=bins, color=c, alpha=0.85)
        ax.axvspan(-0.2, 0.2, color="0.9", zorder=0)
        ax.set_yticks([])
        ax.set_ylabel(m, rotation=0, ha="right", va="center")
        ax.text(1.0, 0.75, f"within 0.2 s {board[m]['picked_within_tolerance']:.0%}, no pick {missed:.0%}",
                transform=ax.transAxes, ha="right", fontsize=8.5)
    axes[0].set_title("Pick error by method, arrival moved (3,000 traces)", loc="left")
    axes[-1].set_xlabel("pick minus analyst's arrival (s), clipped at ±1.5 s; shaded ±0.2 s")
    fig.tight_layout()
    figstyle.save(fig, os.path.join(HERE, "figures", "fig5_pick_errors"))
    print({m: round(float(np.nanmedian(np.abs(board[m]["errors"]))), 3) for m in methods})
