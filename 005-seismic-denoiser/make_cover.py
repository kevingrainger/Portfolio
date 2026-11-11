#-------- make_cover.py -----------------------------------------------------------------
# The card cover: the synthetic 24-sensor array from the notebook (seed 3, road noise),
# filtered 2-12 Hz, then one sensor, the plain average and the optimal stack with a
# cleaned noise covariance.
#
#   python make_cover.py

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
from corrlib import array_stacking as stack

if __name__ == "__main__":
    figstyle.apply()
    X, clean, delays, x_km, exposure = s.synthetic_array(n_sensors=24, n_samples=4200, arrival=3600, road_strength=4.0, seed=3)
    Xf = s.bandpass(X, low=2.0, high=12.0)
    aligned = s.align_known(Xf, delays)
    truth = s.align_known(s.bandpass(clean, low=2.0, high=12.0), delays)[0]
    avg, _ = stack.simple(aligned)
    opt, _ = stack.mvdr(aligned, clean="nonlinear_shrinkage", noise_window=(1400, 1800))
    one = aligned[int(np.argsort(aligned[:, 1400:1800].std(axis=1))[12])]
    t = np.arange(X.shape[1]) / s.FS
    win = (t > 33.5) & (t < 39.5)
    scale = np.abs(truth).max()

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes([0.04, 0.09, 0.42, 0.80])
    cmap = plt.get_cmap("magma")
    for k in range(24):
        tr = Xf[k, win]
        ax.plot(t[win], tr / (4 * Xf[:, win].std()) + k, color=cmap(0.2 + 0.6 * exposure[k] / exposure.max()), lw=0.7)
        ax.plot((3600 + delays[k]) / s.FS, k + 0.6, "v", color="#009E73", ms=5)
    ax.set_yticks([]); ax.set_xlabel("time (s)")
    ax.set_title("24 sensors, filtered 2-12 Hz (arrivals marked)", loc="left", fontsize=11)
    panels = [("One sensor", one, "0.5"), ("Plain average", avg, "#D55E00"), ("Optimal stack, cleaned covariance", opt, "#0072B2")]
    for j, (title, tr, col) in enumerate(panels):
        a = fig.add_axes([0.53, 0.66 - 0.285 * j, 0.45, 0.22])
        a.plot(t[win], truth[win] / scale, "k--", lw=1.2, label="true wave")
        a.plot(t[win], tr[win] / scale, color=col, lw=1.1)
        a.set_yticks([]); a.set_title(title, loc="left", fontsize=11)
        a.set_ylim(-3, 3)
        if j < 2:
            a.set_xticklabels([])
        else:
            a.set_xlabel("time (s)")
        if j == 0:
            a.legend(loc="upper right", fontsize=8.5)
    fig.savefig(os.path.join(HERE, "figures", "cover.png"), dpi=200)
    fig.savefig(os.path.join(HERE, "figures", "cover.pdf"))
    print("wrote figures/cover.png")
