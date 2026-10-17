#!/usr/bin/env python3
#-------- make_figures.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# The figures of 016 (run run_hysteresis.py, run_theory.py, early_warning.py,
# sentinel_patches.py and run_sites.py first).
#
#   fig1_hysteresis      drawn by run_hysteresis.py
#   fig2_theory          the lattice against its mean-field and pair approximations
#   fig3_early_warning   the vegetation stops connecting long before it disappears
#   fig4_patterns        Sentinel-2 at the two sites beside the model lattice
#   fig5_sites           rainfall-driven model against MODIS greenness, Gourma and Fakara

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figstyle                                         # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")
GREEN, WINE, SAND, GREY = "#117733", "#882255", "#DDCC77", "0.45"
LATTICE = ListedColormap(["#a6611a", "#f3e6c4", "#117733"])        # degraded, empty, vegetated


def fig2_theory():
    ca = pd.read_csv(os.path.join(RES, "hysteresis.csv"))
    ca = ca[ca.f == 0.9].assign(model="lattice")
    th = pd.read_csv(os.path.join(RES, "theory.csv"))
    df = pd.concat([ca, th])
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for m, col in [("lattice", WINE), ("pair", GREEN), ("mean field", GREY)]:
        for d, ls in [("drying", "-"), ("wetting", "--")]:
            s = df[(df.model == m) & (df.direction == d)].sort_values("b")
            ax.plot(s.b, s.cover, ls, color=col, lw=1.8)
        ax.plot([], [], "-", color=col, lw=4, label={"lattice": "the lattice (simulation)", "pair": "pair approximation",
                                                     "mean field": "mean field (no space)"}[m])
    ax.plot([], [], "-", color="0.3", label="drying")
    ax.plot([], [], "--", color="0.3", label="wetting")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42))
    ax.set_xlabel("environmental quality b (rainfall)")
    ax.set_ylabel("vegetation cover")
    ax.set_title("Without space, a desert never comes back; with it, it does (f = 0.9)", loc="left")
    ax.set_xlim(0, 1)
    figstyle.save(fig, os.path.join(FIG, "fig2_theory"))


def ccdf(sizes):
    s = np.sort(sizes)[::-1]
    return s, np.arange(1, len(s) + 1) / len(s)


def fig3_early_warning():
    ew = pd.read_csv(os.path.join(RES, "early_warning.csv"))
    pz = np.load(os.path.join(RES, "patches.npz"))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax = axes[0]
    ax.plot(ew.b, ew.cover, color=GREEN, label="vegetation cover")
    ax.plot(ew.b, ew.largest_share, color=WINE, label="largest patch's share")
    ax.set_xlabel("environmental quality b, drying to the right")
    ax.invert_xaxis()
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.3))
    ax.set_title("The network breaks before the cover goes", loc="left", fontsize=10)
    ax = axes[1]
    keys = sorted([k for k in pz.files if k.startswith("sizes_")], reverse=True)
    shades = plt.cm.viridis(np.linspace(0.15, 0.85, len(keys)))
    for k, c in zip(keys, shades):
        s, p = ccdf(pz[k])
        ax.loglog(s, p, color=c, label=f"model, b = {k[6:]}")
    for site, ls in [("gourma", ":"), ("fakara", "-.")]:
        path = os.path.join(DATA, f"s2_{site}.npz")
        if os.path.exists(path):
            s, p = ccdf(np.load(path)["sizes"])
            ax.loglog(s, p, ls, color="0.15", label=f"Sentinel-2, {site.capitalize()}")
    ax.set_xlabel("patch size (cells or 10 m pixels)")
    ax.set_ylabel("share of patches at least this large")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Patch sizes: the large ones vanish first", loc="left", fontsize=10)
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig3_early_warning"))


def fig4_patterns():
    pz = np.load(os.path.join(RES, "patches.npz"))
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.9))
    for ax, site in zip(axes[:2], ["gourma", "fakara"]):
        d = np.load(os.path.join(DATA, f"s2_{site}.npz"))
        n = d["ndvi"][:200, :200]                                   # a 2 x 2 km corner, the lattice's size at 10 m
        ax.imshow(n, cmap="YlGn", vmin=0.05, vmax=0.45)
        ax.set_title(f"{site.capitalize()}, Sentinel-2 NDVI\n{str(d['scene'])[11:19]}, 2 x 2 km", fontsize=8.5)
    for ax, b in zip(axes[2:], ["0.60", "0.42"]):
        ax.imshow(pz[f"grid_{b}"], cmap=LATTICE, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"model lattice, b = {b}\n200 x 200 cells", fontsize=8.5)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig4_patterns"))


def fig5_sites():
    s = pd.read_csv(os.path.join(RES, "sites.csv"))
    nd = pd.read_csv(os.path.join(RES, "ndvi_annual.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.3]))
    for j, site in enumerate(["gourma", "fakara"]):
        g = s[s.site == site]
        r = g[g.d == g.d.min()]
        ax = axes[0, j]
        ax.bar(r.year, r.rain_mm, color=SAND, width=0.85)
        ax.set_title(f"{site.capitalize()}: rainy-season rain (ERA5)", loc="left", fontsize=10)
        ax.set_ylabel("mm, June-September" if j == 0 else "")
        ax = axes[1, j]
        for d, ls in zip(sorted(g.d.unique()), ["-", "--"]):
            x = g[g.d == d]
            ax.plot(x.year, x.cover, ls, color=GREEN, label=f"model cover, degradation d = {d}")
        ax.set_ylabel("model vegetation cover" if j == 0 else "")
        ax.set_ylim(0, 0.8)
        tw = ax.twinx()
        n = nd[nd.site == site]
        tw.plot(n.year, n.ndvi_peak, "o-", ms=3, color=WINE, lw=1, label="MODIS peak NDVI")
        tw.set_ylabel("MODIS growing-season peak NDVI" if j == 1 else "", color=WINE)
        tw.tick_params(axis="y", colors=WINE)
        tw.grid(False)
        tw.spines["right"].set_visible(True)
        if j == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = tw.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig5_sites"))


if __name__ == "__main__":
    figstyle.apply()
    fig2_theory()
    fig3_early_warning()
    fig4_patterns()
    if os.path.exists(os.path.join(RES, "sites.csv")):
        fig5_sites()
