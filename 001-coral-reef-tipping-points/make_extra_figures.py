#-------- make_extra_figures.py ---------------------------------------------------------
# Small figures for the website timeline, drawn from results already saved by
# coral_potts.ipynb, inference/infer_crw.py and run_tipping_crw.py. Nothing is re-simulated
# except one 60 x 60 tolerance field per panel (seeded).
#
#   python make_extra_figures.py

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import figstyle
from coral_crw import CRWReef

figstyle.apply()
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "inference", "results_crw")
SEN, TOL = "#CC79A7", "#0072B2"


def hysteresis_width():
    """Loop width by interaction range, as printed by coral_potts.ipynb."""
    nb = json.load(open(os.path.join(HERE, "coral_potts.ipynb"), encoding="utf8"))
    text = "".join("".join(o.get("text", "")) for c in nb["cells"] for o in c.get("outputs", []))
    rows = re.findall(r"^(.+?)\s+hysteresis width ([\d.]+)", text, flags=re.M)
    labels, w = [r[0].strip() for r in rows], np.array([float(r[1]) for r in rows])
    fig, ax = plt.subplots(figsize=(7, 2.6))
    y = np.arange(len(w))[::-1]
    ax.hlines(y, w[0], w, color="0.75", lw=1)
    ax.plot(w, y, "o", color=figstyle.PALETTE[0], ms=8)
    for yi, wi in zip(y, w):
        ax.text(wi + 0.02, yi, f"{wi:.2f} ({100 * (wi / w[0] - 1):+.0f}%)", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0.75, 1.6)
    ax.set_xlabel("area between warming and cooling paths (cover × °C)")
    ax.set_title("Hysteresis loop width by interaction range")
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig11_hysteresis_width"))
    plt.close(fig)


def tolerance_field():
    """One tolerance field D50(x) at each mode's median parameters (60 x 60, seed 0)."""
    post = np.load(os.path.join(RES, "observed.npz"))["post"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, (name, sel) in zip(axes, (("Sensitive reef", post[:, 0] < 8), ("Tolerant reef", post[:, 0] >= 8))):
        mu, sd = np.median(post[sel, 0]), np.median(post[sel, 1])
        r = CRWReef(mu_D=mu, sigma_D=sd, seed=0)
        d = np.where(r.reef, r.D50, np.nan)
        im = ax.imshow(d, cmap="viridis", vmin=0, vmax=20)
        ax.set_title(name + f"\nμ_D {mu:.1f}, σ_D {sd:.1f} DHW", fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes, label="tolerance D₅₀ (DHW)", shrink=0.85)
    figstyle.save(fig, os.path.join(FIG, "fig12_tolerance_field"))
    plt.close(fig)


def coverage():
    """Share of 800 held-out simulations whose credible interval holds the true value."""
    c = json.load(open(os.path.join(RES, "check.json")))
    lv, cov = np.array(c["levels"]), np.array(c["coverage"])
    names = ["mean tolerance μ_D", "tolerance spread σ_D", "stress sensitivity s", "noise β"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="ideal")
    for k, n in enumerate(names):
        ax.plot(lv, cov[:, k], "o-", ms=3, lw=1.2, label=n)
    ax.set_xlabel("credible interval level")
    ax.set_ylabel("share holding the true value")
    ax.set_title("Coverage on 800 held-out simulated reefs")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig13_coverage"))
    plt.close(fig)


def tip_vs_tolerance():
    """Tipping temperature of each of the 40 posterior draws against its mean tolerance."""
    d = pd.read_csv(os.path.join(RES, "tipping.csv")).drop_duplicates("run")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    jit = np.random.default_rng(0).uniform(-0.015, 0.015, len(d))
    for mode, col in (("sensitive", "#882255"), ("tolerant", "#117733")):
        g = d["mode"] == mode
        ax.plot(d.mu_D[g], d.tip[g] + jit[g.values], "o", color=col, alpha=0.8, label=f"{mode} reef")
    ax.set_xlabel("mean tolerance μ_D (DHW)")
    ax.set_ylabel("tipping point (°C above MMM)")
    ax.set_title("Tipping point against mean tolerance, 40 posterior draws")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig14_tip_vs_tolerance"))
    plt.close(fig)


def february():
    """Bleached share of living coral on 1 February and 16 April 2016, per posterior draw."""
    o = np.load(os.path.join(RES, "observed.npz"))
    p, mode = o["paths"][:, :, 0], o["mode"]
    c = pd.read_csv(os.path.join(HERE, "data", "crw_scott_ashmore_2015-2016.csv"), comment="#", parse_dates=["date"])
    t = c.date.values[:p.shape[1]]
    i_feb = int(np.flatnonzero(t == np.datetime64("2016-02-01"))[0])
    i_apr = int(np.flatnonzero(t == np.datetime64("2016-04-16"))[0])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for m, col, y0, name in ((1, SEN, 1, "sensitive reef"), (0, TOL, 0, "tolerant reef")):
        a, b = p[mode == m, i_feb], p[mode == m, i_apr]
        jy = np.random.default_rng(m).uniform(-0.12, 0.12, len(a))
        ax.plot(a, y0 + jy, "o", color=col, ms=3, alpha=0.4)
        ax.plot(b, y0 + jy, "s", color=col, ms=3, alpha=0.4)
        ax.annotate("", xy=(np.median(b), y0 + 0.28), xytext=(np.median(a), y0 + 0.28),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
        ax.text(np.median(a), y0 + 0.36, f"1 Feb {np.median(a):.0%}", ha="center", fontsize=8.5, color=col)
        ax.text(np.median(b), y0 + 0.36, f"16 Apr {np.median(b):.0%}", ha="center", fontsize=8.5, color=col)
    ax.set_yticks([1, 0]); ax.set_yticklabels(["sensitive reef", "tolerant reef"])
    ax.set_ylim(-0.35, 1.55); ax.set_xlim(0, 1)
    ax.set_xlabel("bleached share of living coral (circles 1 Feb, squares 16 Apr)")
    ax.set_title("Bleached share on 1 February and 16 April 2016")
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig15_february_april"))
    plt.close(fig)


if __name__ == "__main__":
    hysteresis_width()
    tolerance_field()
    coverage()
    tip_vs_tolerance()
    february()
