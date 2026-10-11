#!/usr/bin/env python3
#-------- simulate_theory.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# Figures for Part I beyond percolation itself:
#   may        May's stability bound: eigenvalue discs and the sharp loss of stability
#   assort     degree correlations: disassortative, uncorrelated and assortative networks
#
#   python simulate_theory.py [may] [assort]

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import figstyle
import percolation as P

figstyle.apply()
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
COL = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#555555"]
INK = "#222222"


def may(seed=3):
    rng = np.random.default_rng(seed)
    C = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw=dict(wspace=0.32, width_ratios=[1, 1, 1.35]))
    N = 400
    for ax, x, lab in zip(axes[:2], (0.8, 1.2), ("(a)", "(b)")):
        sigma = x / np.sqrt(N * C)
        ev = np.linalg.eigvals(P.may_matrix(N, C, sigma, rng))
        stable = ev.real.max() < 0
        ax.scatter(ev.real, ev.imag, s=5, color=COL[0], alpha=0.6, lw=0)
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(-1 + x * np.cos(t), x * np.sin(t), color=INK, lw=1.0, ls="--")
        ax.axvline(0, color=COL[2], lw=1.4)
        ax.set_xlim(-2.5, 0.6); ax.set_ylim(-1.55, 1.55); ax.set_aspect("equal")
        ax.set_xlabel(r"Re $\lambda$"); ax.set_ylabel(r"Im $\lambda$")
        ax.set_title(rf"{lab} $\sigma\sqrt{{NC}} = {x}$: " + ("stable" if stable else "unstable"), loc="left")
    xs = np.linspace(0.6, 1.4, 25)
    out = {}
    for k, (N, reps) in enumerate([(25, 400), (100, 200), (400, 60)]):
        p = []
        for x in xs:
            sigma = x / np.sqrt(N * C)
            p.append(np.mean([P.may_stable(N, C, sigma, rng) for _ in range(reps)]))
        axes[2].plot(xs, p, "o-", ms=4, mfc="white", mew=1.2, color=COL[k], label=f"$N = {N}$")
        out[str(N)] = p
    axes[2].axvline(1, color=INK, lw=0.9, ls=(0, (2, 2)))
    axes[2].set_xlabel(r"$\sigma\sqrt{NC}$"); axes[2].set_ylabel("probability of stability")
    axes[2].set_title("(c) The transition sharpens with size", loc="left")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    axes[2].set_ylim(-0.02, 1.02)
    json.dump(dict(x=xs.tolist(), C=C, p_stable=out), open(os.path.join(RES, "may_stability.json"), "w"), indent=2)
    figstyle.save(fig, os.path.join(FIG, "fig1_may_stability"))
    plt.close(fig)


def assort(N=10_000, seed=5, reps=8):
    rng = np.random.default_rng(seed)
    d = P.Degrees.power_law(2.5, kmin=2, kmax=150)
    base = P.configuration_model(d, N, rng)
    nets = {"disassortative": P.rewire(base, N, "disassortative", 20 * len(base), rng),
            "uncorrelated": base,
            "assortative": P.rewire(base, N, "assortative", 20 * len(base), rng)}
    Ts = np.linspace(0.0, 0.7, 29)
    out = {}
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for k, (name, e) in enumerate(nets.items()):
        r = P.assortativity(e, N)
        tc = P.correlated_threshold(e, N)
        S = [np.mean([np.bincount(P.percolate(e, N, t, rng)).max() / N for _ in range(reps)]) for t in Ts]
        out[name] = dict(r=r, Tc=tc, T=Ts.tolist(), S=S)
        ax.plot(Ts, S, marker="o", ms=3.8, mfc="white", mew=1.2, color=COL[[2, 0, 1][k]], ls=["-", "--", "-."][k],
                label=f"{name}, $r = {r:+.2f}$, $T_c = {tc:.3f}$")
        ax.axvline(tc, color=COL[[2, 0, 1][k]], lw=0.8, ls=":")
    ax.set_xlabel(r"transmissibility $T$"); ax.set_ylabel("largest failure cluster")
    ax.set_xlim(0, 0.7); ax.set_ylim(0, 1)
    ax.set_title(r"Same degrees, different wiring (power law, $\gamma = 2.5$, $N = 10^4$)", loc="left")
    ax.legend(loc="upper left")
    json.dump(out, open(os.path.join(RES, "assortativity.json"), "w"), indent=2)
    figstyle.save(fig, os.path.join(FIG, "fig3_disassortativity"))
    plt.close(fig)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["may", "assort"]:
        globals()[w]()
        print(w, "done")
