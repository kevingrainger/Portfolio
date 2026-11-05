#!/usr/bin/env python3
#-------- assimilate.py -------------------------------------------------------------
#-----------------------------------------------------------------------------
# Data assimilation for the olive grove: a twin experiment with a particle filter.
#
# A "true" grove is run with a hidden infection rate (tree_susceptibility = 0.5). Every
# 120 days a satellite-like observation counts its symptomatic trees, with each one seen
# only 80% of the time (crowns hidden, misclassified). The filter starts from 32 groves,
# each with its own guess of the rate drawn from 0.1 to 1.0, runs them forward between
# observations, weights each by how well its symptomatic count matches the observation,
# and resamples (copying the hidden state of the groves that fit, with a little jitter on
# their rate). Can a sequence of images, like the real Gallipoli pair, pin the rate down?
#
# The catch it exposes: symptoms show about two years after infection, so the images say
# nothing about the rate until the first infections become visible.
#
#   python assimilate.py
#
# Output: results/assimilation.json, figures/fig8_assimilation.png

import copy
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from olive_model import OliveGrove, SYMPTOMATIC

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
TRUE_RATE, N_PART, EVERY, YEARS, DETECT = 0.5, 32, 120, 6, 0.8


def symptomatic(g):
    return int((g.tree_state == SYMPTOMATIC).sum())


def main():
    rng = np.random.default_rng(7)
    truth = OliveGrove(seed=11, tree_susceptibility=TRUE_RATE)
    truth.run(0, initial_infected=1)
    parts = []
    for i in range(N_PART):
        g = OliveGrove(seed=100 + i, tree_susceptibility=float(rng.uniform(0.1, 1.0)))
        g.run(0, initial_infected=1)
        parts.append(g)
    days, obs, true_s, mean_r, lo_r, hi_r, ess = [], [], [], [], [], [], []
    for k in range(1, YEARS * 365 // EVERY + 1):
        truth.run(EVERY)
        s_true = symptomatic(truth)
        y = rng.binomial(s_true, DETECT)
        for g in parts:
            g.run(EVERY)
        s = np.array([symptomatic(g) for g in parts])
        sd = np.sqrt(s * DETECT * (1 - DETECT)) + 2.0
        logw = -0.5 * ((y - DETECT * s) / sd) ** 2 - np.log(sd)
        w = np.exp(logw - logw.max()); w /= w.sum()
        rates = np.array([g.tree_susceptibility for g in parts])
        order = np.argsort(rates); cw = np.cumsum(w[order])
        days.append(truth.day); obs.append(int(y)); true_s.append(s_true)
        mean_r.append(float((w * rates).sum()))
        lo_r.append(float(rates[order][np.searchsorted(cw, 0.05)])); hi_r.append(float(rates[order][min(np.searchsorted(cw, 0.95), N_PART - 1)]))
        ess.append(float(1 / (w ** 2).sum()))
        if ess[-1] < N_PART / 2:                                  # resample: copy the groves that fit
            pick = rng.choice(N_PART, N_PART, p=w)
            new = []
            for j in pick:
                g = copy.deepcopy(parts[j])
                g.rng = np.random.default_rng(rng.integers(1 << 31))
                g.tree_susceptibility = float(np.clip(g.tree_susceptibility + rng.normal(0, 0.03), 0.05, 1.2))
                new.append(g)
            parts = new
        print(f"day {truth.day:4d}: observed {y:3d} symptomatic (true {s_true}), rate {mean_r[-1]:.2f} [{lo_r[-1]:.2f}, {hi_r[-1]:.2f}], ESS {ess[-1]:.1f}", flush=True)
    out = dict(true_rate=TRUE_RATE, days=days, observed=obs, true_symptomatic=true_s, rate_mean=mean_r,
               rate_90=[lo_r, hi_r], ess=ess)
    json.dump(out, open(os.path.join(HERE, "results", "assimilation.json"), "w"), indent=1)

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    yrs = np.array(days) / 365
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(wspace=0.3))
    ax = axes[0]
    ax.plot(yrs, true_s, color=NAVY, lw=1.6, label="true symptomatic trees")
    ax.plot(yrs, obs, "o", color=RED, ms=4, label="observed (80% detection)")
    ax.set_xlabel("years since the first infection"); ax.set_ylabel("trees")
    ax.set_title("What the satellite sees", loc="left"); ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    ax.fill_between(yrs, lo_r, hi_r, color=LIGHT, alpha=0.7, lw=0, label="90% of the filter's belief")
    ax.plot(yrs, mean_r, color=BLUE, lw=1.8, label="filter's estimate")
    ax.axhline(TRUE_RATE, color=RED, lw=1, ls="--", label="hidden true rate")
    ax.set_xlabel("years since the first infection"); ax.set_ylabel("infection rate (tree susceptibility)")
    ax.set_title("What the filter learns", loc="left"); ax.legend(frameon=False, fontsize=9)
    fig.savefig(os.path.join(HERE, "figures", "fig8_assimilation.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
