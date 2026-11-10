#!/usr/bin/env python3
#-------- ensemble_likelihood.py ----------------------------------------------------
#-----------------------------------------------------------------------------
# Can the counts identify the infection rate at all? The particle filter in
# assimilate.py has only 32 groves and no shrinkage on the jittered rate, so a wide
# posterior from it could be the filter's fault rather than the data's. This asks the
# question directly, without a filter.
#
# For each rate on a grid from 0.1 to 1.0, 24 groves are run forward from one infected
# tree for six years, and their symptomatic count is recorded every 120 days. The
# likelihood of the twin experiment's observed counts (results/assimilation.json, same
# truth and same observations as the filter) given the rate is the average over the
# groves of the probability of the whole sequence of counts, with the filter's own
# observation model (each tree seen 80% of the time; Gaussian with sd sqrt(s 0.8 0.2) + 2):
#
#   p(y_1..y_K | rate) ~ (1/M) sum_m prod_k N(y_k; 0.8 s_k^(m), sd_k^(m))
#
# Normalised over the grid (flat prior, as in the filter) this is the posterior the
# counts allow. It's computed with the first 2, 4 and 6 years of images.
#
#   python ensemble_likelihood.py
#
# Output: results/ensemble_likelihood.json, figures/fig11_ensemble_likelihood.png

import json
import os
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from olive_model import OliveGrove, SYMPTOMATIC
from assimilate import EVERY, YEARS, DETECT, TRUE_RATE

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
RATES = np.round(np.arange(0.1, 1.01, 0.1), 2)
M = 24
K = YEARS * 365 // EVERY


def run(job):
    rate, seed = job
    g = OliveGrove(seed=seed, tree_susceptibility=float(rate))
    g.run(0, initial_infected=1)
    s = []
    for _ in range(K):
        g.run(EVERY)
        s.append(int((g.tree_state == SYMPTOMATIC).sum()))
    return s


def loglik(y, S):
    """log p(y | rate) from the M runs S (M x K)."""
    sd = np.sqrt(S * DETECT * (1 - DETECT)) + 2.0
    per_run = (-0.5 * ((y - DETECT * S) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)).sum(axis=1)
    return float(logsumexp(per_run) - np.log(len(S)))


def main():
    y_all = np.array(json.load(open(os.path.join(HERE, "results", "assimilation.json")))["observed"])
    jobs = [(r, 1000 + 100 * i + m) for i, r in enumerate(RATES) for m in range(M)]
    with Pool() as pool:
        sims = pool.map(run, jobs)
    S = {r: np.array(sims[i * M:(i + 1) * M]) for i, r in enumerate(RATES)}
    out = dict(rates=RATES.tolist(), runs_per_rate=M, true_rate=TRUE_RATE, horizons={})
    for years in (2, 4, 6):
        k = years * 365 // EVERY
        ll = np.array([loglik(y_all[:k], S[r][:, :k]) for r in RATES])
        post = np.exp(ll - logsumexp(ll))
        cdf = np.cumsum(post)
        out["horizons"][years] = dict(loglik=ll.tolist(), posterior=post.tolist(),
                                      map=float(RATES[post.argmax()]),
                                      range90=[float(RATES[np.searchsorted(cdf, 0.05)]),
                                               float(RATES[min(np.searchsorted(cdf, 0.95), len(RATES) - 1)])])
    out["final_symptomatic"] = {str(r): dict(median=float(np.median(S[r][:, -1])),
                                             q10=float(np.quantile(S[r][:, -1], 0.1)),
                                             q90=float(np.quantile(S[r][:, -1], 0.9))) for r in RATES}
    out["symptomatic_median"] = {str(r): np.median(S[r], axis=0).tolist() for r in RATES}
    json.dump(out, open(os.path.join(HERE, "results", "ensemble_likelihood.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "symptomatic_median"}, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(wspace=0.3))
    ax = axes[0]
    yrs = np.arange(1, K + 1) * EVERY / 365
    for r, col in [(0.2, LIGHT), (0.5, BLUE), (0.9, NAVY)]:
        q = np.quantile(S[r], [0.1, 0.5, 0.9], axis=0)
        ax.fill_between(yrs, DETECT * q[0], DETECT * q[2], color=col, alpha=0.3, lw=0)
        ax.plot(yrs, DETECT * q[1], color=col, lw=1.8, label=f"rate {r}: median and 10-90% of runs")
    ax.plot(yrs, y_all, "o", color=RED, ms=4, label="observed counts (true rate 0.5)")
    ax.set_xlabel("years since the first infection"); ax.set_ylabel("trees seen symptomatic (80% detection)")
    ax.set_title("Symptomatic counts by infection rate", loc="left"); ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    for years, col, ls in [(2, GREY, ":"), (4, LIGHT, "-"), (6, BLUE, "-")]:
        ax.plot(RATES, out["horizons"][years]["posterior"], ls, color=col, lw=1.8, marker="o", ms=4,
                label=f"first {years} years of images")
    ax.axvline(TRUE_RATE, color=RED, lw=1, ls="--", label="hidden true rate")
    ax.set_xlabel("infection rate (tree susceptibility)"); ax.set_ylabel("posterior probability on the grid")
    ax.set_title("Likelihood of the counts against the rate", loc="left"); ax.legend(frameon=False, fontsize=9)
    fig.savefig(os.path.join(HERE, "figures", "fig11_ensemble_likelihood.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
