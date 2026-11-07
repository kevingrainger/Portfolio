#!/usr/bin/env python3
#-------- mlmc.py -------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Multilevel Monte Carlo for the grove.
#
# The paper's grove has a million spittlebugs a hectare, 810,000 on our 90 x 90 m grove.
# That can't be run many times, so the model samples a fraction of them (sampling_rate).
# The answer depends on that fraction, so the quantity wanted is the one at the finest
# level affordable. Multilevel Monte Carlo (Giles 2008) gets it cheaply by writing
#
#     E[Q_L] = E[Q_0] + sum over l of E[Q_l - Q_(l-1)],
#
# with level l sampling 0.0005 x 2^l of the spittlebugs. Most runs go on the cheap coarse
# level; the corrections need few runs if a fine run and its coarse partner move together,
# so that Q_l - Q_(l-1) varies little. That pairing (the coupling) is the whole game.
#
# Coupling here: a fine run's spittlebugs are its coarse partner's spittlebugs plus as many
# again, the coarse ones placed, moving and feeding exactly as in the coarse run (same random
# numbers, from a stream of their own); the extra ones have their own stream. The seed tree
# is the same. Implemented by giving each spittlebug its own random stream (CoupledGrove).
#
#   Q   share of the 256 trees infected after 365 days (one year; by 540 days the denser
#       levels have all taken the whole grove, which leaves nothing to estimate)
#
# For each level: the mean and variance of Q_l and of Q_l - Q_(l-1), and the cost of a run.
# Then the MLMC sample sizes for a target accuracy (N_l proportional to sqrt(V_l / C_l)),
# and the cost against plain Monte Carlo at the finest level for the same accuracy.
#
#   python mlmc.py
#
# Output: results/mlmc.json, figures/fig9_mlmc.png

import json
import os
import time
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from olive_model import OliveGrove, SHRUB, SUSCEPTIBLE, INFECTED, SYMPTOMATIC, STEPS_4, STEPS_8

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
BASE_RATE, LEVELS, DAYS, PAIRS = 0.0005, 5, 365, 30


class CoupledGrove(OliveGrove):
    """The grove with its randomness split so that runs can be paired. Every per-spittlebug
    draw is a full-length array: the first n_shared spittlebugs take theirs from stream A,
    the rest from stream B. A coarse run (n spittlebugs, n_shared = n) and its fine partner
    (2n, n_shared = n) built on the same seed then give spittlebug i < n exactly the same
    numbers, so it starts in the same place and makes the same choices until the trees it
    meets differ. Each tree's symptom delay is drawn once, up front, from a third stream.
    The seed carrier is spittlebug 0 in both."""

    def __init__(self, *a, seed=0, n_shared=None, **k):
        super().__init__(*a, seed=seed, **k)
        self.n_shared = self.n_vectors if n_shared is None else n_shared
        self.rng_a = np.random.default_rng([seed, 1])
        self.rng_b = np.random.default_rng([seed, 2, self.n_vectors])
        self.tree_delay = np.maximum(0, np.random.default_rng([seed, 3]).normal(
            self.symptom_delay_mean, self.symptom_delay_std, len(self.tree_centres)))

    def _u(self):
        n0 = self.n_shared
        return np.concatenate([self.rng_a.random(n0), self.rng_b.random(self.n_vectors - n0)])

    def _i(self, k):
        n0 = self.n_shared
        return np.concatenate([self.rng_a.integers(0, k, n0), self.rng_b.integers(0, k, self.n_vectors - n0)])

    def initialize_vectors(self, initial_infected=1):
        shrubs = np.argwhere(self.cell_type == SHRUB)
        pick = self._i(len(shrubs))
        self.vector_y[:], self.vector_x[:] = shrubs[pick, 0], shrubs[pick, 1]
        self.vector_infected[:] = False
        self.vector_source[:] = -1
        if initial_infected > 0:
            self.vector_infected[0] = True

    def vector_movement(self):
        x, y = self.vector_x, self.vector_y
        u_move, u_go, k8, k4 = self._u(), self._u(), self._i(8), self._i(4)
        tree_here = self.tree_of_cell[y, x]
        on_tree = tree_here >= 0
        on_symptomatic = on_tree & (self.tree_state[np.maximum(tree_here, 0)] >= SYMPTOMATIC)
        repelled = np.where(on_symptomatic)[0]
        if len(repelled):
            ey, ex = self.escape_y[y[repelled], x[repelled]], self.escape_x[y[repelled], x[repelled]]
            has_exit = ey >= 0
            y[repelled[has_exit]], x[repelled[has_exit]] = ey[has_exit], ex[has_exit]
            stuck = repelled[~has_exit]
            if len(stuck):
                step = STEPS_8[k8[stuck]]
                x[stuck] += step[:, 0]
                y[stuck] += step[:, 1]
        movers = np.where(~on_symptomatic & (u_move < 0.5))[0]
        step = STEPS_4[k4[movers]]
        nx, ny = x[movers] + step[:, 0], y[movers] + step[:, 1]
        inside = (nx >= 0) & (nx < self.width) & (ny >= 0) & (ny < self.height)
        movers, nx, ny = movers[inside], nx[inside], ny[inside]
        from_tree = self.tree_of_cell[y[movers], x[movers]]
        to_tree = self.tree_of_cell[ny, nx]
        blocked = (to_tree >= 0) & (self.tree_state[np.maximum(to_tree, 0)] >= SYMPTOMATIC)
        p = np.empty(len(movers))
        shrub_from, shrub_to = from_tree < 0, to_tree < 0
        p[shrub_from & shrub_to] = self.p_ss
        p[shrub_from & ~shrub_to] = self.p_st
        p[~shrub_from & shrub_to] = self.p_ts
        tree_tree = ~shrub_from & ~shrub_to
        p[tree_tree] = np.where(from_tree[tree_tree] == to_tree[tree_tree], self.p_tt, self.p_bb)
        go = ~blocked & (u_go[movers] < p)
        x[movers[go]], y[movers[go]] = nx[go], ny[go]

    def transmission(self):
        u_hit, u_catch = self._u(), self._u()
        tree_here = self.tree_of_cell[self.vector_y, self.vector_x]
        on_tree = tree_here >= 0
        state = np.where(on_tree, self.tree_state[np.maximum(tree_here, 0)], -1)
        hit = np.where(self.vector_infected & (state == SUSCEPTIBLE) & (u_hit < self.tree_susceptibility))[0]
        if len(hit):
            targets, first = np.unique(tree_here[hit], return_index=True)
            self.tree_state[targets] = INFECTED
            self.infection_day[targets] = self.day
            self.symptom_day[targets] = self.day + self.tree_delay[targets]
            self.new_infections_today += len(targets)
        feeding = np.where(~self.vector_infected & (state == INFECTED))[0]
        if len(feeding):
            trees = tree_here[feeding]
            days_infected = self.day - self.infection_day[trees]
            colonisation = np.minimum(1.0, days_infected * self.xylem_propagation_rate / self.twig_length)
            prob = self.vector_susceptibility * np.where(days_infected < 90, colonisation, 1.0)
            caught = feeding[u_catch[feeding] < prob]
            self.vector_infected[caught] = True


def q_of(rate, seed, n_shared=None, coupled=True):
    g = CoupledGrove(seed=seed, sampling_rate=rate, n_shared=n_shared) if coupled else OliveGrove(seed=seed, sampling_rate=rate)
    t = time.time()
    g.run(DAYS, initial_infected=1)
    return float(g.ever_infected()), time.time() - t


def job(args):
    level, seed, coupled = args
    n_coarse = int(810000 * BASE_RATE * 2 ** (level - 1)) if level else None
    qf, cf = q_of(BASE_RATE * 2 ** level, seed, n_shared=n_coarse, coupled=coupled)
    if level == 0:
        return level, seed, qf, np.nan, cf
    qc, cc = q_of(BASE_RATE * 2 ** (level - 1), seed, coupled=coupled)
    return level, seed, qf, qc, cf + cc


def level_stats(res, coupled):
    lv = {}
    for l in range(LEVELS + 1):
        r = [x for x in res if x[0] == l and x[5] == coupled]
        qf = np.array([x[2] for x in r]); qc = np.array([x[3] for x in r]); c = np.array([x[4] for x in r])
        d = qf - qc if l else qf
        lv[l] = dict(rate=BASE_RATE * 2 ** l, spittlebugs=int(810000 * BASE_RATE * 2 ** l), mean_q=float(qf.mean()),
                     var_q=float(qf.var(ddof=1)), mean_diff=float(d.mean()), var_diff=float(d.var(ddof=1)),
                     corr_pair=float(np.corrcoef(qf, qc)[0, 1]) if l and qf.std() > 0 and qc.std() > 0 else None,
                     cost_s=float(c.mean()))
    return lv


def costs(lv, eps):
    V = np.array([lv[l]["var_diff"] for l in lv]); C = np.array([lv[l]["cost_s"] for l in lv])
    N = np.ceil(np.sqrt(V / C) * np.sum(np.sqrt(V * C)) / eps ** 2)
    return float((N * C).sum()), float(lv[LEVELS]["var_q"] / eps ** 2 * lv[LEVELS]["cost_s"]), N


def main():
    jobs = [(l, 1000 + s, cp) for cp in (True, False) for l in range(LEVELS + 1) for s in range(PAIRS)]
    jobs.sort(key=lambda j: -j[0])                                   # longest first
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        res = [r + (j[2],) for r, j in zip(pool.map(job, jobs), jobs)]
    lv_c, lv_u = level_stats(res, True), level_stats(res, False)
    eps_list = [0.04, 0.02, 0.01, 0.005]
    cost = {e: dict(zip(["mlmc_coupled", "plain_mc", "mlmc_seed_only"], [costs(lv_c, e)[0], costs(lv_c, e)[1], costs(lv_u, e)[0]]))
            for e in eps_list}
    out = dict(days=DAYS, pairs_per_level=PAIRS, coupled=lv_c, seed_only=lv_u, cost_s=cost,
               samples_at_0_01=costs(lv_c, 0.01)[2].tolist(), speedup_at_0_01=cost[0.01]["plain_mc"] / cost[0.01]["mlmc_coupled"])
    json.dump(out, open(os.path.join(HERE, "results", "mlmc.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("coupled", "seed_only")}, indent=1))
    for nm, lv in [("coupled", lv_c), ("seed only", lv_u)]:
        print(nm, [(l, round(v["mean_q"], 3), round(v["var_q"], 4), round(v["var_diff"], 4), v["corr_pair"] and round(v["corr_pair"], 2), round(v["cost_s"], 1)) for l, v in lv.items()])

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw=dict(wspace=0.3))
    L = np.arange(LEVELS + 1); bugs = [lv_c[l]["spittlebugs"] for l in L]
    ax = axes[0]
    ax.plot(bugs, [lv_c[l]["var_q"] for l in L], "-o", color=NAVY, lw=1.6, label="Q at each level")
    ax.plot(bugs[1:], [lv_c[l]["var_diff"] for l in L[1:]], "-o", color=RED, lw=1.6, label="fine minus coarse, coupled")
    ax.plot(bugs[1:], [lv_u[l]["var_diff"] for l in L[1:]], "--s", color=BLUE, lw=1.3, ms=5, label="fine minus coarse, same seed only")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("spittlebugs simulated (of 810,000)"); ax.set_ylabel("variance")
    ax.set_title("Variance by level", loc="left"); ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    ax.plot(bugs, [100 * lv_c[l]["mean_q"] for l in L], "-o", color=NAVY, lw=1.6, label=f"share infected at day {DAYS}")
    ax.plot(bugs[1:], [100 * abs(lv_c[l]["mean_diff"]) for l in L[1:]], "-o", color=RED, lw=1.6, label="size of the correction from the level below")
    ax.set_xscale("log"); ax.set_xlabel("spittlebugs simulated (of 810,000)"); ax.set_ylabel("% of trees")
    ax.set_title("Mean by level", loc="left"); ax.legend(frameon=False, fontsize=8.5)
    ax = axes[2]
    e = np.array(eps_list)
    ax.plot(e, [cost[x]["plain_mc"] / 3600 for x in eps_list], "-o", color=NAVY, lw=1.6, label="plain Monte Carlo, finest level")
    ax.plot(e, [cost[x]["mlmc_coupled"] / 3600 for x in eps_list], "-o", color=RED, lw=1.6, label="multilevel, coupled")
    ax.plot(e, [cost[x]["mlmc_seed_only"] / 3600 for x in eps_list], "--s", color=BLUE, lw=1.3, ms=5, label="multilevel, same seed only")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.minorticks_off(); ax.set_xticks(e, [f"{100 * x:g}%" for x in e])
    ax.set_xlabel("target standard error on the share infected (points)"); ax.set_ylabel("computer time, hours (one core)")
    ax.set_title("Cost for a given accuracy", loc="left"); ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(os.path.join(HERE, "figures", "fig9_mlmc.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
