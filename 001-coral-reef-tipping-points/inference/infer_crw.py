#!/usr/bin/env python3
#-------- infer_crw.py ------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Neural posterior estimation for the CRW-driven reef, against the 65 real April 2016
# Scott Reef surveys.
#
#   python inference/infer_crw.py train check observed figures

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))
import figstyle
import mdn
from simulate_bank_crw import HIGH, LABELS, LOW, N_SITES, NAMES, SURVEY_DAY, forcing, observed, simulate

figstyle.apply()
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(HERE, "results_crw")
os.makedirs(RES, exist_ok=True)
COL = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#555555"]
INK = "#222222"
N_TEST = 800
CATS = ["< 30%", "30-60%", "60-90%", "91-100%"]


def features(counts):
    return np.atleast_2d(counts) / N_SITES


def load():
    z = np.load(os.path.join(HERE, "bank_crw.npz"))
    ok = z["done"]
    return z["theta"][ok], features(z["x"][ok])


def split():
    th, x = load()
    n = len(th)
    return (th[:n - 2 * N_TEST], x[:n - 2 * N_TEST]), (th[n - 2 * N_TEST:n - N_TEST], x[n - 2 * N_TEST:n - N_TEST]), \
        (th[n - N_TEST:], x[n - N_TEST:])


def train():
    (th, x), (thv, xv), _ = split()
    best = None
    for hid, K in [((64, 64), 6), ((96, 96), 8)]:
        net = mdn.MDN(x.shape[1], 4, hidden=hid, K=K, seed=1)
        h = net.fit(x, mdn.to_u(th, LOW, HIGH), xv, mdn.to_u(thv, LOW, HIGH), epochs=500, lr=1e-3, patience=50, verbose=False)
        print(hid, K, "val", h[:, 1].min())
        if best is None or h[:, 1].min() < best[0]:
            best = (h[:, 1].min(), net)
    pickle.dump(best[1], open(os.path.join(RES, "mdn.pkl"), "wb"))


def net():
    return pickle.load(open(os.path.join(RES, "mdn.pkl"), "rb"))


def check(n_draw=2000):
    _, _, (tht, xt) = split()
    q = net()
    rng = np.random.default_rng(3)
    levels = np.linspace(0.05, 0.95, 19)
    cov = np.zeros((len(levels), 4)); mean = np.zeros_like(tht); sd = np.zeros_like(tht)
    for i in range(len(tht)):
        s = mdn.to_theta(q.sample(xt[i], n_draw, rng), LOW, HIGH)
        mean[i], sd[i] = s.mean(0), s.std(0)
        for j, a in enumerate(levels):
            lo, hi = np.quantile(s, [(1 - a) / 2, (1 + a) / 2], axis=0)
            cov[j] += (tht[i] >= lo) & (tht[i] <= hi)
    cov /= len(tht)
    prior_sd = (HIGH - LOW) / np.sqrt(12)
    out = dict(levels=levels.tolist(), coverage=cov.tolist(), shrinkage=(1 - (sd.mean(0) / prior_sd) ** 2).tolist())
    json.dump(out, open(os.path.join(RES, "check.json"), "w"), indent=2)
    np.savez(os.path.join(RES, "check_draws.npz"), true=tht, mean=mean, sd=sd)
    print("90% coverage", cov[np.argmin(np.abs(levels - 0.9))], "shrinkage", out["shrinkage"])


def observed_post(n_draw=20000, n_pred=60):
    q = net()
    rng = np.random.default_rng(5)
    xo = features(observed())[0]
    s = mdn.to_theta(q.sample(xo, n_draw, rng), LOW, HIGH)
    qs = np.quantile(s, [0.05, 0.5, 0.95], axis=0)
    out = {n: dict(median=float(qs[1, i]), lo90=float(qs[0, i]), hi90=float(qs[2, i])) for i, n in enumerate(NAMES)}
    out["corr"] = np.corrcoef(s.T).tolist()
    # predictive, separately for the two modes of the posterior: a sensitive reef (mean
    # tolerance below NOAA's 8 DHW) and a tolerant one
    sens = s[:, 0] < 8.0
    out["p_sensitive"] = float(sens.mean())
    preds, paths, mode = [], [], []
    for m, pool in ((1, s[sens]), (0, s[~sens])):
        for k in range(n_pred):
            counts, path, _ = simulate(pool[k], seed=700 + k + 1000 * m, keep=True)
            preds.append(counts); paths.append(path); mode.append(m)
    preds, mode = np.array(preds), np.array(mode)
    out["predictive_counts_mean"] = preds.mean(0).tolist()
    out["predictive_counts_sensitive"] = preds[mode == 1].mean(0).tolist()
    out["predictive_counts_tolerant"] = preds[mode == 0].mean(0).tolist()
    out["observed_counts"] = observed().tolist()
    # tolerance of the colonies: the 10th and 90th percentiles implied by each draw
    out["tolerance_p10_p90"] = dict(p10=np.quantile(s[:, 0] - 1.2816 * s[:, 1], [0.05, 0.5, 0.95]).tolist(),
                                     p90=np.quantile(s[:, 0] + 1.2816 * s[:, 1], [0.05, 0.5, 0.95]).tolist())
    json.dump(out, open(os.path.join(RES, "observed.json"), "w"), indent=2)
    np.savez(os.path.join(RES, "observed.npz"), post=s, preds=preds, paths=np.array(paths), mode=mode)
    print(json.dumps({k: out[k] for k in NAMES + ["p_sensitive", "predictive_counts_sensitive", "predictive_counts_tolerant", "observed_counts", "tolerance_p10_p90"]}, indent=1))


def figures():
    from scipy.stats import norm
    import seaborn as sns
    o = np.load(os.path.join(RES, "observed.npz"))
    info = json.load(open(os.path.join(RES, "observed.json")))
    s, p, mode = o["post"], o["paths"], o["mode"]
    sens = s[:, 0] < 8.0
    c = pd.read_csv(os.path.join(ROOT, "data", "crw_scott_ashmore_2015-2016.csv"), comment="#", parse_dates=["date"])
    dates, T = forcing()
    day = int(np.flatnonzero(dates == np.datetime64(SURVEY_DAY))[0])
    dhw_s = float(c.dhw[day])
    SEN, TOL, THR = COL[2], COL[0], "#8a8a8a"
    lab = {1: f"sensitive reef: mean tolerance < 8 DHW ({100 * sens.mean():.0f}% of posterior)",
           0: f"tolerant reef: mean tolerance > 8 DHW ({100 * (1 - sens.mean()):.0f}%)"}

    # fig8: heat stress, and the two reefs that fit the April surveys
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True, gridspec_kw=dict(hspace=0.12, height_ratios=[1, 1.3]))
    a1.plot(c.date, c.dhw, color=INK, lw=1.6)
    for v in (4, 8):
        a1.axhline(v, color=THR, lw=1, ls="--")
    a1.text(c.date.iloc[2], 4.5, "NOAA: bleaching (4)", fontsize=8, color=THR)
    a1.text(c.date.iloc[2], 8.5, "NOAA: mortality (8)", fontsize=8, color=THR)
    a1.set_ylabel("degree heating weeks")
    a1.set_title("Scott Reef, 2015-16: two reefs fit the April surveys, and they disagree about the summer", loc="left")
    t = c.date.values[:p.shape[1]]
    for m, colr in ((1, SEN), (0, TOL)):
        q = p[mode == m, :, 0]
        a2.fill_between(t, np.quantile(q, 0.1, 0), np.quantile(q, 0.9, 0), color=colr, alpha=0.18, lw=0)
        a2.plot(t, np.median(q, 0), color=colr, lw=2, label=lab[m])
    g = pd.read_csv(os.path.join(ROOT, "data", "gcbd_scott_reef_2016.csv"))
    gd = pd.to_datetime(dict(year=g.year, month=g.month, day=g.day))
    jit = np.random.default_rng(0).uniform(-0.4, 0.4, len(g))
    for dd, lo, hi, jj in zip(gd, g.bleached_min, g.bleached_max, jit):
        a2.plot([dd + pd.Timedelta(days=jj)] * 2, [lo / 100, hi / 100], color=INK, lw=0.8, alpha=0.6)
    a2.plot([], [], color=INK, lw=0.8, label="AIMS site surveys, 13-20 April 2016 (category range)")
    a2.set_ylabel("bleached share of living coral"); a2.set_ylim(0, 1.02)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=8.5, frameon=False)
    figstyle.save(fig, os.path.join(FIG, "fig8_scott_reef_2016"))
    plt.close(fig)

    # fig9: what the surveys say about coral tolerance
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(wspace=0.28))
    a1.hist2d(s[:, 0], s[:, 2], bins=60, range=[[LOW[0], HIGH[0]], [LOW[2], HIGH[2]]],
              cmap=sns.color_palette("crest", as_cmap=True), cmin=1)
    for v in (4, 8):
        a1.axvline(v, color=THR, lw=1, ls="--")
    a1.set_xlabel(LABELS[0]); a1.set_ylabel(LABELS[2])
    a1.set_title("(a) Posterior: tolerance trades off against sensitivity", loc="left")
    x = np.linspace(0, 22, 400)
    rng = np.random.default_rng(1)
    for m, colr in ((1, SEN), (0, TOL)):
        pool = s[sens] if m else s[~sens]
        for k in rng.choice(len(pool), 80, replace=False):
            a2.plot(x, norm.pdf(x, pool[k, 0], pool[k, 1]), color=colr, alpha=0.06, lw=1)
        med = np.median(pool, 0)
        a2.plot(x, norm.pdf(x, med[0], med[1]), color=colr, lw=2.2, label=lab[m].split(":")[0] + f" (median {med[0]:.1f} DHW)")
    for v in (4, 8):
        a2.axvline(v, color=THR, lw=1, ls="--")
    a2.axvline(dhw_s, color=INK, lw=1.2)
    ytop = a2.get_ylim()[1]
    a2.text(4.2, ytop * 0.96, "4", fontsize=8, color=THR, va="top"); a2.text(8.2, ytop * 0.96, "8", fontsize=8, color=THR, va="top")
    a2.text(dhw_s + 0.3, ytop * 0.96, f"survey\n{dhw_s:.1f} DHW", fontsize=8, color=INK, va="top")
    a2.set_xlabel("DHW at which a colony bleaches"); a2.set_ylabel("share of colonies (density)")
    a2.set_title("(b) Tolerance of Scott Reef's corals", loc="left")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5, frameon=False, ncol=1)
    figstyle.save(fig, os.path.join(FIG, "fig9_tolerance"))
    plt.close(fig)


if __name__ == "__main__":
    for w in sys.argv[1:]:
        {"observed": observed_post}.get(w, globals().get(w))()
