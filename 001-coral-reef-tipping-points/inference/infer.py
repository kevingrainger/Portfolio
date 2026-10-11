#!/usr/bin/env python3
#-------- infer.py ----------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Train the posterior network on the simulation bank, check it on simulations whose
# answer is known, apply it to the observed Scott Reef bleaching record, and carry the
# posterior through to the temperature at which the reef tips.
#
#   python inference/infer.py train      fit the MDN (and the rejection baseline)
#   python inference/infer.py check      coverage and recovery on held-out simulations
#   python inference/infer.py observed   posterior for the real record, predictive check
#   python inference/infer.py tipping    tipping temperature for posterior draws
#   python inference/infer.py figures

import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))
import figstyle
import coral_model as cm
import mdn
from simulate_bank import CALIBRATED, HIGH, LABELS, LOW, NAMES, GRID, record, simulate, summarise

figstyle.apply()
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
COL = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#555555"]
INK = "#222222"
N_TEST = 1000
TICKS = [[0.5, 1.0], [0.5, 1.0, 1.5], [30.0, 30.5], [0.1, 0.2, 0.3]]


def features(x):
    """Bleaching in excess of the series' own lowest 10-day level. The observed record
    never falls below a 2% background that the model does not represent, so both
    simulated and observed curves are measured from their own floor."""
    x = np.atleast_2d(x)
    return x - x.min(1, keepdims=True)


def load_bank():
    b = np.load(os.path.join(HERE, "bank.npz"))
    theta, x = b["theta"], b["x"]
    ok = np.isfinite(x).all(1) & (np.abs(x).sum(1) > 0)
    return theta[ok], features(x[ok])


def split():
    theta, x = load_bank()
    n = len(theta)
    te = slice(n - N_TEST, n)
    va = slice(n - 2 * N_TEST, n - N_TEST)
    tr = slice(0, n - 2 * N_TEST)
    return (theta[tr], x[tr]), (theta[va], x[va]), (theta[te], x[te])


def observed_summary():
    _, obs = record()
    return features(summarise(obs))[0]


#-------- the rejection baseline: keep the nearest 2% of simulations ------------------
def rejection(x_obs, theta_bank, x_bank, keep=0.02):
    scale = x_bank.std(0) + 1e-9
    d = np.sqrt((((x_bank - x_obs) / scale) ** 2).sum(1))
    k = max(20, int(keep * len(d)))
    return theta_bank[np.argsort(d)[:k]]


def train():
    (th, x), (thv, xv), _ = split()
    net = mdn.MDN(x.shape[1], 4, hidden=(96, 96), K=8, seed=1)
    hist = net.fit(x, mdn.to_u(th, LOW, HIGH), xv, mdn.to_u(thv, LOW, HIGH), epochs=600, lr=1e-3, patience=50)
    pickle.dump(net, open(os.path.join(RES, "mdn.pkl"), "wb"))
    np.save(os.path.join(RES, "history.npy"), hist)
    print("best val", hist[:, 1].min(), "epochs", len(hist))


def load_net():
    return pickle.load(open(os.path.join(RES, "mdn.pkl"), "rb"))


def check(n_draw=2000, seed=3):
    """Coverage of central credible intervals, per parameter and level, for the network
    and for rejection; posterior means against true values."""
    (th, x), _, (tht, xt) = split()
    net = load_net()
    rng = np.random.default_rng(seed)
    levels = np.linspace(0.05, 0.95, 19)
    cov_net = np.zeros((len(levels), 4)); cov_rej = np.zeros((len(levels), 4))
    mean_net = np.zeros_like(tht); sd_net = np.zeros_like(tht); mean_rej = np.zeros_like(tht)
    for i in range(len(tht)):
        s = mdn.to_theta(net.sample(xt[i], n_draw, rng), LOW, HIGH)
        r = rejection(xt[i], th, x)
        mean_net[i], sd_net[i], mean_rej[i] = s.mean(0), s.std(0), r.mean(0)
        for j, a in enumerate(levels):
            lo, hi = np.quantile(s, [(1 - a) / 2, (1 + a) / 2], axis=0)
            cov_net[j] += (tht[i] >= lo) & (tht[i] <= hi)
            lo, hi = np.quantile(r, [(1 - a) / 2, (1 + a) / 2], axis=0)
            cov_rej[j] += (tht[i] >= lo) & (tht[i] <= hi)
    cov_net /= len(tht); cov_rej /= len(tht)
    prior_sd = (HIGH - LOW) / np.sqrt(12)
    out = dict(levels=levels.tolist(), coverage_net=cov_net.tolist(), coverage_rej=cov_rej.tolist(),
               rmse_net=np.sqrt(((mean_net - tht) ** 2).mean(0)).tolist(),
               rmse_rej=np.sqrt(((mean_rej - tht) ** 2).mean(0)).tolist(),
               shrinkage=(1 - (sd_net.mean(0) / prior_sd) ** 2).tolist(), names=NAMES)
    json.dump(out, open(os.path.join(RES, "check.json"), "w"), indent=2)
    np.savez(os.path.join(RES, "check_draws.npz"), true=tht, mean_net=mean_net, sd_net=sd_net, mean_rej=mean_rej)
    print(json.dumps({k: v for k, v in out.items() if k not in ("coverage_net", "coverage_rej", "levels")}, indent=1))
    i90 = int(np.argmin(np.abs(levels - 0.9)))
    print("90% coverage net", cov_net[i90], "rej", cov_rej[i90])


def observed(n_draw=20000, n_pred=40, seed=5):
    (th, x), _, _ = split()
    net = load_net()
    rng = np.random.default_rng(seed)
    xo = observed_summary()
    s = mdn.to_theta(net.sample(xo, n_draw, rng), LOW, HIGH)
    r = rejection(xo, th, x)
    temps, obs = record()
    # posterior predictive: daily bleached fraction for draws from the posterior, and at the hand-set values
    pred, hand, rej = [], [], []
    pick = np.random.default_rng(seed + 1).choice(len(r), n_pred, replace=False)
    for k in range(n_pred):
        J, sv, T0, b = r[pick[k]]
        _, bl, _ = cm.run_record(temps, seed=500 + k, grid=GRID, coupling_strength=J, temp_sensitivity=sv,
                                 bleaching_thresh=T0, beta=b)
        rej.append(bl)
        J, sv, T0, b = s[k]
        _, bl, _ = cm.run_record(temps, seed=500 + k, grid=GRID, coupling_strength=J, temp_sensitivity=sv,
                                 bleaching_thresh=T0, beta=b)
        pred.append(bl)
        _, bl, _ = cm.run_record(temps, seed=500 + k, grid=GRID)
        hand.append(bl)
    np.savez(os.path.join(RES, "observed.npz"), post=s, rej=r, pred=np.array(pred), hand=np.array(hand),
             pred_rej=np.array(rej), obs=obs, temps=temps)
    rm = lambda a: float(np.median(np.sqrt(((np.array(a) - obs) ** 2).mean(1))))
    q = np.quantile(s, [0.05, 0.5, 0.95], axis=0)
    out = {n: dict(median=float(q[1, i]), lo90=float(q[0, i]), hi90=float(q[2, i]), hand_set=float(CALIBRATED[i]))
           for i, n in enumerate(NAMES)}
    out["corr"] = np.corrcoef(s.T).tolist()
    out["predictive_rmse"] = dict(neural_posterior=rm(pred), rejection=rm(rej), hand_set=rm(hand))
    json.dump(out, open(os.path.join(RES, "observed.json"), "w"), indent=2)
    print(json.dumps({n: out[n] for n in NAMES}, indent=1))


#-------- how hot can it get? the tipping temperature for each posterior draw ---------
ANOMALIES = np.round(np.arange(0.0, 6.01, 0.25), 2)


def _tip(args):
    k, theta, seed = args
    J, s, T0, b = theta
    up, down = cm.hysteresis(ANOMALIES, sweeps=60, seed=seed, grid=GRID, coupling_strength=J,
                             temp_sensitivity=s, bleaching_thresh=T0, beta=b)
    base = T0 - 1.0
    # tipping point: first sustained temperature at which healthy cover falls below half its start
    i = np.flatnonzero(up < 0.5 * up[0])
    T_tip = base + ANOMALIES[i[0]] if len(i) else np.nan
    j = np.flatnonzero(down > 0.5 * up[0])            # recovery: warmest temperature at which cover is back above half
    T_back = base + ANOMALIES[j[-1]] if len(j) else np.nan
    return k, T_tip, T_back, up, down


def tipping(n=80, seed=9):
    post = np.load(os.path.join(RES, "observed.npz"))["post"]
    tasks = [(k, post[k], 900 + k) for k in range(n)] + [(-1, CALIBRATED, 899)]
    with Pool(2) as pool:
        res = pool.map(_tip, tasks)
    tip = np.array([r[1] for r in res if r[0] >= 0]); back = np.array([r[2] for r in res if r[0] >= 0])
    hand = [r for r in res if r[0] < 0][0]
    np.savez(os.path.join(RES, "tipping.npz"), tip=tip, back=back, hand_tip=hand[1], hand_back=hand[2],
             up=np.array([r[3] for r in res if r[0] >= 0]), down=np.array([r[4] for r in res if r[0] >= 0]),
             theta=post[:n], anomalies=ANOMALIES)
    print("tip", np.nanquantile(tip, [0.05, 0.5, 0.95]), "hand", hand[1], "back", np.nanquantile(back, [0.05, 0.5, 0.95]), "nan", np.isnan(tip).sum())


#-------- figures ------------------------------------------------------------------------
def figures():
    o = np.load(os.path.join(RES, "observed.npz"))
    s, r = o["post"], o["rej"]
    # fig5: the posterior, pairwise
    fig, axes = plt.subplots(4, 4, figsize=(8.6, 8.2), gridspec_kw=dict(wspace=0.08, hspace=0.08))
    import seaborn as sns
    cmap = sns.color_palette("crest", as_cmap=True)
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if j > i:
                ax.axis("off"); continue
            if i == j:
                ax.hist(s[:, i], bins=50, range=(LOW[i], HIGH[i]), density=True, color=COL[0], alpha=0.85)
                ax.hist(r[:, i], bins=20, range=(LOW[i], HIGH[i]), density=True, histtype="step", color=INK, lw=1.0)
                ax.axvline(CALIBRATED[i], color=COL[2], lw=1.5)
                ax.set_yticks([])
            else:
                ax.hist2d(s[:, j], s[:, i], bins=45, range=[[LOW[j], HIGH[j]], [LOW[i], HIGH[i]]], cmap=cmap, cmin=1)
                ax.plot(CALIBRATED[j], CALIBRATED[i], marker="*", color=COL[2], ms=11, mec="white", mew=0.8)
            ax.set_xlim(LOW[j], HIGH[j])
            ax.set_xticks(TICKS[j])
            if i == 3:
                ax.set_xlabel(LABELS[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(LABELS[i])
                ax.set_yticks(TICKS[i])
            elif j > 0 or i == 0:
                ax.set_yticklabels([]) if i != j else None
            ax.grid(False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COL[0], label="neural posterior (20,000 draws)"),
                        Line2D([], [], color=INK, lw=1, label="rejection, nearest 2% of the bank"),
                        Line2D([], [], color=COL[2], marker="*", lw=1.5, label="hand-set values of the original model")],
               loc="upper right", bbox_to_anchor=(0.92, 0.86))
    fig.suptitle("What the Scott Reef bleaching record says about the model's parameters", x=0.12, ha="left", y=0.93)
    figstyle.save(fig, os.path.join(FIG, "fig5_posterior"))
    plt.close(fig)

    # fig6: calibration and recovery
    c = json.load(open(os.path.join(RES, "check.json")))
    d = np.load(os.path.join(RES, "check_draws.npz"))
    lev = np.array(c["levels"])
    fig = plt.figure(figsize=(12, 3.9))
    gs = fig.add_gridspec(1, 5, wspace=0.42, width_ratios=[1.25, 1, 1, 1, 1])
    ax = fig.add_subplot(gs[0])
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.9, ls=(0, (2, 2)))
    ax.plot(lev, np.mean(c["coverage_net"], 1), "o-", ms=4, mfc="white", mew=1.2, color=COL[0], label="neural posterior")
    ax.plot(lev, np.mean(c["coverage_rej"], 1), "s--", ms=4, mfc="white", mew=1.2, color=INK, label="rejection")
    ax.set_xlabel("credible level"); ax.set_ylabel("share of true values inside")
    ax.set_title("(a) Calibration, 1,000 tests", loc="left"); ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for k in range(4):
        ax = fig.add_subplot(gs[k + 1])
        ax.scatter(d["true"][:, k], d["mean_net"][:, k], s=4, color=COL[0], alpha=0.4, lw=0)
        ax.plot([LOW[k], HIGH[k]], [LOW[k], HIGH[k]], color=INK, lw=0.8)
        ax.set_xlim(LOW[k], HIGH[k]); ax.set_ylim(LOW[k], HIGH[k])
        ax.set_title(f"({'bcde'[k]}) {LABELS[k]}", loc="left", fontsize=9.5)
        ax.set_xlabel("true"); ax.set_ylabel("recovered" if k == 0 else "")
        ax.text(0.04, 0.96, f"shrinkage {c['shrinkage'][k]:.2f}", transform=ax.transAxes, va="top", fontsize=8)
    figstyle.watermark(fig, "Held-out simulations")
    figstyle.save(fig, os.path.join(FIG, "fig6_inference_check"))
    plt.close(fig)

    # fig7: posterior predictive and the tipping temperature
    t = np.load(os.path.join(RES, "tipping.npz"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.1), gridspec_kw=dict(wspace=0.28, width_ratios=[1.2, 1]))
    days = np.arange(1, len(o["obs"]) + 1)
    for arr, colr, lab in ((o["hand"], COL[2], "hand-set parameters"), (o["pred"], COL[0], "posterior draws")):
        a1.fill_between(days, np.quantile(arr, 0.05, 0), np.quantile(arr, 0.95, 0), color=colr, alpha=0.22, lw=0)
        a1.plot(days, np.median(arr, 0), color=colr, label=f"{lab}, median and 90% band")
    a1.plot(days, o["obs"], color=INK, ls="--", label="observed")
    a1.set_xlabel("day"); a1.set_ylabel("fraction of reef bleached")
    a1.set_title("(a) Posterior predictive check", loc="left")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5)
    tip = t["tip"][np.isfinite(t["tip"])]
    a2.hist(tip, bins=np.arange(np.floor(tip.min() * 4) / 4 - 0.125, tip.max() + 0.375, 0.25), color=COL[0], alpha=0.85,
            edgecolor="white", label="posterior draws")
    a2.axvline(float(t["hand_tip"]), color=COL[2], lw=1.8, label="hand-set parameters")
    a2.axvline(float(o["temps"].max()), color=INK, lw=1.0, ls=(0, (2, 2)), label="hottest day in the record")
    a2.set_xlabel("sustained water temperature at which the reef tips (°C)"); a2.set_ylabel("posterior draws")
    a2.set_title("(b) How hot before it tips?", loc="left")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5)
    figstyle.save(fig, os.path.join(FIG, "fig7_tipping_temperature"))
    plt.close(fig)


if __name__ == "__main__":
    for w in sys.argv[1:]:
        globals()[w]()
