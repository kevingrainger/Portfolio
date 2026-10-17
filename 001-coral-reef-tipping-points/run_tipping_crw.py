#!/usr/bin/env python3
#-------- run_tipping_crw.py -------------------------------------------------------
#-----------------------------------------------------------------------------
# The tipping temperature again, now with error bars from the real-data fit.
#
# The hysteresis experiment of the first version (coral_model.hysteresis) is rerun on the
# model fitted to Scott Reef (coral_crw.CRWReef), once for each of 40 parameter sets drawn
# from the posterior: 20 from the sensitive mode (mean tolerance mu_D < 8 DHW) and 20
# from the tolerant one.
#
# Each run holds the water at MMM + a for 120 days (long enough for the 84-day DHW window
# to fill: a sustained HotSpot a >= 1 C settles at DHW = 12 a), steps a up from 0 to
# 2.5 C in 0.1 C steps, then back down. Recorded at the end of each step: living coral
# (healthy + bleached) as a share of the reef's starting living coral.
#   tipping point   the first warming step at which living coral is below half
#   recovery point  the cooling step at which it is back above half (if ever)
#
#   python run_tipping_crw.py         # about 10 minutes
#
# Output: inference/results_crw/tipping.csv, figures/fig10_tipping_crw

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from coral_crw import MMM, CRWReef                       # noqa: E402
from coral_model import BLEACHED, HEALTHY                 # noqa: E402

LEVELS = np.round(np.arange(0, 2.51, 0.1), 2)
HOLD = 120
PER_MODE = 20


def living(reef):
    return np.mean(np.isin(reef.coral[reef.reef], (HEALTHY, BLEACHED)))


def run(theta, seed):
    mu, sd, s, b = theta
    reef = CRWReef(mu_D=mu, sigma_D=sd, s=s, beta=b, seed=seed)
    for _ in range(60):                                   # settle at a cool baseline
        reef.step(MMM - 1.0)
    start = living(reef)
    up, down = [], []
    for a in LEVELS:
        for _ in range(HOLD):
            reef.step(MMM + a)
        up.append(living(reef) / start)
    for a in LEVELS[::-1]:
        for _ in range(HOLD):
            reef.step(MMM + a)
        down.append(living(reef) / start)
    return np.array(up), np.array(down[::-1])


def main():
    post = np.load(os.path.join(HERE, "inference", "results_crw", "observed.npz"))["post"]
    rng = np.random.default_rng(7)
    sens = post[post[:, 0] < 8]
    tol = post[post[:, 0] >= 8]
    draws = [("sensitive", t) for t in sens[rng.choice(len(sens), PER_MODE, replace=False)]] + \
            [("tolerant", t) for t in tol[rng.choice(len(tol), PER_MODE, replace=False)]]
    rows = []
    for k, (mode, theta) in enumerate(draws):
        up, down = run(theta, seed=100 + k)
        tip = LEVELS[np.argmax(up < 0.5)] if (up < 0.5).any() else np.nan
        rec = LEVELS[np.argmax(down >= 0.5)] if (down >= 0.5).any() else np.nan
        for a, u, d in zip(LEVELS, up, down):
            rows.append(dict(run=k, mode=mode, mu_D=theta[0], sigma_D=theta[1], s=theta[2], beta=theta[3],
                             anomaly=a, up=u, down=d, tip=tip, recover=rec))
        print(f"{k:2d} {mode:9s} mu_D {theta[0]:5.1f}: tips at +{tip} C, recovers {'at +' + str(rec) + ' C' if rec == rec else 'never'}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "inference", "results_crw", "tipping.csv"), index=False)
    s = df.drop_duplicates("run").groupby("mode")[["tip", "recover"]].describe(percentiles=[0.1, 0.5, 0.9])
    print(s.round(2).T)
    plot(df)


def plot(df):
    import matplotlib.pyplot as plt
    import figstyle
    figstyle.apply()
    colours = {"sensitive": "#882255", "tolerant": "#117733"}
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for (mode, run), g in df.groupby(["mode", "run"]):
        ax.plot(g.anomaly, g.up, color=colours[mode], lw=0.8, alpha=0.5)
        ax.plot(g.anomaly, g.down, "--", color=colours[mode], lw=0.6, alpha=0.35)
    for mode, c in colours.items():
        t = df[df["mode"] == mode].drop_duplicates("run").tip
        lab = f"{mode}: +{t.quantile(0.1):.1f} to +{t.quantile(0.9):.1f} C"
        ax.plot([], [], color=c, lw=2, label=lab)
    ax.plot([], [], "--", color="0.4", label="cooling: no recovery")
    ax.axhline(0.5, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("sea temperature held above the warmest-month mean (C)")
    ax.set_ylabel("living coral, share of the start")
    ax.set_ylim(-0.03, 1.1)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.15), fontsize=8.5)
    ax.set_title(f"Scott Reef's tipping point, 40 posterior draws (warmest-month mean {MMM:.1f} C)", loc="left")
    figstyle.save(fig, os.path.join(HERE, "figures", "fig10_tipping_crw"))


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(pd.read_csv(os.path.join(HERE, "inference", "results_crw", "tipping.csv")))
    else:
        main()
