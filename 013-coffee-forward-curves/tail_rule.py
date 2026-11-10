#!/usr/bin/env python3
#-------- tail_rule.py --------------------------------------------------------------
#-----------------------------------------------------------------------------
# A tail rule for the importer: buy forward only when ECMWF's seasonal forecast puts an
# origin in its extreme tail (the forecast anomaly for the next three months above one
# standard deviation: hot or dry in Vietnam, dry in Brazil), otherwise wait. Weather moves
# coffee mainly through rare extremes, so this asks whether the forecast's value is in
# flagging them rather than in calling every quarter.
#
# For every month from 2008 to July 2026: the next three months' price change of each bean (what
# an importer who waited would pay extra), split by whether the forecast flagged the tail.
# The uncertainty comes from a stationary bootstrap (Politis and Romano 1994) over the
# whole monthly series: blocks of consecutive months, of random length with mean BLOCK,
# are drawn with replacement and each month keeps its flag and its return together. The
# three-month returns overlap, so neighbouring months are far from independent, and
# drawing single months (an iid bootstrap) would make the intervals too narrow.
#
#   python tail_rule.py
#
# Output: results/tail_rule.json, figures/fig17_tail_rule.png

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seasonal_backtest import build

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
THRESH = 1.0
BLOCK = 6                       # mean block length in months
N_BOOT = 4000
rng = np.random.default_rng(0)


def stationary_indices(n, block, rng):
    """One stationary-bootstrap resample of 0..n-1: start a new block at a random month
    with probability 1/block, otherwise carry on to the next month (wrapping round)."""
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    new = rng.random(n) < 1 / block
    starts = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if new[t] else (idx[t - 1] + 1) % n
    return idx


def block_ci(r, flag, block=BLOCK, n_boot=N_BOOT, rng=rng):
    """90% interval for the mean of r over flagged months. r and flag are aligned
    monthly arrays (months with no return already dropped); resampling is over time."""
    r, flag = np.asarray(r, float), np.asarray(flag, bool)
    means = []
    for _ in range(n_boot):
        i = stationary_indices(len(r), block, rng)
        f = flag[i]
        if f.sum():
            means.append(r[i][f].mean())
    return [float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))]


def main():
    df = build()["2008":"2026-07"]
    flags = {"Vietnam hot or dry": (df.vietnam_heat_fc > THRESH) | (df.vietnam_dry_fc > THRESH),
             "Brazil dry": df.brazil_dry_fc > THRESH}
    flags["both origins"] = flags["Vietnam hot or dry"] & flags["Brazil dry"]
    flags["either origin"] = flags["Vietnam hot or dry"] | flags["Brazil dry"]
    out = {}
    for bean in ["arabica", "robusta"]:
        r = df[f"{bean}_next3_raw"]
        ok = r.notna() & df.vietnam_heat_fc.notna() & df.brazil_dry_fc.notna()
        out[bean] = {"all_months": dict(n=int(ok.sum()), mean_change=float(r[ok].mean()))}
        for name, f in flags.items():
            m = ok & f
            if m.sum() < 3:
                continue
            x = r[m].values
            out[bean][name] = dict(n=int(m.sum()), mean_change=float(x.mean()), share_up=float((x > 0).mean()),
                                   ci90=block_ci(r[ok].values, f[ok].values),
                                   ci90_block3=block_ci(r[ok].values, f[ok].values, block=3),
                                   other_months=float(r[ok & ~f].mean()))
    json.dump(out, open(os.path.join(RES, "tail_rule.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, bean in zip(axes, ["arabica", "robusta"]):
        o = out[bean]
        names = ["all_months"] + [k for k in ["Brazil dry", "Vietnam hot or dry", "either origin", "both origins"] if k in o]
        labels = ["every month"] + [f"forecast: {k}" for k in names[1:]]
        y = np.arange(len(names))
        means = [100 * o[k]["mean_change"] for k in names]
        cols = [GREY] + [RED if "Vietnam" in k or "both" in k else BLUE for k in names[1:]]
        ax.axvline(means[0], color=GREY, lw=0.9, ls="--")
        for i, k in enumerate(names):
            if "ci90" in o[k]:
                lo, hi = (100 * v for v in o[k]["ci90"])
                ax.plot([lo, hi], [i, i], color=cols[i], lw=2, solid_capstyle="round")
            ax.plot(means[i], i, "o", color=cols[i], ms=8, mec="white", mew=1)
            if i:
                ax.text(max(hi, means[i]) + 0.5, i, f"{o[k]['n']} months", va="center", fontsize=8.5, color=GREY)
        ax.axvline(0, color="0.7", lw=0.8)
        ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_xlabel("price change over the next three months (%)")
        ax.set_title(bean.capitalize(), loc="left")
    fig.savefig(os.path.join(FIG, "fig17_tail_rule.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
