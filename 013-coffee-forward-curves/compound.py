#!/usr/bin/env python3
#-------- compound.py ---------------------------------------------------------------
#-----------------------------------------------------------------------------
# Compound weather events: months when both origins are hit at once. A month is a hit
# in Brazil if its three-month dryness is above one standard deviation or a damaging
# frost night falls in it; in Vietnam if its three-month heat or dryness is above one.
# Each month falls in one of four groups (neither, Brazil only, Vietnam only, both); the
# mean monthly move of each bean, with the real removed, is compared across groups, and
# the "both" group is tested against a block-shuffled null.
#
#   python compound.py
#
# Output: results/compound.json, figures/fig14_compound.png

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
GROUPS = ["neither", "Brazil only", "Vietnam only", "both"]
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
rng = np.random.default_rng(0)


def groups(df):
    br = (df.brazil_dry > 1) | (df.brazil_frost > 0.3)
    vn = (df.vietnam_heat > 1) | (df.vietnam_dry > 1)
    return pd.Series(np.select([br & vn, br & ~vn, ~br & vn], ["both", "Brazil only", "Vietnam only"], "neither"), index=df.index), br, vn


def block_null(y, mask, n=5000, block=12):
    """Mean of y over months like `mask`, with y shuffled in 12-month blocks."""
    out = np.empty(n)
    m = len(y)
    for k in range(n):
        starts = rng.integers(0, m - block, size=m // block + 1)
        ys = np.concatenate([y[s:s + block] for s in starts])[:m]
        out[k] = ys[mask].mean()
    return out


def main():
    df = pd.read_csv(os.path.join(RES, "weather_monthly.csv"), index_col=0, parse_dates=True)
    g, br, vn = groups(df)
    out = dict(p_brazil=float(br.mean()), p_vietnam=float(vn.mean()), p_both=float((br & vn).mean()),
               p_both_if_independent=float(br.mean() * vn.mean()),
               months_both=list(df.index[g == "both"].strftime("%Y-%m")), groups={})
    for bean in ["arabica", "robusta"]:
        y = df[f"{bean}_res"].values
        null = block_null(y, (g == "both").values)
        out["groups"][bean] = {k: dict(mean=float(df.loc[g == k, f"{bean}_res"].mean()), months=int((g == k).sum()))
                               for k in GROUPS}
        out["groups"][bean]["both_p_value"] = float((null >= out["groups"][bean]["both"]["mean"]).mean())
    json.dump(out, open(os.path.join(RES, "compound.json"), "w"), indent=1)

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1, 1.6], wspace=0.25))
    ax = axes[0]
    x = np.arange(len(GROUPS)); w = 0.2
    rs = np.random.default_rng(1)
    for i, (bean, c) in enumerate([("arabica", NAVY), ("robusta", BLUE)]):
        for j, k in enumerate(GROUPS):
            v = 100 * df.loc[g == k, f"{bean}_res"].dropna().values
            xx = j + (i - 0.5) * 2 * w + rs.uniform(-w / 2, w / 2, len(v))
            ax.scatter(xx, v, s=6, color=c, alpha=0.25, lw=0)
            bm = [rs.choice(v, len(v)).mean() for _ in range(2000)]
            ax.plot([j + (i - 0.5) * 2 * w] * 2, np.quantile(bm, [0.05, 0.95]), color=c, lw=2.2)
            ax.plot(j + (i - 0.5) * 2 * w, v.mean(), "o", color=c, ms=8, mec="white", mew=1.2, label=bean if j == 0 else None)
    ax.set_ylim(-20, 25)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(x, [f"{k}\n({out['groups']['arabica'][k]['months']} months)" for k in GROUPS], fontsize=9)
    ax.set_ylabel("monthly return, real removed (%)")
    ax.set_title("Bean returns by weather hits", loc="left"); ax.text(0.99, 0.02, "dots: single months; marker and bar: mean, 90% interval", transform=ax.transAxes, ha="right", fontsize=8, color=GREY); ax.legend(frameon=False)
    ax = axes[1]
    ax.fill_between(df.index, 0, br.astype(int), step="mid", color=BLUE, alpha=0.5, lw=0, label="Brazil hit")
    ax.fill_between(df.index, 0, -vn.astype(int), step="mid", color=RED, alpha=0.5, lw=0, label="Vietnam hit")
    both = df.index[g == "both"]
    ax.scatter(both, np.zeros(len(both)), color=NAVY, s=28, zorder=3, label="both at once")
    ax.set_yticks([-0.5, 0.5], ["Vietnam", "Brazil"]); ax.set_ylim(-1.3, 1.3)
    ax.set_title("Weather hits by origin, 2004-2026", loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0))
    fig.savefig(os.path.join(FIG, "fig14_compound.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
