#!/usr/bin/env python3
#-------- tail_rule.py --------------------------------------------------------------
#-----------------------------------------------------------------------------
# A tail rule for the importer: buy forward only when ECMWF's seasonal forecast puts an
# origin in its extreme tail (the forecast anomaly for the next three months above one
# standard deviation: hot or dry in Vietnam, dry in Brazil), otherwise wait. Weather moves
# coffee mainly through rare extremes, so this asks whether the forecast's value is in
# flagging them rather than in calling every quarter.
#
# For every month from 2008 to 2025: the next three months' price change of each bean (what
# an importer who waited would pay extra), split by whether the forecast flagged the tail.
# A block bootstrap of the flagged months gives the uncertainty.
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
rng = np.random.default_rng(0)


def main():
    df = build()["2008":"2025-09"]
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
            boot = [rng.choice(x, len(x)).mean() for _ in range(4000)]
            out[bean][name] = dict(n=int(m.sum()), mean_change=float(x.mean()), share_up=float((x > 0).mean()),
                                   ci90=[float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
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
        ax.barh(y, means, color=[GREY] + [RED if "Vietnam" in k or "both" in k else BLUE for k in names[1:]], height=0.6)
        for i, k in enumerate(names[1:], 1):
            lo, hi = (100 * v for v in o[k]["ci90"])
            ax.plot([lo, hi], [i, i], color=NAVY, lw=1.5)
            ax.text(max(hi, means[i]) + 0.5, i, f"{o[k]['n']} months", va="center", fontsize=8.5, color=GREY)
        ax.axvline(0, color="0.7", lw=0.8)
        ax.set_yticks(y, labels); ax.invert_yaxis()
        ax.set_xlabel("price change over the next three months (%)")
        ax.set_title(bean.capitalize(), loc="left")
    fig.savefig(os.path.join(FIG, "fig17_tail_rule.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
