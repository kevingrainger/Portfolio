#!/usr/bin/env python3
#-------- event_set.py --------------------------------------------------------------
#-----------------------------------------------------------------------------
# A first stochastic event set for the fleet, and the P95 it implies.
#
# From one year of ERA5 (2023) the daily output of the 25 regions (1 MW each, irradiance x
# 0.8) is split into each site's seasonal mean and its daily anomaly. The anomalies' joint
# behaviour is captured by a Gaussian copula (each site's own anomaly distribution, joined
# by their rank correlation, with day-to-day persistence from an AR(1) of the common
# factor). 2,000 synthetic years are drawn twice: with the fitted correlation, and with the
# sites independent. For each, the fleet's monthly output and its 1-in-20 bad month (the
# monthly P95 shortfall) are read off. The gap is the risk a portfolio priced site by site
# would miss. One year is a short record, so this is the framework, not a calibrated set.
#
#   python event_set.py      (in .venv-anemoi)
#
# Output: results/event_set.json, figures/fig10_event_set.png

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import solar_sites as ss                                       # noqa: E402
from spread_sites import daily_output                          # noqa: E402

NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
YEARS = 2000
rng = np.random.default_rng(0)


def main():
    day, _, _ = daily_output(ss.SITES)
    day = day.loc[day.index.year == 2023]
    month = day.index.month
    mean = day.groupby(month).transform("mean")
    sd = day.groupby(month).transform("std").replace(0, np.nan)
    z = ((day - mean) / sd).fillna(0)
    u = np.column_stack([rankdata(z[c]) / (len(z) + 1) for c in z])
    g = norm.ppf(u)
    R = np.corrcoef(g.T)
    common = g.mean(axis=1)
    phi = float(np.corrcoef(common[:-1], common[1:])[0, 1])          # persistence of the shared weather
    L = np.linalg.cholesky(R + 1e-9 * np.eye(len(R)))
    sorted_z = np.sort(z.values, axis=0)
    n_days, n_sites = z.shape
    mean_v, sd_v = mean.values, sd.fillna(0).values

    def simulate(correlated):
        out = []
        for _ in range(YEARS):
            e = rng.standard_normal((n_days, n_sites))
            ar = np.empty_like(e); ar[0] = e[0]
            for t in range(1, n_days):                            # persistence day to day
                ar[t] = phi * ar[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
            x = ar @ L.T if correlated else ar
            uu = norm.cdf(x)
            zz = np.take_along_axis(sorted_z, np.clip((uu * n_days).astype(int), 0, n_days - 1), axis=0)
            gen = np.clip(mean_v + sd_v * zz, 0, None)
            out.append(pd.Series(gen.sum(axis=1), index=day.index).groupby(month).sum().values)
        return np.array(out)                                       # years x 12 months, MWh for the fleet

    cor, ind = simulate(True), simulate(False)
    actual = day.sum(axis=1).groupby(month).sum().values
    res = {}
    for m in range(12):
        res[int(m + 1)] = dict(mean=float(cor[:, m].mean()), p5_correlated=float(np.quantile(cor[:, m], 0.05)),
                               p5_independent=float(np.quantile(ind[:, m], 0.05)), actual_2023=float(actual[m]))
    short_c = np.array([(res[m]["mean"] - res[m]["p5_correlated"]) / res[m]["mean"] for m in res])
    short_i = np.array([(res[m]["mean"] - res[m]["p5_independent"]) / res[m]["mean"] for m in res])
    out = dict(years=YEARS, persistence=phi, mean_pairwise_rank_corr=float(R[np.triu_indices(len(R), 1)].mean()),
               monthly=res, p95_shortfall_correlated=float(short_c.mean()), p95_shortfall_independent=float(short_i.mean()),
               ratio=float(short_c.mean() / short_i.mean()))
    print(json.dumps({k: v for k, v in out.items() if k != "monthly"}, indent=1))

    mmean = cor.mean(axis=0)
    worst_c = ((mmean - cor) / mmean).max(axis=1)                 # each simulated year's worst month
    worst_i = ((mmean - ind) / mmean).max(axis=1)
    worst_2023 = float(((mmean - actual) / mmean).max())
    out.update(worst_month_1in20_correlated=float(np.quantile(worst_c, 0.95)),
               worst_month_1in20_independent=float(np.quantile(worst_i, 0.95)), worst_month_2023=worst_2023)
    json.dump(out, open(os.path.join(HERE, "results", "event_set.json"), "w"), indent=1)
    print({k: round(out[k], 3) for k in ("worst_month_1in20_correlated", "worst_month_1in20_independent", "worst_month_2023")})

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(wspace=0.25))
    ax = axes[0]
    mths = np.arange(1, 13)
    for lo, hi, a in [(0.05, 0.95, 0.18), (0.25, 0.75, 0.35)]:
        ax.fill_between(mths, np.quantile(cor, lo, axis=0), np.quantile(cor, hi, axis=0), color=RED, alpha=a, lw=0,
                        label=f"correlated sites, {int(lo*100)}-{int(hi*100)}%")
    for q in (0.05, 0.95):
        ax.plot(mths, np.quantile(ind, q, axis=0), color=BLUE, lw=1.1, ls="--", label="independent sites, 5% and 95%" if q == 0.05 else None)
    ax.plot(mths, actual, color=NAVY, lw=1.6, marker="o", ms=4, label="2023 (ERA5)")
    ax.set_xticks(mths, list("JFMAMJJASOND")); ax.set_ylabel("fleet output per month, MWh (25 x 1 MW)")
    ax.set_title("Monthly fleet output, 2,000 simulated years", loc="left"); ax.legend(frameon=False, fontsize=8.5, loc="lower center")
    ax = axes[1]
    for w, col, ls, lab in [(worst_c, RED, "-", "correlated sites"), (worst_i, BLUE, "--", "independent sites")]:
        x = np.sort(w)[::-1]
        rp = (len(x) + 1) / np.arange(1, len(x) + 1)
        ax.plot(rp, 100 * x, color=col, lw=1.8, ls=ls, label=lab)
    ax.axvline(20, color=GREY, lw=0.8, ls=":")
    ax.text(20, ax.get_ylim()[1] * 0.97, " 1 in 20 years", color=GREY, fontsize=9, va="top")
    ax.set_xscale("log"); ax.set_xlim(1, 2000); ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], ["1", "2", "5", "10", "20", "50", "100", "200", "500", "1000"])
    ax.set_xlabel("return period, years"); ax.set_ylabel("worst month of the year, % below its mean")
    ax.set_title("Worst month of the year against return period", loc="left"); ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(os.path.join(HERE, "figures", "fig10_event_set.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
