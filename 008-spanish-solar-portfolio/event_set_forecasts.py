#!/usr/bin/env python3
#-------- event_set_forecasts.py ----------------------------------------------------
#-----------------------------------------------------------------------------
# The two forecast models in the event set.
#
# First idea: treat each model's day-7 forecast as a day that could have happened (the
# UNSEEN approach, which stretches short records with forecast ensembles). It doesn't work
# at a week ahead: the forecasts are still too good. AIFS's day-7 fleet forecast follows
# what happened with a correlation of 0.72, so its days aren't new weather. (IFS days 6-7
# are left out altogether: in the archive part of each day is missing.) New weather needs
# extended-range or seasonal ensembles, weeks to months ahead.
#
# What the forecasts can give the event set now is their errors. A forecast error field is
# a realistic pattern of "the weather turned out differently from what was expected": all
# 25 sites on one day, with the spatial structure the model's mistakes really have. So each
# of the 30 years of ERA5 days (event_set_30y.py) is perturbed with an error field drawn
# from a scored day within a month of the same date of year, from IFS or AIFS, 3 or 5 days
# ahead, in units of each site's normal day:
#
#   perturbed site output = ERA5 output / normal  +  (forecast - ERA5) / normal on the drawn day
#
# Then the 1-in-20 bad day, for the fleet and for single sites, with and without the
# perturbations, over 20 draws of the whole 30 years.
#
#   python event_set_forecasts.py
#
# Output: results/event_set_forecasts.json, figures/fig12_event_set_forecasts.png

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import solar_sites as ss

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
DRAWS = 20
rng = np.random.default_rng(0)


def main():
    sites = list(ss.SITES)
    d = pd.read_csv(os.path.join(HERE, "aifs", "results", "scored_days.csv"), parse_dates=["date"])
    clim = d.pivot(index="date", columns="site", values="clim")[sites]
    era = d.pivot(index="date", columns="site", values="era5")[sites]
    skill = {}
    for m in ["aifs", "ifs"]:
        skill[m] = {}
        for L in range(1, 8):
            f = d.pivot(index="date", columns="site", values=f"lead{L}_{m}")[sites]
            fr, er = f.sum(axis=1) / clim.sum(axis=1), era.sum(axis=1) / clim.sum(axis=1)
            skill[m][L] = float(fr.corr(er))
    errors = {}
    for m in ["aifs", "ifs"]:
        for L in (3, 5):
            f = d.pivot(index="date", columns="site", values=f"lead{L}_{m}")[sites]
            e = ((f - era) / clim).dropna()
            errors[f"{m.upper()}, day {L}"] = e

    base = pd.read_csv(os.path.join(HERE, "data", "era5_daily_1995_2024.csv"), index_col=0, parse_dates=True)[sites].dropna()
    base = base[~((base.index.month == 2) & (base.index.day == 29))]
    normal = base.groupby(base.index.dayofyear).transform("mean").rolling(15, center=True, min_periods=1).mean()
    r = base / normal                                              # each site relative to its normal day
    doy = base.index.dayofyear.values

    def tails(rr):
        fleet = (rr * normal.values).sum(axis=1) / normal.values.sum(axis=1)
        return float(np.quantile(fleet, 0.05)), float(np.mean([np.quantile(rr[:, j], 0.05) for j in range(rr.shape[1])]))

    out = {"skill_fleet_corr_by_lead": skill}
    f0, s0 = tails(r.values)
    out["ERA5, 30 years"] = dict(fleet=f0, single_site=s0)
    for name, e in errors.items():
        edoy = e.index.dayofyear.values
        pools = {k: np.flatnonzero(np.minimum(abs(edoy - k), 365 - abs(edoy - k)) <= 30) for k in range(1, 367)}
        fl, si = [], []
        for _ in range(DRAWS):
            pick = np.array([rng.choice(pools[k]) for k in doy])
            rr = np.clip(r.values + e.values[pick], 0, None)
            a, b = tails(rr); fl.append(a); si.append(b)
        out[name] = dict(fleet=float(np.mean(fl)), fleet_range=[float(min(fl)), float(max(fl))],
                         single_site=float(np.mean(si)), single_site_range=[float(min(si)), float(max(si))],
                         error_sd_site=float(e.values.std()), error_sd_fleet=float(e.mean(axis=1).std()))
    json.dump(out, open(os.path.join(HERE, "results", "event_set_forecasts.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw=dict(wspace=0.55, width_ratios=[0.9, 1, 1]))
    ax = axes[0]
    for m, c, lab in [("aifs", RED, "AIFS (AI)"), ("ifs", BLUE, "IFS (physics)")]:
        Ls = list(range(1, 8)) if m == "aifs" else list(range(1, 6))
        ax.plot(Ls, [skill[m][L] for L in Ls], "-o", color=c, lw=1.7, ms=5, label=lab)
    ax.set_ylim(0, 1.02); ax.set_xlabel("days ahead"); ax.set_ylabel("correlation with what happened")
    ax.set_title("Fleet forecast against what happened", loc="left", fontsize=10.5); ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.text(0.98, 0.30, "IFS days 6-7: broken in the archive", transform=ax.transAxes, ha="right", fontsize=8, color="0.4")
    names = ["ERA5, 30 years"] + list(errors)
    cols = [NAVY] + [RED if n.startswith("AIFS") else BLUE for n in names[1:]]
    mk = ["o"] + ["o" if "day 3" in n else "s" for n in names[1:]]
    for ax, key, title in [(axes[1], "fleet", "The fleet's 1-in-20 bad day"), (axes[2], "single_site", "A single site's 1-in-20 bad day")]:
        for k, (n, c, m_) in enumerate(zip(names, cols, mk)):
            v = 100 * out[n][key]
            if f"{key}_range" in out[n]:
                lo, hi = (100 * x for x in out[n][f"{key}_range"])
                ax.plot([lo, hi], [k, k], color=c, lw=2.4, solid_capstyle="round")
            ax.plot(v, k, m_, color=c, ms=9, mec="white", mew=1.2)
        ax.axvline(100 * out["ERA5, 30 years"][key], color=GREY, lw=0.8, ls=":")
        ax.set_yticks(range(len(names)), ["ERA5 alone"] + [f"+ {n} errors" for n in names[1:]], fontsize=9); ax.invert_yaxis()
        ax.set_xlabel("% of a normal day (range over 20 draws)")
        ax.set_title(title, loc="left", fontsize=10.5)
    fig.savefig(os.path.join(HERE, "figures", "fig12_event_set_forecasts.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
