#!/usr/bin/env python3
#-------- make_figures.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# Figures for the AIFS against IFS evaluation (run fetch_openmeteo.py and
# score_forecasts.py first). Saved into the project's figures/ folder.
#
#   fig5_forecast_skill   skill against lead time, one site and the fleet
#   fig6_error_reach      how far forecast errors reach, against the weather itself
#   fig7_cloudy_week      the fleet's worst-forecast week, day 3

import json
import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import figstyle                                         # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(os.path.dirname(HERE), "figures")
AIFS, IFS, TRUTH = "#882255", "#117733", "0.15"


def fig5_skill():
    s = pd.read_csv(os.path.join(RES, "skill.csv"))
    s = s[s.kind == "raw"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, col, title in zip(axes, ["skill_site", "skill_fleet"], ["One site at a time", "The fleet of 25 sites"]):
        for m, c, lab in [("aifs", AIFS, "AIFS (AI)"), ("ifs", IFS, "IFS (physics)")]:
            g = s[s.model == m]
            ax.plot(g.lead, g[col], "o-", color=c, label=lab)
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_xlabel("days ahead")
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].set_ylabel("skill against climatology\n(1 = perfect, 0 = no better)")
    axes[0].legend(loc="lower left")
    start = pd.Timestamp(json.load(open(os.path.join(RES, "error_reach.json")))["test_from"])
    fig.suptitle(f"Daily solar energy at 25 Spanish sites, forecasts scored from {start:%d %B %Y}", x=0.01, ha="left")
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig5_forecast_skill"))


def fig6_reach():
    r = json.load(open(os.path.join(RES, "error_reach.json")))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.linspace(0, 1000, 200)
    for key, c, lab in [("weather_anomaly", TRUTH, "the weather (daily anomaly)"), ("error_day3_aifs", AIFS, "AIFS error, day 3"),
                        ("error_day3_ifs", IFS, "IFS error, day 3")]:
        v = r[key]
        ax.scatter(v["pairs_d"], v["pairs_r"], s=6, alpha=0.35, color=c, lw=0)
        ax.plot(x, np.exp(-(x / v["L_km"]) ** v["p"]), color=c, lw=2,
                label=f"{lab}: L = {v['L_km']:.0f} km, {v['n_eff']:.1f} independent sites")
    ax.set_xlabel("distance between sites, km")
    ax.set_ylabel("correlation")
    ax.set_ylim(-0.3, 1.05)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("Forecast errors are far more local than the weather", loc="left")
    figstyle.save(fig, os.path.join(FIG, "fig6_error_reach"))


def fig7_week():
    d = pd.read_csv(os.path.join(RES, "scored_days.csv"), parse_dates=["date"])
    fleet = d.groupby("date")[["era5", "clim", "lead3_aifs", "lead3_ifs"]].mean()
    test = fleet[fleet.index >= pd.Timestamp(json.load(open(os.path.join(RES, "error_reach.json")))["test_from"])]
    err = (test.lead3_aifs - test.era5).abs().rolling(7, center=True).mean()
    mid = err.idxmax()
    w = fleet.loc[mid - pd.Timedelta(days=10): mid + pd.Timedelta(days=10)]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(w.index, w.era5, "o-", color=TRUTH, lw=2, label="what happened (ERA5)")
    ax.plot(w.index, w.lead3_aifs, "s--", color=AIFS, label="AIFS, 3 days ahead")
    ax.plot(w.index, w.lead3_ifs, "^--", color=IFS, label="IFS, 3 days ahead")
    ax.plot(w.index, w.clim, ":", color="0.55", label="climatology")
    ax.set_ylabel("fleet mean solar energy, MJ/m2 a day")
    ax.legend(loc="upper center", ncol=2, fontsize=8.5)
    ax.set_ylim(None, ax.get_ylim()[1] + 3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title(f"Around the test period's worst AIFS miss for the fleet ({mid:%d %B %Y})", loc="left", fontsize=10)
    figstyle.save(fig, os.path.join(FIG, "fig7_cloudy_week"))


if __name__ == "__main__":
    figstyle.apply()
    fig5_skill()
    fig6_reach()
    fig7_week()
