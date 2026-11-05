#!/usr/bin/env python3
#-------- spread_sites.py -----------------------------------------------------------
#-----------------------------------------------------------------------------
# The 25 regions sit mostly in the dry south and centre, under much the same weather. How
# much does a fleet gain by adding sites in other weather regimes: the Atlantic north-west
# (A Coruna, Porto), the Cantabrian coast (Bilbao), the Catalan Mediterranean (Barcelona,
# Girona) and the Portuguese west (Lisbon)?
#
# For three fleets (the 25 regions; the 25 plus the six; and a 25-site fleet in which the
# six replace the six most clustered southern regions), each site has 1 MW and output
# proportional to ERA5 irradiance, 2023:
#   effective sites    (sum of eigenvalues)^2 / sum of squared eigenvalues of the
#                      correlation matrix of daily output anomalies (season removed)
#   bad days           the fleet's worst 5% of days, per MW, against the same sites made
#                      independent by shuffling each site's days within the month
#
#   python spread_sites.py      (in .venv-anemoi)
#
# Output: results/spread_sites.json, figures/fig9_spread_sites.png

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import solar_sites as ss                                       # noqa: E402
from revenue_risk import shuffle_within_months                 # noqa: E402
import json as _json                                           # noqa: E402

NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
NEW = {"A Coruna": (43.36, -8.41), "Porto": (41.15, -8.61), "Bilbao": (43.26, -2.93),
       "Barcelona": (41.39, 2.17), "Girona": (41.98, 2.82), "Lisbon": (38.72, -9.14)}


def daily_output(sites):
    from anemoi.datasets import open_dataset
    ds = open_dataset(ss.DATASET)
    lats, lons = np.asarray(ds.latitudes), np.asarray(ds.longitudes)
    idx = np.array([np.argmin(ss.haversine_km(la, lo, lats, lons)) for la, lo in sites.values()])
    v = ds.variables.index("ssrd")
    irr, mid = ss.ssrd_to_irradiance(np.asarray(ds[:, v, 0, :]).T, np.asarray(ds.dates))
    df = pd.DataFrame(np.nan_to_num(irr[idx]).clip(min=0).T / 1000 * 0.8, index=pd.DatetimeIndex(mid), columns=list(sites))
    return df.resample("D").sum(), lats[idx], lons[idx]


def eff_sites(day):
    a = day - day.groupby(day.index.month).transform("mean")
    lam = np.linalg.eigvalsh(np.corrcoef(a.values.T))
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def bad_days(day):
    fleet = day.sum(axis=1) / day.shape[1]
    ind = np.mean([np.quantile(shuffle_within_months(day).sum(axis=1) / day.shape[1], 0.05) for _ in range(300)])
    return float(np.quantile(fleet, 0.05)), float(ind), float(fleet.mean())


def main():
    allsites = {**ss.SITES, **NEW}
    day, lat, lon = daily_output(allsites)
    base = list(ss.SITES)
    south = ["Sevilla", "Cordoba", "Jaen", "Merida", "Puertollano", "Ciudad Real"]     # the most clustered
    fleets = {"25 southern and central regions": base,
              "the 25 plus six in other regimes": base + list(NEW),
              "25 sites, six moved to other regimes": [s for s in base if s not in south] + list(NEW)}
    out = {}
    for name, cols in fleets.items():
        p5, p5_ind, mean = bad_days(day[cols])
        out[name] = dict(sites=len(cols), effective_sites=eff_sites(day[cols]), worst5_per_mw=p5,
                         worst5_if_independent=p5_ind, mean_per_mw=mean, worst5_share_of_mean=p5 / mean)
    json.dump(out, open(os.path.join(HERE, "results", "spread_sites.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], hspace=0.55, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    lines = []
    for nm in ["ne_50m_coastline", "ne_50m_admin_0_boundary_lines_land"]:
        for feat in _json.load(open(os.path.join(HERE, "basemap", f"{nm}.geojson")))["features"]:
            g = feat["geometry"]
            for part in (g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]):
                part = np.asarray(part)
                if ((part[:, 0] > -11) & (part[:, 0] < 5) & (part[:, 1] > 35) & (part[:, 1] < 44.5)).any():
                    ax.plot(part[:, 0], part[:, 1], color="0.45", lw=0.6)
    n0 = len(base)
    ax.scatter(lon[:n0], lat[:n0], s=40, color=BLUE, edgecolor="k", lw=0.4, label="the 25 regions", zorder=3)
    ax.scatter(lon[n0:], lat[n0:], s=70, color=RED, marker="D", edgecolor="k", lw=0.4, label="six in other weather regimes", zorder=3)
    for nm, la, lo in zip(list(NEW), lat[n0:], lon[n0:]):
        ax.annotate(nm, (lo, la), xytext=(4, 4), textcoords="offset points", fontsize=8.5, color=RED)
    ax.set_xlim(-10.5, 4.5); ax.set_ylim(35.5, 44.3); ax.set_aspect(1 / np.cos(np.radians(40)))
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Sites", loc="left")
    short = ["25 regions", "25 + six spread", "25, six moved"]
    styles = [(NAVY, "-"), (BLUE, "--"), (RED, "-")]
    ax = fig.add_subplot(gs[0, 1])
    for (name, cols), lab, (c, ls) in zip(fleets.items(), short, styles):
        a = day[cols] - day[cols].groupby(day.index.month).transform("mean")
        lam = np.sort(np.linalg.eigvalsh(np.corrcoef(a.values.T)))[::-1]
        ax.plot(np.arange(1, len(lam) + 1), 100 * np.cumsum(lam) / lam.sum(), ls, color=c, lw=1.6, marker="o", ms=3,
                label=f"{lab}: {out[name]['effective_sites']:.1f} effective sites")
    ax.set_xlim(0.5, 10.5); ax.set_ylim(40, 101)
    ax.set_xlabel("number of weather patterns (principal components)"); ax.set_ylabel("% of daily variation")
    ax.set_title("Daily variation explained by shared weather patterns", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax = fig.add_subplot(gs[1, 1])
    for (name, cols), lab, (c, ls) in zip(fleets.items(), short, styles):
        f = day[cols].sum(axis=1) / len(cols)
        x = np.sort(f / f.mean())
        ax.plot(100 * x, 100 * np.arange(1, len(x) + 1) / len(x), ls, color=c, lw=1.6, label=lab)
    ax.axhline(5, color=GREY, lw=0.8, ls=":"); ax.text(2, 5.5, "worst 5% of days", color=GREY, fontsize=8.5)
    ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    ax.set_xlabel("fleet output on the day, % of an average day"); ax.set_ylabel("% of days at or below")
    ax.set_title("Low-output days, cumulative share", loc="left", fontsize=10.5)
    fig.savefig(os.path.join(HERE, "figures", "fig9_spread_sites.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
