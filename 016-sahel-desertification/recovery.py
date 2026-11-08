#!/usr/bin/env python3
#-------- recovery.py ---------------------------------------------------------------
#-----------------------------------------------------------------------------
# Did land use stop the Sahel coming back after the 1980s droughts?
#
# A tipping point shows in data as land that doesn't recover when the rain does. So for
# every vegetated grid site:
#
#   recovery        dry-season greenness (AVHRR, March-May) in 1995-2000 minus 1982-86,
#                   the drought years; relative to the site's 1982-86 level, in %
#   rain recovery   the same change in 3-year mean June-September rain (CHIRPS), in %
#   land use        the combined cropland, livestock and population index (covariates.py),
#                   plus its three parts
#
# Recovery is regressed on rain recovery, the land-use index and their product (does heavy
# land use blunt the response to returning rain?), with mean rain and soil as controls, all
# scaled to one standard deviation. Intervals from a block bootstrap over 4-degree bands of
# longitude. Shared satellite changes (NOAA-9 to NOAA-14 in 1995) shift every site alike and
# cancel in the comparison between sites. Then the same for the longer change, 1982-86 to
# 2015-21. Gourma and Fakara are marked.
#
#   python recovery.py
#
# Output: results/recovery.json, results/recovery_sites.csv, figures/fig14_recovery.png

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES, FIG = (os.path.join(HERE, d) for d in ("data", "results", "figures"))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
SITES = {"Gourma": (15.33, -1.55), "Fakara": (13.50, 2.70)}
rng = np.random.default_rng(0)


def fit(t, y, cols):
    X = t[cols].values; X = (X - X.mean(0)) / X.std(0)
    X = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X, t[y].values, rcond=None)
    pred = X @ b
    r2 = 1 - ((t[y] - pred) ** 2).sum() / ((t[y] - t[y].mean()) ** 2).sum()
    return b[1:], float(r2)


def analyse(t, y, label):
    t = t.copy()
    t["rain_x_land"] = ((t.rain_change - t.rain_change.mean()) / t.rain_change.std()) * ((t.land_use - t.land_use.mean()) / t.land_use.std())
    cols = ["rain_change", "land_use", "rain_x_land", "rain_mean", "sand"]
    b, r2 = fit(t, y, cols)
    r2_rain = fit(t, y, ["rain_change", "rain_mean", "sand"])[1]
    band = ((t.lon + 16) // 4).astype(int)
    boot = np.array([fit(pd.concat([t[band == k] for k in rng.choice(band.unique(), band.nunique())]), y, cols)[0] for _ in range(2000)])
    out = {c: dict(beta=float(b[i]), ci90=[float(np.quantile(boot[:, i], 0.05)), float(np.quantile(boot[:, i], 0.95))]) for i, c in enumerate(cols)}
    out["r2_full"], out["r2_rain_and_controls"] = r2, r2_rain
    out["spearman_land_use"] = float(spearmanr(t.land_use, t[y])[0])
    for part in ["crop", "log_tlu", "log_pop"]:
        out[f"spearman_{part}"] = float(spearmanr(t[part], t[y] - np.polyval(np.polyfit(t.rain_change, t[y], 1), t.rain_change))[0])
    return out


def main():
    t = pd.read_csv(os.path.join(RES, "covariates_sites.csv"))
    a = pd.read_csv(os.path.join(DATA, "sahel_avhrr_dry.csv")).pivot(index="year", columns="id", values="avhrr_dry")
    r = pd.read_csv(os.path.join(DATA, "sahel_chirps_jjas.csv")).pivot(index="year", columns="id", values="rain_jjas")
    p3 = r.rolling(3, min_periods=1).mean()
    t = t[t.id.isin(a.columns) & t.id.isin(r.columns)].copy()
    rows = []
    for i in t.id:
        base = a[i].loc[1982:1986].mean(); rb = p3[i].loc[1982:1986].mean()
        rows.append(dict(id=i, recovery=100 * (a[i].loc[1995:2000].mean() / base - 1),
                         long_change=100 * (a[i].loc[2015:2021].mean() / base - 1),
                         rain_change=100 * (p3[i].loc[1995:2000].mean() / rb - 1),
                         rain_change_long=100 * (p3[i].loc[2015:2021].mean() / rb - 1)))
    t = t.merge(pd.DataFrame(rows), on="id").dropna(subset=["recovery", "long_change"])
    out = {"sites": len(t), "recovery_1982_86_to_1995_2000": analyse(t, "recovery", "recovery")}
    tl = t.copy(); tl["rain_change"] = tl.rain_change_long
    out["change_1982_86_to_2015_21"] = analyse(tl, "long_change", "long")
    near = {}
    for nm, (la, lo) in SITES.items():
        k = ((t.lat - la) ** 2 + (t.lon - lo) ** 2).idxmin()
        near[nm] = dict(recovery=float(t.recovery[k]), rain_change=float(t.rain_change[k]), land_use=float(t.land_use[k]),
                        land_use_percentile=float((t.land_use < t.land_use[k]).mean() * 100))
    out["field_sites"] = near
    json.dump(out, open(os.path.join(RES, "recovery.json"), "w"), indent=1)
    t.to_csv(os.path.join(RES, "recovery_sites.csv"), index=False)
    print(json.dumps(out, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw=dict(wspace=0.55, width_ratios=[1.15, 1.15, 0.9]))
    ax = axes[0]
    sc = ax.scatter(t.rain_change, t.recovery, c=t.land_use, cmap="RdBu_r", vmin=-2.2, vmax=2.2, s=22, edgecolor="0.4", lw=0.3)
    for nm in SITES:
        k = ((t.lat - SITES[nm][0]) ** 2 + (t.lon - SITES[nm][1]) ** 2).idxmin()
        ax.plot(t.rain_change[k], t.recovery[k], "*", color="k", ms=13); ax.annotate(nm, (t.rain_change[k], t.recovery[k]), xytext=(5, 5), textcoords="offset points", fontsize=9.5, weight="bold")
    ax.axhline(0, color="0.7", lw=0.8)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045); cb.set_label("land-use index (sd)")
    ax.set_xlabel("rain, 1995-2000 against 1982-86 (%)"); ax.set_ylabel("dry-season greenness, same change (%)")
    ax.set_title("Recovery from the droughts against returning rain", loc="left", fontsize=10.5)
    ax = axes[1]
    t["resid"] = t.recovery - np.polyval(np.polyfit(t.rain_change, t.recovery, 1), t.rain_change)
    ax.scatter(t.land_use, t.resid, s=22, color=NAVY, alpha=0.55, lw=0)
    qs = np.quantile(t.land_use, np.linspace(0, 1, 9)); idx = np.clip(np.digitize(t.land_use, qs[1:-1]), 0, 7)
    ax.plot([t.land_use[idx == q].median() for q in range(8)], [t.resid[idx == q].median() for q in range(8)], "-o", color=RED, lw=1.8, ms=5, label="median of each eighth")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("land-use index (cropland, livestock, people; sd)"); ax.set_ylabel("recovery beyond what rain explains (points)")
    ax.set_title("Recovery left over after rain, against land use", loc="left", fontsize=10.5); ax.legend(frameon=False, fontsize=8.5)
    ax = axes[2]
    lab = {"rain_change": "returning rain", "land_use": "land use", "rain_x_land": "rain x land use", "rain_mean": "mean rain", "sand": "topsoil sand"}
    for j, (key, c, off) in enumerate([("recovery_1982_86_to_1995_2000", RED, -0.13), ("change_1982_86_to_2015_21", BLUE, 0.13)]):
        o = out[key]
        for k, cname in enumerate(lab):
            lo, hi = o[cname]["ci90"]
            ax.plot([lo, hi], [k + off, k + off], color=c, lw=2.2, solid_capstyle="round")
            ax.plot(o[cname]["beta"], k + off, "o", color=c, ms=7, mec="white", mew=1, label=("to 1995-2000" if j == 0 else "to 2015-21") if k == 0 else None)
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_yticks(range(len(lab)), list(lab.values())); ax.invert_yaxis()
    ax.set_xlabel("points of recovery per sd")
    ax.set_title("Effects, 90% interval", loc="left", fontsize=10.5); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.savefig(os.path.join(FIG, "fig14_recovery.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
