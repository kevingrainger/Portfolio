#!/usr/bin/env python3
#-------- score_forecasts.py -------------------------------------------------------
#-----------------------------------------------------------------------------
# Does an AI weather model forecast Spanish solar better than the physics model it is
# trained to imitate, and does a fleet of 25 sites help?
#
# Daily surface solar energy (MJ/m2) at the 25 sites, March 2025 onwards, forecast 1-7
# days ahead by ECMWF's AIFS Single (the AI model) and IFS (the physics model), scored
# against ERA5.
#
#   skill = 1 - MSE(forecast) / MSE(climatology)
# climatology being each site's ERA5 mean for that day of the year, 2000-2024 (+-15 days).
# Scored for one site at a time and for the fleet (the mean of the 25 sites, what a
# portfolio sells).
#
# Corrected forecasts: for each model, site and lead, a straight-line correction
# truth = a + b * forecast, fitted on the first 60% of days and scored on the last 40%
# (model output statistics, the simplest trained step).
#
# Shared weather. The day-3 forecast errors are correlated between every pair of sites
# and fitted against distance with rho(d) = exp(-(d/L)^p), as the weather itself was in
# solar_sites.py. Effective number of independent sites:
#   N_eff = N^2 / sum_ij rho_ij
#
#   python aifs/score_forecasts.py
#
# Output: aifs/results/skill.csv, aifs/results/error_reach.json

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from solar_sites import SITES                                    # noqa: E402

DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
LEADS = range(1, 8)
MODELS = ["aifs", "ifs"]


def load():
    t = pd.read_csv(os.path.join(DATA, "era5_daily.csv"), parse_dates=["date"])
    t["doy"] = t.date.dt.dayofyear
    hist = t[t.date.dt.year <= 2024]
    clim = {}
    for site, g in hist.groupby("site"):
        m = g.groupby("doy").era5.mean().reindex(range(1, 367)).interpolate()
        m = pd.concat([m.iloc[-15:], m, m.iloc[:15]]).rolling(31, center=True).mean().iloc[15:-15]
        clim[site] = m
    t["clim"] = [clim[s].loc[d] for s, d in zip(t.site, t.doy)]
    f = {m: pd.read_csv(os.path.join(DATA, f"forecast_{m}.csv"), parse_dates=["date"]) for m in MODELS}
    df = t.merge(f["aifs"], on=["site", "date"]).merge(f["ifs"], on=["site", "date"], suffixes=("_aifs", "_ifs"))
    return df.dropna()


def skill(err, err_clim):
    return 1 - np.mean(err ** 2) / np.mean(err_clim ** 2)


def haversine(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def reach(wide):
    """Fit rho(d) = exp(-(d/L)^p) to the pairwise correlations of the columns of `wide`."""
    C = wide.corr().values
    names = list(wide.columns)
    d, r = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d.append(haversine(SITES[names[i]], SITES[names[j]]))
            r.append(C[i, j])
    d, r = np.array(d), np.array(r)
    (L, p), _ = curve_fit(lambda x, L, p: np.exp(-(x / L) ** p), d, r, p0=[400, 1], bounds=([10, 0.2], [5000, 2]))
    n = len(names)
    return dict(L_km=L, p=p, n_eff=n ** 2 / C.sum(), pairs_d=d.tolist(), pairs_r=r.tolist())


def main():
    os.makedirs(RES, exist_ok=True)
    df = load()
    dates = np.sort(df.date.unique())
    split = dates[int(0.6 * len(dates))]
    rows = []
    for lead in LEADS:
        # straight-line correction per model, site and lead, fitted before `split`
        for m in MODELS:
            col = f"lead{lead}_{m}"
            df[f"{col}_corr"] = np.nan
            for site, g in df.groupby("site"):
                tr = g[g.date < split]
                b, a = np.polyfit(tr[col], tr.era5, 1)
                df.loc[g.index, f"{col}_corr"] = a + b * g[col]
        test = df[df.date >= split]
        fleet = test.groupby("date").mean(numeric_only=True)
        for m in MODELS:
            for kind, col in [("raw", f"lead{lead}_{m}"), ("corrected", f"lead{lead}_{m}_corr")]:
                rows.append(dict(model=m, kind=kind, lead=lead,
                                 skill_site=skill(test[col] - test.era5, test.clim - test.era5),
                                 skill_fleet=skill(fleet[col] - fleet.era5, fleet.clim - fleet.era5),
                                 rmse_site=np.sqrt(np.mean((test[col] - test.era5) ** 2)),
                                 rmse_fleet=np.sqrt(np.mean((fleet[col] - fleet.era5) ** 2))))
    sk = pd.DataFrame(rows)
    sk.to_csv(os.path.join(RES, "skill.csv"), index=False)
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(sk.pivot_table(index="lead", columns=["model", "kind"], values=["skill_site", "skill_fleet"]))

    # shared weather: how far do forecast errors reach, against the weather itself
    anomaly = df.assign(a=df.era5 - df.clim).pivot(index="date", columns="site", values="a")
    out = {"weather_anomaly": reach(anomaly)}
    for m in MODELS:
        err = df.assign(e=df[f"lead3_{m}"] - df.era5).pivot(index="date", columns="site", values="e")
        out[f"error_day3_{m}"] = reach(err)
    out["test_from"] = str(pd.Timestamp(split).date())
    out["days"] = int(len(dates))
    json.dump(out, open(os.path.join(RES, "error_reach.json"), "w"))
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"{k}: L = {v['L_km']:.0f} km, p = {v['p']:.2f}, N_eff = {v['n_eff']:.1f}")
    df.to_csv(os.path.join(RES, "scored_days.csv"), index=False)


if __name__ == "__main__":
    main()
