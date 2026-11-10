#!/usr/bin/env python3
#-------- event_set_30y.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# The event set with thirty years of weather instead of one.
#
# event_set.py fitted a Gaussian copula to the daily output of 2023 alone, so it could
# only reshuffle that year's weather and never produce a whole cloudy month or year. Here
# the same fleet (25 regions, 1 MW each, ERA5 daily radiation x 0.8) is followed 1995-2024:
#
#   observed           the fleet's output in each calendar month of each of the 30 years;
#                      the 1-in-20 bad month is the 5th percentile of the 30 (interpolated,
#                      between the worst and second-worst year: rough, but real)
#   observed,          the same 30 years with each site's year drawn independently of the
#   sites independent  others (2,000 draws): what the fleet would see if its sites didn't
#                      share weather
#   copula, 30 years   the event_set.py simulation refitted to 30 years of daily anomalies
#                      (each site's own distribution, rank correlation, day-to-day
#                      persistence), 2,000 simulated years
#   copula, 2023       the original, for comparison (results/event_set.json)
#   t copula, 30 years the copula run again with a t copula in place of the Gaussian
#                      (copula_tails.py), which allows joint extremes at any threshold
#
# Shortfall = how far the 1-in-20 month falls below that month's mean, in %.
#
#   python event_set_30y.py
#
# Output: results/event_set_30y.json, figures/fig11_event_set_30y.png

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata, t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
YEARS = 2000
rng = np.random.default_rng(0)


def copula_sim(day, years, nu=None, R=None):
    month = day.index.month
    mean = day.groupby(month).transform("mean"); sd = day.groupby(month).transform("std")
    z = ((day - mean) / sd).fillna(0)
    g = norm.ppf(np.column_stack([rankdata(z[c]) / (len(z) + 1) for c in z]))
    R = np.corrcoef(g.T) if R is None else R; common = g.mean(axis=1)
    phi = float(np.corrcoef(common[:-1], common[1:])[0, 1])
    L = np.linalg.cholesky(R + 1e-9 * np.eye(len(R)))
    # one simulated calendar year: 365 days, each month's mean and sd, the anomaly
    # distribution of that calendar month, pooled over all years
    cal = pd.date_range("2023-01-01", "2023-12-31")
    mm = day.groupby(month).mean(); ss = day.groupby(month).std()
    mean_v, sd_v = mm.loc[cal.month].values, ss.loc[cal.month].values
    sorted_z = {m: np.sort(z[month == m].values, axis=0) for m in range(1, 13)}
    out = np.empty((years, 12))
    for y in range(years):
        e = rng.standard_normal((len(cal), z.shape[1]))
        ar = np.empty_like(e); ar[0] = e[0]
        for t in range(1, len(cal)):
            ar[t] = phi * ar[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
        x = ar @ L.T
        if nu is None:
            u = norm.cdf(x)
        else:                                   # t copula: one chi-squared mixing draw per day
            u = student_t.cdf(x / np.sqrt(rng.chisquare(nu, len(cal)) / nu)[:, None], nu)
        zz = np.empty_like(u)
        for m in range(1, 13):
            i = cal.month == m; sz = sorted_z[m]
            zz[i] = np.take_along_axis(sz, np.clip((u[i] * len(sz)).astype(int), 0, len(sz) - 1), axis=0)
        gen = np.clip(mean_v + sd_v * zz, 0, None).sum(axis=1)
        out[y] = pd.Series(gen, index=cal).groupby(cal.month).sum().values
    return out, phi, float(R[np.triu_indices(len(R), 1)].mean())


def shortfall(x, axis=0):
    m = x.mean(axis=axis)
    return (m - np.quantile(x, 0.05, axis=axis)) / m


def main():
    day = pd.read_csv(os.path.join(HERE, "data", "era5_daily_1995_2024.csv"), index_col=0, parse_dates=True).dropna()
    day = day[~((day.index.month == 2) & (day.index.day == 29))]
    sites = list(day.columns)
    site_month = day.groupby([day.index.year, day.index.month]).sum()           # (year, month) x site
    years = sorted(set(site_month.index.get_level_values(0)))
    obs = np.array([[site_month.loc[(y, m)].sum() for m in range(1, 13)] for y in years])     # years x 12
    ind = np.empty((YEARS, 12))
    for k in range(YEARS):                      # each site's year drawn on its own
        pick = rng.integers(0, len(years), len(sites))
        for m in range(1, 13):
            ind[k, m - 1] = sum(site_month.loc[(years[p], m), s] for p, s in zip(pick, sites))
    sim, phi, rho = copula_sim(day, YEARS)
    # the same with a t copula (copula_tails.py): R from Kendall's tau, nu by pseudo-likelihood
    from copula_tails import anomalies, pseudo_obs, fit_t
    Rt, nu, _ = fit_t(pseudo_obs(anomalies(day)))
    sim_t, _, _ = copula_sim(day, YEARS, nu=nu, R=Rt)
    old = json.load(open(os.path.join(HERE, "results", "event_set.json")))
    s23 = np.array([(old["monthly"][str(m)]["mean"] - old["monthly"][str(m)]["p5_correlated"]) / old["monthly"][str(m)]["mean"] for m in range(1, 13)])
    s_obs, s_ind, s_sim = shortfall(obs), shortfall(ind), shortfall(sim)
    annual = obs.sum(axis=1)
    out = dict(years=f"{years[0]}-{years[-1]}", persistence=phi, mean_pairwise_rank_corr=rho,
               shortfall_by_month={k: [round(float(v), 4) for v in a] for k, a in
                                   [("observed", s_obs), ("observed_sites_independent", s_ind), ("copula_30y", s_sim), ("copula_2023", s23)]},
               mean_shortfall=dict(observed=float(s_obs.mean()), observed_sites_independent=float(s_ind.mean()),
                                   copula_30y=float(s_sim.mean()), copula_2023=float(s23.mean())),
               worst_year=int(years[int(np.argmin(annual))]), worst_year_below_mean=float(1 - annual.min() / annual.mean()),
               annual_1in20_observed=float(1 - np.quantile(annual, 0.05) / annual.mean()),
               annual_1in20_copula_30y=float(1 - np.quantile(sim.sum(axis=1), 0.05) / sim.sum(axis=1).mean()),
               year_2023_rank_of_30=int((annual < annual[years.index(2023)]).sum() + 1),
               t_copula=dict(nu=nu, mean_shortfall=float(shortfall(sim_t).mean()),
                             shortfall_by_month=[round(float(v), 4) for v in shortfall(sim_t)],
                             worst_month_1in20=float(np.quantile(((sim_t.mean(0) - sim_t) / sim_t.mean(0)).max(axis=1), 0.95)),
                             annual_1in20=float(1 - np.quantile(sim_t.sum(axis=1), 0.05) / sim_t.sum(axis=1).mean())),
               worst_month_1in20_copula_30y=float(np.quantile(((sim.mean(0) - sim) / sim.mean(0)).max(axis=1), 0.95)))
    json.dump(out, open(os.path.join(HERE, "results", "event_set_30y.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "shortfall_by_month"}, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw=dict(wspace=0.25))
    ax = axes[0]; mths = np.arange(1, 13)
    ax.plot(mths, 100 * s_obs, "-o", color=NAVY, lw=1.9, ms=5, label="observed, 1995-2024")
    ax.plot(mths, 100 * s_sim, "-", color=RED, lw=1.7, label="copula fitted to 30 years")
    ax.plot(mths, 100 * s23, ":", color=RED, lw=1.5, label="copula fitted to 2023 only")
    ax.plot(mths, 100 * s_ind, "--", color=BLUE, lw=1.4, label="observed years, sites independent")
    ax.set_xticks(mths, list("JFMAMJJASOND")); ax.set_ylabel("1-in-20 bad month, % below the mean")
    ax.set_title("Fleet's 1-in-20 bad month by calendar month", loc="left"); ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    wc = ((sim.mean(0) - sim) / sim.mean(0)).max(axis=1); wo = ((obs.mean(0) - obs) / obs.mean(0)).max(axis=1)
    x = np.sort(wc)[::-1]; ax.plot((len(x) + 1) / np.arange(1, len(x) + 1), 100 * x, color=RED, lw=1.8, label="copula fitted to 30 years")
    x = np.sort(wo)[::-1]; ax.plot((len(x) + 1) / np.arange(1, len(x) + 1), 100 * x, "o", color=NAVY, ms=5, label="observed years")
    ax.set_xscale("log"); ax.set_xlim(1, 2000); ax.minorticks_off()
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], ["1", "2", "5", "10", "20", "50", "100", "200", "500", "1000"])
    ax.axvline(20, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("return period, years"); ax.set_ylabel("worst month of the year, % below its mean")
    ax.set_title("Worst month of the year against return period", loc="left"); ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(os.path.join(HERE, "figures", "fig11_event_set_30y.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
