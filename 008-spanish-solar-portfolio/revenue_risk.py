#!/usr/bin/env python3
#-------- revenue_risk.py -----------------------------------------------------------
#-----------------------------------------------------------------------------
# Correlated risk in money. Each of the 25 sites is given 1 MW of solar, its hourly output
# proxied by ERA5 surface irradiance (output = irradiance / 1000 W/m2 x 0.8 performance
# ratio), and sold at the OMIE Spanish day-ahead price of that hour, 2023.
#
#   capture price   what each site actually earns per MWh, against the average price of
#                   every hour (solar sells into its own glut: cannibalisation)
#   fleet risk      daily fleet output, and its bad days (the 5th percentile), against the
#                   same fleet with each site's days shuffled independently within each month: the bad days a
#                   fleet would have if sites didn't share weather. The gap is the correlated
#                   weather risk that summing site-level P95s misses. Revenue adds a second
#                   shared risk, the price, which is the same for every site (days of near-zero
#                   midday prices hit the whole fleet at once)
#
#   python revenue_risk.py      (in .venv-anemoi)
#
# Output: results/revenue_risk.json, figures/fig8_revenue_risk.png

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

NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
rng = np.random.default_rng(0)


def site_irradiance():
    from anemoi.datasets import open_dataset
    ds = open_dataset(ss.DATASET)
    lats, lons = np.asarray(ds.latitudes), np.asarray(ds.longitudes)
    idx = np.array([np.argmin(ss.haversine_km(la, lo, lats, lons)) for la, lo in ss.SITES.values()])
    v = ds.variables.index("ssrd")
    ssrd = np.asarray(ds[:, v, 0, :])
    irr, mid = ss.ssrd_to_irradiance(ssrd.T, np.asarray(ds.dates))
    return pd.DataFrame(np.nan_to_num(irr[idx]).T, index=pd.DatetimeIndex(mid), columns=list(ss.SITES))


def shuffle_within_months(df):
    """Each site's days shuffled independently, but only among days of the same month, so the
    season stays and only the shared day-to-day weather is broken."""
    months = pd.to_datetime(pd.Series(df.index)).dt.month.values
    out = np.empty(df.shape)
    for j, c in enumerate(df.columns):
        col = df[c].values.copy()
        for m in np.unique(months):
            i = np.flatnonzero(months == m)
            col[i] = col[rng.permutation(i)]
        out[:, j] = col
    return out


def main():
    irr = site_irradiance()
    utc = irr.index.floor("h")
    irr.index = (utc.tz_localize("UTC") if utc.tz is None else utc).tz_convert("Europe/Madrid")
    gen = (irr.clip(lower=0) / 1000 * 0.8).groupby(level=0).mean()          # MWh per hour per MW
    p = pd.read_csv(os.path.join(HERE, "prices", "data", "omie_es_hourly.csv"), parse_dates=["date"])
    p.index = (p.date + pd.to_timedelta(p.hour, unit="h")).dt.tz_localize("Europe/Madrid", ambiguous="NaT", nonexistent="NaT")
    p = p[p.index.notna()].price_es
    p = p[~p.index.duplicated()]
    d = gen.join(p.rename("price"), how="inner").dropna()
    sites = list(ss.SITES)
    base = d.price.mean()
    capture = {s: float((d[s] * d.price).sum() / d[s].sum()) for s in sites}
    rev = d[sites].multiply(d.price, axis=0).groupby(d.index.date).sum()     # euros a day per MW
    fleet = rev.sum(axis=1)
    p5 = float(np.quantile(fleet, 0.05))
    shuffled = []
    for _ in range(300):
        shuffled.append(np.quantile(shuffle_within_months(rev).sum(axis=1), 0.05))
    p5_ind = float(np.mean(shuffled))
    site_p5_sum = float(sum(np.quantile(rev[s], 0.05) for s in sites))
    egy = d[sites].groupby(d.index.date).sum()                                # MWh a day per MW, by site
    gen_day = egy.sum(axis=1)
    e_p5 = float(np.quantile(gen_day, 0.05))
    e_p5_ind = float(np.mean([np.quantile(shuffle_within_months(egy).sum(axis=1), 0.05) for _ in range(300)]))
    day_price = d.price.where(d.index.hour.isin(range(10, 18))).groupby(d.index.date).mean()
    price_day = d.price.groupby(d.index.date).mean()
    out = dict(hours=len(d), baseload_price=float(base), capture_price_mean=float(np.mean(list(capture.values()))),
               capture_ratio=float(np.mean(list(capture.values())) / base), capture_price=capture,
               fleet_daily_mean=float(fleet.mean()), fleet_p5=p5, fleet_p5_if_independent=p5_ind,
               sum_of_site_p5=site_p5_sum, energy_daily_mean=float(gen_day.mean()), energy_p5=e_p5,
               energy_p5_if_independent=e_p5_ind, days_midday_price_below_10=int((day_price < 10).sum()),
               corr_daily_output_price=float(np.corrcoef(gen_day, price_day.reindex(gen_day.index))[0, 1]))
    json.dump(out, open(os.path.join(HERE, "results", "revenue_risk.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "capture_price"}, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(wspace=0.3))
    ax = axes[0]
    hour = d.groupby(d.index.hour)
    ax.plot(hour.price.mean().index, hour.price.mean(), color=NAVY, lw=1.8, label="average price")
    ax2 = ax.twinx()
    ax2.fill_between(hour[sites].mean().index, 0, hour[sites].mean().mean(axis=1), color=LIGHT, alpha=0.6, lw=0, label="solar output")
    ax.set_xlabel("hour of the day (Spanish time)"); ax.set_ylabel("€ per MWh"); ax2.set_ylabel("output per MW")
    ax2.spines["right"].set_visible(True); ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_title("Price and solar output by hour, 2023", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left"); ax2.legend(frameon=False, fontsize=9, loc="upper right")
    ax = axes[1]
    sh = shuffle_within_months(egy).sum(axis=1)
    bins = np.linspace(0, max(gen_day.max(), sh.max()), 45)
    ax.hist(sh, bins=bins, color=LIGHT, alpha=0.8, label="sites independent (same season)")
    ax.hist(gen_day, bins=bins, color=RED, alpha=0.55, label="actual fleet")
    ax.axvline(e_p5_ind, color=BLUE, lw=1.5, ls="--"); ax.axvline(e_p5, color=RED, lw=1.5, ls="--")
    ax.set_xlabel("fleet output per day, MWh (25 sites x 1 MW)"); ax.set_ylabel("days")
    ax.set_title("Daily fleet output, 2023", loc="left"); ax.legend(frameon=False, fontsize=9)
    fig.savefig(os.path.join(HERE, "figures", "fig8_revenue_risk.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
