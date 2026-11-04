#!/usr/bin/env python3
#-------- event_chile_2026.py -------------------------------------------------------
#-----------------------------------------------------------------------------
# Event study: the central-Chile winter storm of 15-21 July 2026 (CRU), which halted
# Codelco's Andina surface operations and El Teniente's open pit and restricted shipping
# at central ports. Northern Chile was unaffected, so it is the control.
#
#   ports    IMIF PortWatch daily dry-bulk exports, seven-day mean, indexed to the six weeks
#            before the storm: central ports (San Antonio, Quintero/Ventanas) against
#            northern copper ports (Antofagasta, Mejillones, Caldera)
#   heat     Sentinel-2 hot pixels per clear pass at the central smelters (Caletones, which
#            smelts El Teniente's ore, and Chagres) against the northern ones (Chuquicamata,
#            Altonorte), before and after the storm, set against the same weeks of 2019-2025
#   price    COMEX copper, daily
#
# For each signal: when the fall shows, how deep it goes, and when it recovers.
#
#   python event_chile_2026.py
#
# Output: results/chile_2026.json, figures/fig5_chile_2026.png

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES, FIG = (os.path.join(HERE, d) for d in ("data", "results", "figures"))
STORM = ("2026-07-15", "2026-07-21")
PRE = ("2026-06-01", "2026-07-14")
POST = ("2026-07-22", "2026-09-15")
CENTRAL_PORTS = {"port1058": "San Antonio", "port90": "Quintero (Ventanas)"}
NORTH_PORTS = {"port56": "Antofagasta", "port728": "Mejillones", "port1031": "Caldera"}
CENTRAL_SMELTERS = ["Caletones (Codelco)", "Chagres (Anglo American)"]
NORTH_SMELTERS = ["Chuquicamata (Codelco)", "Altonorte (Glencore)"]
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"


def ports():
    d = pd.read_csv(os.path.join(DATA, "ports_daily.csv"), parse_dates=["date"])
    out = {}
    for group, ids in [("central", CENTRAL_PORTS), ("north", NORTH_PORTS)]:
        s = d[d.portid.isin(ids)].groupby("date").export_dry_bulk.sum().asfreq("D").fillna(0)
        s7 = s.rolling(7).mean()
        out[group] = s7 / s7[PRE[0]:PRE[1]].mean()
    return pd.DataFrame(out)


def heat():
    p = pd.read_csv(os.path.join(DATA, "smelter_passes.csv"), parse_dates=["date"])
    p = p[(p.clear >= 0.9) & p.error.isna()] if "error" in p else p[p.clear >= 0.9]
    rows = {}
    for group, sites in [("central", CENTRAL_SMELTERS), ("north", NORTH_SMELTERS)]:
        g = p[p.site.isin(sites)]
        def window(year, a, b):
            a, b = a.replace("2026", str(year)), b.replace("2026", str(year))
            return g[(g.date >= a) & (g.date <= b)].hot
        pre, post = window(2026, *PRE), window(2026, *POST)
        base = [(window(y, *POST).mean() / window(y, *PRE).mean()) for y in range(2019, 2026)
                if len(window(y, *PRE)) and len(window(y, *POST))]
        storm_gap = g[(g.date > "2026-07-01") & (g.date < "2026-09-01")].date.sort_values()
        gaps = storm_gap.diff().dt.days
        rows[group] = dict(pre_mean=float(pre.mean()), post_mean=float(post.mean()), ratio_2026=float(post.mean() / pre.mean()),
                           ratio_2019_2025=float(np.nanmean(base)), passes_pre=int(len(pre)), passes_post=int(len(post)),
                           longest_gap_days=int(gaps.max()) if len(gaps) > 1 else None,
                           last_clear_before=str(g[g.date < STORM[0]].date.max().date()),
                           first_clear_after=str(g[g.date > STORM[1]].date.min().date()))
    return rows, p


def main():
    os.makedirs(RES, exist_ok=True)
    pt = ports()
    c = pt.central["2026-07-01":"2026-09-30"]
    low = c[c < 0.5]
    first_low = str(low.index.min().date()) if len(low) else None
    recovered = c[(c.index > pd.Timestamp(STORM[1])) & (c >= 0.9)]
    hrows, passes = heat()
    hg = pd.read_csv(os.path.join(DATA, "hg_front.csv"), parse_dates=["Date"], index_col="Date").HG
    pr = lambda d: float(hg[:d].iloc[-1])
    out = dict(storm=STORM,
               ports=dict(central_min=float(c["2026-07-10":"2026-08-15"].min()),
                          central_min_date=str(c["2026-07-10":"2026-08-15"].idxmin().date()),
                          north_min=float(pt.north["2026-07-10":"2026-08-15"].min()),
                          first_day_below_half=first_low,
                          recovered_to_90pct=str(recovered.index.min().date()) if len(recovered) else None),
               heat=hrows,
               price=dict(before=pr("2026-07-14"), storm_end=pr(STORM[1]), two_weeks_after=pr("2026-08-04"),
                          storm_return=pr(STORM[1]) / pr("2026-07-14") - 1,
                          return_to_aug4=pr("2026-08-04") / pr("2026-07-14") - 1))
    json.dump(out, open(os.path.join(RES, "chile_2026.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    a, b = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-09-30")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.4), sharex=True, gridspec_kw=dict(hspace=0.15))
    for ax in axes:
        ax.axvspan(pd.Timestamp(STORM[0]), pd.Timestamp(STORM[1]), color=RED, alpha=0.15, lw=0)
    ax = axes[0]
    h = hg[a:b]; ax.plot(h.index, h, color=NAVY, lw=1.6)
    ax.set_ylabel("$ per lb"); ax.set_title("COMEX copper", loc="left")
    ax.text(pd.Timestamp(STORM[0]), 1.02, "central Chile storm, 15-21 July", transform=ax.get_xaxis_transform(), color=RED, fontsize=9)
    ax = axes[1]
    ax.plot(pt.index, pt.central, color=RED, lw=1.6, label="central (San Antonio, Quintero)")
    ax.plot(pt.index, pt.north, color=BLUE, lw=1.6, label="north (Antofagasta, Mejillones, Caldera)")
    ax.axhline(1, color="0.8", lw=0.8)
    ax.set_ylabel("dry-bulk exports\n(7-day mean, pre-storm = 1)"); ax.set_title("Port exports", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax = axes[2]
    for sites, col, lab in [(CENTRAL_SMELTERS, RED, "central (Caletones, Chagres)"), (NORTH_SMELTERS, BLUE, "north (Chuquicamata, Altonorte)")]:
        g = passes[passes.site.isin(sites) & (passes.date >= a) & (passes.date <= b)]
        for site, m in zip(sites, ["o", "s"]):
            gg = g[g.site == site]
            ax.plot(gg.date, gg.hot, m, color=col, ms=6, alpha=0.85, label=site.split(" (")[0])
    ax.set_ylabel("hot pixels per clear pass"); ax.set_title("Smelter heat (Sentinel-2)", loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper right")
    ax.set_xlim(a, b)
    fig.align_ylabels(axes)
    fig.savefig(os.path.join(FIG, "fig5_chile_2026.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
