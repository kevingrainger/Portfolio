#!/usr/bin/env python3
#-------- make_figures.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# The three figures of 013 (run fetch_data.py and run_analysis.py first).
#
#   fig1_frost_2021   the July 2021 frost: what the forecasts said, what the price did
#   fig2_events       price paths around every frost night and every first warning
#   fig3_curve        how far along the curve a convenience-yield shock reaches

import json
import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figstyle                                                  # noqa: E402
from coffee_data import cold_actual, front_month, warning        # noqa: E402
from schwartz import loading                                     # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
GREEN, WINE, GREY, SAND = "#117733", "#882255", "0.35", "#DDCC77"


def fig1_frost_2021():
    a, b = "2021-06-15", "2021-08-10"
    W, _ = warning("gfs")
    C = cold_actual()
    P = front_month()[a:b]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.2]))
    ax = axes[0]
    ax.bar(W[a:b].index, W[a:b].values, width=0.8, color=SAND, label="frost warning in that day's forecast (next 7 days)")
    ax.plot(C[a:b].index, C[a:b].values, "o", ms=4, color=WINE, label="frost that night (ERA5)")
    ax.set_ylabel("cold excess below 5 C,\nproduction-weighted")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot(P.index, P.values, color=GREEN)
    ax.set_ylabel("arabica front month,\nUS cents/lb")
    for d, txt, xy in [("2021-07-16", "first warning", (-70, 18)), ("2021-07-20", "frost night", (-60, 30)),
                       ("2021-07-26", "harder frost forecast\nover the weekend", (-190, -8))]:
        j = P.index.searchsorted(pd.Timestamp(d))
        ax.annotate(txt, (P.index[j], P.iloc[j]), xytext=xy, textcoords="offset points", fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].set_title("July 2021: the forecasts warned, the price moved most once the damage was known", loc="left")
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig1_frost_2021"))


def fig2_events():
    ev = pd.read_csv(os.path.join(RES, "events.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, kind, col, title in zip(axes, ["frost night (ERA5)", "first warning (GFS)"], [WINE, GREEN],
                                    ["Aligned on the frost nights of 2000, 2019 and 2021", "Aligned on the first forecast warning, 2021-2026"]):
        sub = ev[ev.kind == kind]
        for e, g in sub.groupby("event"):
            ax.plot(g.k, 100 * g.cum, color=col, alpha=0.35, lw=1)
            if not kind.startswith("frost"):
                ax.annotate(e[:7], (g.k.iloc[-1], 100 * g.cum.iloc[-1]),
                            xytext=(3, 0), textcoords="offset points", fontsize=7.5, color="0.35", va="center")
        m = sub.groupby("k").cum.mean()
        ax.plot(m.index, 100 * m.values, color=col, lw=2.4, label="average")
        ax.axvline(0, color="0.6", lw=0.8, ls=":")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("trading days from the event")
        ax.legend(loc="upper left")
        ax.set_xlim(-10, 13)
    axes[0].set_ylabel("cumulative change in price, %")
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig2_events"))


def fig3_curve():
    s = json.load(open(os.path.join(RES, "schwartz.json")))
    p = s["no_weather"]
    st = pd.read_csv(os.path.join(RES, "schwartz_states.csv"), index_col=0, parse_dates=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw=dict(width_ratios=[1, 1.3]))
    tau = np.linspace(0.01, 4, 200)
    ax = axes[0]
    ax.plot(tau, loading(p["kappa"], tau) / tau, color=GREEN)
    ax.set_xlabel("contract maturity, years")
    ax.set_ylabel("share of a convenience-yield shock\npassed to the contract's price")
    ax.set_title(f"How far along the curve (kappa = {p['kappa']:.2f})", loc="left", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax = axes[1]
    ax.plot(st.index, 100 * st.delta, color=WINE)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylabel("convenience yield, % a year")
    ax.set_title("The convenience yield the curve implies, 2024-2026", loc="left", fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig3_curve"))


def fig4_timeline():
    """26 years of arabica with every frost night and every dry spell marked (run drivers.py first)."""
    P = front_month()["2000":]
    C = cold_actual()["2000":]
    m = pd.read_csv(os.path.join(RES, "drivers_monthly.csv"), index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(P.index, P.values, color=GREEN, lw=1.1)
    ax.set_yscale("log")
    ax.set_yticks([50, 100, 200, 400]); ax.set_yticklabels(["50", "100", "200", "400"])
    ax.set_ylabel("arabica front month, US cents/lb (log scale)")
    top = P.max() * 1.9
    rows = [(top * 1.00, "frost night (ERA5)", C[C >= 0.3].index, WINE),
            (top * 0.86, "Brazil dry (3-month rain, driest 10%)", m.index[m.brazil_dry > 1.28], SAND),
            (top * 0.74, "Vietnam dry (driest 10%)", m.index[m.vietnam_dry > 1.28], GREY)]
    for y, lab, days, col in rows:
        ax.scatter(days, np.full(len(days), y), marker="|", s=90, color=col, lw=2.2 if col != WINE else 2.8)
        ax.text(pd.Timestamp("2001-03-01"), y, lab, fontsize=8, color=col if col != SAND else "#8a7a2a", va="center")
    for d, txt, xy in [("2014-02-21", "2014: drought in Minas Gerais", (-175, 45)),
                       ("2021-07-23", "July 2021 frost", (-95, 62)),
                       ("2025-02-10", "2024-25: dry Brazil and Vietnam,\nthin stocks", (pd.Timestamp("2018-09-01"), 62))]:
        j = P.index.searchsorted(pd.Timestamp(d))
        ax.annotate(txt, (P.index[j], P.iloc[j]), xytext=xy, textcoords="data" if isinstance(xy[0], pd.Timestamp)
                    else "offset points", fontsize=8.5, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
    ax.set_ylim(P.min() * 0.85, top * 1.25)
    ax.set_title("Arabica since 2000: frost nights are rare, and the big moves mostly come without them", loc="left")
    figstyle.save(fig, os.path.join(FIG, "fig4_timeline"))


def fig5_curve_heatmap():
    """The shape of the forward curve each day, 2024-2026: each contract against the front month."""
    from coffee_data import DELIVERY
    F = pd.read_csv(os.path.join(HERE, "data", "kc_contracts.csv"), index_col=0, parse_dates=True)
    F = F[F.notna().sum(axis=1) >= 2]                                # days with at least two live contracts
    front = front_month().reindex(F.index)
    taus = np.linspace(0, 3.6, 73)
    grid = np.full((len(taus), len(F)), np.nan)
    for j, (d, row) in enumerate(F.iterrows()):
        pts = []
        for tic, v in row.dropna().items():
            code, yy = tic[2], int(tic[3:5])
            T = pd.Timestamp(2000 + yy, DELIVERY[code], 15)
            pts.append(((T - d).days / 365.25, np.log(v / front[d])))
        pts = sorted(p for p in pts if p[0] > 0)
        if len(pts) >= 2 and np.isfinite(front[d]):
            x, y = map(np.array, zip(*pts))
            ok = (taus >= x.min()) & (taus <= x.max())
            grid[ok, j] = np.interp(taus[ok], x, y)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    lim = np.nanpercentile(np.abs(grid), 98)
    im = ax.pcolormesh(F.index, taus, 100 * grid, cmap="PRGn", vmin=-100 * lim, vmax=100 * lim, shading="auto")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("contract price against the\nfront month, % (log)")
    ax.set_ylabel("years to delivery")
    ax.grid(False)
    ax.set_title("The forward curve, 2024-2026: purple is backwardation, deferred coffee cheaper than coffee now",
                 loc="left", fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    figstyle.save(fig, os.path.join(FIG, "fig5_curve_heatmap"))


def fig6_frost_calendar():
    """Every frost-season day since 1990, coloured by the production-weighted cold excess."""
    from matplotlib.colors import LinearSegmentedColormap
    C = cold_actual()
    C = C[(C.index.month >= 5) & (C.index.month <= 9)]
    tab = pd.DataFrame({"y": C.index.year, "d": C.index.dayofyear, "c": C.values}).pivot(index="y", columns="d", values="c")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    cmap = LinearSegmentedColormap.from_list("frost", ["#ffffff", "#e8c4d4", WINE, "#3d0f26"])
    im = ax.pcolormesh(tab.columns, tab.index, tab.values, cmap=cmap, vmin=0, vmax=1.2, shading="auto")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("cold excess below 5 C, production-weighted")
    starts = [pd.Timestamp(2001, mo, 1).dayofyear for mo in range(5, 10)]
    ax.set_xticks(starts, ["May", "Jun", "Jul", "Aug", "Sep"])
    ax.invert_yaxis(); ax.grid(False)
    ax.set_ylabel("year")
    ax.set_title("Frost in Brazil's coffee belt, 1990-2026: a handful of nights a decade", loc="left")
    figstyle.save(fig, os.path.join(FIG, "fig6_frost_calendar"))


def fig7_drivers():
    """How much of the monthly price movement each driver explains (Shapley shares of R-squared)."""
    s = json.load(open(os.path.join(RES, "drivers.json")))
    names = ["Brazilian real", "Vietnam dry and heat", "Brazil dry", "frost"]
    vals = [100 * s["shapley"][n] for n in names]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.barh(names[::-1], vals[::-1], color=[WINE, SAND, SAND, SAND][::-1], height=0.6)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.15, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("share of the variance of monthly returns explained, %")
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_title(f"What moves arabica month to month, {s['start'][:4]}-{s['end'][:4]}: "
                 f"all four explain {100 * s['r2_total']:.0f}%", loc="left", fontsize=10)
    figstyle.save(fig, os.path.join(FIG, "fig7_drivers"))


if __name__ == "__main__":
    figstyle.apply()
    fig1_frost_2021()
    fig2_events()
    fig3_curve()
    fig4_timeline()
    fig5_curve_heatmap()
    fig6_frost_calendar()
    fig7_drivers()
