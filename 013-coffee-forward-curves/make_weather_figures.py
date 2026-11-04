#!/usr/bin/env python3
#-------- make_weather_figures.py ---------------------------------------------------
# Figures for the weather and coffee price analysis (weather_signal.py):
#   fig8_stacked        arabica and robusta, the real, Brazil's dryness, Vietnam's heat and
#                       dryness and El Nino on one time axis, with the two droughts and the
#                       2021 frost shaded through every panel
#   fig9_scatter        each month's weather in Brazil and Vietnam, coloured by the bean's
#                       move once the real is out; the largest moves outlined and labelled
#   fig10_lags          correlation of each hazard with each bean against lag, with the
#                       95% band from the effective sample size
#
#   python make_weather_figures.py

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
EVENTS = [("2013-12-01", "2014-04-30", "Brazil drought 2014"),
          ("2021-07-01", "2021-08-31", "Brazil frost 2021"),
          ("2023-10-01", "2024-06-30", "Vietnam heat and drought 2024")]

plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False,
                     "axes.spines.right": False})


def shade(ax, label=False):
    for k, (a, b, name) in enumerate(EVENTS):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=RED, alpha=0.12, lw=0)
        if label:
            ax.text(pd.Timestamp(a), 1.02 + 0.09 * (k % 2), name, transform=ax.get_xaxis_transform(),
                    fontsize=9, color=RED, ha="right" if k == 1 else "left")


def stacked(df, px):
    fig, axes = plt.subplots(5, 1, figsize=(12, 10.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.6, 0.8, 1, 1, 0.9], hspace=0.12))
    ax = axes[0]
    ax.plot(px.index, px.arabica, color=NAVY, lw=1.6, label="arabica")
    ax.plot(px.index, px.robusta, color=LIGHT, lw=1.6, label="robusta")
    ax.set_ylabel("US$ per kg"); ax.legend(frameon=False, loc="upper left"); shade(ax, label=True)
    ax = axes[1]
    ax.plot(df.index, df.brl.fillna(0).cumsum(), color=GREY, lw=1.2)
    ax.set_ylabel("reais per $\n(log, cumulative)"); shade(ax)
    ax = axes[2]
    ax.fill_between(df.index, 0, df.brazil_dry, where=df.brazil_dry > 0, color=BLUE, alpha=0.8, lw=0, label="drier than usual")
    ax.fill_between(df.index, 0, df.brazil_dry, where=df.brazil_dry <= 0, color=LIGHT, alpha=0.6, lw=0, label="wetter than usual")
    ax.set_ylabel("Brazil\ndryness (z)"); ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9); shade(ax)
    ax = axes[3]
    ax.plot(df.index, df.vietnam_heat, color=RED, lw=1.2, label="heat")
    ax.plot(df.index, df.vietnam_dry, color=BLUE, lw=1.2, label="dryness")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_ylabel("Vietnam (z)"); ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9); shade(ax)
    ax = axes[4]
    ax.fill_between(df.index, 0, df.enso, where=df.enso > 0, color=RED, alpha=0.55, lw=0)
    ax.fill_between(df.index, 0, df.enso, where=df.enso <= 0, color=BLUE, alpha=0.55, lw=0)
    ax.set_ylabel("El Nino\n(ONI, °C)"); shade(ax)
    fig.align_ylabels(axes)
    fig.savefig(os.path.join(FIG, "fig8_stacked.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def scatter(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, bean in zip(axes, ["arabica", "robusta"]):
        y = df[f"{bean}_res"]
        lim = np.percentile(np.abs(y), 98)
        norm = colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        sc = ax.scatter(df.vietnam_heat, df.brazil_dry, c=y, cmap="RdBu_r", norm=norm, s=22, edgecolor="none", alpha=0.85)
        big = y.abs().nlargest(8).index
        ax.scatter(df.loc[big, "vietnam_heat"], df.loc[big, "brazil_dry"], s=70, facecolor="none",
                   edgecolor=NAVY, lw=1.3)
        for t in big:
            ax.annotate(t.strftime("%b %Y"), (df.loc[t, "vietnam_heat"], df.loc[t, "brazil_dry"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8.5, color=NAVY)
        ax.axhline(0, color="0.85", lw=0.8); ax.axvline(0, color="0.85", lw=0.8)
        ax.set_xlabel("Vietnam heat, three months (z)")
        ax.set_title(bean.capitalize(), loc="left")
    axes[0].set_ylabel("Brazil dryness, three months (z)")
    fig.colorbar(sc, ax=axes, shrink=0.85, label="log return, real removed")
    fig.savefig(os.path.join(FIG, "fig9_scatter.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def lag_plot(lags):
    names = {"brazil_frost": "Brazil frost", "brazil_dry": "Brazil dryness", "vietnam_dry": "Vietnam dryness",
             "vietnam_heat": "Vietnam heat", "enso": "El Nino"}
    cols = {"brazil_frost": LIGHT, "brazil_dry": BLUE, "vietnam_dry": NAVY, "vietnam_heat": RED, "enso": GREY}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, bean in zip(axes, ["arabica", "robusta"]):
        d = lags[lags.bean == bean]
        band = d.band.mean()
        ax.axhspan(-band, band, color="0.92", lw=0, label="95% band (no relationship)")
        for h, g in d.groupby("hazard"):
            ax.plot(g.lag, g.r, marker="o", ms=4, color=cols[h], lw=1.5, label=names[h])
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("months the weather leads the price")
        ax.set_title(bean.capitalize(), loc="left")
    axes[0].set_ylabel("Gaussian-rank correlation")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.savefig(os.path.join(FIG, "fig10_lags.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(os.path.join(RES, "weather_monthly.csv"), index_col=0, parse_dates=True)
    from weather_signal import pink_sheet, full_panel
    px = pink_sheet()["2004-01":]
    stacked(full_panel(), px)       # every month, so gaps in the exchange-rate series don't bridge the weather
    scatter(df)
    lag_plot(pd.read_csv(os.path.join(RES, "lag_correlations.csv")))
    print("figures written")


if __name__ == "__main__":
    main()


def dependence_figure():
    import json
    d = json.load(open(os.path.join(RES, "dependence.json")))
    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.2, 1.2], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    names = {"robusta|vietnam_heat": "Robusta, Vietnam heat", "arabica|vietnam_heat": "Arabica, Vietnam heat",
             "arabica|brazil_dry": "Arabica, Brazil dryness", "robusta|brazil_dry": "Robusta, Brazil dryness"}
    keys = list(names)
    vals = [d["tail"][k]["tail"] for k in keys]
    ax.barh(range(len(keys)), vals, color=[RED, RED, BLUE, BLUE], height=0.6)
    ax.axvline(0.10, color=GREY, lw=1, ls=":", label="independent")
    ax.axvline(d["tail"][keys[0]]["null_95"], color=NAVY, lw=1, ls="--", label="95% of shuffles")
    ax.set_yticks(range(len(keys)), [names[k] for k in keys]); ax.invert_yaxis()
    ax.set_xlabel("share of extreme-weather months\nthat were top-10% price rises")
    ax.set_title("Tail dependence", loc="left"); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for i, (pair, title) in enumerate([("robusta_vietnam_heat", "Robusta and Vietnam heat"),
                                       ("arabica_brazil_dry", "Arabica and Brazil dryness")]):
        z = np.load(os.path.join(RES, f"coherence_{pair}.npz"))
        t = pd.to_datetime(z["time"]); P = z["periods"]
        ax = fig.add_subplot(gs[i + 1])
        m = ax.pcolormesh(t, P, z["R2"], cmap="Blues", vmin=0, vmax=1, shading="auto")
        ax.contour(t, P, z["R2"] / z["sig95"], levels=[1], colors=RED, linewidths=1.1)
        ax.fill_between(t, z["coi"], P.max(), color="white", alpha=0.55, lw=0)
        ax.set_yscale("log"); ax.set_ylim(P.min(), P.max()); ax.invert_yaxis()
        ax.set_yticks([6, 12, 24, 48, 96], ["6", "12", "24", "48", "96"])
        ax.set_ylabel("period (months)"); ax.set_title(title, loc="left")
        ax.spines[["top", "right"]].set_visible(True)
    fig.colorbar(m, ax=fig.axes[1:], shrink=0.85, label="wavelet coherence")
    fig.savefig(os.path.join(FIG, "fig11_dependence.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    dependence_figure()
