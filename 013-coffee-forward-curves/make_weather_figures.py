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
#   fig11_dependence    tail dependence (empirical and Gumbel copula), and the arabica-robusta
#                       correlation through time from DCC-GARCH, weather events shaded
#   fig12_bean_tail     arabica against robusta with joint spikes marked, and the Gumbel tail
#                       dependence between them on a rolling eight-year window
#   fig13_copula        the fitted Gumbel and Gaussian copulas of arabica against robusta as
#                       density rings in normal-score space, the months on top, joint spikes red
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
        ax.set_title(f"{bean.capitalize()} return (real removed) vs weather", loc="left")
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
        ax.set_title(f"{bean.capitalize()}: correlation with weather by lag", loc="left")
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
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    names = {"robusta|vietnam_heat": "Robusta, Vietnam heat", "arabica|vietnam_heat": "Arabica, Vietnam heat",
             "arabica|brazil_dry": "Arabica, Brazil dryness", "robusta|brazil_dry": "Robusta, Brazil dryness"}
    keys = list(names)
    y = np.arange(len(keys))
    ax.barh(y, [d["tail"][k]["tail"] for k in keys], color=[RED, RED, BLUE, BLUE], height=0.55, alpha=0.85,
            label="empirical, top 10%")
    lam = np.array([d["copula"][k]["tail_lambda"] for k in keys])
    lo = np.array([d["copula"][k]["tail_lambda_90"][0] for k in keys])
    hi = np.array([d["copula"][k]["tail_lambda_90"][1] for k in keys])
    ax.errorbar(lam, y, xerr=[lam - lo, hi - lam], fmt="D", color=NAVY, ms=5, capsize=3, label="Gumbel copula, 90% interval")
    ax.axvline(0.10, color=GREY, lw=1, ls=":", label="chance (empirical)")
    ax.set_yticks(y, [names[k] for k in keys]); ax.invert_yaxis()
    ax.set_xlabel("upper-tail dependence")
    ax.set_title("Tail dependence, weather vs price", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=3)
    ax = fig.add_subplot(gs[1])
    s = pd.read_csv(os.path.join(RES, "dcc.csv"), index_col=0, parse_dates=True)
    ax.plot(s.index, s.dcc, color=NAVY, lw=1.5)
    ax.axhline(s.dcc.mean(), color=GREY, lw=0.9, ls="--")
    shade(ax, label=False)
    ax.set_ylabel("correlation")
    ax.set_title("Arabica-robusta correlation (DCC-GARCH)", loc="left")
    fig.savefig(os.path.join(FIG, "fig11_dependence.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def bean_tail_figure():
    import json
    d = json.load(open(os.path.join(RES, "dependence.json")))["beans_tail"]
    r = pd.read_csv(os.path.join(RES, "bean_tail_rolling.csv"), parse_dates=["month"])
    df = pd.read_csv(os.path.join(RES, "weather_monthly.csv"), index_col=0, parse_dates=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1, 1.5], wspace=0.3))
    ax = axes[0]
    x, y = df.arabica_res, df.robusta_res
    both = (x >= x.quantile(0.9)) & (y >= y.quantile(0.9))
    ax.scatter(x, y, s=14, color=LIGHT, edgecolor="none")
    ax.scatter(x[both], y[both], s=40, color=RED, edgecolor="none", label="both in their top 10%")
    for t in x[both].index:
        if t.year in (2014, 2024):
            ax.annotate(t.strftime("%b %Y"), (x[t], y[t]), xytext=(4, 3), textcoords="offset points", fontsize=8.5, color=NAVY)
    ax.axvline(x.quantile(0.9), color=GREY, lw=0.8, ls=":"); ax.axhline(y.quantile(0.9), color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("arabica, log return (real removed)"); ax.set_ylabel("robusta, log return (real removed)")
    ax.set_title("Arabica vs robusta monthly returns", loc="left"); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax = axes[1]
    ax.plot(r.month, r.tail_lambda, color=NAVY, lw=1.6)
    ax.axhline(d["tail_lambda"], color=GREY, lw=0.9, ls="--")
    shade(ax)
    ax.set_ylabel("upper-tail dependence (Gumbel)")
    ax.set_title("Arabica-robusta tail dependence, rolling 8 years", loc="left")
    fig.savefig(os.path.join(FIG, "fig12_bean_tail.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def copula_figure():
    """Copula density rings in normal-score space, Gumbel against Gaussian, months on top."""
    import json
    from scipy.stats import norm, multivariate_normal, rankdata
    d = json.load(open(os.path.join(RES, "dependence.json")))["beans_tail"]
    df = pd.read_csv(os.path.join(RES, "weather_monthly.csv"), index_col=0, parse_dates=True)
    u = rankdata(df.arabica_res) / (len(df) + 1); v = rankdata(df.robusta_res) / (len(df) + 1)
    zu, zv = norm.ppf(u), norm.ppf(v)
    g = np.linspace(-3.2, 3.2, 241); X, Y = np.meshgrid(g, g)
    U, V = norm.cdf(X), norm.cdf(Y)
    th, rho = d["theta"], d["gaussian_rho"]

    def gumbel_density(U, V):
        a, b = -np.log(U), -np.log(V); sm = a ** th + b ** th; A = sm ** (1 / th)
        c = np.exp(-A + np.log(A + th - 1) + (1 - 2 * th) / th * np.log(sm) + (th - 1) * (np.log(a) + np.log(b)) + a + b)
        return c * norm.pdf(X) * norm.pdf(Y)

    gauss = multivariate_normal([0, 0], [[1, rho], [rho, 1]])
    dens = {"Gumbel copula": gumbel_density(U, V), "Gaussian copula": gauss.pdf(np.dstack([X, Y]))}
    q = 0.9
    p_both = {"Gumbel copula": 1 - 2 * q + np.exp(-(2 * (-np.log(q)) ** th) ** (1 / th)),
              "Gaussian copula": 1 - 2 * q + gauss.cdf([norm.ppf(q), norm.ppf(q)])}
    emp = float(((u > q) & (v > q)).mean())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    for ax, (name, Z) in zip(axes, dens.items()):
        ax.contourf(X, Y, Z, levels=12, cmap="Blues", alpha=0.9)
        ax.contour(X, Y, Z, levels=12, colors=NAVY, linewidths=0.5, alpha=0.6)
        k = norm.ppf(q)
        ax.add_patch(plt.Rectangle((k, k), 3.2 - k, 3.2 - k, color=RED, alpha=0.15, lw=0))
        both = (u > q) & (v > q)
        ax.scatter(zu[~both], zv[~both], s=9, color="white", edgecolor=NAVY, lw=0.4)
        ax.scatter(zu[both], zv[both], s=26, color=RED, edgecolor="white", lw=0.5)
        ax.text(0.03, 0.97, f"chance both are top 10%:\n  {name.split()[0]} {p_both[name]:.1%}\n"
                            f"  observed {emp:.1%}\n  independent 1.0%",
                transform=ax.transAxes, va="top", fontsize=9, color=NAVY,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal")
        ax.set_xlabel("arabica return, normal score")
        ax.set_title(f"{name}, arabica vs robusta", loc="left")
    axes[0].set_ylabel("robusta return, normal score")
    fig.savefig(os.path.join(FIG, "fig13_copula.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    dependence_figure()
    bean_tail_figure()
    copula_figure()
