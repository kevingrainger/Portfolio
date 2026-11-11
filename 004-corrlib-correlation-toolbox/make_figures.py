#-------- make_figures.py ---------------------------------------------------------------
# Redraws the README and website figures in the portfolio style, from the same steps as
# corrlib_showcase.ipynb. Stock prices come from data_stock_closes.csv, a local copy of
# the notebook's yfinance download (39 tickers with a full 2015-2024 record, not
# committed). The simulated panels use the library's own seeded simulators.
#
#   python make_figures.py

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, leaves_list

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import figstyle
from corrlib import Correlator, plotting
from corrlib import array_stacking as stack
from corrlib import correlation_measures as cm
from corrlib import random_matrix_cleaning as rmt
from corrlib import sampling_estimators as se

FIG = os.path.join(HERE, "figures")
SECTORS = {"Energy": ["XOM", "CVX", "COP", "SLB", "OXY"], "Banks": ["JPM", "BAC", "GS", "MS", "C"],
           "Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"], "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "UNH"],
           "Staples": ["PG", "KO", "PEP", "WMT", "COST"], "Industrials": ["CAT", "BA", "GE", "HON", "UNP"],
           "Utilities": ["NEE", "DUK", "SO", "D", "AEP"], "Property": ["AMT", "PLD", "SPG", "O"]}
SECTOR_COLOURS = dict(zip(SECTORS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]))


def load():
    raw = pd.read_csv(os.path.join(HERE, "data_stock_closes.csv"), index_col=0, parse_dates=True).dropna(axis=1, how="any")
    returns = np.log(raw).diff().dropna()
    return returns, returns.values.T, list(returns.columns)


def save(fig, name):
    figstyle.save(fig, os.path.join(FIG, name))
    plt.close(fig)


if __name__ == "__main__":
    figstyle.apply()
    returns, X, names = load()
    N, T = X.shape
    market = X.mean(axis=0)

    # fig1: four simulated pairs with similar Pearson correlations
    fig = cm.plot_same_number_different_reality(target=0.6)
    save(fig, "fig1_same_number_different_reality")

    # fig2: tail dependence, two banks against a bank and a utility
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for (a, b), colour in ((("JPM", "BAC"), "#000000"), (("JPM", "SO"), "#D55E00")):
        q, chi = cm.chi_curve(returns[a].values, returns[b].values)
        r = cm.pearson(np.vstack([returns[a].values, returns[b].values]))[0, 1]
        ax.plot(q, chi, "o-", ms=3, color=colour, label=f"{a} and {b} (correlation {r:.2f})")
    ax.plot(np.linspace(0.30, 0.01, 30), np.linspace(0.30, 0.01, 30), "--", color="#0072B2", lw=1, label="independence")
    ax.invert_xaxis()
    ax.set_xlabel("tail quantile (further into the tail to the right)")
    ax.set_ylabel("share of A's worst days that were B's worst days")
    ax.set_title("Tail dependence of daily returns, 2015-2024")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig2_tail_dependence")

    # fig3: the Epps effect on simulated ticks with a true correlation of 0.7
    noisy, _ = se.simulate_noisy_ticks(n=30000, true_correlation=0.7)
    fig = se.plot_epps_curve(noisy, true_correlation=0.7)
    fig.set_size_inches(8, 4)
    fig.tight_layout()
    save(fig, "fig3_epps_effect")

    # fig4: the full-sample eigenvalue spectrum against the noise band
    fig, ax = plt.subplots(figsize=(8, 4))
    rmt.plot_spectrum(cm.pearson(X), T, title=f"Eigenvalues of {N} stocks, 2015-2024", ax=ax)
    save(fig, "fig4_noise_band")

    # fig5 and hero: the Correlator, market regressed out
    c = Correlator(measure="spearman", remove=[market], clean="nonlinear_shrinkage")
    c.fit(X, labels=names)
    save(c.plot(), "fig5_correlator_summary")
    configs = {"Nothing removed": Correlator(measure="spearman", clean="nonlinear_shrinkage"),
               "Market regressed out": Correlator(measure="spearman", remove=[market], clean="nonlinear_shrinkage"),
               "Top mode removed": Correlator(measure="spearman", remove=1, clean="nonlinear_shrinkage")}
    mats = [k.fit(X, labels=names) for k in configs.values()]
    save(plotting.compare(mats, list(configs), labels=names), "hero")

    # fig6: stacking gain against the number of sensors (simulated array, as in the notebook)
    rng = np.random.default_rng(7)
    t = np.arange(2000); arrival = 900
    wave = np.sin(2 * np.pi * (t - arrival) / 60) * np.exp(-((t - arrival) / 150.0) ** 2); wave[t < arrival] = 0
    shared = np.cumsum(rng.normal(0, 0.25, 2000)); shared -= shared.mean()
    traces = np.array([wave + 1.8 * rng.normal(size=2000) + rng.uniform(0.3, 1.2) * shared for _ in range(12)])
    fig = stack.plot_gain_vs_sensors(traces, (0, 700), clean="nonlinear_shrinkage")
    fig.set_size_inches(8, 4)
    fig.tight_layout()
    save(fig, "fig6_stacking")

    # cover: Gaussian-rank correlation, market removed and cleaned, clustered, with sectors marked
    c = Correlator(measure="gaussian_rank", remove=[market], clean="nonlinear_shrinkage", effective_n="bartlett")
    C = c.fit(X, labels=names)
    L = plotting._linkage(C)
    order = leaves_list(L)
    sector_of = {s: k for k, v in SECTORS.items() for s in v}
    fig = plt.figure(figsize=(10, 5.3))
    ax_d = fig.add_axes([0.06, 0.80, 0.50, 0.17])
    dendrogram(L, ax=ax_d, color_threshold=0, above_threshold_color="0.35", no_labels=True)
    ax_d.axis("off")
    ax_s = fig.add_axes([0.06, 0.775, 0.50, 0.02])
    ax_s.imshow([[matplotlib.colors.to_rgb(SECTOR_COLOURS[sector_of[names[i]]]) for i in order]], aspect="auto")
    ax_s.axis("off")
    ax = fig.add_axes([0.06, 0.06, 0.50, 0.71])
    im = ax.imshow(C[np.ix_(order, order)], cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(range(N)); ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=6)
    ax.set_yticks(range(N)); ax.set_yticklabels([names[i] for i in order], fontsize=6)
    ax.grid(False)
    cb = fig.colorbar(im, cax=fig.add_axes([0.58, 0.06, 0.015, 0.71]))
    cb.set_label("correlation, market removed")
    fig.text(0.66, 0.90, f"{N} US stocks, daily returns 2015-2024", fontsize=13)
    fig.text(0.66, 0.84, "Market regressed out, cleaned by nonlinear shrinkage,\nordered by clustering", fontsize=9.5, color="0.3", va="top")
    fig.legend([Patch(color=v) for v in SECTOR_COLOURS.values()], list(SECTOR_COLOURS), loc="lower left",
               bbox_to_anchor=(0.66, 0.06), frameon=False, fontsize=9.5)
    fig.savefig(os.path.join(FIG, "cover.png"), dpi=200)
    fig.savefig(os.path.join(FIG, "cover.pdf"))
    plt.close(fig)
    # fig7: minimum-variance portfolio test, predicted against realised risk, one year in and one year out
    methods = {"raw": "raw", "clip": "clipped", "linear_shrinkage": "Ledoit-Wolf", "nonlinear_shrinkage": "nonlinear shrinkage"}
    res = {m: ([], []) for m in methods}
    for s0 in range(0, T - 500, 250):
        tr, te = X[:, s0:s0 + 250], X[:, s0 + 250:s0 + 500]
        Ctr = np.cov(tr)
        for m in methods:
            w = stack.min_variance(Ctr, clean=m, n_observations=250, X=tr)
            res[m][0].append(np.sqrt(w @ Ctr @ w * 252)); res[m][1].append((w @ te).std() * np.sqrt(252))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    y = np.arange(len(methods))[::-1]
    for yi, m in zip(y, methods):
        p, r = 100 * np.mean(res[m][0]), 100 * np.mean(res[m][1])
        ax.plot([p, r], [yi, yi], color="0.8", lw=2, zorder=1)
        ax.plot(p, yi, "o", color="#0072B2", ms=8, zorder=2, label="predicted" if yi == y[0] else None)
        ax.plot(r, yi, "o", color="#D55E00", ms=8, zorder=2, label="realised next year" if yi == y[0] else None)
        ax.text(r + 0.15, yi, f"x{r / p:.2f}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(list(methods.values()))
    ax.set_xlabel("annualised volatility of the minimum-variance portfolio (%)")
    ax.set_title("Predicted and realised risk by cleaner, 9 one-year windows")
    ax.set_xlim(9.5, 17); ax.legend(fontsize=8.5, loc="center left")
    fig.tight_layout()
    save(fig, "fig7_portfolio_test")
    # fig8: the spectrum from one year of data, where the noise band is much wider
    fig, ax = plt.subplots(figsize=(8, 3.6))
    rmt.plot_spectrum(cm.pearson(X[:, -250:]), 250, title=f"Eigenvalues of {N} stocks, 2024 only", ax=ax)
    save(fig, "fig8_one_year_spectrum")

    # fig9: the simulated array, one sensor against the optimal stack
    noise_levels = traces[:, :700].std(axis=1)
    single = traces[np.argsort(noise_levels)[len(noise_levels) // 2]]
    s_mvdr, _ = stack.mvdr(traces, clean="nonlinear_shrinkage", noise_window=(0, 700))
    fig, axes = plt.subplots(2, 1, figsize=(8, 3.6), sharex=True)
    for ax, (title, tr) in zip(axes, (("One sensor (median noise)", single), ("Optimal weights, cleaned covariance", s_mvdr))):
        ax.plot(t, tr, color="k", lw=0.6)
        ax.axvline(arrival, color="#D55E00", ls="--", lw=1)
        ax.set_title(f"{title}, noise reduction {stack.gain(tr, single, (0, 700)):.1f} dB", loc="left", fontsize=10)
        ax.set_yticks([])
    axes[1].set_xlabel("sample")
    fig.tight_layout()
    save(fig, "fig9_stacked_trace")
    print("effective rank:", {k: round(rmt.effective_rank(m), 2) for k, m in zip(configs, mats)})
