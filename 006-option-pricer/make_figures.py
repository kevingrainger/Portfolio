#-------- make_figures.py ---------------------------------------------------------------
# Redraws the website and README figures in the portfolio style. It runs the code cells
# of options_03_asian_softs.ipynb (all simulations seeded, prices from data/) and the
# convergence test of archive/options_01_solver_european.ipynb, then draws from the
# results. Nothing new is computed beyond what the notebooks compute.
#
#   python make_figures.py

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figstyle

FIG = os.path.join(HERE, "figures")
COL = {"coffee": "#6b3e26", "cocoa": "#c0392b", "sugar": "#2e86c1"}


def run_cells(path, upto=None, cwd=HERE):
    """Execute a notebook's code cells in one namespace, with plotting switched off."""
    nb = json.load(open(path, encoding="utf8"))
    ns = {}
    show, old = plt.show, os.getcwd()
    plt.show = lambda *a, **k: plt.close("all")
    os.chdir(cwd)
    sys.path.insert(0, cwd)
    sys.path.insert(0, os.path.join(HERE, "archive"))
    try:
        for i, c in enumerate(nb["cells"]):
            if upto is not None and i > upto:
                break
            if c["cell_type"] == "code":
                exec("".join(c["source"]), ns)
                plt.close("all")
    finally:
        plt.show = show
        os.chdir(old)
    return ns


def save(fig, name):
    figstyle.save(fig, os.path.join(FIG, name))
    plt.close(fig)


if __name__ == "__main__":
    g = run_cells(os.path.join(HERE, "options_03_asian_softs.ipynb"), upto=34)
    figstyle.apply()

    # fig8: seasonal volatility, dot and whisker with 95% intervals
    seasonal, se = g["seasonal"], g["seasonal_se"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, name in zip(axes, seasonal):
        m = np.arange(1, 13)
        ax.axhline(1, color="0.5", lw=0.8)
        ax.errorbar(m, seasonal[name], yerr=1.96 * se[name], fmt="o", color=COL[name], ms=5, capsize=2, lw=1)
        ax.set_xticks(m); ax.set_xticklabels(list("JFMAMJJASOND"))
        ax.set_title(name.capitalize(), loc="left")
    axes[0].set_ylabel("volatility relative to an average month")
    fig.tight_layout()
    save(fig, "fig8_seasonal_vol")

    # fig9: control variate convergence
    mc, cp, mm_price = g["mc"], g["checkpoints"], g["mm_price"]
    run = lambda x: np.array([x[:n].mean() for n in cp])
    band = lambda x: np.array([2 * x[:n].std() / np.sqrt(n) for n in cp])
    a, c = mc["arithmetic_payoffs"], mc["corrected_payoffs"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(cp, run(a) - band(a), run(a) + band(a), color="0.6", alpha=0.3)
    ax.plot(cp, run(a), color="0.35", lw=1.2, label="plain Monte Carlo, ±2 s.e.")
    ax.fill_between(cp, run(c) - band(c), run(c) + band(c), color="#D55E00", alpha=0.3)
    ax.plot(cp, run(c), color="#D55E00", lw=1.5, label="geometric control variate, ±2 s.e.")
    ax.axhline(mm_price, color="#0072B2", ls="--", lw=1, label="moment matched")
    ax.set_xscale("log")
    ax.set_xlabel("number of simulated paths"); ax.set_ylabel("Asian call price (US cents/lb)")
    ax.set_title("Coffee Asian call price against number of paths")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig9_control_variate")

    # hero: one path, futures price and the two deltas
    dates, path, K = g["dates"], g["path"], g["coffee"]["K"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    for ax in axes:
        ax.axvspan(g["AVG_START"], g["EXPIRY"], color="0.92", zorder=0)
    axes[0].plot(dates, path, "k-", lw=1)
    axes[0].axhline(K, color="#D55E00", ls="--", lw=1, label="strike")
    axes[0].set_ylabel("futures price (US cents/lb)"); axes[0].legend(loc="upper left", fontsize=8.5)
    axes[0].set_title("One simulated coffee path, with the averaging period shaded", loc="left")
    axes[1].plot(dates, g["euro_d"], color="0.5", lw=1.5, label="European")
    axes[1].plot(dates, g["asian_d"], color="#D55E00", lw=1.5, label="Asian")
    axes[1].set_ylabel("delta (futures per option)"); axes[1].legend(loc="upper left", fontsize=8.5)
    axes[1].set_title("Delta along the same path", loc="left")
    fig.tight_layout()
    save(fig, "hero")

    # fig10: extra hedging error from rebalancing weekly in one phase, dot plot
    ph = g["phases"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(ph))
    ax.plot(x, ph["Asian"], "o-", color="#D55E00", ms=7, label="Asian")
    ax.plot(x, ph["European"], "s-", color="0.45", ms=6, label="European")
    ax.axvline(2.5, color="k", ls="--", lw=1)
    ax.text(2.55, ax.get_ylim()[1] * 0.95, "averaging starts", fontsize=8.5, va="top")
    ax.set_xticks(x); ax.set_xticklabels(ph.index, fontsize=8.5)
    ax.set_ylabel("extra error variance / premium²")
    ax.set_title("Extra hedging error from rebalancing weekly in one phase")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig10_hedging_risk_timing")

    # fig11: hedging error against rebalancing frequency
    res, prem, freqs, n_steps = g["results"], g["premium"], g["frequencies"], g["n_steps"]
    n_reb = np.array([n_steps / f for f in freqs])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.linspace(-12, 12, 81)
    for kind, colour in (("European", "0.5"), ("Asian", "#D55E00")):
        e = res[(kind, 1)]
        axes[0].hist(e, bins=bins, color=colour, alpha=0.55, density=True, label=f"{kind}, sd {e.std():.2f}")
        sds = np.array([res[(kind, f)].std() for f in freqs])
        slope = np.polyfit(np.log(n_reb), np.log(sds), 1)[0]
        axes[1].loglog(n_reb, sds / prem[kind], "o-", color=colour, label=f"{kind}, slope {slope:.2f}")
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("hedging error at expiry (US cents/lb)"); axes[0].set_ylabel("density")
    axes[0].set_title("Hedging error, daily rebalancing", loc="left"); axes[0].legend(fontsize=8.5)
    axes[1].set_xlabel("rebalances over the option's life"); axes[1].set_ylabel("sd of error / premium")
    axes[1].set_title("Hedging error against rebalancing", loc="left"); axes[1].legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig11_hedging_error")

    # fig12: three volatility curves with the same total variance
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for (label, s), colour in zip(g["curves"].items(), ["0.5", "#D55E00", "#0072B2"]):
        ax.plot(dates[:-1], s, color=colour, lw=1.5, label=label)
    ax.axvspan(g["AVG_START"], g["EXPIRY"], color="0.92", zorder=0)
    ax.set_ylabel("volatility σ(t)")
    ax.set_title("Three coffee volatility curves with the same total variance")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig12_vol_curves")

    # fig7: twenty years of softs futures, rebased price and 3-month realised volatility
    prices, rets = g["prices"], g["returns"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for name in prices:
        axes[0].plot(prices[name] / prices[name].iloc[0] * 100, color=COL[name], lw=0.9, label=name)
        axes[1].plot(rets[name].rolling(63).std() * np.sqrt(252) * 100, color=COL[name], lw=0.9)
    axes[0].set_yscale("log"); axes[0].set_yticks([50, 100, 200, 400, 800])
    axes[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0].set_ylabel("price, 2005 = 100"); axes[0].legend(fontsize=8.5, loc="upper left")
    axes[0].set_title("ICE coffee, cocoa and sugar front-month futures, 2005-2025", loc="left")
    axes[1].set_ylabel("3-month realised vol (%)")
    fig.tight_layout()
    save(fig, "fig7_softs_history")

    # fig14: Asian and European price under the three volatility curves
    cmp_ = g["compare"]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    y = np.arange(len(cmp_))[::-1]
    ax.plot(cmp_["european"], y, "s", color="0.45", ms=7, label="European")
    ax.plot(cmp_["asian"], y, "o", color="#D55E00", ms=8, label="Asian")
    for yi, (lab, row) in zip(y, cmp_.iterrows()):
        ax.text(row["asian"] - 0.3, yi + 0.18, f"{row['asian']:.2f} ({100 * row['asian vs flat']:+.1f}%)", ha="right", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels(cmp_.index)
    ax.set_ylim(-0.6, len(cmp_) - 0.3); ax.set_xlim(29.5, 41)
    ax.set_xlabel("option price (US cents/lb)")
    ax.set_title("Coffee option prices under three volatility curves")
    ax.legend(fontsize=8.5, loc="center")
    fig.tight_layout()
    save(fig, "fig14_seasonality_price")

    # fig2: convergence of the finite-difference solver on a European option
    h = run_cells(os.path.join(HERE, "archive", "options_01_solver_european.ipynb"), upto=20)
    figstyle.apply()
    sizes, errors, slope, icpt = h["sizes"], h["errors"], h["slope"], h["intercept"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.loglog(sizes, errors, "o-", color="k", ms=5, label="measured error")
    ax.loglog(sizes, np.exp(icpt) * np.array(sizes, dtype=float) ** slope, "--", color="#D55E00", lw=1, label=f"fitted slope {slope:.2f}")
    ax.set_xlabel("grid points (space and time together)"); ax.set_ylabel("absolute error against the closed form")
    ax.set_title("Finite-difference error against grid size, European option")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig2_pde_convergence")
    print("done; convergence order", round(-slope, 2))

    # fig13: Večeř's PDE against Monte Carlo with a control variate (the checks in vecer.py)
    sys.path.insert(0, HERE)
    import vecer
    dt, n_obs = 1 / 252, 127
    fix = np.arange(64, 127)
    tt = np.arange(n_obs - 1) * dt
    cases = {"flat 35%": np.full(n_obs - 1, 0.35),
             "rising 25% to 45%": 0.25 + 0.20 * np.clip((tt - 0.15) / 0.3, 0, 1)}
    fig, ax = plt.subplots(figsize=(7, 3))
    for j, (name, sg) in enumerate(cases.items()):
        for i, k in enumerate((0.9, 1.0, 1.1)):
            disc = np.exp(-0.04 * 0.5)
            pde = vecer.vecer_call(100.0, 100 * k, sg, fix, n_obs, dt, discount=disc)
            mcp, mse = vecer.monte_carlo(100.0, 100 * k, sg, fix, n_obs, dt, discount=disc)
            y = i + 0.15 * (1 - 2 * j)
            ax.errorbar(pde - mcp, y, xerr=mse, fmt="o", color=["#0072B2", "#D55E00"][j], capsize=3,
                        label=name if i == 0 else None)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["K = 0.9 F₀", "K = F₀", "K = 1.1 F₀"])
    ax.set_xlabel("PDE price minus Monte Carlo price (±1 standard error)")
    ax.set_title("Večeř PDE against Monte Carlo, six Asian calls")
    ax.set_ylim(-0.5, 2.9)
    ax.legend(fontsize=8.5, loc="upper center", ncol=2)
    fig.tight_layout()
    save(fig, "fig13_vecer_check")
