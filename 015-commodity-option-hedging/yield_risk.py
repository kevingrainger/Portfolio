#!/usr/bin/env python3
#-------- yield_risk.py -------------------------------------------------------------
#-----------------------------------------------------------------------------
# Yield risk for the wheat producer of wheat_2010.py. Revenue is price times harvest:
#   R = Q F_T + h (F_0 - F_T),     Q = expected harvest x (1 + yield shock)
# with h the share of the expected harvest sold forward. If the harvest falls in the same
# drought that lifts the price (correlation rho < 0 between yield and price), the crop
# partly hedges itself and a full futures hedge over-hedges. For several rho, as seen on
# 3 May 2010 (same price volatility, yield volatility 15% a year), the hedge ratio that
# minimises the spread of revenue is found by Monte Carlo.
#
#   rho = 0      an Italian producer whose harvest doesn't depend on Russia's weather
#   rho < 0      a producer inside the drought, like a Russian farm in 2010
#
#   python yield_risk.py
#
# Output: results/yield_risk.json, figures/fig8_yield_risk.png

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
SIG_Q = 0.15


def main():
    w = json.load(open(os.path.join(RES, "wheat_2010.json")))
    F0, sig, tau = w["F0"], w["sigma"], w["tau_years"]
    rng = np.random.default_rng(0)
    n = 200000
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    FT = F0 * np.exp(sig * np.sqrt(tau) * z1 - 0.5 * sig ** 2 * tau)
    H = np.linspace(0, 1.2, 61)
    out, curves = {}, {}
    for rho in [0.0, -0.3, -0.6]:
        zq = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        Q = 1 + SIG_Q * zq
        sd = np.array([np.std(Q * FT + h * (F0 - FT)) for h in H])
        h_opt = float(H[sd.argmin()])
        curves[rho] = sd
        out[str(rho)] = dict(optimal_hedge_ratio=h_opt, sd_unhedged=float(sd[0]), sd_full_hedge=float(sd[np.argmin(abs(H - 1))]),
                             sd_optimal=float(sd.min()))
    # 2010 in hindsight: an Italian producer with a normal harvest, a Russian-style producer
    # whose harvest fell a third as the price rose 39%
    FT10 = w["FT"]
    hind = {}
    for name, q in [("Italian producer, normal harvest", 1.0), ("producer in the drought, harvest down a third", 0.67)]:
        hind[name] = {f"{h:.0%} sold forward": float(q * FT10 + h * (F0 - FT10)) for h in [0.0, 0.5, 1.0]}
    out["hindsight_2010_eur_per_expected_tonne"] = hind
    json.dump(out, open(os.path.join(RES, "yield_risk.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw=dict(wspace=0.3))
    ax = axes[0]
    for rho, c in [(0.0, NAVY), (-0.3, BLUE), (-0.6, RED)]:
        ax.plot(100 * H, curves[rho], color=c, lw=1.8, label=f"yield-price correlation {rho:+.1f}")
        ax.plot(100 * out[str(rho)]["optimal_hedge_ratio"], curves[rho].min(), "o", color=c, ms=6)
    ax.set_xlabel("share of expected harvest sold forward (%)"); ax.set_ylabel("spread of revenue, € per expected tonne")
    ax.set_title("Revenue risk against hedge ratio", loc="left"); ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    names = list(hind); x = np.arange(3); wbar = 0.38
    for i, (name, c) in enumerate(zip(names, [NAVY, RED])):
        vals = list(hind[name].values())
        ax.bar(x + (i - 0.5) * wbar, vals, wbar, color=c, label=name)
    ax.set_xticks(x, list(hind[names[0]].keys())); ax.set_ylabel("revenue, € per expected tonne")
    ax.set_title("2010 in hindsight", loc="left"); ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    fig.savefig(os.path.join(FIG, "fig8_yield_risk.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
