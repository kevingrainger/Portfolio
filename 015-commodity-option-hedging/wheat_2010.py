#!/usr/bin/env python3
#-------- wheat_2010.py -------------------------------------------------------------
#-----------------------------------------------------------------------------
# A wheat producer's hedge through Russia's 2010 drought and export ban.
#
# The producer: a US winter-wheat farm that, on 3 May 2010, hedges the harvest it will
# sell on 31 August 2010. CBOT wheat futures (front month, Yahoo ZW=F) stand in for the
# cash price, so basis is ignored; all revenue is per bushel. Options are European, expire
# on the sale date and are priced by Black (1976) at the 60-day volatility seen before
# 3 May (no option quotes are free for 2010), with r = 0.2%.
#
# The shock: a record heatwave in western Russia through July 2010, then the grain
# export ban announced on 5 August. Wheat rose from about $4.60 to $6.60 a bushel.
#
# The eight standard producer hedges:
#   no hedge         sell at harvest at whatever the price is
#   short futures    sell the whole crop forward on 3 May
#   50% futures      sell half forward, leave half open
#   long put         buy an at-the-money put: a floor, upside kept, premium paid
#   zero-cost collar buy a 95% put, sell a call whose premium pays for it: floor and cap
#   put spread       buy the at-the-money put, sell an 85% put: a cheaper, partial floor
#   three-way        a put spread (95% / 85%) paid for by selling a call: the fence
#   layered          sell a quarter of the crop forward at the start of May, June, July
#                    and August
#
# Each is scored on what happened in 2010, and against a Monte Carlo of what could have
# happened as seen on 3 May (driftless lognormal paths at the same volatility), so the
# hindsight answer can be set beside the decision a producer could have made.
#
#   python wheat_2010.py
#
# Output: results/wheat_2010.json, results/wheat_2010_prices.csv

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from hedging import black

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
START, SALE = "2010-05-03", "2010-08-31"
LAYERS = ["2010-05-03", "2010-06-01", "2010-07-01", "2010-08-02"]
R = 0.002
EVENTS = {"2010-07-01": "Russian heatwave builds", "2010-08-05": "Russia bans grain exports"}


def prices():
    path = os.path.join(DATA, "wheat_2009_2011.csv")
    if not os.path.exists(path):
        import yfinance as yf
        d = yf.download("ZW=F", start="2009-06-01", end="2011-03-01", progress=False, auto_adjust=False)
        (d["Close"].squeeze() / 100).rename("wheat").to_csv(path)       # cents to dollars a bushel
    return pd.read_csv(path, index_col=0, parse_dates=True).squeeze()


def put(F, K, tau, s):
    c = black(F, K, tau, s, R)[0]
    return c - np.exp(-R * tau) * (F - K)                    # put-call parity on futures


def call(F, K, tau, s):
    return black(F, K, tau, s, R)[0]


def setup(F0, tau, s):
    """Strikes and premiums for every option structure, fixed on 3 May."""
    P_atm, P95, P85 = put(F0, F0, tau, s), put(F0, 0.95 * F0, tau, s), put(F0, 0.85 * F0, tau, s)
    Kc_collar = brentq(lambda K: call(F0, K, tau, s) - P95, F0, 3 * F0)
    Kc_fence = brentq(lambda K: call(F0, K, tau, s) - (P95 - P85), F0, 3 * F0)
    return dict(P_atm=P_atm, P95=P95, P85=P85, Kc_collar=Kc_collar, Kc_fence=Kc_fence)


def revenue(F0, FT, layers, k):
    """Revenue per bushel at sale for each strategy; FT can be an array of outcomes."""
    put_pay = lambda K: np.maximum(K - FT, 0)
    call_pay = lambda K: np.maximum(FT - K, 0)
    return {
        "no hedge": FT,
        "short futures": np.full_like(FT, F0),
        "50% futures": 0.5 * F0 + 0.5 * FT,
        "long put": FT + put_pay(F0) - k["P_atm"],
        "zero-cost collar": FT + put_pay(0.95 * F0) - call_pay(k["Kc_collar"]),
        "put spread": FT + put_pay(F0) - put_pay(0.85 * F0) - (k["P_atm"] - k["P85"]),
        "three-way": FT + put_pay(0.95 * F0) - put_pay(0.85 * F0) - call_pay(k["Kc_fence"]),
        "layered": layers,
    }


def main():
    os.makedirs(RES, exist_ok=True)
    F = prices()
    F0 = float(F[:START].iloc[-1])
    FT = float(F[:SALE].iloc[-1])
    ret = np.log(F).diff()
    sigma = float(ret[:START].iloc[-60:].std() * np.sqrt(252))
    tau = (pd.Timestamp(SALE) - pd.Timestamp(START)).days / 365
    k = setup(F0, tau, sigma)
    layer_prices = [float(F[:d].iloc[-1]) for d in LAYERS]
    real = {s: float(v) for s, v in revenue(F0, np.array(FT), np.mean(layer_prices), k).items()}

    # what could have happened, as seen on 3 May: driftless paths, layers sold along each path
    rng = np.random.default_rng(0)
    n, days = 20000, int(round(tau * 252))
    z = rng.standard_normal((n, days))
    paths = F0 * np.exp(np.cumsum(sigma / np.sqrt(252) * z - 0.5 * sigma ** 2 / 252, axis=1))
    idx = [0] + [int(round((pd.Timestamp(d) - pd.Timestamp(START)).days / 365 * 252)) - 1 for d in LAYERS[1:]]
    lay = np.column_stack([np.full(n, F0)] + [paths[:, i] for i in idx[1:]]).mean(axis=1)
    sim = revenue(F0, paths[:, -1], lay, k)
    ex_ante = {}
    for s, v in sim.items():
        v = np.asarray(v)
        ex_ante[s] = dict(mean=float(v.mean()), p5=float(np.quantile(v, 0.05)), p95=float(np.quantile(v, 0.95)),
                          cvar5=float(v[v <= np.quantile(v, 0.05)].mean()),
                          percentile_2010=float((v <= real[s]).mean()))
    out = dict(F0=F0, FT=FT, change=FT / F0 - 1, sigma=sigma, tau_years=tau, setup=k, layer_prices=layer_prices,
               realised=real, ex_ante=ex_ante,
               best_in_hindsight=max(real, key=real.get),
               sale_price_percentile=float((paths[:, -1] <= FT).mean()))
    json.dump(out, open(os.path.join(RES, "wheat_2010.json"), "w"), indent=1)
    F["2010-01":"2010-12"].to_csv(os.path.join(RES, "wheat_2010_prices.csv"))
    print(json.dumps({k_: out[k_] for k_ in ["F0", "FT", "change", "sigma", "setup", "realised",
                                             "best_in_hindsight", "sale_price_percentile"]}, indent=1))
    for s in ex_ante:
        print(f"{s:18s} mean {ex_ante[s]['mean']:.2f}  5% {ex_ante[s]['p5']:.2f}  CVaR {ex_ante[s]['cvar5']:.2f}  2010 at {ex_ante[s]['percentile_2010']:.0%}")


if __name__ == "__main__":
    main()
