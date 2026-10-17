#!/usr/bin/env python3
#-------- monte_carlo.py -----------------------------------------------------------
#-----------------------------------------------------------------------------
# Which feature of a real market breaks which hedge? The same 3-month at-the-money
# call is hedged on simulated futures from four worlds that all have 25% volatility on
# average, so the option is fairly priced in each:
#   gbm        Black's own world: constant volatility, no jumps
#   jumps      Merton jumps, four a year, 5% each, most of the variance still diffusive
#   heston     volatility that wanders (Heston), mean-reverting over about four months
#   scheduled  one 15% jump, up or down, on a known day mid-life (a tariff ruling, a crop report)
#
#   python monte_carlo.py
#
# Output: results/monte_carlo.csv

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hedging import DAYS, garch_path, garch_term_vol, hedge_trade, realised_vol, simulate   # noqa: E402

SIGMA = 0.25
N_DAYS = 63
N_PATHS = 2000
COST = 5e-4
MODELS = ["gbm", "jumps", "heston", "scheduled"]
STRATEGIES = ["static", "weekly", "move", "daily", "garch", "leland", "band"]
# GARCH(1,1) with typical commodity persistence, long-run volatility SIGMA
ALPHA, BETA = 0.05, 0.93
OMEGA = (1 - ALPHA - BETA) * SIGMA ** 2 / DAYS


def main():
    rows = []
    for model in MODELS:
        paths = simulate(model, N_PATHS, N_DAYS, sigma=SIGMA, seed=1)
        H = paths.shape[1] - N_DAYS - 1
        for p in paths:
            ret = np.diff(np.log(p))
            vol = realised_vol(ret)[H - 1:H - 1 + N_DAYS]
            s2 = garch_path(ret[:H + N_DAYS - 1], (OMEGA, ALPHA, BETA), SIGMA ** 2 / DAYS)[H:H + N_DAYS]
            vol_g = np.array([garch_term_vol(s2[t], (OMEGA, ALPHA, BETA), N_DAYS - t) for t in range(N_DAYS)])
            F = p[H:]
            for s in STRATEGIES:
                out = hedge_trade(F, vol_g if s == "garch" else vol, N_DAYS, s, k=COST, sigma_price=SIGMA)
                rows.append(dict(model=model, strategy=s, err=out["pnl"] / out["premium"],
                                 cost=out["cost"] / out["premium"], trades=out["trades"]))
        print(f"{model}: done")
    df = pd.DataFrame(rows)
    summ = df.groupby(["model", "strategy"]).agg(sd_err=("err", "std"), mean_err=("err", "mean"),
                                                 cost=("cost", "mean"), trades=("trades", "mean")).reset_index()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    summ.to_csv(os.path.join(HERE, "results", "monte_carlo.csv"), index=False)
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(summ.pivot(index="strategy", columns="model", values="sd_err").loc[STRATEGIES, MODELS])


if __name__ == "__main__":
    main()
