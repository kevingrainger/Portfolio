#!/usr/bin/env python3
#-------- backtest.py --------------------------------------------------------------
#-----------------------------------------------------------------------------
# Every month since 2002, for six commodities, sell a 3-month at-the-money call on the
# futures and hedge it to expiry with each strategy in hedging.py.
#
# Commodities (continuous front-month futures from Yahoo Finance): copper (COMEX HG),
# gold (COMEX GC), wheat (CBOT ZW), corn (CBOT ZC), arabica coffee (ICE KC) and cocoa
# (ICE CC). No energy. The continuous series jumps a little at each contract roll; the
# jump is the same for every strategy, so it does not favour one over another.
#
# No free history of commodity option prices exists, so each option is sold at its
# trailing 60-day realised volatility. The test is of hedging, not of option pricing.
#
# A trade is a SHOCK trade when the volatility over its life turns out at least 1.5
# times the volatility it was sold at: the market did something the seller did not
# expect. Everything else is CALM. The same rule for every commodity.
#
# Hedging error and trading cost are reported as a share of the premium received, so
# commodities with different volatilities can be pooled: an error of 1.0 means the
# hedge missed by as much as the option was sold for.
#
#   python backtest.py
#
# Output: results/trades.csv, results/summary.csv, results/by_commodity.csv

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from hedging import (DAYS, garch_fit, garch_path, garch_term_vol,   # noqa: E402
                     hedge_trade, realised_vol)

DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
N_DAYS = 63                                   # 3 months of trading days
COST = 5e-4                                   # 5 bp of notional per futures traded
SHOCK_RATIO = 1.5
COMMODITIES = {"HG=F": "copper", "GC=F": "gold", "ZW=F": "wheat", "ZC=F": "corn",
               "KC=F": "coffee", "CC=F": "cocoa"}
STRATEGIES = ["none", "static", "weekly", "move", "daily", "garch", "leland", "band"]


def load_prices():
    path = os.path.join(DATA, "front_months.csv")
    if not os.path.exists(path):
        import yfinance as yf
        os.makedirs(DATA, exist_ok=True)
        yf.download(list(COMMODITIES), start="2000-01-01", progress=False, auto_adjust=False)["Close"].to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def garch_variance(dates, ret):
    """GARCH(1,1) refitted every January on the previous five years and used, without
    refitting, through that year. Returns the one-day-ahead variance for each day and
    the parameters by year."""
    s2 = np.full(len(ret) + 1, np.nan)
    params = {}
    for year in range(dates[0].year + 2, dates[-1].year + 1):
        i0 = np.searchsorted(dates, pd.Timestamp(f"{year}-01-01"))
        i_fit = np.searchsorted(dates, pd.Timestamp(f"{year - 5}-01-01"))
        p = garch_fit(ret[i_fit:i0 - 1])
        params[year] = p
        i1 = np.searchsorted(dates, pd.Timestamp(f"{year + 1}-01-01"))
        path = garch_path(ret[i0 - 1:i1 - 1], p, np.var(ret[i_fit:i0 - 1]))
        s2[i0:i1] = path[1:i1 - i0 + 1]
    return s2, params


def run_commodity(name, F):
    F = F.dropna()
    F = F[F > 0]
    dates, f = F.index, F.values
    ret = np.diff(np.log(f))
    vol_known = np.concatenate([[np.nan], realised_vol(ret)])   # known at the close of day t
    s2, params = garch_variance(dates, ret)
    rows = []
    for d in pd.date_range(dates[0] + pd.DateOffset(years=2), dates[-1], freq="MS"):
        i = np.searchsorted(dates, d)
        if i + N_DAYS >= len(f):
            break
        Fp = f[i:i + N_DAYS + 1]
        sig = vol_known[i:i + N_DAYS]
        p = params[dates[i].year]
        sig_g = np.array([garch_term_vol(s2[i + t], p, N_DAYS - t) for t in range(N_DAYS)])
        realised = np.std(np.diff(np.log(Fp))) * np.sqrt(DAYS)
        regime = "shock" if realised >= SHOCK_RATIO * sig[0] else "calm"
        for s in STRATEGIES:
            out = hedge_trade(Fp, sig_g if s == "garch" else sig, N_DAYS, s, k=COST, sigma_price=sig[0])
            rows.append(dict(commodity=name, start=dates[i].date(), end=dates[i + N_DAYS].date(), strategy=s,
                             regime=regime, sigma_sold=sig[0], sigma_realised=realised,
                             move=Fp[-1] / Fp[0] - 1, **out))
    return rows


def cvar(x, q=0.05):
    x = np.sort(x)
    return x[:max(1, int(np.ceil(q * len(x))))].mean()


def main():
    prices = load_prices()
    rows = []
    for tic, name in COMMODITIES.items():
        rows += run_commodity(name, prices[tic])
        print(f"{name}: done")
    trades = pd.DataFrame(rows)
    trades["err"] = trades.pnl / trades.premium                # hedging error, share of premium
    trades["cost_p"] = trades.cost / trades.premium
    os.makedirs(RESULTS, exist_ok=True)
    trades.to_csv(os.path.join(RESULTS, "trades.csv"), index=False)

    def summarise(g):
        return pd.Series(dict(n=len(g), mean_err=g.err.mean(), sd_err=g.err.std(), cvar5=cvar(g.err.values),
                              mean_cost=g.cost_p.mean(), trades=g.trades.mean()))
    summ = trades.groupby(["regime", "strategy"]).apply(summarise, include_groups=False).reset_index()
    by_c = trades.groupby(["commodity", "regime", "strategy"]).apply(summarise, include_groups=False).reset_index()
    summ.to_csv(os.path.join(RESULTS, "summary.csv"), index=False)
    by_c.to_csv(os.path.join(RESULTS, "by_commodity.csv"), index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(summ)
        print(trades.drop_duplicates(["commodity", "start"]).groupby("commodity").regime.value_counts().unstack())


if __name__ == "__main__":
    main()
