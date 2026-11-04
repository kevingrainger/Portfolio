#!/usr/bin/env python3
#-------- dependence.py -------------------------------------------------------------
#-----------------------------------------------------------------------------
# Dependence between weather and the coffee price beyond a single correlation:
#
#   partial correlation   each bean against each hazard with the Brazilian real held fixed
#                         (Gaussian-rank, so outliers don't drive it), corrlib
#   tail dependence       of the most extreme weather months (top 10%), what share were also
#                         months of the largest price rises? Independence gives 10%; a
#                         block-shuffle of the price gives the 95% level, corrlib
#   Gumbel copula         the joint distribution of hazard and price move with each one's own
#                         distribution taken out (ranks), fitted by maximum likelihood. The
#                         Gumbel family has upper-tail dependence lambda = 2 - 2^(1/theta),
#                         the chance one is extreme given the other is; Gaussian has none.
#   DCC-GARCH             the correlation between arabica and robusta as a time series of its
#                         own (Engle 2002): GARCH(1,1) for each bean, then a dynamic
#                         conditional correlation fitted to the standardised returns. Does the
#                         link between the beans jump during weather events, just when an
#                         importer would want to switch origin?
#
#   python dependence.py
#
# Output: results/dependence.json, results/dcc.csv

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "004-corrlib-correlation-toolbox"))
from corrlib import correlation_measures as cm                     # noqa: E402

from weather_signal import full_panel, pink_sheet, strip           # noqa: E402

RES = os.path.join(HERE, "results")
HAZARDS = ["brazil_dry", "vietnam_dry", "vietnam_heat", "enso"]
PAIRS = [("robusta", "vietnam_heat"), ("arabica", "vietnam_heat"), ("arabica", "brazil_dry"), ("robusta", "brazil_dry")]
EVENTS = [("2013-12", "2014-04"), ("2021-07", "2021-08"), ("2023-10", "2024-06")]
rng = np.random.default_rng(0)


def panel():
    """Every month 2004-2026: both beans' returns with the real taken out, and the hazards."""
    p = full_panel()
    p = p.join(np.log(pink_sheet()).diff().reindex(p.index))
    fx = p.brl.fillna(0.0)                       # a few months missing from Yahoo: treat as no change
    for bean in ["arabica", "robusta"]:
        p[f"{bean}_res"], _ = strip(p[bean].values, fx.values)
    return p.dropna(subset=["arabica", "robusta"] + HAZARDS)


def partials(p):
    out = {}
    for bean in ["arabica", "robusta"]:
        for h in HAZARDS:
            X = np.vstack([p[bean].values, p[h].values, p.brl.fillna(0).values])
            out[f"{bean}|{h}"] = float(cm.partial_from_matrix(cm.gaussian_rank(X))[0, 1])
    return out


def block_resample(n, block=12):
    starts = rng.integers(0, n - block, size=n // block + 1)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def tails(p, q=0.10, n_boot=2000):
    out = {}
    for bean, h in PAIRS:
        x, y = p[h].values, p[f"{bean}_res"].values
        lam = cm.tail_dependence(x, y, quantile=q, lower=False)
        null = np.array([cm.tail_dependence(x, y[block_resample(len(y))], quantile=q, lower=False)
                         for _ in range(n_boot)])
        out[f"{bean}|{h}"] = dict(tail=float(lam), independent=q, null_95=float(np.quantile(null, 0.95)),
                                  p_value=float((null >= lam).mean()))
    return out


#-------- Gumbel copula ------------------------------------------------------------------
def gumbel_nll(theta, u, v):
    a, b = -np.log(u), -np.log(v)
    s = a ** theta + b ** theta
    A = s ** (1 / theta)
    logc = (-A + np.log(A + theta - 1) + (1 - 2 * theta) / theta * np.log(s)
            + (theta - 1) * (np.log(a) + np.log(b)) + a + b)
    return -logc.sum()


def gaussian_copula_nll(rho, u, v):
    x, y = norm.ppf(u), norm.ppf(v)
    return -(-0.5 * np.log(1 - rho ** 2) - (rho ** 2 * (x ** 2 + y ** 2) - 2 * rho * x * y) / (2 * (1 - rho ** 2))).sum()


def fit_gumbel(x, y):
    u, v = rankdata(x) / (len(x) + 1), rankdata(y) / (len(y) + 1)
    g = minimize_scalar(gumbel_nll, bounds=(1.0001, 5), args=(u, v), method="bounded")
    n = minimize_scalar(gaussian_copula_nll, bounds=(-0.95, 0.95), args=(u, v), method="bounded")
    return g.x, g.fun, n.x, n.fun


def copulas(p, n_boot=500):
    out = {}
    for bean, h in PAIRS:
        x, y = p[h].values, p[f"{bean}_res"].values
        theta, nll_g, rho, nll_n = fit_gumbel(x, y)
        boot = []
        for _ in range(n_boot):
            i = block_resample(len(x))
            boot.append(2 - 2 ** (1 / fit_gumbel(x[i], y[i])[0]))
        out[f"{bean}|{h}"] = dict(theta=float(theta), tail_lambda=float(2 - 2 ** (1 / theta)),
                                  tail_lambda_90=[float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
                                  gaussian_rho=float(rho), aic_gumbel=float(2 + 2 * nll_g),
                                  aic_gaussian=float(2 + 2 * nll_n))
    return out


#-------- DCC-GARCH ----------------------------------------------------------------------
def dcc(p):
    from arch import arch_model
    Z, vols = [], {}
    for bean in ["arabica", "robusta"]:
        r = 100 * p[f"{bean}_res"]
        fit = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
        Z.append(fit.std_resid.values)
        vols[bean] = fit.params.to_dict()
    Z = np.column_stack(Z)
    Qbar = np.cov(Z.T)

    def path(ab):
        a, b = ab
        Q = Qbar.copy()
        rho, nll = np.empty(len(Z)), 0.0
        for t in range(len(Z)):
            if t > 0:
                Q = (1 - a - b) * Qbar + a * np.outer(Z[t - 1], Z[t - 1]) + b * Q
            d = np.sqrt(np.diag(Q))
            R = Q / np.outer(d, d)
            rho[t] = R[0, 1]
            nll += 0.5 * (np.log(np.linalg.det(R)) + Z[t] @ np.linalg.solve(R, Z[t]))
        return rho, nll

    res = minimize(lambda ab: path(ab)[1] if ab.sum() < 0.999 and (ab > 0).all() else 1e9,
                   x0=[0.05, 0.9], method="Nelder-Mead")
    rho, _ = path(res.x)
    s = pd.Series(rho, index=p.index, name="dcc")
    in_event = pd.Series(False, index=p.index)
    for a, b in EVENTS:
        in_event[a:b] = True
    s.to_frame().assign(event=in_event).to_csv(os.path.join(RES, "dcc.csv"))
    return dict(a=float(res.x[0]), b=float(res.x[1]), mean=float(s.mean()),
                in_events=float(s[in_event].mean()), outside=float(s[~in_event].mean()),
                min=float(s.min()), max=float(s.max()), garch=vols)


def main():
    p = panel()
    out = dict(months=len(p), partial=partials(p), tail=tails(p), copula=copulas(p), dcc=dcc(p))
    json.dump(out, open(os.path.join(RES, "dependence.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["copula", "dcc"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
