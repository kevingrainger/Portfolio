#!/usr/bin/env python3
#-------- lr_bootstrap.py -----------------------------------------------------------
#-----------------------------------------------------------------------------
# Bootstrap p-value for the GARCH-MIDAS likelihood ratio with the lag shape w left free.
# Under theta = 0, w drops out of the likelihood, so this LR isn't chi-squared with 2
# degrees of freedom (Davies 1977, 1987). Instead we simulate it under the null:
#
#   1. fit the null (theta = 0) to the real returns; keep its standardised residuals
#   2. build a new return series from the null model, drawing the shocks from those
#      residuals with replacement (so the fat tails of the real shocks are kept)
#   3. fit the null and the full model (theta and w free, the same starting points as
#      power.fit_garch_midas) to it, and record the LR
#
# The p-value is the share of simulated LRs at least as large as the real one.
#
#   python lr_bootstrap.py
#
# Output: results/real_stage2_bootstrap.json

import json
import os

import numba
import numpy as np
from scipy.optimize import minimize

import run_real as rr

HERE = os.path.dirname(os.path.abspath(__file__))
K = 6
N_BOOT = 499


@numba.njit(cache=True)
def nll(params, r, month_idx, X, K):
    mu, a, b, m, theta, w = params
    if a < 0 or b < 0 or a + b >= 0.999 or w < 1:
        return 1e10
    k = np.arange(1, K + 1)
    phi = (1 - k / (K + 1)) ** (w - 1)
    phi = phi / phi.sum()
    log_tau_m = np.full(len(X), m)
    for i in range(K, len(X)):
        s = 0.0
        for j in range(K):
            s += phi[j] * X[i - 1 - j]
        log_tau_m[i] = m + theta * s
    out = 0.0
    g = 1.0
    for t in range(len(r)):
        tau = np.exp(log_tau_m[month_idx[t]])
        if t > 0:
            e_prev = r[t - 1] - mu
            tau_prev = np.exp(log_tau_m[month_idx[t - 1]])
            g = (1 - a - b) + a * e_prev ** 2 / tau_prev + b * g
        if month_idx[t] >= K:
            v = tau * g
            out += 0.5 * (np.log(v) + (r[t] - mu) ** 2 / v)
    return out


def fit_null(r, mi, X, x0):
    f = lambda q: nll(np.array([q[0], q[1], q[2], q[3], 0.0, 1.0]), r, mi, X, K)
    return minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=8000, xatol=1e-8, fatol=1e-8))


def fit_full(r, mi, X):
    best = None
    for theta0 in (0.0, 0.1, 0.3, -0.15):
        for w0 in (1.5, 4.0):
            x0 = [0.0, 0.05, 0.9, np.log(np.var(r)) - theta0 * X.mean(), theta0, w0]
            res = minimize(lambda p: nll(np.asarray(p), r, mi, X, K), x0, method="Nelder-Mead",
                           options=dict(maxiter=8000, xatol=1e-7, fatol=1e-7))
            if best is None or res.fun < best.fun:
                best = res
    return best


def lr(r, mi, X):
    full = fit_full(r, mi, X)
    null = fit_null(r, mi, X, full.x[:4])
    null2 = fit_null(r, mi, X, [0.0, 0.05, 0.9, np.log(np.var(r))])
    n0 = min(null.fun, null2.fun)
    return 2 * (n0 - full.fun), (null if null.fun <= null2.fun else null2)


def simulate_null(q, z, mi, rng):
    mu, a, b, m = q
    tau = np.exp(m)
    shocks = rng.choice(z, len(mi))
    r = np.empty(len(mi))
    g = 1.0
    for t in range(len(mi)):
        if t > 0:
            g = (1 - a - b) + a * (r[t - 1] - mu) ** 2 / tau + b * g
        r[t] = mu + np.sqrt(tau * g) * shocks[t]
    return r


def main():
    df, mi, extreme, r = rr.load()
    X = extreme.values.astype(float)
    lr_real, null = lr(r, mi, X)
    mu, a, b, m = null.x
    tau = np.exp(m)
    g = np.empty(len(r)); g[0] = 1.0
    for t in range(1, len(r)):
        g[t] = (1 - a - b) + a * (r[t - 1] - mu) ** 2 / tau + b * g[t - 1]
    z = (r - mu) / np.sqrt(tau * g)
    z = (z[mi >= K] - z[mi >= K].mean()) / z[mi >= K].std()
    rng = np.random.default_rng(0)
    sims = []
    for i in range(N_BOOT):
        sims.append(lr(simulate_null(null.x, z, mi, rng), mi, X)[0])
        if (i + 1) % 50 == 0:
            print(i + 1, "of", N_BOOT, "largest so far", round(max(sims), 2))
    sims = np.array(sims)
    out = dict(LR=float(lr_real), replicates=N_BOOT,
               p_value=float((1 + (sims >= lr_real).sum()) / (N_BOOT + 1)),
               null_LR_quantiles={q: float(np.quantile(sims, q)) for q in (0.5, 0.9, 0.95, 0.99)},
               null_LR_max=float(sims.max()),
               chi2_2_quantiles={0.95: 5.99, 0.99: 9.21})
    json.dump(out, open(os.path.join(HERE, "results", "real_stage2_bootstrap.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
