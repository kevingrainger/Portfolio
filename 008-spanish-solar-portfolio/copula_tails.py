#!/usr/bin/env python3
#-------- copula_tails.py -----------------------------------------------------------
#-----------------------------------------------------------------------------
# Do the sites have their bad days together more often than the Gaussian copula allows?
# A Gaussian copula is tail-independent: as the threshold goes further into the tail,
# the chance that two sites are both in it, given one is, falls to zero. A t copula
# keeps a share of joint extremes at any threshold. The event sets (event_set.py,
# event_set_30y.py) read a 1-in-20 tail off a Gaussian copula, so this checks it.
#
# On the 1995-2024 daily anomalies of the 25 regions (each calendar month's mean and
# standard deviation removed, then ranks):
#
#   chi(q)      for each pair of sites, P(both below their q-quantile) / q, the finite-
#               level measure of joint bad days (Coles, Heffernan and Tawn 1999), at
#               q = 10%, 5%, 1%, averaged over the 300 pairs; empirical, and from each
#               fitted copula (simulated, 400,000 days); the same for sunny days
#               (both above their 1 - q quantile)
#   t copula    correlation from Kendall's tau (R = sin(pi tau / 2)), degrees of freedom
#               nu by maximum pseudo-likelihood over a grid
#
# Days are treated as independent in the fit, which they aren't (persistence 0.66),
# so the likelihood is a pseudo-likelihood and only the point estimate of nu is used.
#
#   python copula_tails.py
#
# Output: results/copula_tails.json

import json
import os

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import kendalltau, norm, rankdata, t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
QS = (0.10, 0.05, 0.01)
NU_GRID = np.array([2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 100])


def anomalies(day):
    month = day.index.month
    mean = day.groupby(month).transform("mean"); sd = day.groupby(month).transform("std")
    return ((day - mean) / sd).fillna(0)


def pseudo_obs(z):
    return np.column_stack([rankdata(z[c]) / (len(z) + 1) for c in z])


def kendall_R(u):
    d = u.shape[1]
    R = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            R[i, j] = R[j, i] = np.sin(np.pi * kendalltau(u[:, i], u[:, j])[0] / 2)
    w, v = np.linalg.eigh(R)                         # nearest positive-definite, if needed
    if w.min() < 1e-6:
        R = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T
        s = np.sqrt(np.diag(R)); R = R / np.outer(s, s)
    return R


def t_copula_loglik(u, R, nu):
    d = u.shape[1]
    x = student_t.ppf(u, nu)
    Rinv = np.linalg.inv(R)
    _, logdet = np.linalg.slogdet(R)
    q = np.einsum("ij,jk,ik->i", x, Rinv, x)
    joint = gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log(nu * np.pi) - logdet / 2 \
        - (nu + d) / 2 * np.log1p(q / nu)
    marg = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
            - (nu + 1) / 2 * np.log1p(x ** 2 / nu)).sum(axis=1)
    return float((joint - marg).sum())


def gauss_copula_loglik(u, R):
    x = norm.ppf(u)
    Rinv = np.linalg.inv(R)
    _, logdet = np.linalg.slogdet(R)
    q = np.einsum("ij,jk,ik->i", x, Rinv - np.eye(len(R)), x)
    return float((-logdet / 2 - q / 2).sum())


def fit_t(u):
    R = kendall_R(u)
    ll = np.array([t_copula_loglik(u, R, nu) for nu in NU_GRID])
    return R, float(NU_GRID[ll.argmax()]), dict(zip(NU_GRID.tolist(), ll.tolist()))


def chi_lower(u, qs=QS):
    """Mean over pairs of P(U_i < q, U_j < q) / q."""
    out = {}
    iu = np.triu_indices(u.shape[1], 1)
    for q in qs:
        b = (u < q).astype(float)
        both = (b.T @ b) / len(u)
        out[q] = float((both[iu] / q).mean())
    return out


def sample(R, n, rng, nu=None):
    x = rng.standard_normal((n, len(R))) @ np.linalg.cholesky(R).T
    if nu is None:
        return norm.cdf(x)
    x = x / np.sqrt(rng.chisquare(nu, n) / nu)[:, None]
    return student_t.cdf(x, nu)


def main():
    rng = np.random.default_rng(0)
    day = pd.read_csv(os.path.join(HERE, "data", "era5_daily_1995_2024.csv"), index_col=0, parse_dates=True).dropna()
    day = day[~((day.index.month == 2) & (day.index.day == 29))]
    u = pseudo_obs(anomalies(day))
    Rg = np.corrcoef(norm.ppf(u).T)
    Rt, nu, ll = fit_t(u)
    out = dict(days=int(len(u)), sites=int(u.shape[1]), nu=nu, t_loglik_by_nu=ll,
               gauss_loglik=gauss_copula_loglik(u, Rg),
               chi_lower=dict(empirical=chi_lower(u),
                              gaussian=chi_lower(sample(Rg, 400_000, rng)),
                              t=chi_lower(sample(Rt, 400_000, rng, nu))),
               chi_upper=dict(empirical=chi_lower(1 - u),
                              gaussian=chi_lower(1 - sample(Rg, 400_000, rng)),
                              t=chi_lower(1 - sample(Rt, 400_000, rng, nu))))
    json.dump(out, open(os.path.join(HERE, "results", "copula_tails.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "t_loglik_by_nu"}, indent=1))


if __name__ == "__main__":
    main()
