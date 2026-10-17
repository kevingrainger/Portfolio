#!/usr/bin/env python3
#-------- vecer.py -----------------------------------------------------------------
#-----------------------------------------------------------------------------
# Asian options by Večeř's one-dimensional PDE (Večeř 2001, J. Comput. Finance 4(4);
# Večeř 2002, Risk 15(6)), for a call on the average of a futures price over a set of
# daily fixings, with time-dependent volatility.
#
# Hold q(t) futures, where q(t) is the share of fixings still to come after time t
# (1 before the averaging window, falling by 1/n at each fixing). Futures cost nothing
# to hold, so starting from wealth X_0 = F_0 - K the position is worth exactly A - K at
# expiry, A being the average. With the futures price itself as numeraire, the ratio
# Z = X / F is a martingale with dZ = (q(t) - Z) sigma(t) dW, so the price solves
#
#     u_t + 1/2 sigma(t)^2 (q(t) - z)^2 u_zz = 0,     u(T, z) = max(z, 0)
#     price = discount * F_0 * u(0, z_0),             z_0 = 1 - K / F_0
#
# one space dimension whatever the number of fixings. Crank-Nicolson in time, with
# u = 0 far below and u = z far above (where the payoff is certainly zero or certainly
# in the money).
#
#   python vecer.py        # checks against Monte Carlo with a geometric control variate

import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import norm


def vecer_call(F0, K, sigma, fix_idx, n_obs, dt, discount=1.0, nz=1201, z_lo=-1.5, z_hi=2.5, substeps=8):
    """sigma: per-step volatility, length n_obs - 1 (step k runs from observation k to k+1).
    fix_idx: indices of the observations that enter the average."""
    fix = np.zeros(n_obs, dtype=bool)
    fix[np.asarray(fix_idx)] = True
    n = fix.sum()
    # q during step k: fixings at observations k+1 .. n_obs-1, as a share of all fixings
    after = np.cumsum(fix[::-1])[::-1]                    # fixings at observation >= k
    q_step = np.r_[after[1:], 0][:n_obs - 1] / n
    z = np.linspace(z_lo, z_hi, nz)
    h = z[1] - z[0]
    u = np.maximum(z, 0.0)
    tau = dt / substeps
    for k in range(n_obs - 2, -1, -1):                    # backwards from expiry
        a = 0.5 * sigma[k] ** 2 * (q_step[k] - z) ** 2 / h ** 2 * tau
        ab = np.zeros((3, nz))                            # (I - A/2), banded
        ab[0, 2:] = -0.5 * a[1:-1]
        ab[1, 1:-1] = 1 + a[1:-1]
        ab[2, :-2] = -0.5 * a[1:-1]
        ab[1, 0] = ab[1, -1] = 1.0
        for _ in range(substeps):
            rhs = u.copy()
            rhs[1:-1] = u[1:-1] + 0.5 * a[1:-1] * (u[2:] - 2 * u[1:-1] + u[:-2])
            rhs[0], rhs[-1] = 0.0, z[-1]
            u = solve_banded((1, 1), ab, rhs)
    z0 = 1 - K / F0
    return discount * F0 * np.interp(z0, z, u)


def monte_carlo(F0, K, sigma, fix_idx, n_obs, dt, discount=1.0, n_paths=400_000, seed=0):
    """Plain arithmetic Asian by simulation, with the exact geometric Asian as control variate."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_obs - 1))
    lr = -0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * z
    logF = np.log(F0) + np.c_[np.zeros(n_paths), np.cumsum(lr, axis=1)]
    Fx = np.exp(logF[:, fix_idx])
    arith = discount * np.maximum(Fx.mean(1) - K, 0)
    geo = discount * np.maximum(np.exp(logF[:, fix_idx].mean(1)) - K, 0)
    V = np.r_[0, np.cumsum(sigma ** 2 * dt)][fix_idx]
    mean = np.log(F0) - 0.5 * V.mean()
    var = np.minimum.outer(V, V).sum() / len(V) ** 2
    d1 = (mean - np.log(K) + var) / np.sqrt(var)
    geo_exact = discount * (np.exp(mean + 0.5 * var) * norm.cdf(d1) - K * norm.cdf(d1 - np.sqrt(var)))
    c = np.cov(arith, geo)
    beta = c[0, 1] / c[1, 1]
    cv = arith - beta * (geo - geo_exact)
    return cv.mean(), cv.std() / np.sqrt(n_paths)


def main():
    dt, n_obs = 1 / 252, 127                              # six months of trading days
    fix_idx = np.arange(64, 127)                          # average over the last three months
    t = np.arange(n_obs - 1) * dt
    cases = {"flat 35%": np.full(n_obs - 1, 0.35),
             "seasonal (rising into the window)": 0.25 + 0.20 * np.clip((t - 0.15) / 0.3, 0, 1)}
    print(f"{'volatility':36s} {'K/F0':>5s} {'PDE':>9s} {'Monte Carlo':>18s} {'gap (s.e.)':>10s}")
    for name, sig in cases.items():
        for k in (0.9, 1.0, 1.1):
            pde = vecer_call(100.0, 100 * k, sig, fix_idx, n_obs, dt, discount=np.exp(-0.04 * 0.5))
            mc, se = monte_carlo(100.0, 100 * k, sig, fix_idx, n_obs, dt, discount=np.exp(-0.04 * 0.5))
            print(f"{name:36s} {k:5.2f} {pde:9.4f} {mc:10.4f} +/- {se:.4f} {(pde - mc) / se:+10.1f}")


if __name__ == "__main__":
    main()
