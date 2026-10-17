#-------- approximations.py --------------------------------------------------------
#-----------------------------------------------------------------------------
# Two analytic stand-ins for the lattice in sahel_ca.py, to see what the space adds.
#
# Mean field: every cell sees the landscape average, q_{+|i} = rho_+. Three ODEs for the
# densities rho_+, rho_0, rho_- (they sum to 1).
#
# Pair approximation (Matsuda et al. 1992; for this model, Kéfi et al. 2007 TPB): track
# the densities of neighbouring pairs rho_ab, so a cell knows something about its own
# neighbours. A cell in state a whose neighbour is in state n sees, on average,
#     q_{+|a,n} = [1(n = +) + (z - 1) q_{+|a}] / z,     q_{+|a} = rho_{a+} / rho_a,   z = 4
# (the other z - 1 neighbours are guessed from a's pair statistics, closing the hierarchy).
# The rates are those of sahel_ca.py with q replaced by these averages:
#   drho_ab/dt = sum_{c != a} [rho_cb W(c -> a | b) - rho_ab W(a -> c | b)]
#              + sum_{c != b} [rho_ac W(c -> b | a) - rho_ab W(b -> c | a)]
#
# Each is swept in environmental quality b up and down, continuing from the last steady
# state, exactly as the lattice was.

import numpy as np
from scipy.integrate import solve_ivp

from sahel_ca import DEFAULTS

P, E, D = 0, 1, 2           # vegetated, empty, degraded
Z = 4


def rates(q, rho_p, b, p):
    """W[x, y]: rate at which a cell in state x becomes y, given local vegetated share q."""
    W = np.zeros((3, 3))
    W[E, P] = max((p["delta"] * rho_p + (1 - p["delta"]) * q) * (b - p["c"] * rho_p), 0.0)
    W[P, E] = p["m"]
    W[E, D] = p["d"]
    W[D, E] = p["r"] + p["f"] * q
    return W


def mean_field_rhs(t, rho, b, p):
    W = rates(rho[P], rho[P], b, p)
    return W.T @ rho - W.sum(axis=1) * rho


def pair_rhs(t, y, b, p):
    R = np.maximum(y.reshape(3, 3), 0.0)
    R = 0.5 * (R + R.T)
    rho = R.sum(axis=1)
    rho_p = rho[P]
    qp = np.where(rho > 1e-12, R[:, P] / np.maximum(rho, 1e-12), 0.0)     # q_{+|a}
    Wn = np.empty((3, 3, 3, 3))                                          # Wn[a, n] = rates of a with neighbour n
    for a in range(3):
        for n in range(3):
            q = ((n == P) + (Z - 1) * qp[a]) / Z
            Wn[a, n] = rates(q, rho_p, b, p)
    dR = np.zeros((3, 3))
    for a in range(3):
        for b_ in range(3):
            g = 0.0
            for c in range(3):
                if c != a:
                    g += R[c, b_] * Wn[c, b_][c, a] - R[a, b_] * Wn[a, b_][a, c]
                if c != b_:
                    g += R[a, c] * Wn[c, a][c, b_] - R[a, b_] * Wn[b_, a][b_, c]
            dR[a, b_] = g
    return dR.ravel()


def steady(kind, b, start, p, T=3000.0):
    rhs = mean_field_rhs if kind == "mean field" else pair_rhs
    sol = solve_ivp(rhs, (0, T), start, args=(b, p), method="LSODA", rtol=1e-7, atol=1e-10)
    return sol.y[:, -1]


def sweep(kind, b_grid, f=0.9, seed_cover=0.01, **params):
    p = {**DEFAULTS, **params, "f": f}
    if kind == "mean field":
        state = np.array([0.6, 0.2, 0.2])
    else:
        rho = np.array([0.6, 0.2, 0.2])
        state = np.outer(rho, rho).ravel()                               # start uncorrelated
    rows = []
    for direction, bs in [("drying", b_grid), ("wetting", b_grid[::-1])]:
        for b in bs:
            cover = state[P] if kind == "mean field" else state.reshape(3, 3).sum(axis=1)[P]
            if direction == "wetting" and cover < 1e-6:
                # same seed rain as the lattice: a little vegetation scattered at random
                if kind == "mean field":
                    state = np.array([seed_cover, state[E], state[D] - seed_cover])
                else:
                    rho = state.reshape(3, 3).sum(axis=1)
                    rho = np.array([seed_cover, rho[E], rho[D] - seed_cover])
                    state = np.outer(rho, rho).ravel()
            state = steady(kind, b, state, p)
            cover = state[P] if kind == "mean field" else state.reshape(3, 3).sum(axis=1)[P]
            rows.append(dict(model=kind, f=f, direction=direction, b=b, cover=float(cover)))
    return rows
