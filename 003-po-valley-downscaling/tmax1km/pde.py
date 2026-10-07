#-------- pde.py -----------------------------------------------------------------------
#-----------------------------------------------------------------------------
# The reduced heat budget, solved once per day as a steady state.
#
#   u . grad(theta) - kappa * laplacian(theta)
#       + (theta - theta_s) / tau_s(x) + (theta - theta_E) / tau_a = q(x)
#   theta = theta_E on the boundary
#
# theta is afternoon air temperature reduced to sea level. Wind carries it, turbulence
# spreads it, the surface below pulls it towards the surface temperature theta_s on a
# time tau_s, and the large-scale atmosphere pulls it back to ERA5-Land's theta_E on a
# time tau_a. q is whatever heating the equation leaves out.
#
# Land cover enters through the surface coupling: rates add, so
#   1 / tau_s(x) = sum_c f_c(x) / tau_c
#
# On the grid (first-order upwind for the wind, five-point Laplacian) this is the
# sparse linear system  A theta = b + q.  Units: km, hours, kelvin.

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .grid import CLASSES

M2S_TO_KM2H = 3.6e-3                                   # 1 m^2/s in km^2/h
BOUNDS = dict(kappa=(10 * M2S_TO_KM2H, 1e4 * M2S_TO_KM2H), tau=(0.25, 48.0))


@dataclass
class Params:
    kappa: float = 300 * M2S_TO_KM2H                   # km^2/h
    tau_a: float = 6.0                                 # h
    tau_c: np.ndarray = field(default_factory=lambda: np.full(len(CLASSES), 6.0))   # h, one per land-cover class

    def pack(self):
        """Parameters as the vector the optimiser works on: their logarithms."""
        return np.log(np.concatenate([[self.kappa, self.tau_a], self.tau_c]))

    @staticmethod
    def unpack(v):
        v = np.exp(np.asarray(v, float))
        return Params(v[0], v[1], v[2:].copy())

    @staticmethod
    def bounds():
        return [tuple(np.log(BOUNDS["kappa"]))] + [tuple(np.log(BOUNDS["tau"]))] * (1 + len(CLASSES))

    @staticmethod
    def names():
        return ["kappa", "tau_a"] + [f"tau[{c}]" for c in CLASSES]

    def surface_rate(self, frac):
        """1 / tau_s(x), per hour."""
        return np.tensordot(1.0 / self.tau_c, frac, axes=1)

    def table(self):
        import pandas as pd
        return pd.Series(np.concatenate([[self.kappa / M2S_TO_KM2H, self.tau_a], self.tau_c]),
                         index=["kappa (m^2/s)", "tau_a (h)"] + [f"tau {c} (h)" for c in CLASSES])


def laplacian(grid):
    """Positive semi-definite grid Laplacian L (per km^2) with no-flux edges:
    theta' L theta is the sum of squared differences between neighbouring cells."""
    h2 = grid.dx ** 2
    idx = np.arange(grid.n).reshape(grid.shape)
    pairs = [(idx[:, :-1].ravel(), idx[:, 1:].ravel()), (idx[:-1, :].ravel(), idx[1:, :].ravel())]
    rows, cols, vals = [], [], []
    for a, b in pairs:
        rows += [a, b, a, b]; cols += [a, b, b, a]
        vals += [np.ones(a.size), np.ones(a.size), -np.ones(a.size), -np.ones(a.size)]
    return sp.csc_matrix((np.concatenate(vals) / h2, (np.concatenate(rows), np.concatenate(cols))), shape=(grid.n, grid.n))


def assemble(grid, kappa, rate, u, v):
    """The matrix A. Interior rows hold the discretised operator; rows of the outer ring
    are the identity, which pins theta to the boundary value carried in b.

    Upwind differencing takes the gradient from the side the air arrives from, so every
    off-diagonal entry is negative and the diagonal dominates: the solution can never
    overshoot its inputs."""
    nx, h = grid.nx, grid.dx
    idx = np.arange(grid.n).reshape(grid.shape)
    inner = grid.interior()
    c = idx[inner]
    up, vp = np.maximum(u[inner], 0), np.maximum(v[inner], 0)            # flow towards +x, +y
    um, vm = np.maximum(-u[inner], 0), np.maximum(-v[inner], 0)          # flow towards -x, -y
    k = kappa / h ** 2
    west, east, south, north = c - 1, c + 1, c - nx, c + nx
    diag = (up + um + vp + vm) / h + 4 * k + rate[inner]
    rows = np.concatenate([c, c, c, c, c, idx[~inner]])
    cols = np.concatenate([c, west, east, south, north, idx[~inner]])
    vals = np.concatenate([diag, -up / h - k, -um / h - k, -vp / h - k, -vm / h - k, np.ones((~inner).sum())])
    return sp.csc_matrix((vals, (rows, cols)), shape=(grid.n, grid.n))


def factorise(A):
    """Sparse LU. A is diagonally dominant, so no pivoting is needed for stability; turning
    it off and ordering on the pattern of A + A' halves both the time and the fill-in."""
    return splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0)


class DaySystem:
    """One day's linear system, factorised once.

      solve(rhs)       A^-1 rhs
      solve_T(rhs)     A^-T rhs   (the adjoint: how a reading depends on forcing everywhere)
      theta(q)         solution for a given missing-physics field q (interior cells only)
    """

    def __init__(self, grid, params, frac, fields):
        self.grid = grid
        self.inner = grid.interior().ravel()
        self.rate_s = params.surface_rate(frac)
        self.fields = fields
        rate = self.rate_s + 1.0 / params.tau_a
        self.A = assemble(grid, params.kappa, rate, fields["u"], fields["v"])
        self.lu = factorise(self.A)
        b = self.rate_s * fields["theta_s"] + fields["theta_E"] / params.tau_a
        self.b = np.where(self.inner, b.ravel(), fields["theta_E"].ravel())

    def solve(self, rhs):
        return self.lu.solve(np.asarray(rhs, float))

    def solve_T(self, rhs):
        return self.lu.solve(np.asarray(rhs, float), trans="T")

    def theta(self, q=None):
        rhs = self.b if q is None else self.b + np.where(self.inner, np.ravel(q), 0.0)
        return self.solve(rhs)


#-------- fitting the physics-only model (M1) ---------------------------------------------
def misfit(vec, grid, frac, days_fields, cells, y_days):
    """Mean squared error of the physics-only solution at the stations, and its gradient
    with respect to the log-parameters.

    The gradient comes from one extra (adjoint) solve per day rather than one per
    parameter. With r = H theta - y, solve A' lam = H' r; then for any parameter p
        dJ/dp = lam' (db/dp - dA/dp theta).
    For the rates this is a sum over cells of lam * (target - theta), and for kappa it is
    lam' applied to the discrete Laplacian of theta."""
    p = Params.unpack(vec)
    inner = grid.interior()
    h2 = grid.dx ** 2
    total, count, grad = 0.0, 0, np.zeros(len(vec))
    for fields, y in zip(days_fields, y_days):
        ok = np.isfinite(y)
        if not ok.any():
            continue
        sysd = DaySystem(grid, p, frac, fields)
        theta = sysd.theta()
        r = theta[cells[ok]] - y[ok]
        total += r @ r; count += ok.sum()
        rhs = np.zeros(grid.n); np.add.at(rhs, cells[ok], r)
        lam = sysd.solve_T(rhs).reshape(grid.shape)
        th = theta.reshape(grid.shape)
        lap = np.zeros(grid.shape)
        lap[1:-1, 1:-1] = (th[1:-1, :-2] + th[1:-1, 2:] + th[:-2, 1:-1] + th[2:, 1:-1] - 4 * th[1:-1, 1:-1]) / h2
        li = np.where(inner, lam, 0.0)
        g_kappa = (li * lap).sum()                                        # dA/dkappa theta = -laplacian(theta)
        g_ra = (li * (fields["theta_E"] - th)).sum()
        g_rc = np.array([(li * f * (fields["theta_s"] - th)).sum() for f in frac])
        grad += np.concatenate([[p.kappa * g_kappa, -g_ra / p.tau_a], -g_rc / p.tau_c])
    return total / count, 2 * grad / count


def length_scales(params, frac, speed_kmh):
    """How far the air carries a temperature anomaly before the relaxation terms erase it:
    by the wind, |u| * tau, and by mixing, sqrt(kappa * tau), both in km. tau is the
    combined relaxation time of a cell, 1 / (1/tau_s + 1/tau_a)."""
    tau = 1.0 / (params.surface_rate(frac) + 1.0 / params.tau_a)
    return dict(tau_h=tau, advective_km=speed_kmh * tau, diffusive_km=np.sqrt(params.kappa * tau))
