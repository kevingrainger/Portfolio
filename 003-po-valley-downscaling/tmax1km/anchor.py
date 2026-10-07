#-------- anchor.py --------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Making the surface pass through every station exactly.
#
# The PDE alone will not reproduce the stations: the equation is missing physics. The
# missing part is the field q. Of all the q that make the solution equal the station
# value at every station cell, take the one closest to a prior guess q_hat:
#
#   minimise   (q - q_hat)' W (q - q_hat)
#   subject to A theta = b + q   and   H theta = y
#
# W = I + l_q^2 L penalises both the size and the roughness of the correction (L is the
# grid Laplacian, l_q a smoothness length). This is a quadratic problem with linear
# constraints, so it has a closed form:
#
#   theta_0 = A^-1 (b + q_hat)                     solution with the prior forcing
#   d       = y - H theta_0                        what the stations disagree by
#   G       = H A^-1                               one adjoint solve per station
#   q       = q_hat + W^-1 G' (G W^-1 G')^-1 d     smallest correction that closes the gap
#   theta   = A^-1 (b + q)
#
# Row i of G is station i's footprint: how much one unit of heating in each cell would
# change that station's reading. It stretches upwind.
#
# This is the strong-constraint variational analysis of Sasaki (1970), with the
# observations rather than the dynamics imposed exactly.

import numpy as np
import scipy.sparse as sp
from .pde import factorise, laplacian


class Smoother:
    """W = I + l_q^2 L, factorised once and reused every day."""

    def __init__(self, grid, l_q):
        self.l_q = l_q
        self.lu = factorise((sp.identity(grid.n, format="csc") + l_q ** 2 * laplacian(grid)).tocsc())

    def solve(self, rhs):
        return self.lu.solve(np.asarray(rhs, float))


def footprints(sysd, cells):
    """G = H A^-1 restricted to interior cells: one row per station."""
    G = np.empty((len(cells), sysd.grid.n))
    for i, c in enumerate(cells):
        h = np.zeros(sysd.grid.n); h[c] = 1.0
        G[i] = np.where(sysd.inner, sysd.solve_T(h), 0.0)
    return G


class Anchor:
    """Everything the anchored solve needs for one day, computed once for all stations.

    Because G and W^-1 G' are built station by station, any subset of stations can then
    be used as anchors at the cost of one small dense solve. That is what makes spatially
    blocked cross-validation affordable: the held-out fold changes, the factorisations
    do not."""

    def __init__(self, sysd, cells, smoother, q_hat=None):
        self.sysd, self.cells, self.q_hat = sysd, np.asarray(cells), q_hat
        self.G = footprints(sysd, cells)
        self.WG = np.column_stack([smoother.solve(g) for g in self.G])          # W^-1 G'
        self.S = self.G @ self.WG                                               # G W^-1 G'
        self.theta0 = sysd.theta(q_hat)

    def multipliers(self, y, use):
        d = y[use] - self.theta0[self.cells[use]]
        return np.linalg.solve(self.S[np.ix_(use, use)], d)

    def at_stations(self, y, use):
        """Anchored solution read at every station cell, anchoring only on stations `use`.
        H theta = H theta_0 + S[:, use] lam, so no field solve is needed."""
        return self.theta0[self.cells] + self.S[:, use] @ self.multipliers(y, use)

    def forcing(self, y, use):
        """The missing-physics field q for a set of anchors."""
        dq = self.WG[:, use] @ self.multipliers(y, use)
        return dq if self.q_hat is None else dq + np.ravel(self.q_hat)

    def solve(self, y, use=None, check=True):
        """Full anchored field: returns theta and q on the grid.
        With check, verifies the constraint itself: the surface equals the stations."""
        use = np.isfinite(y) if use is None else use
        q = self.forcing(y, use)
        theta = self.sysd.theta(q)
        if check:
            gap = np.abs(theta[self.cells[use]] - y[use]).max()
            assert gap < 1e-6, f"anchors missed by {gap:.2e} K"
        return theta.reshape(self.sysd.grid.shape), q.reshape(self.sysd.grid.shape)
