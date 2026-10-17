#-------- sahel_ca.py --------------------------------------------------------------
#-----------------------------------------------------------------------------
# Dryland vegetation as a cellular automaton in which plants help plants
# (Kéfi et al. 2007, Nature 449:213; Kéfi et al. 2007, Theor. Pop. Biol. 71:367).
#
# Each cell is vegetated (+), empty but fertile (0) or degraded (-). With rho_+ the
# share of the whole landscape that is vegetated and q_{+|i} the share of cell i's four
# neighbours that are vegetated:
#
#   0 -> +   rate  [delta rho_+ + (1 - delta) q_{+|i}] (b - c rho_+)   seeds from far (delta) or near,
#                                                                       limited by environmental quality b
#                                                                       (rainfall) and competition c
#   + -> 0   rate  m                                                    plant death
#   0 -> -   rate  d                                                    soil degradation
#   - -> 0   rate  r + f q_{+|i}                                        recovery, helped by neighbouring
#                                                                       plants: FACILITATION
#
# f is the neighbour coupling: as in the Ising model, a cell's next state depends on
# its neighbours, and being next to plants lowers the barrier for bare soil to recover.
#
# Time runs in continuous-time steps of dt: in each step every cell is visited once in
# random order and changes with probability rate * dt (asynchronous updating, as in the
# original). The lattice is periodic.

import numpy as np
from numba import njit

VEG, EMPTY, DEGRADED = 1, 0, -1

# Kéfi et al. (2007, Theor. Pop. Biol.) defaults
DEFAULTS = dict(delta=0.1, c=0.3, m=0.1, d=0.2, r=0.0001, f=0.9)


@njit(cache=True)
def _sweep(grid, b, c, m, d, r, f, delta, dt, order):
    n = grid.shape[0]
    nveg = 0
    for x in range(n * n):
        if grid.flat[x] == 1:
            nveg += 1
    rho = nveg / (n * n)
    for idx in order:
        i, j = idx // n, idx % n
        q = 0.25 * ((grid[(i + 1) % n, j] == 1) + (grid[(i - 1) % n, j] == 1)
                    + (grid[i, (j + 1) % n] == 1) + (grid[i, (j - 1) % n] == 1))
        s = grid[i, j]
        u = np.random.random()
        if s == 0:
            grow = max((delta * rho + (1 - delta) * q) * (b - c * rho), 0.0) * dt
            if u < grow:
                grid[i, j] = 1
                nveg += 1
            elif u < grow + d * dt:
                grid[i, j] = -1
        elif s == 1:
            if u < m * dt:
                grid[i, j] = 0
                nveg -= 1
        else:
            if u < (r + f * q) * dt:
                grid[i, j] = 0
        rho = nveg / (n * n)
    return grid


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


class Landscape:
    def __init__(self, n=100, b=0.6, cover=0.5, seed=0, dt=0.2, **params):
        self.n, self.b, self.dt = n, b, dt
        self.p = {**DEFAULTS, **params}
        self.rng = np.random.default_rng(seed)
        _seed(seed)
        u = self.rng.random((n, n))
        self.grid = np.where(u < cover, VEG, np.where(u < cover + (1 - cover) / 2, EMPTY, DEGRADED)).astype(np.int8)
        self.t = 0.0

    def step(self, years=1.0):
        p = self.p
        for _ in range(int(round(years / self.dt))):
            order = self.rng.permutation(self.n * self.n)
            _sweep(self.grid, self.b, p["c"], p["m"], p["d"], p["r"], p["f"], p["delta"], self.dt, order)
            self.t += self.dt
        return self

    def cover(self):
        return float(np.mean(self.grid == VEG))

    def degraded(self):
        return float(np.mean(self.grid == DEGRADED))


#-------- patterns ---------------------------------------------------------------------
def patch_sizes(grid):
    """Sizes of the vegetated clusters (4-neighbour connectivity, periodic edges joined)."""
    from scipy.ndimage import label
    lab, k = label(grid == VEG)
    if k == 0:
        return np.array([], dtype=int)
    # join clusters that touch across the periodic boundary
    parent = np.arange(k + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in zip(np.r_[lab[0, :], lab[:, 0]], np.r_[lab[-1, :], lab[:, -1]]):
        if a and b:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(a) for a in range(k + 1)])
    sizes = np.bincount(roots[lab.ravel()], minlength=k + 1)
    sizes[0] = 0
    return sizes[sizes > 0]


def percolates(grid):
    """True if one vegetated cluster connects the top edge to the bottom edge (open
    boundaries): the vegetation spans the landscape."""
    from scipy.ndimage import label
    lab, _ = label(grid == VEG)
    top, bottom = set(lab[0][lab[0] > 0]), set(lab[-1][lab[-1] > 0])
    return bool(top & bottom)
