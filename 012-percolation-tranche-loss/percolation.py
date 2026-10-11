#-------- percolation.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# Bond percolation on configuration-model networks, the Watts threshold cascade,
# and the tranche losses each implies, beside the Vasicek one-factor model.
#
# Theory:  Newman, Strogatz & Watts (2001); Callaway et al. (2000); Watts (2002);
#          Vasicek (2002).  The tranche-loss link is the contribution of this note.

import numpy as np
from scipy import optimize, sparse, stats
from scipy.sparse.csgraph import connected_components


#-------- degree distributions --------------------------------------------------
class Degrees:
    """A degree distribution p_k on k = 0..kmax, with its generating functions."""

    def __init__(self, pk):
        pk = np.asarray(pk, float)
        self.pk = pk / pk.sum()
        self.k = np.arange(len(pk))
        self.mean = (self.k * self.pk).sum()
        self.mean2 = (self.k ** 2 * self.pk).sum()

    @classmethod
    def poisson(cls, c, kmax=None):
        kmax = kmax or int(c + 12 * np.sqrt(c) + 20)
        return cls(stats.poisson.pmf(np.arange(kmax + 1), c))

    @classmethod
    def power_law(cls, gamma, kmin=1, kmax=200):
        k = np.arange(kmax + 1).astype(float)
        pk = np.where(k >= kmin, np.maximum(k, 1) ** -gamma, 0.0)
        return cls(pk)

    def G0(self, x):
        return np.polyval(self.pk[::-1], x)

    def G1(self, x):
        q = self.k[1:] * self.pk[1:] / self.mean        # excess-degree distribution
        return np.polyval(q[::-1], x)

    @property
    def Tc(self):
        """Percolation threshold  T_c = <k> / (<k^2> - <k>)."""
        return self.mean / (self.mean2 - self.mean)


def giant_component(deg, T):
    """Size S of the giant failure cluster at transmissibility T.

    u is the probability that following one link does not lead to the giant
    cluster:  u = 1 - T + T G1(u),  and  S = 1 - G0(u)."""
    if T <= deg.Tc:
        return 0.0
    f = lambda u: 1 - T + T * deg.G1(u) - u
    u = optimize.brentq(f, 0.0, 1 - 1e-12)
    return float(1 - deg.G0(u))


#-------- tranche loss ----------------------------------------------------------
def tranche_payoff(L, a, d):
    """Fraction of tranche [a, d] written down for portfolio loss L."""
    return np.clip(L - a, 0.0, d - a) / (d - a)


def percolation_tranche_el(S, lam, lgd, a, d):
    """Expected tranche loss in the large-network limit.

    Idiosyncratic failures arrive as Poisson(lam) seeds. If any seed lies in the
    giant cluster (probability 1 - exp(-lam S)) the loss is lgd * S; otherwise it
    is negligible. So the loss distribution is two-point."""
    p_hit = 1 - np.exp(-lam * S)
    return p_hit * tranche_payoff(lgd * S, a, d)


def percolation_mean_loss(S, lam, lgd, N):
    return lgd * (S * (1 - np.exp(-lam * S)) + lam / N)


def vasicek_tranche_el(p, rho, lgd, a, d, n=4001):
    """Expected tranche loss under Vasicek's one-factor model, by quadrature over Z."""
    z = np.linspace(-8, 8, n)
    w = stats.norm.pdf(z) * (z[1] - z[0])
    pd_z = stats.norm.cdf((stats.norm.ppf(p) - np.sqrt(rho) * z) / np.sqrt(1 - rho))
    return float((tranche_payoff(lgd * pd_z, a, d) * w).sum())


def vasicek_loss_pdf(x, p, rho):
    """Density of the default fraction x under Vasicek (large homogeneous pool)."""
    x = np.clip(x, 1e-12, 1 - 1e-12)
    q = stats.norm.ppf(x)
    return (np.sqrt((1 - rho) / rho)
            * np.exp(0.5 * q ** 2 - (np.sqrt(1 - rho) * q - stats.norm.ppf(p)) ** 2 / (2 * rho)))


#-------- simulation ------------------------------------------------------------
def configuration_model(deg, N, rng):
    """Random network with degree sequence drawn from p_k; self-loops and repeated
    links are dropped (negligible for N large). Returns an edge array (m, 2)."""
    ks = rng.choice(deg.k, size=N, p=deg.pk)
    if ks.sum() % 2:
        ks[rng.integers(N)] += 1
    stubs = np.repeat(np.arange(N), ks)
    rng.shuffle(stubs)
    e = stubs.reshape(-1, 2)
    e = e[e[:, 0] != e[:, 1]]
    e = np.unique(np.sort(e, axis=1), axis=0)
    return e


def percolate(edges, N, T, rng):
    """Keep each link with probability T; return the cluster label of every node."""
    keep = edges[rng.random(len(edges)) < T]
    A = sparse.coo_matrix((np.ones(len(keep)), (keep[:, 0], keep[:, 1])), shape=(N, N))
    return connected_components(A, directed=False)[1]


def simulate_losses(deg, N, T, lam, n_trials, rng, new_network_every=1):
    """Monte Carlo failed fraction: a fresh network and fresh percolation per trial,
    Poisson(lam) seed failures, everything connected to a seed fails."""
    out = np.empty(n_trials)
    edges = None
    for i in range(n_trials):
        if edges is None or i % new_network_every == 0:
            edges = configuration_model(deg, N, rng)
        labels = percolate(edges, N, T, rng)
        sizes = np.bincount(labels)
        m = rng.poisson(lam)
        seeds = rng.integers(N, size=m)
        hit = np.unique(labels[seeds])
        out[i] = sizes[hit].sum() / N
    return out


def largest_cluster(deg, N, T, rng):
    edges = configuration_model(deg, N, rng)
    labels = percolate(edges, N, T, rng)
    return np.bincount(labels).max() / N


#-------- Watts (2002) threshold cascade -----------------------------------------
def adjacency(edges, N):
    """CSR adjacency (indptr, indices) of an undirected edge list."""
    A = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(N, N))
    A = (A + A.T).tocsr()
    return A.indptr, A.indices


def watts_cascade(adj, N, phi, rng, seed=None):
    """Watts (2002): a node fails once at least a fraction phi of its partners have
    failed. One random seed. Returns the final failed fraction."""
    indptr, indices = adj
    kdeg = np.diff(indptr)
    failed = np.zeros(N, bool)
    count = np.zeros(N, int)
    s = int(rng.integers(N)) if seed is None else seed
    failed[s] = True
    queue = [s]
    while queue:
        v = queue.pop()
        for w in indices[indptr[v]:indptr[v + 1]]:
            if failed[w]:
                continue
            count[w] += 1
            if count[w] >= phi * kdeg[w]:
                failed[w] = True
                queue.append(w)
    return failed.mean()


def watts_condition(deg, phi):
    """Global cascades are possible when  sum_k k(k-1) rho_k p_k > <k>,
    with rho_k = 1 if a single failed partner tips a degree-k node (k <= 1/phi)."""
    k = deg.k
    rho = (k >= 1) & (k <= np.floor(1 / phi + 1e-12))
    return (k * (k - 1) * rho * deg.pk).sum() / deg.mean


#-------- many seeds: failed fraction with an initial default fraction p ----------
def failed_fraction(deg, T, p):
    """Large-network failed fraction when each firm defaults on its own with
    probability p and each link transmits with probability T.

    v is the probability that a link, followed outward, does not lead to failure:
        v = 1 - T + T (1 - p) G1(v),     failed fraction = 1 - (1 - p) G0(v).
    As p -> 0 this tends to the giant-cluster size S above T_c and to 0 below."""
    f = lambda v: 1 - T + T * (1 - p) * deg.G1(v) - v
    v = optimize.brentq(f, 0.0, 1.0)
    return float(1 - (1 - p) * deg.G0(v))


def amplification(deg, T):
    """Mean size of the failure cluster set off by one default, below T_c:
       <s> = 1 + T <k> / (1 - T G1'(1)),   G1'(1) = (<k^2> - <k>) / <k>.
    It diverges at T_c."""
    g1p = (deg.mean2 - deg.mean) / deg.mean
    return np.where(T < deg.Tc, 1 + T * deg.mean / (1 - T * g1p), np.inf)


#-------- network-factor model: the economy sets the transmissibility -------------
def T_of_Z(Z, T_bar, beta):
    """Transmissibility in economic state Z (Z large = bad year), a probit link."""
    return stats.norm.cdf((stats.norm.ppf(T_bar) + np.sqrt(beta) * Z) / np.sqrt(1 - beta))


def network_loss_grid(deg, p, lgd, T_bar, beta, n=2001):
    z = np.linspace(-7, 7, n)
    w = stats.norm.pdf(z) * (z[1] - z[0])
    T = T_of_Z(z, T_bar, beta)
    L = lgd * np.array([failed_fraction(deg, t, p) for t in T])
    return z, w, T, L


def network_tranche_el(grid, a, d):
    z, w, T, L = grid
    return float((tranche_payoff(L, a, d) * w).sum())


#-------- May (1972): stability of a large random system -------------------------
def may_matrix(N, C, sigma, rng):
    """Community matrix  A = -I + B:  each species regulates itself at unit rate, and
    each ordered pair interacts with probability C, with strength drawn N(0, sigma^2)."""
    B = rng.normal(0.0, sigma, (N, N)) * (rng.random((N, N)) < C)
    np.fill_diagonal(B, 0.0)
    return B - np.eye(N)


def may_stable(N, C, sigma, rng):
    """Locally stable if every eigenvalue has negative real part."""
    return np.linalg.eigvals(may_matrix(N, C, sigma, rng)).real.max() < 0


#-------- degree correlations: disassortative and assortative networks -------------
def degrees_of(edges, N):
    return np.bincount(edges.ravel(), minlength=N)


def assortativity(edges, N):
    """Pearson correlation of the degrees at either end of a link (Newman 2002)."""
    k = degrees_of(edges, N)
    x = np.concatenate([k[edges[:, 0]], k[edges[:, 1]]]).astype(float)
    y = np.concatenate([k[edges[:, 1]], k[edges[:, 0]]]).astype(float)
    return float(np.corrcoef(x, y)[0, 1])


def rewire(edges, N, mode, n_swaps, rng):
    """Degree-preserving rewiring towards assortative or disassortative mixing
    (Xulvi-Brunet & Sokolov 2004). Two links are drawn; their four ends are sorted by
    degree and re-paired either high-high / low-low (assortative) or high-low
    (disassortative). Self-loops and repeated links are refused."""
    e = [tuple(x) for x in edges]
    present = set(e)
    k = degrees_of(edges, N)
    m = len(e)
    for _ in range(n_swaps):
        i, j = rng.integers(m, size=2)
        if i == j:
            continue
        a, b = e[i]; c, d = e[j]
        nodes = sorted({a, b, c, d}, key=lambda v: k[v])
        if len(nodes) < 4:
            continue
        lo1, lo2, hi1, hi2 = nodes
        if mode == "assortative":
            new = [(hi1, hi2), (lo1, lo2)]
        else:
            new = [(hi2, lo1), (hi1, lo2)]
        new = [tuple(sorted(x)) for x in new]
        if any(x in present for x in new):
            continue
        present.discard(e[i]); present.discard(e[j])
        e[i], e[j] = new
        present.update(new)
    return np.array(e)


def correlated_threshold(edges, N):
    """Bond-percolation threshold on a network with degree correlations:
       T_c = 1 / Lambda,  Lambda the largest eigenvalue of  C_{kk'} = (k' - 1) P(k'|k),
    where P(k'|k) is the probability that a link from a degree-k node ends at degree k'
    (Boguna & Pastor-Satorras 2002; Vazquez & Moreno 2003). For uncorrelated networks
    Lambda = (<k^2> - <k>) / <k>, recovering the earlier formula."""
    k = degrees_of(edges, N)
    ks = np.unique(k[k > 0])
    pos = {v: i for i, v in enumerate(ks)}
    E = np.zeros((len(ks), len(ks)))
    for a, b in edges:
        E[pos[k[a]], pos[k[b]]] += 1
        E[pos[k[b]], pos[k[a]]] += 1
    P = E / E.sum(1, keepdims=True)
    M = P * (ks[None, :] - 1)
    return float(1.0 / np.abs(np.linalg.eigvals(M)).max())
