#-------- rhine.py --------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Case study A: the Rhine low water of 2018 as a seed set on a spatial supply network.
#
# Firms sit at the real industrial centres of the Rhine basin and of Germany. Each buys
# from suppliers chosen by sector (an input-output matrix) and by distance (a gravity
# kernel). When the Rhine at Kaub falls, firms that receive bulk inputs by barge past
# Kaub lose them with a probability that rises as the water falls. A disrupted
# supplier disrupts each customer with probability T: the share of that input that
# cannot be replaced by rail or road in time. The question is how close 2018 came to
# the percolation threshold of this network.
#
# Anchors: Kaub 78 cm, below which barges must light-load (CCNR); 40 cm, below which
# barge transport is barely viable; the 2018 low of about 25 cm (October 2018); and
# Ademmer, Jannsen & Meuchelbock (2023): 30 days below 78 cm reduce German industrial
# production by about 1%.

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order

# (name, lon, lat, size weight, exposure to Kaub, country)
# exposure: 1 = bulk traffic must pass Kaub (upstream of the Middle Rhine),
#           0.35 = on the Rhine downstream of Kaub (shallower water, but not the bottleneck),
#           0 = inland, no barge dependence
HUBS = [
    ("Rotterdam",    4.48, 51.92, 1.2, 0.00, "NL"),
    ("Duisburg",     6.76, 51.43, 1.4, 0.35, "DE"),
    ("Düsseldorf",   6.77, 51.23, 0.9, 0.35, "DE"),
    ("Cologne",      6.96, 50.94, 1.0, 0.35, "DE"),
    ("Dortmund",     7.47, 51.51, 0.9, 0.20, "DE"),
    ("Koblenz",      7.59, 50.36, 0.3, 0.60, "DE"),
    ("Mainz",        8.27, 50.00, 0.6, 1.00, "DE"),
    ("Frankfurt",    8.68, 50.11, 0.9, 1.00, "DE"),
    ("Ludwigshafen", 8.43, 49.48, 1.3, 1.00, "DE"),
    ("Karlsruhe",    8.40, 49.01, 0.6, 1.00, "DE"),
    ("Stuttgart",    9.18, 48.78, 1.1, 0.50, "DE"),
    ("Strasbourg",   7.75, 48.58, 0.4, 1.00, "FR"),
    ("Basel",        7.59, 47.56, 0.6, 1.00, "CH"),
    ("Hamburg",      9.99, 53.55, 1.0, 0.00, "DE"),
    ("Bremen",       8.80, 53.08, 0.5, 0.00, "DE"),
    ("Hanover",      9.73, 52.37, 0.6, 0.00, "DE"),
    ("Berlin",      13.40, 52.52, 0.8, 0.00, "DE"),
    ("Leipzig",     12.37, 51.34, 0.6, 0.00, "DE"),
    ("Nuremberg",   11.08, 49.45, 0.6, 0.00, "DE"),
    ("Munich",      11.58, 48.14, 1.0, 0.00, "DE"),
]
KAUB = (7.76, 50.09)

SECTORS = ["Chemicals", "Coke & fuels", "Basic metals", "Energy", "Building materials",
           "Machinery", "Motor vehicles", "Other manufacturing"]
# Share of each sector's bulk inputs that arrives by inland waterway (an assumption,
# ordered by the sectors named in reports on 2018: chemicals, fuels, steel, coal power).
BARGE = np.array([0.6, 0.8, 0.6, 0.7, 0.5, 0.05, 0.05, 0.10])


def default_io():
    """Input shares A[s, t]: share of sector t's intermediate inputs bought from sector s.
    A stylised matrix, to be replaced by the Destatis 2018 input-output table."""
    A = np.array([
        # Chem  Fuel  Metal Enrg  Bld   Mach  Veh   Other   <- buyer
        [0.40, 0.10, 0.05, 0.02, 0.10, 0.08, 0.10, 0.15],   # Chemicals
        [0.15, 0.30, 0.05, 0.25, 0.10, 0.02, 0.02, 0.05],   # Coke & fuels
        [0.03, 0.02, 0.45, 0.02, 0.10, 0.30, 0.25, 0.10],   # Basic metals
        [0.15, 0.20, 0.15, 0.50, 0.15, 0.05, 0.05, 0.10],   # Energy
        [0.02, 0.02, 0.05, 0.01, 0.40, 0.02, 0.02, 0.05],   # Building materials
        [0.10, 0.10, 0.10, 0.10, 0.10, 0.40, 0.21, 0.15],   # Machinery
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.30, 0.02],   # Motor vehicles
        [0.15, 0.26, 0.15, 0.10, 0.15, 0.10, 0.05, 0.38],   # Other manufacturing
    ])
    return A / A.sum(0, keepdims=True)


# sector mix at each kind of hub: Rhine hubs lean to bulk industry
MIX_RIVER = np.array([0.22, 0.10, 0.14, 0.10, 0.08, 0.14, 0.07, 0.15])
MIX_INLAND = np.array([0.08, 0.03, 0.06, 0.06, 0.07, 0.25, 0.20, 0.25])


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


class Economy:
    """N firms at the hubs, each with a sector, an output size, an exposure to Kaub and
    a list of suppliers. Links point supplier -> customer."""

    def __init__(self, N=4000, k_in=4.0, length_km=250.0, io=None, seed=0):
        rng = np.random.default_rng(seed)
        self.io = default_io() if io is None else io
        w = np.array([h[3] for h in HUBS]); w = w / w.sum()
        hub = rng.choice(len(HUBS), size=N, p=w)
        self.hub = hub
        lon = np.array([h[1] for h in HUBS])[hub] + rng.normal(0, 0.12, N)
        lat = np.array([h[2] for h in HUBS])[hub] + rng.normal(0, 0.08, N)
        self.lon, self.lat = lon, lat
        river = np.array([h[4] > 0 for h in HUBS])[hub]
        self.sector = np.array([rng.choice(len(SECTORS), p=MIX_RIVER if r else MIX_INLAND) for r in river])
        self.exposure = np.array([h[4] for h in HUBS])[hub] * BARGE[self.sector]
        self.germany = np.array([h[5] == "DE" for h in HUBS])[hub]
        self.output = rng.lognormal(0.0, 1.0, N)
        self.N = N

        # suppliers: sector by the input-output column of the buyer, place by gravity
        src, dst = [], []
        by_sector = [np.flatnonzero(self.sector == s) for s in range(len(SECTORS))]
        n_in = rng.poisson(k_in, N)
        for j in range(N):
            if n_in[j] == 0:
                continue
            sec = rng.choice(len(SECTORS), size=n_in[j], p=self.io[:, self.sector[j]])
            for s in sec:
                pool = by_sector[s]
                if len(pool) == 0:
                    continue
                d = haversine_km(lon[j], lat[j], lon[pool], lat[pool])
                p = np.exp(-d / length_km); p[pool == j] = 0
                i = pool[rng.choice(len(pool), p=p / p.sum())]
                src.append(i); dst.append(j)
        e = np.unique(np.column_stack([src, dst]), axis=0)
        self.edges = e

    def threshold(self):
        """Directed percolation threshold  T_c = <k> / <j k>,  j = in-degree, k = out-degree
        (Newman, Strogatz & Watts 2001), valid when degrees at the two ends of a link
        are uncorrelated."""
        j = np.bincount(self.edges[:, 1], minlength=self.N)
        k = np.bincount(self.edges[:, 0], minlength=self.N)
        return float(k.mean() / (j * k).mean())

    def disrupted(self, kaub_cm, T, rng, severity=1.0):
        """One realisation: which firms lose a month of output at a given Kaub level and
        transmissibility. `severity` is the chance that a missed barge delivery costs
        the month's output rather than being bridged from stock. Returns a boolean array."""
        seed = rng.random(self.N) < severity * self.exposure * barge_shortfall(kaub_cm)
        keep = self.edges[rng.random(len(self.edges)) < T]
        N = self.N
        # a super-source joined to every seed; everything it reaches is disrupted
        rows = np.concatenate([keep[:, 0], np.full(seed.sum(), N)])
        cols = np.concatenate([keep[:, 1], np.flatnonzero(seed)])
        G = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N + 1, N + 1))
        reached = breadth_first_order(G, N, directed=True, return_predecessors=False)
        out = np.zeros(N + 1, bool); out[reached] = True
        return out[:N]

    def output_loss(self, mask):
        """Share of German industrial output in disrupted firms."""
        g = self.germany
        return float(self.output[mask & g].sum() / self.output[g].sum())


def mean_loss(econ, kaub_cm, T, severity, runs, rng):
    return float(np.mean([econ.output_loss(econ.disrupted(kaub_cm, T, rng, severity)) for _ in range(runs)]))


def calibrate_T(econ, severity, target=0.01, kaub_cm=55.0, runs=200, seed=0):
    """Transmissibility at which a low-water month costs `target` of German output."""
    lo, hi = 0.0, 0.6
    if mean_loss(econ, kaub_cm, 0.0, severity, runs, np.random.default_rng(seed)) > target:
        return np.nan
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if mean_loss(econ, kaub_cm, mid, severity, runs, np.random.default_rng(seed)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def barge_shortfall(kaub_cm, mid=78.0, width=12.0):
    """Probability that a barge-dependent firm misses a delivery, as a function of the
    Kaub gauge: one half at 78 cm (light-loading begins in earnest), near one by 40 cm
    (barge transport barely viable), near zero above about 120 cm. An assumption."""
    return 1.0 / (1.0 + np.exp((kaub_cm - mid) / width))
