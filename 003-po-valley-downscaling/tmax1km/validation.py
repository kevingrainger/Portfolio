#-------- validation.py ----------------------------------------------------------------
#-----------------------------------------------------------------------------
# Honest scoring: spatially blocked hold-out, baselines, and skill tables.
#
# Stations a few kilometres apart read almost the same temperature, so leaving out one
# station at random tests very little. Instead the domain is cut into 50 km blocks and
# whole blocks are held out together: a held-out station has no anchor in its own block.
# Every model is scored only at stations it never saw, in a year (2023) it never saw.

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ELEVATION_BANDS = [("< 200 m", -1e9, 200.0), ("200-800 m", 200.0, 800.0), ("> 800 m", 800.0, 1e9)]
DISTANCE_BANDS = [("< 30 km", 0.0, 30.0), ("30-60 km", 30.0, 60.0), ("> 60 km", 60.0, 1e9)]
MODELS = {"B0": "ERA5-Land, bilinear + lapse rate", "B1": "regression + kriged residuals", "B2": "E-OBS",
          "B3": "XGBoost on station residuals", "M1": "physics only (q = 0)", "M2": "physics, anchored",
          "M3": "physics, anchored, XGBoost prior for q"}


def spatial_folds(stations, grid, block_km=50.0, k=5, seed=0):
    """Assign stations to k folds by 50 km block: blocks are shuffled, then dealt out to
    whichever fold currently holds the fewest stations."""
    bx = np.floor((stations.x.values - grid.x0) / block_km).astype(int)
    by = np.floor((stations.y.values - grid.y0) / block_km).astype(int)
    block = bx * 1000 + by
    ids, counts = np.unique(block, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(ids))
    order = order[np.argsort(-counts[order], kind="stable")]                   # big blocks first
    size, fold_of = np.zeros(k, int), {}
    for j in order:
        f = int(np.argmin(size))
        fold_of[ids[j]] = f
        size[f] += counts[j]
    return np.array([fold_of[b] for b in block]), block


def nearest_anchor_km(stations, fold):
    """For each station, distance to the nearest station outside its own fold."""
    xy = np.column_stack([stations.x.values, stations.y.values])
    dist = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    dist[fold[:, None] == fold[None, :]] = np.inf
    return dist.min(1)


def rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.nanmean(e ** 2)))


def skill(pred, models=None, band=None):
    """RMSE and bias per model, optionally split by a banding column.
    pred has one row per held-out station-day with an 'obs' column and one per model."""
    models = [m for m in MODELS if m in pred.columns] if models is None else models
    groups = [("all", pred)] if band is None else [(k, g) for k, g in pred.groupby(band, sort=True, observed=True)]
    rows = []
    for name, g in groups:
        for m in models:
            e = g[m] - g.obs
            rows.append(dict(group=name, model=m, rmse=rmse(e), bias=float(e.mean()), n=int(e.notna().sum())))
    return pd.DataFrame(rows)


def add_bands(pred):
    pred = pred.copy()
    for col, source, bands in (("elevation band", "elevation", ELEVATION_BANDS), ("anchor distance", "anchor_km", DISTANCE_BANDS)):
        labels = np.full(len(pred), "", dtype=object)
        for label, lo, hi in bands:
            labels[(pred[source] >= lo) & (pred[source] < hi)] = label
        pred[col] = pd.Categorical(labels, [b[0] for b in bands], ordered=True)
    return pred


#-------- B1: regression on covariates, then ordinary kriging of what is left ------------
class RegressionKriging:
    """The standard geostatistical downscaler. A linear regression explains what it can
    of the station-minus-ERA5 difference from terrain and land cover; the remainder is
    interpolated between stations by ordinary kriging with an exponential variogram."""

    def fit(self, X, r, xy, day):
        self.lo, self.hi = X.min(0), X.max(0)                 # a line is not trusted beyond the data that drew it
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = self._design(X)
        self.beta = np.linalg.lstsq(Z, r, rcond=None)[0]
        e = r - Z @ self.beta
        # pooled empirical variogram over days, then a fit of nugget + sill * (1 - exp(-h/a))
        h_all, g_all = [], []
        for d in np.unique(day):
            m = day == d
            if m.sum() < 3:
                continue
            i, j = np.triu_indices(m.sum(), 1)
            h_all.append(np.hypot(*(xy[m][i] - xy[m][j]).T)); g_all.append(0.5 * (e[m][i] - e[m][j]) ** 2)
        h, g = np.concatenate(h_all), np.concatenate(g_all)
        edges = np.linspace(0, 250, 13)
        mid = 0.5 * (edges[1:] + edges[:-1])
        emp = np.array([g[(h >= a) & (h < b)].mean() if ((h >= a) & (h < b)).any() else np.nan for a, b in zip(edges[:-1], edges[1:])])
        ok = np.isfinite(emp)
        fit = least_squares(lambda p: p[0] + p[1] * (1 - np.exp(-mid[ok] / p[2])) - emp[ok], [0.1, max(emp[ok].max(), 0.1), 60.0],
                            bounds=([0, 1e-3, 5.0], [np.inf, np.inf, 400.0]))
        self.nugget, self.sill, self.range_km = fit.x
        return self

    def _design(self, X):
        return np.column_stack([np.ones(len(X)), (np.clip(X, self.lo, self.hi) - self.mu) / self.sd])

    def trend(self, X):
        return self._design(X) @ self.beta

    def predict(self, X_train, r_train, xy_train, X_test, xy_test):
        """One day: trend at the test points plus the kriged residual of the training points."""
        e = r_train - self.trend(X_train)
        cov = lambda h: self.sill * np.exp(-h / self.range_km)
        n = len(e)
        K = np.ones((n + 1, n + 1)); K[-1, -1] = 0.0
        K[:n, :n] = cov(np.hypot(*(xy_train[:, None, :] - xy_train[None, :, :]).transpose(2, 0, 1))) + self.nugget * np.eye(n)
        k = np.ones((n + 1, len(xy_test)))
        k[:n] = cov(np.hypot(*(xy_train[:, None, :] - xy_test[None, :, :]).transpose(2, 0, 1)))
        w = np.linalg.solve(K, k)
        return self.trend(X_test) + w[:n].T @ e


def paired_bootstrap(pred, a, b, n=2000, seed=0):
    """Is model a better than model b, or did it get lucky with these stations?
    Stations are resampled with replacement (errors at one station are not independent
    from day to day, so the station is the unit). Returns the RMSE difference a - b and
    its 95% interval; an interval that excludes zero is a difference the network supports."""
    rng = np.random.default_rng(seed)
    g = pred.groupby("station")
    sq = np.column_stack([g.apply(lambda d: ((d[m] - d.obs) ** 2).sum()) for m in (a, b)])
    cnt = g.size().values
    diff = lambda idx: np.sqrt(sq[idx, 0].sum() / cnt[idx].sum()) - np.sqrt(sq[idx, 1].sum() / cnt[idx].sum())
    k = len(cnt)
    draws = np.array([diff(rng.integers(0, k, k)) for _ in range(n)])
    return float(diff(np.arange(k))), tuple(np.quantile(draws, [0.025, 0.975]))
