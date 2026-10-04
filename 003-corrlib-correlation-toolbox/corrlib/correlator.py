#!/usr/bin/env python3
#-------- corrlib : correlator.py --------------------------------------------
#-----------------------------------------------------------------------------
# One object, configured once per application.
#
#   from corrlib import Correlator
#   solar = Correlator(measure="gaussian_rank", remove=[clear_sky],
#                      clean="nonlinear_shrinkage", effective_n="bartlett")
#   C = solar.fit(X, labels=site_names)
#   solar.plot()
#
# The pipeline order is fixed, and the order is the point:
#
#   1. measure   transform each row (nothing / ranks / normal scores)
#   2. remove    regress out known factors           (costs observations)
#   3. matrix    Pearson on what is left, or Tyler's estimator if robust
#   4. remove    take out the top k shared modes      (costs variables)
#   5. clean     random matrix cleaning, with q computed from what is left
#   6. output    the correlation matrix, or partial correlations from it
#
# Removing shared factors BEFORE cleaning matters: a dominant mode (the market,
# the sun) inflates the top eigenvalue and distorts the noise band for everything
# else. Partial correlations come AFTER cleaning, because inverting a noisy matrix
# amplifies exactly the noise the cleaner is there to remove.

import numpy as np

from . import correlation_measures as cm
from . import random_matrix_cleaning as rmt
from . import sampling_estimators as se
from . import factor_removal as fr


MEASURES = {
    "pearson": lambda X: np.asarray(X, dtype=float),
    "spearman": lambda X: cm._rank_rows(np.asarray(X, dtype=float)),
    "gaussian_rank": cm.gaussian_scores,
}


class Correlator:
    def __init__(self, measure="pearson", remove=None, clean="raw", robust=False,
                 effective_n=None, output="correlation"):
        if measure not in MEASURES:
            raise ValueError(f"measure must be one of {sorted(MEASURES)}")
        if clean is not None and clean not in rmt.CLEANERS:
            raise ValueError(f"clean must be one of {sorted(rmt.CLEANERS)}")
        if effective_n not in (None, "bartlett"):
            raise ValueError("effective_n must be None or 'bartlett'")
        if output not in ("correlation", "partial"):
            raise ValueError("output must be 'correlation' or 'partial'")

        self.measure = measure
        self.remove = remove
        self.clean = clean or "raw"
        self.robust = robust
        self.effective_n = effective_n
        self.output = output

    #-------- fit ----------------------------------------------------------------
    # X is (n_variables, n_observations) as everywhere in corrlib. A pandas
    # DataFrame is read the pandas way - columns are variables, rows are time -
    # and its column names become the labels.
    def fit(self, X, labels=None):
        if hasattr(X, "columns"):
            labels = list(X.columns) if labels is None else labels
            X = X.values.T
        X = np.asarray(X, dtype=float)
        N, T = X.shape
        self.labels_ = list(labels) if labels is not None else [str(i) for i in range(N)]
        self.n_variables_, self.n_observations_ = N, T

        transform = MEASURES[self.measure]
        Z = transform(X)
        self.measured_ = rmt.tyler_correlation(Z) if self.robust else cm.pearson(Z)   # before anything is removed

        #known factors, transformed the same way as the data so like is compared with like
        factors, n_modes = fr.parse_remove(self.remove, T)
        self.loadings_ = None
        n_obs = T
        if factors is not None:
            Z, self.loadings_ = fr.regress_out(Z, transform(factors))
            n_obs -= factors.shape[0]

        if self.effective_n == "bartlett":
            n_obs = se.effective_n(Z) - (T - n_obs)
        self.n_effective_ = float(n_obs)

        self.raw_ = rmt.tyler_correlation(Z) if self.robust else cm.pearson(Z)

        self.residual_, self.modes_, self.mode_eigenvalues_ = fr.remove_modes_matrix(self.raw_, n_modes)
        self.n_removed_ = n_modes
        self.q_ = (N - n_modes) / self.n_effective_

        #standardised data with the modes projected out - what Ledoit-Wolf measures noise on
        Zs = Z - Z.mean(axis=1, keepdims=True)
        sd = Zs.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        Zs = Zs / sd
        if n_modes:
            self.mode_series_ = self.modes_.T @ Zs     # each removed mode through time
            Zs = Zs - self.modes_ @ self.mode_series_
        else:
            self.mode_series_ = np.zeros((0, T))

        cleaned = rmt.clean(self.residual_, self.clean, self.n_effective_,
                            n_removed=n_modes, X=Zs)
        self.cleaned_ = cleaned
        self.matrix_ = cm.partial_from_matrix(cleaned) if self.output == "partial" else cleaned
        return self.matrix_

    #-------- results -------------------------------------------------------------
    def effective_rank(self):
        return rmt.effective_rank(self.cleaned_)

    def describe_removal(self):
        factors, n_modes = fr.parse_remove(self.remove, self.n_observations_)
        parts = []
        if factors is not None:
            parts.append(f"{factors.shape[0]} factor{'s' * (factors.shape[0] > 1)}")
        if n_modes:
            parts.append(f"{n_modes} mode{'s' * (n_modes > 1)}")
        return " + ".join(parts) if parts else "nothing"

    def summary(self):
        return {
            "measure": self.measure,
            "removed": self.describe_removal(),
            "clean": self.clean,
            "robust": self.robust,
            "n_variables": self.n_variables_,
            "n_observations": self.n_observations_,
            "n_effective": round(self.n_effective_, 1),
            "q": round(self.q_, 3),
            "effective_rank_measured": round(rmt.effective_rank(self.measured_), 2),
            "effective_rank_final": round(self.effective_rank(), 2),
        }

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.matrix_, index=self.labels_, columns=self.labels_)

    def plot(self):
        from . import plotting
        return plotting.summary(self)
