#!/usr/bin/env python3
#-------- corrlib : correlation_measures.py ---------------------------------
#-----------------------------------------------------------------------------
# The different questions people call "correlation".
# Everything here takes X with shape (n_variables, n_observations) - so rows are
# assets, or stations, or solar sites. Same shape everywhere in the library.
#
# Most of these exist in scipy. They are written out here because the whole point
# of the project is showing the maths, and none of them are more than ten lines.

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtri                         # inverse of the normal cdf
from scipy.stats import rankdata, kendalltau


#-------- Pearson : the straight line one -----------------------------------
#-----------------------------------------------------------------------------
# Standardise each row (subtract its mean, divide by its standard deviation) and
# the correlation matrix is just the average product of the standardised rows.
# That is all correlation is - the covariance of two things measured in units of
# their own spread.
def pearson(X):
    X = np.asarray(X, dtype=float)
    X_centered = X - X.mean(axis=1, keepdims=True)      # remove each row's mean
    row_sd = X_centered.std(axis=1, keepdims=True)      # each row's own spread
    row_sd[row_sd == 0] = 1.0                           # flat rows would divide by zero
    Z = X_centered / row_sd                             # now every row has sd 1
    return (Z @ Z.T) / X.shape[1]                       # average product of pairs


#-------- Rank correlations : only the ordering matters ----------------------
#-----------------------------------------------------------------------------
# Replace every value by its position in the sorted list, then do Pearson on the
# positions. One freak day can no longer dominate, because the biggest value is
# only ever "rank n", however extreme it was.
#
# Tied values share the average of their ranks. That matters more than it
# sounds: solar output is exactly zero for half of every day, and giving those
# zeros different ranks would invent an ordering that does not exist.
def _rank_rows(X):
    return rankdata(X, axis=1)                          # 1..n, ties averaged


def spearman(X):
    return pearson(_rank_rows(np.asarray(X, dtype=float)))


#-------- Gaussian rank : ranks, then put back on a bell curve ---------------
#-----------------------------------------------------------------------------
# Rank each row, turn the ranks into the matching quantiles of a normal
# distribution, then take Pearson. The ordering is all that is used, so one
# freak day cannot dominate - but unlike Spearman, the result is on the same
# scale as an ordinary correlation of Gaussian data, so the noise band and
# every cleaner apply to it unchanged.
#
# This is the rigorous choice for bounded or skewed data, like a solar clear-sky
# index that piles up just below 1 on clear days. It is also known as the
# normal-scores or van der Waerden correlation, and it is what a Gaussian copula
# fit does under the hood.
def gaussian_scores(X):
    X = np.asarray(X, dtype=float)
    n = X.shape[1]
    return ndtri((_rank_rows(X) - 0.5) / n)             # rank -> normal quantile


def gaussian_rank(X):
    return pearson(gaussian_scores(X))


# Kendall counts pairs instead of ranking. For every pair of days, do the two
# series move the same way or opposite ways? tau = (agreements - disagreements)
# over the number of pairs.
#
# Counting every pair directly is O(n^2); scipy does the same count in
# O(n log n) by sorting, which is the only reason it is used here.
def kendall(x, y):
    tau, _ = kendalltau(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return tau


# The full matrix, so Kendall obeys the same contract as everything else.
def kendall_matrix(X):
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    K = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            K[i, j] = K[j, i] = kendall(X[i], X[j])
    return K


# Kendall's tau converts straight to a correlation for elliptical data, and the
# conversion needs no moments at all - which is why it survives fat tails.
# This is the robust way into the cleaning step in random_matrix_cleaning.py.
def kendall_to_pearson(tau):
    return np.sin(np.pi * tau / 2.0)


#-------- Partial correlation : with everything else taken out ---------------
#-----------------------------------------------------------------------------
# Two bank stocks look correlated. Is that a bank thing, or is it just that
# everything moves with the market? Partial correlation answers that.
# The trick: invert the correlation matrix, and the inverse holds the answer
# directly - each off-diagonal entry, normalised, is the correlation between two
# variables with every other variable already held fixed.
#
# Inverting amplifies noise, so with limited data clean the matrix first and use
# partial_from_matrix - which is what the Correlator does.
def partial(X):
    return partial_from_matrix(pearson(X))


def partial_from_matrix(C):
    P = np.linalg.pinv(np.asarray(C, dtype=float))      # pinv in case C is singular
    d = np.sqrt(np.diag(P))
    partial_corr = -P / np.outer(d, d)                  # minus sign comes out of the algebra
    np.fill_diagonal(partial_corr, 1.0)
    return partial_corr


#-------- Distance correlation : related in ANY shape ------------------------
#-----------------------------------------------------------------------------
# Pearson only sees straight lines. Two series can be perfectly related (y = x^2)
# and score zero. Distance correlation scores zero only if they are genuinely
# independent.
# How it works: instead of comparing values, compare all the distances between
# pairs of values. If x and y are related, days that are far apart in x tend to
# be far apart in y as well.
def distance_correlation(x, y):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = len(x)

    a = np.abs(x - x.T)                                 # all pairwise distances in x
    b = np.abs(y - y.T)                                 # all pairwise distances in y

    # double centering - take off row means, column means, add back the grand mean
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()

    dcov2 = (A * B).sum() / (n * n)                     # distance covariance, squared
    dvar_x = (A * A).sum() / (n * n)
    dvar_y = (B * B).sum() / (n * n)

    if dvar_x <= 0 or dvar_y <= 0:
        return 0.0
    return np.sqrt(dcov2 / np.sqrt(dvar_x * dvar_y))


# The full matrix. Builds an n x n distance table per pair, so keep n to a few
# thousand observations.
def distance_correlation_matrix(X):
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    D = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = D[j, i] = distance_correlation(X[i], X[j])
    return D


#-------- Tail dependence : do they crash together? --------------------------
#-----------------------------------------------------------------------------
# A separate question from correlation, and the one that matters for risk.
# Take the worst q% of days for x. What fraction of those were also in the worst
# q% for y? If the two were independent you would get q back. If they crash
# together you get much more.
def tail_dependence(x, y, quantile=0.05, lower=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if lower:
        x_extreme = x <= np.quantile(x, quantile)       # x in its own worst tail
        y_extreme = y <= np.quantile(y, quantile)
    else:
        x_extreme = x >= np.quantile(x, 1 - quantile)
        y_extreme = y >= np.quantile(y, 1 - quantile)

    if x_extreme.sum() == 0:
        return 0.0
    return np.mean(y_extreme[x_extreme])                # of x's bad days, how many were bad for y


# The same number as a curve rather than one value. Push further into the tail
# (smaller q) and watch what happens. If it settles on something positive the
# assets genuinely crash together. If it slides to zero they do not, and ordinary
# correlation was overstating the joint risk.
def chi_curve(x, y, quantiles=None):
    if quantiles is None:
        quantiles = np.linspace(0.30, 0.01, 30)
    chi = [tail_dependence(x, y, q) for q in quantiles]
    return np.array(quantiles), np.array(chi)


#-------- Lead and lag : who moves first? ------------------------------------
#-----------------------------------------------------------------------------
# Slide one series against the other and correlate at each shift. The shift with
# the strongest correlation is the delay between them.
# Used two ways in this portfolio: which market leads which, and how long a
# seismic wave takes to travel between two stations.
def lead_lag(x, y, max_lag=20):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.zeros(len(lags))

    for k, lag in enumerate(lags):
        if lag < 0:
            a, b = x[-lag:], y[:lag]                    # x shifted back
        elif lag > 0:
            a, b = x[:-lag], y[lag:]                    # y shifted back
        else:
            a, b = x, y
        if len(a) > 2:
            out[k] = pearson(np.vstack([a, b]))[0, 1]

    best_lag = lags[np.argmax(np.abs(out))]             # strongest relationship, either sign
    return lags, out, best_lag


#-------- Rolling correlation : does it hold through time? -------------------
#-----------------------------------------------------------------------------
# One number for ten years of data hides everything interesting. This is the
# cheapest way to see a relationship break down.
def rolling(x, y, window=60):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(len(x), np.nan)
    for t in range(window, len(x)):
        out[t] = pearson(np.vstack([x[t - window:t], y[t - window:t]]))[0, 1]
    return out


#-------- Plot : same number, different reality ------------------------------
#-----------------------------------------------------------------------------
# Four data sets, all with roughly the same Pearson correlation, all completely
# different. This is the figure that justifies everything above existing.
def plot_same_number_different_reality(target=0.6, n=400, seed=0):
    rng = np.random.default_rng(seed)

    #1. the honest case - a tilted cloud
    a = rng.normal(size=n)
    clean_x = a
    clean_y = target * a + np.sqrt(1 - target**2) * rng.normal(size=n)

    #2. no relationship at all, plus a few extreme days doing all the work
    out_x = rng.normal(size=n)
    out_y = rng.normal(size=n)
    out_x[:12] = rng.normal(4, 0.4, 12)                 # the outliers
    out_y[:12] = rng.normal(4, 0.4, 12)

    #3. two separate regimes, neither of which is correlated on its own
    reg_x = np.concatenate([rng.normal(-1.2, 0.5, n // 2), rng.normal(1.2, 0.5, n // 2)])
    reg_y = np.concatenate([rng.normal(-1.2, 0.5, n // 2), rng.normal(1.2, 0.5, n // 2)])

    #4. a fan - correlation is real but the spread grows
    fan_x = rng.normal(size=n)
    fan_y = target * fan_x + np.abs(fan_x) * rng.normal(size=n) * 0.8

    panels = [("straight line", clean_x, clean_y),
              ("twelve days doing the work", out_x, out_y),
              ("two regimes", reg_x, reg_y),
              ("widening fan", fan_x, fan_y)]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (title, px, py) in zip(axes, panels):
        r = pearson(np.vstack([px, py]))[0, 1]
        tail = tail_dependence(px, py, 0.05)
        ax.scatter(px, py, s=6, alpha=0.5, color='k')
        ax.set_title(f"{title}\nPearson {r:.2f}   tail dep {tail:.2f}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return fig
