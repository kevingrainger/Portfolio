#!/usr/bin/env python3
#-------- corrlib : sampling_estimators.py ----------------------------------
#-----------------------------------------------------------------------------
# Getting a correlation out of messy data.
#
# The problem this file solves: two assets do not trade at the same instant.
# Measure them second by second and you keep catching one just after it moved
# and the other just before. They look unrelated. Zoom out to five minutes and
# the correlation reappears.
#
# That is the Epps effect, and it is a measurement artefact - the market did not
# change, the measurement did. Every function here is a different way of
# refusing to be fooled by it.

import numpy as np
import matplotlib.pyplot as plt


#-------- Estimator 1 : realised covariance ---------------------------------
#-----------------------------------------------------------------------------
# The naive method. Force everything onto a common clock, take returns, sum the
# products. It has the Epps bias built in.
#
# It is in the library as a real option and not as a strawman, because when the
# data is noisy enough this plain method beats the clever ones.
def realised(prices, sampling_step=1):
    prices = np.asarray(prices, dtype=float)
    sampled = prices[:, ::sampling_step]                # take every nth point
    returns = np.diff(np.log(sampled), axis=1)          # log returns on that grid
    return returns @ returns.T                          # sum of products = realised covariance


#-------- Estimator 2 : lead-lag adjusted -----------------------------------
#-----------------------------------------------------------------------------
# If asset A's move shows up in asset B one step later, the plain estimator
# misses it entirely. Adding the products of neighbouring steps catches it.
# This is the simplest possible Epps correction and it recovers most of the loss.
def lead_lag_adjusted(prices, sampling_step=1):
    prices = np.asarray(prices, dtype=float)
    sampled = prices[:, ::sampling_step]
    r = np.diff(np.log(sampled), axis=1)

    C = r @ r.T                                         # same step
    C = C + r[:, 1:] @ r[:, :-1].T                      # A now against B one step back
    C = C + r[:, :-1] @ r[:, 1:].T                      # and the other way round
    return C


#-------- Estimator 3 : Hayashi-Yoshida -------------------------------------
#-----------------------------------------------------------------------------
# The proper answer to irregular timestamps: do not build a grid at all.
#
# Take every return of asset A and every return of asset B. If their time
# intervals OVERLAP, multiply them and add to the total. If they do not overlap,
# ignore the pair. No interpolation, no resampling, no Epps bias.
#
# Inputs are raw ticks - a time array and a price array per asset, different
# lengths allowed, which is the whole point.
def hayashi_yoshida(t1, p1, t2, p2):
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    r1 = np.diff(np.log(np.asarray(p1, dtype=float)))   # returns of asset 1
    r2 = np.diff(np.log(np.asarray(p2, dtype=float)))

    total = 0.0
    j = 0
    for i in range(len(r1)):
        start1, end1 = t1[i], t1[i + 1]                 # the interval this return covers
        # walk asset 2's intervals forward until they can possibly overlap
        while j < len(r2) and t2[j + 1] <= start1:
            j += 1
        k = j
        while k < len(r2) and t2[k] < end1:             # every interval that overlaps
            total += r1[i] * r2[k]
            k += 1
    return total


# The full covariance matrix from raw ticks - one time array and one price array
# per asset, all different lengths. Each pair is done by the overlap rule above;
# the diagonal is each asset's own realised variance.
def hayashi_yoshida_matrix(times, prices):
    N = len(times)
    C = np.zeros((N, N))
    for i in range(N):
        r = np.diff(np.log(np.asarray(prices[i], dtype=float)))
        C[i, i] = np.sum(r ** 2)
        for j in range(i + 1, N):
            C[i, j] = C[j, i] = hayashi_yoshida(times[i], prices[i], times[j], prices[j])
    return C


#-------- Estimator 4 : block averaging -------------------------------------
#-----------------------------------------------------------------------------
# The noise in a tick price is fast and the signal is slow, so average locally
# before measuring anything. Each averaged block has much less noise in it
# because the noise cancels and the signal does not.
#
# This is a low pass filter with a different name, which is worth saying out loud
# because it is the same idea as the seismic denoising work.
#
# Named honestly: this is plain block averaging, which also means sampling less
# often. It is not the pre-averaging estimator of Jacod et al. (2009), which
# uses overlapping weighted windows and a bias correction.
def block_averaged(prices, window=10):
    prices = np.asarray(prices, dtype=float)
    N, T = prices.shape
    n_blocks = T // window
    blocks = prices[:, :n_blocks * window].reshape(N, n_blocks, window)
    smoothed = blocks.mean(axis=2)                      # one value per block
    r = np.diff(np.log(smoothed), axis=1)
    return r @ r.T


#-------- The Epps curve ----------------------------------------------------
#-----------------------------------------------------------------------------
# Measure the correlation at every sampling speed and plot it. Fast sampling on
# the left, slow on the right. The curve climbs from near zero up to the true
# value, and where it flattens tells you how often you can honestly measure.
def epps_curve(prices, steps=None):
    if steps is None:
        steps = np.unique(np.logspace(0, 2.4, 25).astype(int))

    correlations = []
    for step in steps:
        C = realised(prices, sampling_step=int(step))
        d = np.sqrt(np.diag(C))
        d[d == 0] = 1.0
        correlations.append((C / np.outer(d, d))[0, 1]) # the first pair, as an example
    return np.array(steps), np.array(correlations)


#-------- Simulating ticks for the demonstration ----------------------------
#-----------------------------------------------------------------------------
# To show the Epps effect you need data where you already know the true answer.
# So: build two correlated price paths, then hide them behind two problems that
# real tick data has - a small random error on every printed price, and trades
# arriving at random times rather than on a clock.
def simulate_noisy_ticks(n=20000, true_correlation=0.7, noise=0.0004, seed=1):
    rng = np.random.default_rng(seed)

    z1 = rng.normal(size=n)
    z2 = true_correlation * z1 + np.sqrt(1 - true_correlation ** 2) * rng.normal(size=n)

    true_path = np.cumsum(np.vstack([z1, z2]) * 0.0005, axis=1)   # the real prices
    prices = np.exp(true_path) * 100.0
    observed = prices * (1 + rng.normal(0, noise, size=prices.shape))  # what gets printed
    return observed, prices


#-------- Plot : watch the correlation melt ---------------------------------
#-----------------------------------------------------------------------------
# The figure that justifies this whole file. Sample fast and the measured
# correlation collapses toward zero even though nothing about the market changed.
def plot_epps_curve(prices, true_correlation=None):
    steps, corr = epps_curve(prices)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, corr, 'o-', color='k', ms=4, lw=1.2, label='measured correlation')
    if true_correlation is not None:
        ax.axhline(true_correlation, color='r', ls='--', lw=1.2, label='true correlation')

    ax.set_xscale('log')
    ax.set_xlabel('sampling interval (ticks per sample) - slower to the right')
    ax.set_ylabel('measured correlation')
    ax.set_title('The Epps effect: sample too fast and the correlation disappears')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


#-------- Effective sample size : how much independent data is there? --------
#-----------------------------------------------------------------------------
# Hourly solar output or daily returns are not independent from one observation
# to the next - today looks like yesterday. So 5,000 hours of data carry far less
# information than 5,000 independent draws, and every confidence interval built
# on n = 5,000 is too narrow. The noise band is too narrow too, since q uses n.
#
# For two series with autocorrelations rho_x(k) and rho_y(k), the variance of
# their sample correlation behaves as if there were
#
#       n_eff = n / (1 + 2 sum_k rho_x(k) rho_y(k))
#
# independent observations (Bartlett 1946; Bayley & Hammersley 1946). For a whole
# matrix, average the product over all pairs - which is the square of the average
# autocorrelation, summed over lags.
#
# The sum is cut off at max_lag, by default sqrt(n), because far lags are pure
# noise and summing them adds error without adding information.
def _autocorrelations(X, max_lag):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=1, keepdims=True)
    var = (Xc ** 2).mean(axis=1)
    var[var == 0] = 1.0
    T = X.shape[1]
    return np.array([(Xc[:, k:] * Xc[:, :T - k]).mean(axis=1) / var
                     for k in range(1, max_lag + 1)])  # shape (max_lag, n_variables)


def effective_n(X, max_lag=None):
    X = np.asarray(X, dtype=float)
    T = X.shape[1]
    if max_lag is None:
        max_lag = int(np.sqrt(T))
    rho = _autocorrelations(X, max_lag)
    mean_rho = rho.mean(axis=1)                         # average autocorrelation at each lag
    inflation = 1 + 2 * np.sum(mean_rho ** 2)
    return T / max(inflation, 1.0)


# The same idea for one pair, used when a single correlation needs an error bar.
def effective_n_pair(x, y, max_lag=None):
    X = np.vstack([x, y])
    T = X.shape[1]
    if max_lag is None:
        max_lag = int(np.sqrt(T))
    rho = _autocorrelations(X, max_lag)
    return T / max(1 + 2 * np.sum(rho[:, 0] * rho[:, 1]), 1.0)


#-------- Block bootstrap : an error bar without a formula ------------------
#-----------------------------------------------------------------------------
# Resample the data in contiguous blocks rather than single observations, so
# whatever autocorrelation is inside a block survives the resampling. Recompute
# the correlation on each resample; the spread of the answers is the error bar.
# No distributional assumptions at all, which is why it is the check on n_eff.
def block_bootstrap(X, statistic, block_length=None, n_boot=500, seed=0):
    X = np.asarray(X, dtype=float)
    T = X.shape[1]
    if block_length is None:
        block_length = max(1, int(round(T ** (1 / 3))))  # a standard rule of thumb
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block_length))

    results = []
    for _ in range(n_boot):
        starts = rng.integers(0, T - block_length + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_length)[None, :]).ravel()[:T]
        results.append(statistic(X[:, idx]))
    return np.array(results)
