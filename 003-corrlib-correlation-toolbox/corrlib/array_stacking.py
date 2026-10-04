#!/usr/bin/env python3
#-------- corrlib : array_stacking.py ---------------------------------------
#-----------------------------------------------------------------------------
# Many sensors, one shared signal.
#
# Ten seismometers record the same earthquake plus their own local noise - a
# lorry outside one, a factory near another. You want the earthquake, and you
# have ten noisy copies of it.
#
# Averaging works if the noise is independent: the signal survives, the noise
# partly cancels, and with N sensors the noise falls by sqrt(N). But two stations
# on the same road share the same traffic, and shared noise survives averaging
# exactly as well as the signal does. So the real question is what WEIGHTED
# combination is best, given that the noise is correlated - and answering that
# needs a covariance matrix, which is why this lives in corrlib.
#
# Same file, two directions:
#   extract  - sensors known, signal unknown      (seismic stations -> earthquake)
#   explain  - driver known, exposures unknown    (weather -> solar farm output)

import numpy as np
import matplotlib.pyplot as plt

from . import random_matrix_cleaning as rmt
from . import correlation_measures as cm


#-------- Stack 1 : plain average -------------------------------------------
#-----------------------------------------------------------------------------
# Every sensor weighted equally. The baseline, and genuinely hard to beat when
# there are few samples to estimate anything better from.
def simple(X):
    X = np.asarray(X, dtype=float)
    weights = np.ones(X.shape[0]) / X.shape[0]
    return weights @ X, weights


#-------- Stack 2 : weighted by how good each sensor is ----------------------
#-----------------------------------------------------------------------------
# A sensor sitting next to a motorway should count for less. Weight each one by
# its own signal-to-noise ratio, measured from a quiet window before the arrival
# and a loud window after it.
def snr_weighted(X, noise_window, signal_window):
    X = np.asarray(X, dtype=float)
    noise_power = X[:, noise_window[0]:noise_window[1]].var(axis=1)
    signal_power = X[:, signal_window[0]:signal_window[1]].var(axis=1)
    noise_power[noise_power <= 0] = 1e-12

    snr = signal_power / noise_power
    weights = snr / snr.sum()
    return weights @ X, weights


#-------- Stack 3 : the optimal weights -------------------------------------
#-----------------------------------------------------------------------------
# The best possible weighted combination when the noise is correlated:
#
#       w = inverse(C) a / (a' inverse(C) a)
#
# where a is the steering vector - how the signal appears at each sensor. If the
# signal hits everything at once with the same strength, a is all ones.
#
# This is the Capon beamformer from array processing (1969). It is also, exactly,
# the minimum variance portfolio from finance (1952) - see min_variance below.
# Two fields, seventeen years apart, same equation.
#
# The inverse is the dangerous part. A covariance matrix estimated from limited
# data has small noisy eigenvalues, and inverting turns them into huge ones, so
# the weights explode. Cleaning the matrix first is what stops that, and this is
# where random_matrix_cleaning earns its place.
#
# Which data the covariance comes from matters. Estimated from the whole trace,
# the earthquake itself is inside C, and the method then actively suppresses
# anything that looks like it - with the steering vector even slightly wrong,
# it partly cancels the signal it is meant to keep (array processing calls this
# MPDR rather than MVDR). So pass noise_window: a stretch before the arrival,
# where there is only noise, and C is estimated from that alone.
def mvdr(X, steering=None, clean="linear_shrinkage", noise_window=None):
    X = np.asarray(X, dtype=float)
    N, T = X.shape

    if steering is None:
        steering = np.ones(N)                           # signal arrives the same everywhere

    noise = X if noise_window is None else X[:, noise_window[0]:noise_window[1]]
    C = np.cov(noise)
    C = _apply_cleaner(C, noise.shape[1], clean, X=noise)

    C_inv = np.linalg.pinv(C)
    weights = C_inv @ steering / (steering @ C_inv @ steering)
    return weights @ X, weights


# The same formula with the steering vector set to all ones and no signal to
# extract. Kept as its own function because the portfolio projects call it by
# this name, and because putting them side by side is the point.
#
# Pass the data X as well if you have it - Ledoit-Wolf then measures the noise
# directly instead of assuming Gaussian data.
def min_variance(C, clean=None, n_observations=None, X=None):
    C = np.asarray(C, dtype=float)
    if clean is not None:
        C = _apply_cleaner(C, n_observations, clean, X=X)

    ones = np.ones(C.shape[0])
    C_inv = np.linalg.pinv(C)
    return C_inv @ ones / (ones @ C_inv @ ones)


def _apply_cleaner(C, n_observations, clean, X=None):
    return rmt.clean(C, clean, n_observations, X=X)


#-------- Stack 4 : the common mode -----------------------------------------
#-----------------------------------------------------------------------------
# Sometimes you do not know the steering vector, because you do not know where
# the signal came from. In that case let the data decide: the strongest pattern
# of joint movement across the sensors IS the shared signal.
#
# That pattern is the top eigenvector of the covariance matrix, and projecting
# onto it is the same eigen-denoising as in Seismic_denoising.py - here with the
# rest of the library's cleaning available behind it.
#
# Returns (stacked trace, weights) like every other stacker: the weights are the
# top eigenvector scaled to sum to one, so the stack keeps the signal's size.
# With reconstruct=True it instead returns every sensor's trace rebuilt from the
# top n_components modes - the denoised array rather than one stacked trace.
def common_mode(X, n_components=1, reconstruct=False):
    X = np.asarray(X, dtype=float)
    X_centered = X - X.mean(axis=1, keepdims=True)

    C = np.cov(X_centered)
    eigenvals, eigenvecs = rmt.eigen_spectrum(C)

    if reconstruct:
        keep = eigenvecs[:, :n_components]              # the strongest joint patterns
        return keep @ (keep.T @ X_centered), keep       # project onto them, rebuild

    top = eigenvecs[:, 0]
    top = top * np.sign(top.sum())                      # eigenvectors come with an arbitrary sign
    weights = top / top.sum()
    return weights @ X, weights


#-------- Aligning before stacking ------------------------------------------
#-----------------------------------------------------------------------------
# If the wave reaches sensor 2 half a second after sensor 1, averaging them
# smears the arrival out. So estimate the delay first, shift each trace back,
# then stack. This is delay-and-sum, the oldest method in array processing.
#
# The delay estimate is just lead_lag from correlation_measures - the same
# function that asks which market moves first.
def align(X, reference=0, max_lag=200):
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    shifts = np.zeros(N, dtype=int)

    for i in range(N):
        if i == reference:
            continue
        _, _, best_lag = cm.lead_lag(X[reference], X[i], max_lag=max_lag)
        shifts[i] = best_lag

    aligned = np.zeros_like(X)
    for i in range(N):
        aligned[i] = _shift(X[i], -shifts[i])           # slide each trace onto the reference
    return aligned, shifts


# Shift a trace by a whole number of samples, filling the gap with zeros. Not
# np.roll, which wraps the end of the trace round to the start - for a seismogram
# that would paste the coda in front of the arrival.
def _shift(x, k):
    out = np.zeros_like(x)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


#-------- The other direction : explain, do not extract ---------------------
#-----------------------------------------------------------------------------
# Here the shared driver is already known - a regional irradiance index, say -
# and what you want is how strongly each site responds to it, plus how much of
# its behaviour is left unexplained.
#
# The loading is a regression slope, and a site with a small loading is one that
# genuinely diversifies your fleet. That is the input the solar project needs.
#
# Several drivers at once are fine - pass them as rows, e.g. the top few weather
# modes - and each site gets one loading per driver. One driver passed as a flat
# array gives one loading per site, as before.
#
# This is the same regression as factor_removal.regress_out, read from the other
# end: there the residual is the point, here the loadings are.
def explain(X, drivers):
    X = np.asarray(X, dtype=float)
    single = np.ndim(drivers) == 1
    D = np.atleast_2d(np.asarray(drivers, dtype=float))

    Dc = D - D.mean(axis=1, keepdims=True)
    Xc = X - X.mean(axis=1, keepdims=True)

    loadings = Xc @ Dc.T @ np.linalg.pinv(Dc @ Dc.T)    # slope of each site on each driver
    explained = loadings @ Dc                           # the part the drivers account for
    residuals = Xc - explained                          # whatever is left over

    var = Xc.var(axis=1)
    var[var == 0] = 1.0
    shared_fraction = 1 - residuals.var(axis=1) / var
    if single:
        loadings = loadings[:, 0]
    return loadings, residuals, shared_fraction


#-------- Scoring : did the noise actually go down? -------------------------
#-----------------------------------------------------------------------------
# Array gain in decibels - how much better the stack is than one sensor alone.
# Every method above gets reported through this, so they are comparable.
def gain(stacked, single, noise_window):
    noise_stack = stacked[noise_window[0]:noise_window[1]].std()
    noise_single = single[noise_window[0]:noise_window[1]].std()
    if noise_stack <= 0:
        return np.inf
    return 20 * np.log10(noise_single / noise_stack)


#-------- Plot : where each method stops working ----------------------------
#-----------------------------------------------------------------------------
# Stack 1 sensor, then 2, then 3, and watch the noise fall. With independent
# noise the plain average tracks the theoretical sqrt(N) line. Switch the noise
# to correlated and it stalls - more sensors stop helping - while the optimal
# weights carry on. Cleaning is what stops the optimal weights blowing up once
# there are too many sensors for the amount of data.
def plot_gain_vs_sensors(X, noise_window, clean="linear_shrinkage"):
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    counts = np.arange(2, N + 1)

    gains = {"plain average": [], "optimal weights": [], "optimal + cleaned": []}
    for n in counts:
        sub = X[:n]
        s_simple, _ = simple(sub)
        s_mvdr, _ = mvdr(sub, clean=None, noise_window=noise_window)
        s_clean, _ = mvdr(sub, clean=clean, noise_window=noise_window)

        gains["plain average"].append(gain(s_simple, sub[0], noise_window))
        gains["optimal weights"].append(gain(s_mvdr, sub[0], noise_window))
        gains["optimal + cleaned"].append(gain(s_clean, sub[0], noise_window))

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, series in gains.items():
        ax.plot(counts, series, 'o-', ms=4, lw=1.2, label=name)
    ax.plot(counts, 10 * np.log10(counts), 'k--', lw=1,
            label='theory: independent noise')

    ax.set_xlabel('number of sensors stacked')
    ax.set_ylabel('noise reduction (dB)')
    ax.set_title('Stacking: what helps, and when it stops helping')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
