#!/usr/bin/env python3
#-------- corrlib : factor_removal.py ----------------------------------------
#-----------------------------------------------------------------------------
# Taking out the shared influences before measuring what is left.
#
# Two bank stocks look correlated. Is that a bank thing, or is it just that the
# whole economy moves them both? Two seismometers look correlated - an earthquake,
# or the same lorry passing both? Two solar farms look correlated - shared cloud,
# or just that the sun rises at both?
#
# Same question every time, and two ways to answer it:
#
#   regress_out    you know what the shared influence is (a market index, a gas
#                  price, the clear-sky profile). Regress every variable on it and
#                  keep what is left.
#   remove_modes   you don't know what it is, only that it is there. The strongest
#                  patterns of joint movement are the eigenvectors with the biggest
#                  eigenvalues; take the top k out.
#
# (The third way, partial correlation, holds every OTHER variable fixed rather
# than a chosen factor. It lives in correlation_measures.)
#
# The cost of each, and this is the part that keeps the noise theory honest:
#
#   regressing out k known factors projects every row onto the same smaller
#   space in TIME - so it costs k observations, and n_observations drops to n - k
#
#   removing k modes deletes k directions in VARIABLE space - so it costs k
#   variables, the residual matrix has rank N - k, and the noise band has to be
#   computed for N - k variables
#
# Both shift q. Forgetting either makes the cleaners treat noise as signal.

import numpy as np

from .random_matrix_cleaning import eigen_spectrum


#-------- Known factors : regress them out -----------------------------------
#-----------------------------------------------------------------------------
# Ordinary least squares, all variables at once. Each variable gets its own
# loading on each factor - how strongly it responds - and the residual is the
# part the factors cannot explain.
#
#   X : (n_variables, n_observations)      factors : (n_factors, n_observations)
#
# Returns the residuals and the loadings (n_variables, n_factors).
def regress_out(X, factors):
    X = np.asarray(X, dtype=float)
    F = np.atleast_2d(np.asarray(factors, dtype=float))
    if F.shape[1] != X.shape[1]:
        raise ValueError(f"factors have {F.shape[1]} observations, X has {X.shape[1]}")

    Xc = X - X.mean(axis=1, keepdims=True)
    Fc = F - F.mean(axis=1, keepdims=True)
    loadings = Xc @ Fc.T @ np.linalg.pinv(Fc @ Fc.T)   # slope of every variable on every factor
    residuals = Xc - loadings @ Fc
    return residuals, loadings


#-------- Unknown factors : remove the strongest modes ------------------------
#-----------------------------------------------------------------------------
# Matrix version: subtract the top k eigen-components and rescale back to a
# correlation matrix. What is left is the correlation each pair has beyond the
# shared modes.
#
# Returns the residual correlation matrix, the removed eigenvectors (each one is
# a pattern of loadings across the variables - the "market" usually looks like
# all positive, all similar), and the removed eigenvalues.
def remove_modes_matrix(C, n_modes):
    C = np.asarray(C, dtype=float)
    if n_modes == 0:
        return C.copy(), np.zeros((C.shape[0], 0)), np.zeros(0)

    eigenvals, eigenvecs = eigen_spectrum(C)
    U = eigenvecs[:, :n_modes]
    lam = eigenvals[:n_modes]
    residual = C - U @ np.diag(lam) @ U.T

    d = np.sqrt(np.clip(np.diag(residual), 1e-12, None))
    residual = residual / np.outer(d, d)                # back to ones on the diagonal
    return residual, U, lam


# Data version: project the standardised data off the top k modes. Gives the
# residual series themselves, and the time series of each removed mode - which
# is often worth looking at. "What is the shared signal?" can be as interesting
# as what is left once it is gone.
def remove_modes(X, n_modes):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=1, keepdims=True)
    sd = Xc.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Z = Xc / sd

    C = (Z @ Z.T) / Z.shape[1]
    _, eigenvecs = eigen_spectrum(C)
    U = eigenvecs[:, :n_modes]
    mode_series = U.T @ Z                               # each mode's value through time
    residuals = (Z - U @ mode_series) * sd              # back in the original units
    return residuals, U, mode_series


#-------- The "remove" argument, as the Correlator reads it -------------------
#-----------------------------------------------------------------------------
# remove can be:
#   None                       nothing removed
#   an int k                   remove the top k modes
#   an array                   regress out that factor (or factors, one per row)
#   a list of the above        all of them - known factors first, then modes
#
# Splits it into the known factors (one array, stacked) and the total number of
# modes, so the caller can do each at the right stage.
def parse_remove(remove, n_observations):
    if remove is None:
        return None, 0
    items = remove if isinstance(remove, (list, tuple)) else [remove]

    factors, n_modes = [], 0
    for item in items:
        if isinstance(item, (int, np.integer)):
            n_modes += int(item)
        else:
            arr = np.atleast_2d(np.asarray(item, dtype=float))
            if arr.shape[1] != n_observations and arr.shape[0] == n_observations:
                arr = arr.T                             # accept (T, k) as well as (k, T)
            factors.append(arr)
    F = np.vstack(factors) if factors else None
    return F, n_modes
