#!/usr/bin/env python3
#-------- corrlib : random_matrix_cleaning.py -------------------------------
#-----------------------------------------------------------------------------
# How much of a correlation matrix is real?
#
# With 100 assets there are 4,950 pairwise correlations to estimate, and maybe
# 250 days to estimate them from. There is not enough information, so most of
# what comes out is noise. Random matrix theory tells you exactly what the
# eigenvalues look like when there is genuinely nothing there, so anything inside
# that range can be treated as noise and anything outside it is probably real.
#
# The Marchenko-Pastur cleaner here is the one from Energy_portfolio_manager,
# tidied and given the rest of the ladder around it.
#
# Every cleaner has the same signature:
#
#       cleaner(C, n_observations, n_removed=0, X=None) -> cleaned C
#
# C can be a correlation or a covariance matrix - each cleaner works on the
# correlation and puts the original variances back at the end. n_removed is the
# number of shared modes taken out by factor_removal; those directions are left
# at zero rather than "cleaned" back into existence. X is the data, which only
# Ledoit-Wolf uses.

import numpy as np
import matplotlib.pyplot as plt


#-------- The noise band ----------------------------------------------------
#-----------------------------------------------------------------------------
# q is the single most important number in this file: variables divided by
# observations. q small means plenty of data and little noise. q near 1 means
# the matrix is mostly noise.
#
# For pure noise the eigenvalues fall between these two edges and nowhere else.
#
# If k shared modes have been removed, only N - k directions are left to carry
# noise, so the band is computed for N - k variables.
def noise_band(n_variables, n_observations, variance=1.0, n_removed=0):
    q = (n_variables - n_removed) / n_observations
    lambda_minus = variance * (1 - np.sqrt(q)) ** 2
    lambda_plus = variance * (1 + np.sqrt(q)) ** 2
    return lambda_minus, lambda_plus


# The shape of the noise, not just its edges. Used to draw the theory curve on
# top of the measured histogram - if the bulk sits under this curve, the bulk is
# noise, and that is the whole argument made visually.
def marchenko_pastur_density(x, n_variables, n_observations, variance=1.0, n_removed=0):
    q = (n_variables - n_removed) / n_observations
    lo, hi = noise_band(n_variables, n_observations, variance, n_removed)
    x = np.asarray(x, dtype=float)
    density = np.zeros_like(x)
    inside = (x > lo) & (x < hi)                        # the density is zero outside the band
    density[inside] = np.sqrt((hi - x[inside]) * (x[inside] - lo)) / (2 * np.pi * q * variance * x[inside])
    return density


#-------- How big is the noise? ---------------------------------------------
#-----------------------------------------------------------------------------
# The band's edges scale with the noise variance, and that is not 1. Real
# structure (a market mode, sector blocks) takes its share of the total, and the
# noise only gets what is left. Laloux et al. (1999) subtract the biggest
# eigenvalue and share out the rest.
#
# This uses the median instead: the median eigenvalue of pure noise is a known
# multiple of the noise variance, so matching the measured median to it gives
# the variance. A handful of big modes cannot drag a median around, however big
# they are.
#
# (The first attempt here iterated - estimate the variance, drop everything past
# the edge, re-average, repeat. Clean on synthetic data, but on real returns,
# where weak structure fades continuously into noise, each round pulled the edge
# lower until nearly everything counted as signal. The median does not cascade.)
def _noise_median(q):
    lo, hi = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    #integrate in u where x = lo + (hi - lo) u^2: at q = 1 the density blows up like
    #1/sqrt(x) at zero, and this substitution makes the integrand smooth there
    u = np.linspace(0, 1, 20001)[1:]
    x = lo + (hi - lo) * u ** 2
    density = np.sqrt(np.clip((hi - x) * (x - lo), 0, None)) / (2 * np.pi * q * x)
    integrand = density * 2 * (hi - lo) * u              # density * dx/du
    cdf = max(0.0, 1 - 1 / q) + np.concatenate([[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(u))])
    if cdf[0] >= 0.5:                                   # more than half the eigenvalues are zero
        return None
    return np.interp(0.5, cdf, x)


def bulk_variance(eigenvals, n_observations):
    vals = np.asarray(eigenvals, dtype=float)
    median_unit = _noise_median(len(vals) / n_observations)
    if median_unit is None:
        return vals.mean()                              # pure noise has mean = variance
    return np.median(vals) / median_unit


#-------- Eigen decomposition, biggest first --------------------------------
#-----------------------------------------------------------------------------
# numpy returns eigenvalues smallest first, which is the opposite of how anyone
# thinks about them. Flip once here so nothing downstream has to remember.
def eigen_spectrum(C):
    eigenvals, eigenvecs = np.linalg.eigh(np.asarray(C, dtype=float))
    return eigenvals[::-1], eigenvecs[:, ::-1]


# Covariance -> correlation and back. Every cleaner works on the correlation,
# because that is what the noise theory describes, then restores the variances.
def _to_correlation(C):
    sd = np.sqrt(np.diag(C))
    sd[sd == 0] = 1.0
    return C / np.outer(sd, sd), sd


# Cleaning changes the diagonal slightly, and a correlation matrix has to have
# ones down the diagonal. Rescale so it does, then put the original variances
# back, keeping the off-diagonal structure.
def _restore_diagonal(cleaned, original):
    d = np.sqrt(np.abs(np.diag(cleaned)))
    d[d == 0] = 1.0
    corr = cleaned / np.outer(d, d)
    scale = np.sqrt(np.diag(original))
    return corr * np.outer(scale, scale)


#-------- Cleaner 0 : do nothing --------------------------------------------
#-----------------------------------------------------------------------------
# The baseline. Every other method has to beat this, and sometimes none of them
# do, which is worth finding out.
def raw(C, n_observations=None, n_removed=0, X=None):
    return np.asarray(C, dtype=float)


#-------- Cleaner 1 : clip the noise ----------------------------------------
#-----------------------------------------------------------------------------
# Everything inside the noise band is indistinguishable from nothing, so replace
# all of it with one flat value - their average, which keeps the total variance
# the same. The eigenvalues sticking out above the band are left alone.
#
# The noise variance is not 1. If there is a big market mode it has taken its
# share of the total, so the noise gets what is left (Laloux et al. 1999) - see
# bulk_variance. Using 1 puts the band edge too high and throws away real
# structure.
def clip(C, n_observations, n_removed=0, X=None):
    C = np.asarray(C, dtype=float)
    R, sd = _to_correlation(C)
    eigenvals, eigenvecs = eigen_spectrum(R)
    live = len(eigenvals) - n_removed                   # directions that still exist
    vals = eigenvals[:live].copy()

    variance = bulk_variance(vals, n_observations)
    _, lambda_plus = noise_band(live, n_observations, variance)

    noise = vals < lambda_plus
    if noise.sum() > 0:
        vals[noise] = vals[noise].mean()                # one flat value for the whole bulk

    cleaned_vals = np.concatenate([vals, eigenvals[live:]])   # removed directions stay at zero
    cleaned = eigenvecs @ np.diag(cleaned_vals) @ eigenvecs.T
    return _restore_diagonal(cleaned, R) * np.outer(sd, sd)


#-------- Cleaner 2 : linear shrinkage (Ledoit-Wolf) ------------------------
#-----------------------------------------------------------------------------
# Pull every eigenvalue the same fraction of the way toward the average. Simple,
# hard to beat, and it never produces a matrix you cannot invert.
#
# The fraction is not a guess. Ledoit and Wolf (2004) worked out the value that
# minimises the expected error, and it depends on how noisy each entry of the
# matrix is:
#
#       intensity = (how much the sample matrix wobbles around the truth)
#                   / (how far the sample matrix is from the target)
#
# The wobble needs the data X. Without X it is estimated assuming the data is
# roughly Gaussian, which is fine for most uses and slightly under-shrinks for
# fat-tailed data.
#
# Worth knowing: in array processing the identical operation is called diagonal
# loading, because adding a constant to the diagonal is exactly what this does.
# Two fields, two names, same fix.
def ledoit_wolf_intensity(C, n_observations, X=None, n_removed=0):
    C = np.asarray(C, dtype=float)
    target = _shrinkage_target(C, n_removed)
    distance = np.sum((C - target) ** 2)                # how far from the target
    if distance <= 0:
        return 0.0

    if X is not None:
        Xc = np.asarray(X, dtype=float)
        Xc = Xc - Xc.mean(axis=1, keepdims=True)
        row_sd = Xc.std(axis=1, keepdims=True)
        row_sd[row_sd == 0] = 1.0
        Xc = Xc / row_sd * np.sqrt(np.diag(C))[:, None]    # match C's scale
        T = Xc.shape[1]
        #average of |x x' - C|^2 over days, without building T matrices
        wobble = (np.sum(np.sum(Xc ** 2, axis=0) ** 2) / T - np.sum(C ** 2)) / T
    else:
        #the same quantity if the data were Gaussian: (tr(C)^2 + tr(C^2)) / T
        wobble = (np.trace(C) ** 2 + np.sum(C ** 2)) / n_observations

    return float(np.clip(wobble / distance, 0.0, 1.0))


# The "everything average" matrix. If modes were removed it lives only in the
# directions that are left, so shrinkage cannot put the removed modes back.
def _shrinkage_target(C, n_removed=0):
    N = C.shape[0]
    if n_removed == 0:
        return np.eye(N) * np.trace(C) / N
    _, eigenvecs = eigen_spectrum(C)
    live = N - n_removed
    projector = eigenvecs[:, :live] @ eigenvecs[:, :live].T
    return projector * np.trace(C) / live


def linear_shrinkage(C, n_observations, n_removed=0, X=None, intensity=None):
    C = np.asarray(C, dtype=float)
    if intensity is None:
        intensity = ledoit_wolf_intensity(C, n_observations, X, n_removed)
    shrunk = (1 - intensity) * C + intensity * _shrinkage_target(C, n_removed)
    if n_removed == 0:
        return shrunk
    #with modes removed the target is not unit-diagonal, so put the variances back
    return _restore_diagonal(shrunk, C)


#-------- Cleaner 3 : nonlinear shrinkage (the current standard) ------------
#-----------------------------------------------------------------------------
# The idea one level past clipping: the eigenVECTORS are noisy too, not just the
# eigenvalues. Random matrix theory can say how far a sample eigenvector has
# drifted from the true one, and the answer depends on where its eigenvalue sits.
# So each eigenvalue gets its own shrinkage - hard if its eigenvector is badly
# corrupted, barely any if it is trustworthy.
#
# This is the rotationally invariant estimator (Bun, Bouchaud & Potters 2017),
# and it is the same object as Ledoit-Peche nonlinear shrinkage from statistics.
# The optimal shrunk eigenvalue is
#
#       d_k = lambda_k / |1 - q + q z_k s(z_k)|^2
#
# where s is the Stieltjes transform of the spectrum - an average of
# 1/(z - eigenvalue), the standard tool for asking "how crowded is the spectrum
# near here". Everything RMT knows about the noise comes through it.
#
# How s is estimated matters. The first version here smoothed it with a fixed
# small imaginary part, and on the test suite it did WORSE than no cleaning at
# all once q got near 1. This version is Ledoit & Wolf's (2020) analytical
# estimator: the same formula, with the density and its Hilbert transform
# estimated by a kernel whose width adapts to each eigenvalue. It beats raw at
# every q tested and is the best cleaner on structured data.
#
# One detail that took finding: a correlation matrix from T centred observations
# has rank T - 1, not T. The formula needs the true number of non-zero
# eigenvalues, so it counts them rather than trusting n_observations.
def nonlinear_shrinkage(C, n_observations, n_removed=0, X=None):
    C = np.asarray(C, dtype=float)
    R, sd = _to_correlation(C)
    eigenvals, eigenvecs = eigen_spectrum(R)
    live = len(eigenvals) - n_removed
    shrunk = _analytical_shrinkage(eigenvals[:live], n_observations)

    cleaned_vals = np.concatenate([shrunk, eigenvals[live:]])   # removed directions stay at zero
    cleaned = eigenvecs @ np.diag(cleaned_vals) @ eigenvecs.T
    return _restore_diagonal(cleaned, R) * np.outer(sd, sd)


# Ledoit & Wolf (2020), Annals of Statistics 48, 3043 - transcribed from their
# reference implementation. Takes eigenvalues biggest first, returns them shrunk.
def _analytical_shrinkage(vals_desc, n_observations):
    lam = np.sort(np.clip(vals_desc, 0, None))          # ascending, as in the paper
    p = len(lam)
    rank = int(np.sum(lam > 1e-10 * lam.max()))
    n = max(2, int(round(n_observations)))
    if rank < p:
        n = min(n, rank)                                # only this many eigenvalues carry information

    lam_nz = lam[max(0, p - n):]                        # the non-zero part of the spectrum
    h = n ** (-1 / 3)                                   # kernel bandwidth
    H = h * lam_nz[None, :]                             # ... scaled to each eigenvalue
    x = (lam_nz[:, None] - lam_nz[None, :]) / H

    #kernel density of the spectrum, and its Hilbert transform
    density = (3 / 4 / np.sqrt(5)) * np.mean(np.maximum(1 - x ** 2 / 5, 0) / H, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        hilbert = (-3 / 10 / np.pi) * x + (3 / 4 / np.sqrt(5) / np.pi) * (1 - x ** 2 / 5) \
            * np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
    edge = np.isclose(np.abs(x), np.sqrt(5))
    hilbert[edge] = (-3 / 10 / np.pi) * x[edge]
    hilbert = np.mean(hilbert / H, axis=1)

    if p <= n:
        c = p / n
        shrunk = lam_nz / ((np.pi * c * lam_nz * density) ** 2
                           + (1 - c - np.pi * c * lam_nz * hilbert) ** 2)
    else:
        #more variables than information: the zero eigenvalues get one shared value
        hilbert0 = (1 / np.pi) * (3 / 10 / h ** 2 + 3 / 4 / np.sqrt(5) / h * (1 - 1 / 5 / h ** 2)
                                  * np.log((1 + np.sqrt(5) * h) / (1 - np.sqrt(5) * h))) * np.mean(1 / lam_nz)
        shrunk0 = 1 / (np.pi * (p - n) / n * hilbert0)
        shrunk1 = lam_nz / (np.pi ** 2 * lam_nz ** 2 * (density ** 2 + hilbert ** 2))
        shrunk = np.concatenate([np.full(p - n, shrunk0), shrunk1])

    return shrunk[::-1]                                 # back to biggest first


#-------- Choosing a cleaner by name ----------------------------------------
#-----------------------------------------------------------------------------
# So an application can switch method with a string - the Correlator and the
# stacking module both go through here.
CLEANERS = {
    "raw": raw,
    "clip": clip,
    "linear_shrinkage": linear_shrinkage,
    "ledoit_wolf": linear_shrinkage,
    "nonlinear_shrinkage": nonlinear_shrinkage,
}


def clean(C, method, n_observations, n_removed=0, X=None):
    if method is None:
        method = "raw"
    if method not in CLEANERS:
        raise ValueError(f"unknown cleaner '{method}', choose from {sorted(CLEANERS)}")
    return CLEANERS[method](C, n_observations, n_removed=n_removed, X=X)


#-------- Tyler's estimator : the fix for fat tails -------------------------
#-----------------------------------------------------------------------------
# Everything above assumes light tails. Financial returns do not have light
# tails, so the noise band is the wrong shape and you are cleaning against the
# wrong benchmark.
#
# Tyler's fix is neat: use only the DIRECTION of each observation and throw away
# its size. A crash day still tells you which way things moved together, but it
# no longer dominates the estimate by sheer magnitude. Feed the result into the
# cleaners above and the theory applies again (Frahm & Jaekel 2005).
#
# It is a fixed point iteration - guess a matrix, reweight the data by it, build
# a new matrix, repeat until nothing changes. Needs more observations than
# variables. Returns the shape, scaled to trace N; tyler_correlation turns that
# into a correlation matrix.
def tyler(X, max_iterations=100, tol=1e-6):
    X = np.asarray(X, dtype=float)
    N, T = X.shape
    X = X - X.mean(axis=1, keepdims=True)

    C = np.eye(N)                                       # start from "no structure"
    for iteration in range(max_iterations):
        C_inv = np.linalg.pinv(C)
        # how far each observation sits from the centre, measured in the current matrix
        d2 = np.einsum('it,ij,jt->t', X, C_inv, X)
        d2[d2 <= 0] = 1e-12
        C_new = (N / T) * (X / d2) @ X.T                # each day weighted by 1/distance
        C_new = C_new / np.trace(C_new) * N             # fix the scale, Tyler only gives shape

        if np.max(np.abs(C_new - C)) < tol:
            C = C_new
            break
        C = C_new
    return C


def tyler_correlation(X, max_iterations=100, tol=1e-6):
    R, _ = _to_correlation(tyler(X, max_iterations, tol))
    return R


#-------- Effective rank : how many independent things are there? -----------
#-----------------------------------------------------------------------------
# One number for "how much diversification do I actually have". If everything
# were independent this equals the number of variables. If everything moved as
# one it equals 1.
#
# This is the function the solar project needs - the effective number of
# independent sites in a generation fleet. Removed modes have zero eigenvalues
# and drop out on their own.
def effective_rank(C):
    eigenvals, _ = eigen_spectrum(C)
    eigenvals = eigenvals[eigenvals > 1e-10 * eigenvals.max()]
    return (eigenvals.sum() ** 2) / (eigenvals ** 2).sum()


#-------- Plot : the spectrum against the theory ----------------------------
#-----------------------------------------------------------------------------
# The whole argument in one figure. Histogram of the measured eigenvalues, the
# pure noise prediction drawn on top, and the band edge marked. Anything to the
# right of the edge is signal - usually one big spike, which is the market
# itself moving everything together.
def plot_spectrum(C, n_observations, title="Eigenvalue spectrum vs pure noise",
                  n_removed=0, ax=None):
    R, _ = _to_correlation(np.asarray(C, dtype=float))
    N = R.shape[0]
    eigenvals, _ = eigen_spectrum(R)
    live_vals = eigenvals[:N - n_removed]
    variance = bulk_variance(live_vals, n_observations)
    lo, hi = noise_band(N, n_observations, variance, n_removed)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    fig = ax.figure
    ax.hist(live_vals, bins=max(20, N // 2), density=True, color='0.75',
            edgecolor='k', linewidth=0.4, label='measured eigenvalues')

    x = np.linspace(max(lo, 1e-6), hi, 400)
    ax.plot(x, marchenko_pastur_density(x, N, n_observations, variance, n_removed), 'r-', lw=2,
            label='pure noise prediction')
    ax.axvline(hi, color='r', ls='--', lw=1, label=f'noise band edge = {hi:.2f}')

    #anything past the edge gets marked, since that is the actual signal
    for lam in live_vals[live_vals > hi]:
        ax.axvline(lam, color='b', lw=1, alpha=0.7)

    q = (N - n_removed) / n_observations
    ax.set_xlabel('eigenvalue')
    ax.set_ylabel('density')
    ax.set_title(f"{title}   (q = {q:.2f},  effective rank = {effective_rank(R):.1f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


#-------- Plot : the four cleaners side by side -----------------------------
#-----------------------------------------------------------------------------
# Same matrix, four treatments. The point to look for is how much structure
# survives each one - raw looks busy because noise looks like structure.
def plot_cleaner_comparison(C, n_observations, X=None):
    methods = [("raw", raw(C)),
               ("clipped", clip(C, n_observations)),
               ("linear shrinkage", linear_shrinkage(C, n_observations, X=X)),
               ("nonlinear shrinkage", nonlinear_shrinkage(C, n_observations))]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, (name, M) in zip(axes, methods):
        im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f"{name}\neffective rank {effective_rank(M):.1f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8)
    return fig
