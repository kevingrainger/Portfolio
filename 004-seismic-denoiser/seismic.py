#!/usr/bin/env python3
#-------- seismic.py ------------------------------------------------------------
#-----------------------------------------------------------------------------
# One denoising problem, two toolkits.
#
# Part A - real data. Single-station traces from the Southern California Seismic
# Network (SCSN), 6 s at 100 Hz, each centred on its P-wave arrival (sample 300),
# with a first-motion polarity label (0 up, 1 down, 2 unknown). Denoisers from
# geophysics (bandpass, wavelets) and from finance (random-matrix cleaning) are
# scored on what matters: does the arrival get picked at the right time, and does
# the first motion keep its sign?
#
# Part B - synthetic array. Many sensors, one wave, noise that is partly shared
# between neighbours (a road) and partly local. The best weighted stack needs the
# noise covariance - the same Sigma^-1 a / (a' Sigma^-1 a) as a minimum-variance
# portfolio - and corrlib supplies it.

import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "003-corrlib-correlation-toolbox"))

FS = 100.0                     # samples per second
ARRIVAL = 300                  # the P pick sits at the centre of every trace
SCSN_FILE = "scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5"


#-------- Part A: loading ------------------------------------------------------------
def load_scsn(path=SCSN_FILE, n=3000, snr_quantile=(0.0, 0.2), seed=0):
    """Small (M <= 3.5), local (<= 50 km) CI-network events with known polarity, in a
    band of SNR (as quantiles of the whole file). Returns X, Y, snr, mag, dist."""
    import h5py
    with h5py.File(path, "r") as h:
        sn, mag, dist, snr, Y = h["sncls"][:], h["mag"][:], h["dist"][:], h["snr"][:], h["Y"][:]
        network = np.array([s.split(b".")[0] for s in sn])
        lo, hi = np.quantile(snr, snr_quantile)
        keep = (network == b"CI") & (mag <= 3.5) & (dist <= 50) & (Y != 2) & (snr >= lo) & (snr <= hi)
        idx = np.flatnonzero(keep)
        idx = np.sort(np.random.default_rng(seed).choice(idx, size=min(n, len(idx)), replace=False))
        X = h["X"][idx]
    return X.astype(float), Y[idx], snr[idx], mag[idx].astype(float), dist[idx].astype(float)


#-------- Part A: denoisers ----------------------------------------------------------
# Zero-phase (forwards and backwards) so the filter cannot shift the arrival -
# a causal filter delays the onset and would fail the pick test for that reason alone.
def bandpass(X, low=1.0, high=20.0, order=4):
    sos = butter(order, [low, high], btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


# Soft-threshold the wavelet coefficients; noise level from the median absolute
# deviation of the finest scale (Donoho & Johnstone 1994).
def wavelet_denoise(X, wavelet="db4", level=4):
    import pywt
    X = np.atleast_2d(X)
    out = np.empty_like(X)
    for i, x in enumerate(X):
        coeffs = pywt.wavedec(x, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(x)))
        coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
        out[i] = pywt.waverec(coeffs, wavelet)[:len(x)]
    return out


# The original project's random-matrix denoiser. Rows are traces from DIFFERENT
# events, all lined up on the P arrival, so the top eigenvectors learn a generic
# P-wave shape and each trace is projected onto it. A template method - it works
# only because the traces are pre-aligned, and the scores should be read that way.
def rmt_denoise(X, min_components=5):
    X = np.atleast_2d(X)
    centred = X - X.mean(axis=1, keepdims=True)
    scale = centred.std(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    Z = centred / scale
    n_traces, n_samples = Z.shape
    C = (Z.T @ Z) / n_traces                            # sample-by-sample covariance across traces
    vals, vecs = np.linalg.eigh(C)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    edge = (1 + np.sqrt(n_samples / n_traces)) ** 2     # Marchenko-Pastur edge
    k = max(min_components, int(np.sum(vals > edge)))
    keep = vecs[:, :k]
    return (Z @ keep @ keep.T) * scale


METHODS = {
    "raw": lambda X: X,
    "bandpass": bandpass,
    "wavelet": wavelet_denoise,
    "RMT template": rmt_denoise,
    "wavelet + RMT": lambda X: rmt_denoise(wavelet_denoise(X)),
}


#-------- Part A: scoring ------------------------------------------------------------
# STA/LTA (Allen 1978): short-term over long-term average of energy. The first
# crossing of the threshold after the LTA has warmed up is the pick.
def sta_lta_pick(X, sta=0.1, lta=1.0, threshold=6.0, start=1.0):
    X = np.atleast_2d(X)
    ns, nl = int(sta * FS), int(lta * FS)
    energy = X ** 2
    c = np.cumsum(np.pad(energy, ((0, 0), (1, 0))), axis=1)
    sta_v = (c[:, ns:] - c[:, :-ns]) / ns               # window ending at each sample
    lta_v = (c[:, nl:] - c[:, :-nl]) / nl
    n = X.shape[1]
    ratio = np.full(X.shape, np.nan)
    #align both to the sample at the END of their windows; LTA lags behind STA
    ratio[:, nl - 1:] = sta_v[:, nl - ns:] / np.maximum(lta_v, 1e-20)
    ratio[:, :int(start * FS)] = np.nan
    over = ratio > threshold
    picks = np.where(over.any(axis=1), over.argmax(axis=1), -1)
    return picks


def pick_error_seconds(picks, arrivals=ARRIVAL):
    err = (picks - np.asarray(arrivals)) / FS
    return np.where(picks >= 0, err, np.nan)


# First motion: does the ground move up or down first? The mean over the first three
# samples after the arrival against the mean of the ten before it. The window was
# tuned on high-SNR traces, where it agrees with the analysts' labels 97% of the time -
# so on noisy traces, a drop in agreement is the noise, not the method.
def first_motion(X, arrivals=ARRIVAL, before=10, start=1, after=3):
    X = np.atleast_2d(X)
    a = np.broadcast_to(np.asarray(arrivals), (X.shape[0],))
    rows = np.arange(X.shape[0])[:, None]
    post = X[rows, a[:, None] + start + np.arange(after)].mean(axis=1)
    pre = X[rows, a[:, None] - before + np.arange(before)].mean(axis=1)
    return np.where(post - pre > 0, 0, 1)               # 0 up, 1 down - the dataset's labels


# Every trace in the file has its arrival at sample 300. A method that learns from
# many traces at once can learn "something happens at 300" and paint it in. Cutting
# a random 4.5 s window out of each 6 s trace moves the arrival to anywhere between
# 1.5 s and 3 s, so every method has to find it from the trace itself - the honest
# test. (A window, not a circular shift: shifting would wrap the end of each trace
# round to its start and paste earthquake coda in front of the arrival.)
def random_shift(X, window=450, seed=0):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, X.shape[1] - window + 1, size=X.shape[0])
    starts = np.minimum(starts, ARRIVAL - 150)          # keep the arrival at least 1.5 s in
    cut = np.array([x[k:k + window] for x, k in zip(X, starts)])
    return cut, ARRIVAL - starts


# Signal: the second after the arrival. Noise: everything at least half a second
# before it.
def snr_db(X, arrivals=ARRIVAL):
    X = np.atleast_2d(X)
    a = np.broadcast_to(np.asarray(arrivals), (X.shape[0],))
    out = np.empty(X.shape[0])
    for i, (x, k) in enumerate(zip(X, a)):
        noise = x[max(0, k - 250):k - 50]
        signal = x[k:k + 100]
        out[i] = 10 * np.log10(max(signal.var(), 1e-20) / max(noise.var(), 1e-20))
    return out


def scoreboard(X, Y, arrivals=ARRIVAL, methods=METHODS, tolerance=0.2):
    rows = {}
    raw_snr = snr_db(X, arrivals)
    for name, method in methods.items():
        D = method(X)
        err = pick_error_seconds(sta_lta_pick(D), arrivals)
        good = np.abs(err) <= tolerance
        rows[name] = dict(
            picked_within_tolerance=np.mean(good),
            median_abs_error_s=np.nanmedian(np.abs(err)),
            bias_s=np.nanmedian(err[good]) if good.any() else np.nan,
            polarity_correct=np.mean(first_motion(D, arrivals) == Y),
            snr_gain_db=np.median(snr_db(D, arrivals) - raw_snr),
            errors=err,
        )
    return rows


#-------- Part B: a synthetic array --------------------------------------------------
def ricker(t, f=6.0):
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def synthetic_array(n_sensors=16, n_samples=3000, arrival=2200, spacing_km=0.5, slowness_s_per_km=0.25,
                    road_km=2.0, road_strength=2.0, local_noise=1.0, correlated=True, seed=0):
    """Sensors on a line; a plane wave crosses them with known delays; noise is a shared
    'road' source whose strength falls with distance from the road, plus local noise.
    Returns traces, the clean wave at each sensor, delays (samples), positions, exposures."""
    rng = np.random.default_rng(seed)
    x = np.arange(n_sensors) * spacing_km
    delays = np.round(x * slowness_s_per_km * FS).astype(int)
    t = np.arange(n_samples) / FS
    clean = np.array([ricker(t - (arrival + d) / FS) for d in delays])

    exposure = road_strength * np.exp(-np.abs(x - road_km) / 2.0) if correlated else np.zeros(n_sensors)
    traffic = np.convolve(rng.standard_normal(n_samples + 50), np.ones(5) / np.sqrt(5), "same")[:n_samples]
    noise = exposure[:, None] * traffic[None, :] + local_noise * rng.standard_normal((n_sensors, n_samples))
    return clean + noise, clean, delays, x, exposure


def align_known(X, delays):
    """Undo known moveout (zero-padded shift) so the wave lines up across sensors."""
    from corrlib.array_stacking import _shift
    return np.array([_shift(x, -d) for x, d in zip(X, delays)])


#-------- Part B: scoring a stack honestly --------------------------------------------
# The weights are fitted on one stretch of noise and judged on a DIFFERENT stretch.
# Judging on the same stretch would reward overfitting - exactly the trap that makes
# in-sample portfolio risk look lower than it turns out to be.
def stack_gains(n_sensors, train_samples, correlated=True, seed=0, test=(1800, 3400), arrival=3600,
                cleaners=("raw", "linear_shrinkage", "nonlinear_shrinkage"), road_strength=4.0):
    from corrlib import array_stacking as stack
    from corrlib import random_matrix_cleaning as rmt

    X, clean, delays, x, exposure = synthetic_array(n_sensors=n_sensors, n_samples=4200, arrival=arrival,
                                                    correlated=correlated, seed=seed, road_strength=road_strength)
    A = align_known(X, delays)
    noise_only = A - align_known(clean, delays)
    train = noise_only[:, test[0] - train_samples:test[0]]
    held_out = noise_only[:, test[0]:test[1]]

    single = np.median(held_out.std(axis=1))            # a typical sensor on its own
    out = {"average": 20 * np.log10(single / held_out.mean(axis=0).std())}
    ones = np.ones(n_sensors)
    for name in cleaners:
        C = np.cov(train)
        C = rmt.clean(C, name, train_samples, X=train)
        C_inv = np.linalg.pinv(C)
        w = C_inv @ ones / (ones @ C_inv @ ones)        # distortionless: the wave passes at unit gain
        out[f"optimal ({name})"] = 20 * np.log10(single / (w @ held_out).std())
    return out
