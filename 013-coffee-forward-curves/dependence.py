#!/usr/bin/env python3
#-------- dependence.py -------------------------------------------------------------
#-----------------------------------------------------------------------------
# Three measures of dependence between weather and the coffee price, beyond a single
# correlation coefficient:
#
#   partial correlation   each bean against each hazard with the Brazilian real held fixed
#                         (Gaussian-rank, so outliers don't drive it), corrlib
#   tail dependence       of the most extreme weather months (top 10%), what share were
#                         also months of the largest price rises? Independence gives 10%;
#                         a block-shuffle of the price gives the 95% level, corrlib
#   wavelet coherence     a Morlet wavelet transform of both series, so the correlation is
#                         resolved in time and in period (months to years); significance
#                         from AR(1) surrogates (Torrence & Compo 1998; Grinsted et al. 2004)
#
#   python dependence.py
#
# Output: results/dependence.json, results/coherence_<pair>.npz

import json
import os
import sys

import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "004-corrlib-correlation-toolbox"))
from corrlib import correlation_measures as cm                     # noqa: E402

from weather_signal import full_panel, pink_sheet, strip           # noqa: E402

RES = os.path.join(HERE, "results")
HAZARDS = ["brazil_dry", "vietnam_dry", "vietnam_heat", "enso"]
PAIRS = [("robusta", "vietnam_heat"), ("arabica", "brazil_dry"), ("robusta", "brazil_dry"), ("arabica", "vietnam_heat")]
rng = np.random.default_rng(0)


def panel():
    """Every month 2004-2026: both beans' returns with the real taken out, and the hazards."""
    p = full_panel()
    ret = np.log(pink_sheet()).diff().reindex(p.index)
    p = p.join(ret)
    fx = p.brl.fillna(0.0)                       # a few months missing from Yahoo: treat as no change
    for bean in ["arabica", "robusta"]:
        p[f"{bean}_res"], _ = strip(p[bean].values, fx.values)
    return p.dropna(subset=["arabica", "robusta"] + HAZARDS)


def partials(p):
    out = {}
    for bean in ["arabica", "robusta"]:
        for h in HAZARDS:
            X = np.vstack([p[bean].values, p[h].values, p.brl.fillna(0).values])
            out[f"{bean}|{h}"] = float(cm.partial_from_matrix(cm.gaussian_rank(X))[0, 1])
    return out


def block_shuffle(y, block=12):
    n = len(y)
    starts = rng.integers(0, n - block, size=n // block + 1)
    return np.concatenate([y[s:s + block] for s in starts])[:n]


def tails(p, q=0.10, n_boot=2000):
    out = {}
    for bean, h in PAIRS:
        x, y = p[h].values, p[f"{bean}_res"].values
        lam = cm.tail_dependence(x, y, quantile=q, lower=False)
        null = np.array([cm.tail_dependence(x, block_shuffle(y), quantile=q, lower=False) for _ in range(n_boot)])
        out[f"{bean}|{h}"] = dict(tail=float(lam), independent=q, null_95=float(np.quantile(null, 0.95)),
                                  p_value=float((null >= lam).mean()))
    return out


#-------- wavelet coherence ------------------------------------------------------------
PERIODS = np.geomspace(4, 96, 48)                          # months
WAVELET = "cmor1.5-1.0"


def cwt(x):
    fc = pywt.central_frequency(WAVELET)
    scales = PERIODS * fc
    W, _ = pywt.cwt((x - x.mean()) / x.std(), scales, WAVELET)
    return W, scales


def smooth(A, scales):
    out = np.empty(A.shape, dtype=complex)
    for i, s in enumerate(scales):                         # in time, in proportion to the scale
        out[i] = gaussian_filter1d(A[i].real, s / 2, mode="nearest") + 1j * gaussian_filter1d(A[i].imag, s / 2, mode="nearest")
    return uniform_filter1d(out.real, 3, axis=0) + 1j * uniform_filter1d(out.imag, 3, axis=0)


def coherence(x, y):
    Wx, sc = cwt(x)
    Wy, _ = cwt(y)
    Sxy = smooth(Wx * np.conj(Wy) / sc[:, None], sc)
    Sxx = smooth(np.abs(Wx) ** 2 / sc[:, None], sc).real
    Syy = smooth(np.abs(Wy) ** 2 / sc[:, None], sc).real
    return np.abs(Sxy) ** 2 / (Sxx * Syy), np.angle(Sxy)


def ar1(x, n):
    x = (x - x.mean()) / x.std()
    a = np.corrcoef(x[:-1], x[1:])[0, 1]
    e = rng.standard_normal(n)
    out = np.empty(n); out[0] = e[0]
    for t in range(1, n):
        out[t] = a * out[t - 1] + np.sqrt(1 - a * a) * e[t]
    return out


def wavelet_pairs(p, n_sur=200):
    summary = {}
    t = np.arange(len(p))
    coi = np.minimum(t, t[::-1]) * np.sqrt(2) + 1              # cone of influence (months) for a Morlet
    for bean, h in PAIRS[:2]:
        x, y = p[h].values, p[f"{bean}_res"].values
        R2, phase = coherence(x, y)
        sur = np.array([coherence(ar1(x, len(x)), ar1(y, len(y)))[0] for _ in range(n_sur)])
        sig95 = np.quantile(sur, 0.95, axis=0)
        np.savez(os.path.join(RES, f"coherence_{bean}_{h}.npz"), R2=R2, phase=phase, sig95=sig95,
                 periods=PERIODS, coi=coi, time=p.index.values.astype("datetime64[M]").astype(str))
        inside = PERIODS[:, None] < coi[None, :]
        sig = (R2 > sig95) & inside
        summary[f"{bean}|{h}"] = dict(share_significant=float(sig.sum() / inside.sum()),
                                      periods_most_significant=float(PERIODS[sig.sum(axis=1).argmax()]))
    return summary


def main():
    p = panel()
    out = dict(months=len(p), partial=partials(p), tail=tails(p), wavelet=wavelet_pairs(p))
    json.dump(out, open(os.path.join(RES, "dependence.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
