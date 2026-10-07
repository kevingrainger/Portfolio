#-------- step0.py ---------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Step 0: the solver is tested on a problem whose answer is known before it touches data.
#
# A fake truth is built FROM the PDE with chosen parameters and a known missing-physics
# field q, then sampled at station locations. Two things must hold:
#   1. the anchored solve returns the station values exactly (to 1e-6 K);
#   2. fitting the physics-only model to those stations recovers the parameters
#      approximately - approximately, because the fit assumes q = 0 and the truth has
#      q != 0, exactly the situation on real data.
#
# The test runs on the real land-cover map coarsened to 4 km so that it takes seconds.

import numpy as np
from scipy.optimize import minimize

from . import placeholder
from .anchor import Anchor, Smoother
from .grid import CLASSES, block_mean
from .pde import M2S_TO_KM2H, DaySystem, Params, misfit

TRUE = Params(kappa=400 * M2S_TO_KM2H, tau_a=5.0, tau_c=np.array([1.5, 4.0, 10.0, 6.0, 3.0, 8.0]))
SKIN = np.array([9.0, 6.0, 1.5, 5.0, -4.0, 8.0])           # surface minus air, K, by class


def fake_days(grid, frac, n_days, rng, wind_scale=1.0):
    """Inputs for n_days: a smooth large-scale field, a surface temperature set by land
    cover, and a wind that changes direction and strength from day to day."""
    out = []
    for _ in range(n_days):
        theta_E = 32 + 2.0 * placeholder.smooth_field(grid.shape, 150, rng, grid.dx)
        theta_s = theta_E + np.tensordot(SKIN, frac, axes=1) * rng.uniform(0.6, 1.1) \
            + 1.0 * placeholder.smooth_field(grid.shape, 20, rng, grid.dx)
        speed, heading = wind_scale * 8.0 * np.exp(rng.normal(0, 0.6)), rng.uniform(0, 2 * np.pi)
        out.append(dict(theta_E=theta_E, theta_s=theta_s, u=np.full(grid.shape, speed * np.cos(heading)),
                        v=np.full(grid.shape, speed * np.sin(heading))))
    return out


def run(static, n_days=30, n_stations=40, q_size=0.3, coarsen=4, seed=0, wind_scale=1.0, start=None):
    """Returns the anchor error, the true and fitted parameters, and how well the
    recovered q matches the planted one where the stations can see it."""
    rng = np.random.default_rng(seed)
    from .grid import Grid
    grid = Grid().coarsen(coarsen)
    frac = block_mean(static["fractions"].astype(float), coarsen)
    frac = frac / frac.sum(0)
    days = fake_days(grid, frac, n_days, rng, wind_scale)
    q_true = q_size * placeholder.smooth_field(grid.shape, 40, rng, grid.dx)            # K/h, the same every day

    inner = grid.interior()
    land = np.flatnonzero((frac[CLASSES.index("water")] < 0.5) & inner)
    cells = rng.choice(land, n_stations, replace=False)
    truth = [DaySystem(grid, TRUE, frac, f).theta(q_true) for f in days]
    y = [t[cells] for t in truth]

    # 1. anchors are reproduced exactly, starting from the WRONG parameters and no prior q
    guess = Params() if start is None else start
    smoother = Smoother(grid, 15.0)
    gaps, q_corr = [], []
    for f, yd in zip(days[:5], y[:5]):
        anchor = Anchor(DaySystem(grid, TRUE, frac, f), cells, smoother)
        theta, q = anchor.solve(yd, check=False)
        gaps.append(np.abs(theta.ravel()[cells] - yd).max())
        weight = np.abs(anchor.G).sum(0).reshape(grid.shape)                             # where stations can see
        seen = weight > np.quantile(weight, 0.9)
        q_corr.append(np.corrcoef(q[seen], q_true[seen])[0, 1] if q_size > 0 else np.nan)
    wrong = Anchor(DaySystem(grid, guess, frac, days[0]), cells, smoother).solve(y[0], check=False)[0]
    gaps.append(np.abs(wrong.ravel()[cells] - y[0]).max())

    # 2. the physics-only fit recovers the parameters approximately
    fit = minimize(misfit, guess.pack(), args=(grid, frac, days, cells, y), jac=True, method="L-BFGS-B",
                   bounds=Params.bounds(), options=dict(maxiter=200))
    est = Params.unpack(fit.x)
    share = np.array([frac[i].ravel()[cells].mean() for i in range(len(CLASSES))])       # stations' land-cover mix
    return dict(anchor_gap=max(gaps), true=TRUE, fitted=est, rmse_start=np.sqrt(misfit(guess.pack(), grid, frac, days, cells, y)[0]),
                rmse_fit=np.sqrt(fit.fun), q_correlation=float(np.mean(q_corr)), station_share=share, grid=grid)


def recovery_table(result):
    """True against fitted, with the two combinations the data actually pin down."""
    import pandas as pd
    t, f = result["true"], result["fitted"]
    rows = [("kappa (m^2/s)", t.kappa / M2S_TO_KM2H, f.kappa / M2S_TO_KM2H, np.nan), ("tau_a (h)", t.tau_a, f.tau_a, np.nan)]
    rows += [(f"tau {c} (h)", t.tau_c[i], f.tau_c[i], result["station_share"][i]) for i, c in enumerate(CLASSES)]
    rows += [(f"tau {c} / tau_a", t.tau_c[i] / t.tau_a, f.tau_c[i] / f.tau_a, result["station_share"][i]) for i, c in enumerate(CLASSES)]
    rows += [("sqrt(kappa tau_a) (km)", np.sqrt(t.kappa * t.tau_a), np.sqrt(f.kappa * f.tau_a), np.nan)]
    df = pd.DataFrame(rows, columns=["parameter", "true", "fitted", "station land-cover share"]).set_index("parameter")
    df["fitted / true"] = df.fitted / df.true
    return df
