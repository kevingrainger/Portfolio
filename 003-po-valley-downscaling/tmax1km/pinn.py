#-------- pinn.py -------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# The U-Net trained on thermometers, with and without the heat budget in its loss.
#
#   loss = mean over training stations of (theta_hat - y)^2  +  pinball loss for the 5% and 95%
#          + lambda * mean over interior land cells of r^2,     r = A theta_hat - b
#
# A theta = b is the same discretised heat budget the physics model solves (pde.py):
#   u . grad(theta) - kappa lap(theta) + (theta - theta_s)/tau_s(x) + (theta - theta_E)/tau_a = q
# with r playing the part of q, the heating the equation leaves out. lambda = 0 is a
# purely data-driven network. lambda > 0 asks the network to agree with the stations and,
# everywhere else, to break the heat budget as little as it can: the soft-constraint
# form of physics-informed learning (Raissi, Perdikaris & Karniadakis 2019; Karniadakis
# et al. 2021). The physics term needs no labels, so it reaches every cell between the
# stations, which is where a data-only network has nothing to learn from.

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import unet


def residual(theta, theta_E, theta_s, u, v, rate, kappa, tau_a, h=1.0):
    """A theta - b on the interior of a (batch, H, W) tile; same upwind / five-point
    stencil as pde.assemble. Returns (batch, H-2, W-2)."""
    c = theta[:, 1:-1, 1:-1]
    west, east = theta[:, 1:-1, :-2], theta[:, 1:-1, 2:]
    south, north = theta[:, :-2, 1:-1], theta[:, 2:, 1:-1]
    uc, vc = u[:, 1:-1, 1:-1], v[:, 1:-1, 1:-1]
    up, um, vp, vm = jnp.maximum(uc, 0), jnp.maximum(-uc, 0), jnp.maximum(vc, 0), jnp.maximum(-vc, 0)
    k = kappa / h ** 2
    rate_c = rate[:, 1:-1, 1:-1]
    diag = (up + um + vp + vm) / h + 4 * k + rate_c + 1.0 / tau_a
    Ath = diag * c - (up / h + k) * west - (um / h + k) * east - (vp / h + k) * south - (vm / h + k) * north
    b = rate_c * theta_s[:, 1:-1, 1:-1] + theta_E[:, 1:-1, 1:-1] / tau_a
    return Ath - b


def loss_fn(p, x, phys, ys, ms, land, lam, kappa, tau_a):
    """x: standardised inputs; phys: raw (theta_E, theta_s, u, v, rate) tiles;
    ys, ms: station readings and their mask (training stations only); land: cells where
    the physics is enforced."""
    out = unet.apply(p, x)
    theta_E = phys[..., 0]
    theta = theta_E + out[..., 0]
    nm = jnp.maximum(ms.sum(), 1.0)
    data = (((theta - ys) ** 2) * ms).sum() / nm
    pin = 0.0
    for k, q in enumerate(unet.QUANTILES, start=1):
        e = ys - (theta_E + out[..., k])
        pin = pin + (jnp.maximum(q * e, (q - 1) * e) * ms).sum() / nm
    r = residual(theta, theta_E, phys[..., 1], phys[..., 2], phys[..., 3], phys[..., 4], kappa, tau_a)
    lm = land[:, 1:-1, 1:-1]
    phys_term = ((r ** 2) * lm).sum() / jnp.maximum(lm.sum(), 1.0)
    return data + pin + lam * phys_term, (data, phys_term)


def make_step(opt, lam, kappa, tau_a):
    @jax.jit
    def step(p, s, x, phys, ys, ms, land):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, x, phys, ys, ms, land, lam, kappa, tau_a)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, l, aux
    return step


def station_crops(X, PHYS, Y, M, LAND, days, centres, size, batch, rng):
    """Windows centred near a randomly chosen training station, so every window carries
    labels; the physics term covers the whole window regardless."""
    H, W = X.shape[1:3]
    xb, pb, yb, mb, lb = [], [], [], [], []
    for _ in range(batch):
        d = days[rng.integers(len(days))]
        cy, cx = centres[rng.integers(len(centres))]
        i = int(np.clip(cy - rng.integers(8, size - 8), 0, H - size))
        j = int(np.clip(cx - rng.integers(8, size - 8), 0, W - size))
        sl = (slice(i, i + size), slice(j, j + size))
        xb.append(X[d][sl]); pb.append(PHYS[d][sl]); yb.append(Y[d][sl]); mb.append(M[d][sl]); lb.append(LAND[sl])
    return tuple(np.stack(a) for a in (xb, pb, yb, mb, lb))
