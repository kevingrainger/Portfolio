#-------- unet.py ------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# A small U-Net in JAX: coarse daily fields and static maps in, a 1 km temperature field
# out, with a 5% and a 95% quantile beside the mean.
#
# A U-Net (Ronneberger, Fischer & Brox 2015) is an encoder that halves the resolution
# twice while widening the channels, then a decoder that doubles it back, with each
# decoder level also given the encoder's output at the same resolution ("skip
# connections"). The encoder sees large-scale context; the skips keep the fine detail.
# It is the standard image-to-image network in statistical downscaling (Bano-Medina et
# al. 2020). JAX is used because it is CPU-friendly and is what DeepMind's GraphCast
# is written in.

import jax
import jax.numpy as jnp
import numpy as np
import optax

QUANTILES = (0.05, 0.95)


def _conv_init(key, cin, cout, k=3):
    w = jax.random.normal(key, (k, k, cin, cout)) * jnp.sqrt(2.0 / (k * k * cin))
    return dict(w=w, b=jnp.zeros(cout))


def init(key, c_in, widths=(16, 32, 64), c_out=3):
    """Two 3x3 convolutions per level on the way down and up; a 1x1 head."""
    keys = iter(jax.random.split(key, 64))
    p = dict(down=[], up=[])
    c = c_in
    for w in widths:
        p["down"].append([_conv_init(next(keys), c, w), _conv_init(next(keys), w, w)])
        c = w
    for w in widths[-2::-1]:
        p["up"].append([_conv_init(next(keys), c + w, w), _conv_init(next(keys), w, w)])
        c = w
    p["head"] = _conv_init(next(keys), c, c_out, k=1)
    return p


def n_params(p):
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))


def _conv(x, c):
    y = jax.lax.conv_general_dilated(x, c["w"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + c["b"]


def _block(x, convs):
    for c in convs:
        x = jax.nn.gelu(_conv(x, c))
    return x


def _down(x):
    n, h, w, c = x.shape
    return x.reshape(n, h // 2, 2, w // 2, 2, c).mean((2, 4))


def _up(x):
    return jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)


def apply(p, x):
    """x: (batch, H, W, channels) with H, W divisible by 4. Returns (batch, H, W, 3):
    mean residual, 5% quantile offset, 95% quantile offset."""
    skips = []
    for i, convs in enumerate(p["down"]):
        x = _block(x, convs)
        if i < len(p["down"]) - 1:
            skips.append(x)
            x = _down(x)
    for convs, s in zip(p["up"], skips[::-1]):
        x = _block(jnp.concatenate([_up(x), s], -1), convs)
    out = _conv(x, p["head"])
    mean = out[..., 0]
    lo = mean - jax.nn.softplus(out[..., 1])          # quantiles kept on the right side of the mean
    hi = mean + jax.nn.softplus(out[..., 2])
    return jnp.stack([mean, lo, hi], -1)


def loss_fn(p, x, y, mask):
    """Mean squared error on the mean, pinball loss on the two quantiles, over land."""
    out = apply(p, x)
    m = mask.astype(out.dtype)
    nm = jnp.maximum(m.sum(), 1.0)
    mse = (((out[..., 0] - y) ** 2) * m).sum() / nm
    pin = 0.0
    for k, q in enumerate(QUANTILES, start=1):
        e = y - out[..., k]
        pin = pin + ((jnp.maximum(q * e, (q - 1) * e)) * m).sum() / nm
    return mse + pin, (mse, pin)


def make_step(opt):
    @jax.jit
    def step(p, s, x, y, mask):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, x, y, mask)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, l, aux
    return step


@jax.jit
def predict(p, x):
    return apply(p, x)


def random_crops(X, Y, M, days, size, batch, rng):
    """Random size x size windows from random training days, mostly on land."""
    H, W = X.shape[1:3]
    xb, yb, mb = [], [], []
    while len(xb) < batch:
        d = days[rng.integers(len(days))]
        i, j = rng.integers(0, H - size + 1), rng.integers(0, W - size + 1)
        m = M[i:i + size, j:j + size]
        if m.mean() < 0.3:
            continue
        xb.append(X[d, i:i + size, j:j + size]); yb.append(Y[d, i:i + size, j:j + size]); mb.append(m)
    return np.stack(xb), np.stack(yb), np.stack(mb)
