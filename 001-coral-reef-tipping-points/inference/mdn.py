#-------- mdn.py ----------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Neural posterior estimation with a mixture density network, written in NumPy.
#
# The network maps summaries x of a simulated reef to a mixture of K Gaussians over
# the parameters, in an unbounded coordinate u = logit((theta - low) / (high - low)):
#
#     q(u | x) = sum_k pi_k(x) N(u; mu_k(x), diag sigma_k(x)^2)
#
# Trained by minimising  -E[log q(u | x)]  over the simulation bank, this q converges
# to the posterior p(u | x) under the prior the bank was drawn from (Papamakarios &
# Murray 2016; Bishop 1994). Gradients are written out by hand and checked against
# finite differences in test_mdn().

import numpy as np


def to_u(theta, low, high):
    p = np.clip((theta - low) / (high - low), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def to_theta(u, low, high):
    return low + (high - low) / (1 + np.exp(-u))


class MDN:
    def __init__(self, n_in, n_out, hidden=(64, 64), K=6, seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden]
        self.W = [rng.normal(0, np.sqrt(1 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.K, self.D = K, n_out
        n_head = K + 2 * K * n_out                      # logits, means, log sigmas
        self.Wo = rng.normal(0, 0.1 * np.sqrt(1 / sizes[-1]), (sizes[-1], n_head))
        self.bo = np.zeros(n_head)
        self.bo[K:K + K * n_out] = rng.normal(0, 1.0, K * n_out)    # spread the initial means
        self.x_mean = self.x_std = None

    #-------- parameters as one flat list, for the optimiser --------------------------
    def params(self):
        return self.W + self.b + [self.Wo, self.bo]

    def set_scaling(self, x):
        self.x_mean, self.x_std = x.mean(0), x.std(0) + 1e-8

    #-------- forward pass ---------------------------------------------------------------
    def forward(self, x):
        h = (x - self.x_mean) / self.x_std
        cache = [h]
        for W, b in zip(self.W, self.b):
            h = np.tanh(h @ W + b)
            cache.append(h)
        o = h @ self.Wo + self.bo
        K, D = self.K, self.D
        logits = o[:, :K]
        mu = o[:, K:K + K * D].reshape(-1, K, D)
        logsig = np.clip(o[:, K + K * D:].reshape(-1, K, D), -7, 5)
        return logits, mu, logsig, cache

    @staticmethod
    def _log_softmax(z):
        z = z - z.max(1, keepdims=True)
        return z - np.log(np.exp(z).sum(1, keepdims=True))

    def log_prob(self, x, u):
        logits, mu, logsig, _ = self.forward(x)
        return self._log_prob_parts(logits, mu, logsig, u)[0]

    def _log_prob_parts(self, logits, mu, logsig, u):
        logpi = self._log_softmax(logits)
        z = (u[:, None, :] - mu) / np.exp(logsig)
        comp = -0.5 * (z ** 2).sum(2) - logsig.sum(2) - 0.5 * self.D * np.log(2 * np.pi)
        a = logpi + comp
        m = a.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(a - m).sum(1))
        gamma = np.exp(a - lse[:, None])                  # responsibilities
        return lse, logpi, z, gamma

    #-------- loss and gradients ---------------------------------------------------------
    def loss_and_grads(self, x, u):
        n = len(x)
        logits, mu, logsig, cache = self.forward(x)
        lse, logpi, z, gamma = self._log_prob_parts(logits, mu, logsig, u)
        loss = -lse.mean()
        # d(-log q)/d head outputs
        d_logits = (np.exp(logpi) - gamma) / n
        d_mu = -(gamma[:, :, None] * z / np.exp(logsig)) / n
        d_logsig = -(gamma[:, :, None] * (z ** 2 - 1)) / n
        d_o = np.concatenate([d_logits, d_mu.reshape(n, -1), d_logsig.reshape(n, -1)], 1)
        gWo = cache[-1].T @ d_o
        gbo = d_o.sum(0)
        dh = d_o @ self.Wo.T
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for l in range(len(self.W) - 1, -1, -1):
            da = dh * (1 - cache[l + 1] ** 2)
            gW[l] = cache[l].T @ da
            gb[l] = da.sum(0)
            dh = da @ self.W[l].T
        return loss, gW + gb + [gWo, gbo]

    #-------- training with Adam, early stopping on a validation set ---------------------
    def fit(self, x, u, x_val, u_val, epochs=400, batch=256, lr=1e-3, patience=40, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        self.set_scaling(x)
        P = self.params()
        m = [np.zeros_like(p) for p in P]
        v = [np.zeros_like(p) for p in P]
        t, best, best_state, wait, history = 0, np.inf, None, 0, []
        for ep in range(epochs):
            idx = rng.permutation(len(x))
            for s in range(0, len(x), batch):
                j = idx[s:s + batch]
                _, G = self.loss_and_grads(x[j], u[j])
                t += 1
                for p, g, mi, vi in zip(P, G, m, v):
                    mi *= 0.9; mi += 0.1 * g
                    vi *= 0.999; vi += 0.001 * g * g
                    p -= lr * (mi / (1 - 0.9 ** t)) / (np.sqrt(vi / (1 - 0.999 ** t)) + 1e-8)
            tr = -self.log_prob(x, u).mean()
            va = -self.log_prob(x_val, u_val).mean()
            history.append((tr, va))
            if va < best - 1e-4:
                best, best_state, wait = va, [p.copy() for p in P], 0
            else:
                wait += 1
                if wait > patience:
                    break
            if verbose and ep % 20 == 0:
                print(f"epoch {ep:4d}  train {tr:.3f}  val {va:.3f}", flush=True)
        for p, q in zip(P, best_state):
            p[...] = q
        return np.array(history)

    #-------- sampling ----------------------------------------------------------------------
    def sample(self, x, n, rng):
        """n draws of u from q(u | x) for a single summary vector x."""
        logits, mu, logsig, _ = self.forward(x[None, :])
        pi = np.exp(self._log_softmax(logits))[0]
        k = rng.choice(self.K, size=n, p=pi)
        return mu[0, k] + np.exp(logsig[0, k]) * rng.normal(size=(n, self.D))


def test_mdn(seed=0):
    """Hand-written gradients against central finite differences."""
    rng = np.random.default_rng(seed)
    net = MDN(5, 3, hidden=(7, 6), K=3, seed=seed)
    x, u = rng.normal(size=(11, 5)), rng.normal(size=(11, 3))
    net.set_scaling(x)
    _, G = net.loss_and_grads(x, u)
    worst = 0.0
    for p, g in zip(net.params(), G):
        for _ in range(5):
            i = tuple(rng.integers(s) for s in p.shape)
            old = p[i]
            p[i] = old + 1e-6; lp, _ = net.loss_and_grads(x, u)
            p[i] = old - 1e-6; lm, _ = net.loss_and_grads(x, u)
            p[i] = old
            fd = (lp - lm) / 2e-6
            worst = max(worst, abs(fd - g[i]) / max(1e-8, abs(fd) + abs(g[i])))
    return worst
