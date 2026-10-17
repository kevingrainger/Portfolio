#-------- schwartz.py --------------------------------------------------------------
#-----------------------------------------------------------------------------
# The Schwartz (1997) two-factor commodity model, fitted to coffee futures curves with a
# Kalman filter, with an optional weather term in the convenience yield.
#
# The spot price S and the convenience yield delta (the benefit of holding physical
# coffee rather than a contract) move together:
#   dS/S   = (mu - delta) dt + sigma1 dW1
#   ddelta = [kappa (alpha - delta) + beta W(t)] dt + sigma2 dW2,     dW1 dW2 = rho dt
# where W(t) is the frost warning in that day's forecasts (beta = 0 without weather).
#
# Futures prices are then linear in the hidden state x = (ln S, delta):
#   ln F(t, T) = ln S - delta (1 - e^{-kappa tau}) / kappa + A(tau),    tau = T - t
#   A(tau) = (r - a + sigma2^2/(2 kappa^2) - rho sigma1 sigma2/kappa) tau
#            + sigma2^2 (1 - e^{-2 kappa tau}) / (4 kappa^3)
#            + (a kappa + rho sigma1 sigma2 - sigma2^2/kappa)(1 - e^{-kappa tau}) / kappa^2
# with a = alpha - lambda/kappa (lambda: the market price of convenience-yield risk).
# The loading (1 - e^{-kappa tau}) / kappa says how far along the curve a change in the
# convenience yield reaches: a frost lifts delta, and contracts with tau << 1/kappa feel
# it fully while distant ones barely move.
#
# Kalman filter (state x_t, observations y_t = today's log futures, NaN dropped):
#   x_{t+1} = c_t + T x_t + eta,   y_t = d_t + Z_t x_t + eps,   eps ~ N(0, h^2 I)
# The log-likelihood of the one-step prediction errors is maximised over the parameters.

import numpy as np
from scipy.optimize import minimize

DT = 1.0 / 252
R = 0.045               # US 3-month bill rate, 2024-2026 average (FRED DTB3), held fixed
NAMES = ["mu", "kappa", "alpha", "sigma1", "sigma2", "rho", "lam", "h", "beta"]


def loading(kappa, tau):
    return (1 - np.exp(-kappa * tau)) / kappa


def A(p, tau):
    mu, kappa, alpha, s1, s2, rho, lam = p[:7]
    a = alpha - lam / kappa
    e1, e2 = 1 - np.exp(-kappa * tau), 1 - np.exp(-2 * kappa * tau)
    return ((R - a + 0.5 * s2 ** 2 / kappa ** 2 - rho * s1 * s2 / kappa) * tau
            + 0.25 * s2 ** 2 * e2 / kappa ** 3
            + (a * kappa + rho * s1 * s2 - s2 ** 2 / kappa) * e1 / kappa ** 2)


def kalman(p, logF, tau, W=None, return_states=False):
    mu, kappa, alpha, s1, s2, rho, lam, h, beta = p
    Tm = np.array([[1.0, -DT], [0.0, 1 - kappa * DT]])
    Q = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]]) * DT
    first = np.where(~np.isnan(logF[:, 0]))[0][0]
    x = np.array([logF[first, 0], 0.0])
    P = np.diag([0.1, 0.5])
    ll, states = 0.0, []
    for t in range(logF.shape[0]):
        if t > 0:
            w = 0.0 if W is None else W[t - 1]
            x = np.array([x[0] + (mu - 0.5 * s1 ** 2) * DT - x[1] * DT,
                          x[1] + (kappa * (alpha - x[1]) + beta * w) * DT])
            P = Tm @ P @ Tm.T + Q
        ok = ~np.isnan(logF[t])
        if ok.sum():
            ta = tau[t, ok]
            Z = np.column_stack([np.ones(ok.sum()), -loading(kappa, ta)])
            v = logF[t, ok] - (A(p, ta) + Z @ x)
            Sm = Z @ P @ Z.T + h ** 2 * np.eye(ok.sum())
            Si = np.linalg.inv(Sm)
            K = P @ Z.T @ Si
            x = x + K @ v
            P = P - K @ Z @ P
            ll += -0.5 * (np.linalg.slogdet(Sm)[1] + v @ Si @ v + ok.sum() * np.log(2 * np.pi))
        states.append(x.copy())
    return (ll, np.array(states)) if return_states else ll


BOUNDS = [(-1, 1), (0.05, 10), (-1, 1), (0.05, 1.5), (0.02, 3), (-0.95, 0.95), (-3, 3), (1e-4, 0.1), (-50, 50)]


def fit(logF, tau, W=None, start=None):
    """Maximum likelihood. beta is fixed at 0 when W is None."""
    x0 = np.array(start if start is not None else [0.0, 1.0, 0.0, 0.35, 0.4, 0.5, 0.0, 0.01, 0.0])
    free = list(range(9)) if W is not None else list(range(8))

    def nll(z):
        p = x0.copy()
        p[free] = z
        val = -kalman(p, logF, tau, W)
        return val if np.isfinite(val) else 1e12
    res = minimize(nll, x0[free], method="L-BFGS-B", bounds=[BOUNDS[i] for i in free])
    p = x0.copy()
    p[free] = res.x
    return p, -res.fun
