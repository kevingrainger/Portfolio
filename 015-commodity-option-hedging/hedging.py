#-------- hedging.py ---------------------------------------------------------------
#-----------------------------------------------------------------------------
# Delta hedging a short commodity call, several ways.
#
# The option is a call on a commodity futures contract, priced by Black (1976):
#   C     = e^{-r tau} [F N(d1) - K N(d2)],   d1 = [ln(F/K) + sigma^2 tau/2] / (sigma sqrt(tau)),  d2 = d1 - sigma sqrt(tau)
#   Delta = e^{-r tau} N(d1),   Gamma = e^{-r tau} n(d1) / (F sigma sqrt(tau))
#
# The trade: sell one call at the money for 3 months, collect the premium, hold
# h(t) futures and re-hedge on a rule, pay a proportional cost k on every futures
# traded. At expiry, pay the option out. What is left is the hedging error:
#   PnL = C_0 - (F_T - K)^+ + sum_t h(t) [F(t+1) - F(t)] - k sum_t |h(t) - h(t-1)| F(t)
# (Futures cost nothing to enter, so there is no financing term on the hedge; the
# small discounting of the premium is ignored.) Everything is per unit of notional F_0.
#
# Strategies differ in the volatility they feed Black's delta and in when they trade:
#   none      no hedge
#   static    Black's delta on the day of sale, then left alone
#   weekly    realised volatility (60 trading days), re-hedged every 5 days
#   move      re-hedged only when the futures price has moved 2% since the last trade
#   daily     realised volatility, re-hedged every day
#   garch     a GARCH(1,1) forecast of the volatility to expiry, re-hedged daily
#   leland    Leland (1985): volatility raised to pay for costs,
#               sigma_L^2 = sigma^2 [1 + sqrt(2/pi) k / (sigma sqrt(dt))]
#   band      Whalley & Wilmott (1997): trade only when the position leaves a band
#               around Delta of half-width H = (3 e^{-r tau} k F Gamma^2 / (2 gamma))^{1/3},
#             and then only back to the band's edge. gamma is risk aversion.

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

DAYS = 252


#-------- Black (1976) -----------------------------------------------------------------
def black(F, K, tau, sigma, r=0.0):
    tau = np.maximum(tau, 1e-10)
    sd = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
    d2 = d1 - sd
    disc = np.exp(-r * tau)
    price = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    delta = disc * norm.cdf(d1)
    gamma = disc * norm.pdf(d1) / (F * sd)
    return price, delta, gamma


#-------- volatility estimates -------------------------------------------------------
def realised_vol(returns, window=60):
    """Annualised standard deviation of daily log returns over the trailing window, known
    at the close of each day (so it can be used for the next day's hedge)."""
    r2 = np.convolve(returns ** 2, np.ones(window), "full")[:len(returns)]
    n = np.minimum(np.arange(1, len(returns) + 1), window)
    return np.sqrt(r2 / n * DAYS)


def garch_fit(returns):
    """GARCH(1,1), sigma2_t = omega + alpha r_{t-1}^2 + beta sigma2_{t-1}, by Gaussian
    maximum likelihood (Bollerslev 1986). Returns (omega, alpha, beta)."""
    r = returns - returns.mean()
    v0 = r.var()

    def nll(p):
        omega, alpha, beta = p
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e10
        s2 = np.empty_like(r)
        s2[0] = v0
        for t in range(1, len(r)):
            s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        return 0.5 * np.sum(np.log(s2) + r ** 2 / s2)

    res = minimize(nll, [v0 * 0.05, 0.05, 0.9], method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-8, fatol=1e-6))
    return res.x


def garch_path(returns, params, v0):
    """Conditional daily variance for each day, given the returns up to the day before."""
    omega, alpha, beta = params
    s2 = np.empty(len(returns) + 1)
    s2[0] = v0
    for t in range(len(returns)):
        s2[t + 1] = omega + alpha * returns[t] ** 2 + beta * s2[t]
    return s2


def garch_term_vol(s2_next, params, n_days):
    """Annualised volatility expected over the next n_days, from GARCH's mean reversion
    toward its long-run variance omega / (1 - alpha - beta)."""
    omega, alpha, beta = params
    p = alpha + beta
    vbar = omega / (1 - p)
    k = np.arange(n_days)
    return np.sqrt(np.mean(vbar + p ** k * (s2_next - vbar)) * DAYS)


#-------- one hedged trade -----------------------------------------------------------
def hedge_trade(F, sigma_path, n_days, strategy, k=5e-4, r=0.0, gamma_ra=10.0, sigma_price=None,
                move=0.02, record=False):
    """Sell an at-the-money call on F[0] expiring after n_days, hedge it with `strategy`.

    F           futures closes, F[0] the day of sale, F[n_days] expiry
    sigma_path  the volatility the strategy believes on each day (length n_days)
    sigma_price the volatility the premium was sold at (defaults to sigma_path[0])
    Returns a dict of P&L, trading cost and the number of trades, per unit of F[0]."""
    K = F[0]
    sp = sigma_path[0] if sigma_price is None else sigma_price
    premium = black(F[0], K, n_days / DAYS, sp, r)[0]
    h_prev, cost, trades, hedge_pnl = 0.0, 0.0, 0, 0.0
    F_last = F[0]
    dt = 1.0 / DAYS
    path = []
    for t in range(n_days):
        tau = (n_days - t) / DAYS
        s = sigma_path[t]
        if strategy == "none":
            h = 0.0
        elif strategy == "static":
            h = black(F[0], K, n_days / DAYS, s, r)[1] if t == 0 else h_prev
        elif strategy in ("daily", "garch"):
            h = black(F[t], K, tau, s, r)[1]
        elif strategy == "weekly":
            h = black(F[t], K, tau, s, r)[1] if t % 5 == 0 else h_prev
        elif strategy == "move":
            if t == 0 or abs(F[t] / F_last - 1) >= move:
                h, F_last = black(F[t], K, tau, s, r)[1], F[t]
            else:
                h = h_prev
        elif strategy == "leland":
            sL = s * np.sqrt(1 + np.sqrt(2 / np.pi) * k / (s * np.sqrt(dt)))
            h = black(F[t], K, tau, sL, r)[1]
        elif strategy == "band":
            _, d, g = black(F[t], K, tau, s, r)
            H = (3 * np.exp(-r * tau) * k * F[t] * g ** 2 / (2 * gamma_ra)) ** (1 / 3)
            h = np.clip(h_prev, d - H, d + H)
        else:
            raise ValueError(strategy)
        if h != h_prev:
            cost += k * abs(h - h_prev) * F[t]
            trades += 1
        hedge_pnl += h * (F[t + 1] - F[t])
        h_prev = h
        if record:
            option_now = black(F[t + 1], K, max(tau - dt, 0.0), s, r)[0] if t < n_days - 1 else max(F[n_days] - K, 0.0)
            path.append((h, (premium - option_now + hedge_pnl - cost) / K))
    payoff = max(F[n_days] - K, 0.0)
    pnl = premium - payoff + hedge_pnl - cost
    out = dict(pnl=pnl / K, cost=cost / K, trades=trades, premium=premium / K)
    if record:
        out["path"] = np.array(path)            # columns: position held, running P&L (marked to model)
    return out


#-------- simulated markets, to separate what breaks which strategy --------------------
def simulate(model, n_paths, n_days, sigma=0.25, seed=0, history=60, **kw):
    """Futures paths with `history` days before the option is sold (so trailing volatility
    can be measured) and n_days after. Returns array (n_paths, history + n_days + 1), F0 = 1
    on the day of sale.
      gbm        constant volatility sigma: Black's own world
      jumps      Merton (1976): Poisson jumps, lam per year, log-size N(0, jump_sd), with the
                 diffusion volatility lowered so total variance matches sigma^2
      heston     Heston (1993): variance v mean-reverts to sigma^2 at rate kappa with
                 volatility of variance xi, correlation rho with the price
      scheduled  constant volatility, plus one jump of size `jump` on a known day mid-life
                 (a tariff decision, a crop report)"""
    rng = np.random.default_rng(seed)
    n = history + n_days
    dt = 1.0 / DAYS
    z = rng.standard_normal((n_paths, n))
    if model == "gbm":
        lr = -0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * z
    elif model == "jumps":
        lam, jsd = kw.get("lam", 4.0), kw.get("jump_sd", 0.05)
        s_d = np.sqrt(max(sigma ** 2 - lam * jsd ** 2, 1e-4))
        nj = rng.poisson(lam * dt, (n_paths, n))
        J = rng.standard_normal((n_paths, n)) * jsd * np.sqrt(nj) - nj * 0.5 * jsd ** 2
        lr = -0.5 * s_d ** 2 * dt + s_d * np.sqrt(dt) * z + J
    elif model == "heston":
        kappa, xi, rho = kw.get("kappa", 3.0), kw.get("xi", 0.8), kw.get("rho", 0.3)
        theta = sigma ** 2
        w = rho * z + np.sqrt(1 - rho ** 2) * rng.standard_normal((n_paths, n))
        v = np.full(n_paths, theta)
        lr = np.empty((n_paths, n))
        for t in range(n):
            vp = np.maximum(v, 0.0)
            lr[:, t] = -0.5 * vp * dt + np.sqrt(vp * dt) * z[:, t]
            v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * w[:, t]
    elif model == "scheduled":
        lr = -0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * z
        sign = rng.choice([-1.0, 1.0], n_paths)
        lr[:, history + n_days // 2] += sign * kw.get("jump", 0.15)
    else:
        raise ValueError(model)
    logF = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(lr, axis=1)], axis=1)
    return np.exp(logF - logF[:, [history]])
