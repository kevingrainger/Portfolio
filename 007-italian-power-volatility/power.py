#!/usr/bin/env python3
#-------- power.py ------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Italian power prices: weather -> price, weather -> volatility, forecast -> volatility.
#
#   Stage 1   what drives the daily price? XGBoost against a linear baseline,
#             time-ordered cross-validation, SHAP for the shapes
#   Stage 2   what drives the volatility? GARCH-MIDAS (Engle, Ghysels & Sohn 2013):
#             daily GARCH clustering on top of a slow monthly level set by weather
#   Stage 3   does it still work on a FORECAST of the weather, and how far ahead?
#
# Until the real series are pulled (GME prices, ERA5 weather, TTF gas, Terna load),
# simulate() builds data in exactly their shape with the relationships PLANTED,
# so each stage is tested on whether it recovers a known answer.

import numpy as np
import pandas as pd
from scipy.optimize import minimize

#-------- planted truths for the simulation -----------------------------------------
TRUE = dict(
    gas_elasticity=0.55,          # log price per log gas
    heat_curve=0.0045,            # U-shape in temperature around 18 C
    load_coef=0.30,
    renew_coef=-0.25,
    garch_alpha=0.08, garch_beta=0.86,
    midas_m=-6.6, midas_theta=0.12, midas_w=3.0, midas_K=6,
    forecast_noise_by_lead=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],   # sd of forecast error in "extreme days"
)


#-------- MIDAS weights ----------------------------------------------------------------
# Rather than one free weight per lagged month, a beta curve with one shape parameter:
# w = 1 is flat, larger w puts the weight on recent months. Two numbers instead of
# twelve is what makes the model estimable at all.
def beta_weights(K, w):
    k = np.arange(1, K + 1)
    raw = (1 - k / (K + 1)) ** (w - 1)
    return raw / raw.sum()


def simulate(start="2015-01-01", end="2024-12-31", seed=0, **overrides):
    """Daily data in the shape of the real series, with the relationships planted.
    Any entry of TRUE can be overridden, e.g. midas_theta=0.35 for a stronger weather effect."""
    T = {**TRUE, **overrides}
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    n = len(days)
    doy = days.dayofyear.values

    temp = 15 - 9 * np.cos(2 * np.pi * (doy - 20) / 365.25) + _ar1(rng, n, 0.8, 2.5)     # C, Italy-wide mean
    heatwave = np.zeros(n)
    for year in np.unique(days.year):                   # a few summer heatwaves a year
        for _ in range(rng.integers(1, 4)):
            centre = np.flatnonzero((days.year == year) & (days.month.isin([6, 7, 8])))
            c = rng.choice(centre)
            heatwave[max(0, c - 4):c + 5] += rng.uniform(4, 8)
    temp = temp + heatwave
    log_gas = np.log(20) + np.cumsum(rng.normal(0, 0.02, n))
    crisis = (days >= "2021-09-01") & (days <= "2022-12-31")                         # the 2021-22 gas crisis
    log_gas = log_gas + np.where(crisis, np.log(5) * np.sin(np.pi * (np.cumsum(crisis) / crisis.sum())), 0)
    load = 1 + 0.08 * np.cos(2 * np.pi * (doy - 15) / 365.25 * 2) + 0.004 * np.maximum(temp - 24, 0) ** 2 \
        - 0.06 * (days.dayofweek >= 5) + _ar1(rng, n, 0.6, 0.02)
    renew = 0.25 + 0.12 * np.sin(2 * np.pi * (doy - 80) / 365.25) + _ar1(rng, n, 0.5, 0.05)

    df = pd.DataFrame(dict(temp=temp, log_gas=log_gas, gas=np.exp(log_gas), load=load, renewables=renew,
                           weekend=(days.dayofweek >= 5).astype(int), month=days.month), index=days)

    #monthly extreme-weather index: days above 28 C or below 3 C (Italy-wide daily mean)
    extreme = ((df.temp > 28) | (df.temp < 3)).groupby(df.index.to_period("M")).sum()
    df["extreme_month"] = extreme.reindex(df.index.to_period("M")).values

    #volatility: GARCH-MIDAS with the planted parameters, the monthly level driven by
    #the extreme-weather index of the previous K months
    ex = extreme.values.astype(float)
    phi = beta_weights(T["midas_K"], T["midas_w"])
    log_tau = np.full(len(ex), T["midas_m"])
    for m in range(T["midas_K"], len(ex)):
        log_tau[m] = T["midas_m"] + T["midas_theta"] * phi @ ex[m - T["midas_K"]:m][::-1]
    tau = np.exp(log_tau)[df.index.to_period("M").map(dict(zip(extreme.index, range(len(ex))))).values]
    a, b = T["garch_alpha"], T["garch_beta"]
    g, shocks = 1.0, np.empty(n)
    for t in range(n):
        shocks[t] = np.sqrt(tau[t] * g) * rng.standard_normal()
        g = (1 - a - b) + a * shocks[t] ** 2 / tau[t] + b * g
    df["true_tau"] = tau

    log_price = (2.2 + T["gas_elasticity"] * df.log_gas + T["heat_curve"] * (df.temp - 18) ** 2
                 + T["load_coef"] * (df.load - 1) * 5 + T["renew_coef"] * (df.renewables - 0.25) * 4
                 - 0.05 * df.weekend + np.cumsum(shocks) * 0 + shocks.cumsum() * 0)
    #the unexplained part of the price follows a persistent process driven by the shocks
    resid = np.zeros(n)
    for t in range(1, n):
        resid[t] = 0.9 * resid[t - 1] + shocks[t]
    df["price"] = np.exp(log_price + resid)
    df["shock"] = shocks
    return df, extreme


def _ar1(rng, n, phi, sd):
    x = np.zeros(n)
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


#-------- Stage 1: weather -> price ----------------------------------------------------
FEATURES = ["temp", "log_gas", "load", "renewables", "weekend", "month"]


# Gas sets the Italian price level - gas plants are usually the marginal generator.
# Trees cannot extrapolate: a model that has never seen 2022 gas prices cannot predict
# 2022 power prices. So the default target is log(price / gas) - what everything
# except gas adds on top - and gas re-enters as a known multiplier.
def time_ordered_cv(df, model_factory, target="log_price_over_gas", folds=7, min_train_years=3):
    """Expanding window: train on everything before each test year, never after."""
    if target == "log_price":
        y = np.log(df.price).values
    elif target == "log_price_over_gas":
        y = np.log(df.price).values - df.log_gas.values
    else:
        y = df[target].values
    X = df[FEATURES].values
    years = df.index.year.values
    test_years = np.unique(years)[min_train_years:][:folds]
    preds = np.full(len(y), np.nan)
    for yr in test_years:
        train, test = years < yr, years == yr
        model = model_factory()
        model.fit(X[train], y[train])
        preds[test] = model.predict(X[test])
    ok = np.isfinite(preds)
    rmse = np.sqrt(np.mean((preds[ok] - y[ok]) ** 2))
    return preds, rmse


#-------- Stage 2: GARCH-MIDAS -----------------------------------------------------------
def garch_midas_nll(params, r, month_idx, X_month, K):
    mu, a, b, m, theta, w = params
    if a < 0 or b < 0 or a + b >= 0.999 or w < 1:
        return 1e10
    phi = beta_weights(K, w)
    log_tau_m = np.full(len(X_month), m)
    for i in range(K, len(X_month)):
        log_tau_m[i] = m + theta * phi @ X_month[i - K:i][::-1]
    tau = np.exp(log_tau_m)[month_idx]
    e = r - mu
    g = np.empty(len(r))
    g[0] = 1.0
    for t in range(1, len(r)):
        g[t] = (1 - a - b) + a * e[t - 1] ** 2 / tau[t - 1] + b * g[t - 1]
    var = tau * g
    keep = month_idx >= K                               # skip the burn-in months
    return 0.5 * np.sum(np.log(var[keep]) + e[keep] ** 2 / var[keep])


def fit_garch_midas(r, month_idx, X_month, K=6):
    """Maximum likelihood from several starting points - the surface has flat ridges
    in (theta, w), and one start can stop on the wrong side of one."""
    best = None
    for theta0 in (0.0, 0.1, 0.3):
        for w0 in (1.5, 4.0):
            x0 = [0.0, 0.05, 0.9, np.log(np.var(r)) - theta0 * X_month.mean(), theta0, w0]
            res = minimize(garch_midas_nll, x0, args=(r, month_idx, X_month, K), method="Nelder-Mead",
                           options=dict(maxiter=8000, xatol=1e-7, fatol=1e-7))
            if best is None or res.fun < best.fun:
                best = res
    return dict(zip(["mu", "alpha", "beta", "m", "theta", "w"], best.x)), best


def garch_midas_variance(params, r, month_idx, X_month, K=6):
    mu, a, b, m, theta, w = [params[k] for k in ["mu", "alpha", "beta", "m", "theta", "w"]]
    phi = beta_weights(K, w)
    log_tau_m = np.full(len(X_month), m)
    for i in range(K, len(X_month)):
        log_tau_m[i] = m + theta * phi @ X_month[i - K:i][::-1]
    tau = np.exp(log_tau_m)[month_idx]
    e = r - mu
    g = np.empty(len(r))
    g[0] = 1.0
    for t in range(1, len(r)):
        g[t] = (1 - a - b) + a * e[t - 1] ** 2 / tau[t - 1] + b * g[t - 1]
    return tau * g, tau


def fit_garch11(r):
    """Plain GARCH(1,1): the no-weather null."""
    def nll(p):
        mu, w0, a, b = p
        if w0 <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        e = r - mu
        h = np.empty(len(r))
        h[0] = np.var(r)
        for t in range(1, len(r)):
            h[t] = w0 + a * e[t - 1] ** 2 + b * h[t - 1]
        return 0.5 * np.sum(np.log(h) + e ** 2 / h)
    res = minimize(nll, [0.0, np.var(r) * 0.05, 0.05, 0.9], method="Nelder-Mead", options=dict(maxiter=4000))
    return dict(zip(["mu", "omega", "alpha", "beta"], res.x))


def garch11_variance(p, r):
    e = r - p["mu"]
    h = np.empty(len(r))
    h[0] = np.var(r)
    for t in range(1, len(r)):
        h[t] = p["omega"] + p["alpha"] * e[t - 1] ** 2 + p["beta"] * h[t - 1]
    return h


#-------- scoring -------------------------------------------------------------------------
def qlike(realised, forecast):
    """The standard loss for volatility forecasts: robust to noise in the realised proxy (Patton 2011)."""
    ratio = realised / forecast
    return np.mean(ratio - np.log(ratio) - 1)


def diebold_mariano(loss_a, loss_b):
    """t-statistic for 'model A has lower loss than B' (negative = A better), Newey-West variance."""
    d = loss_a - loss_b
    n = len(d)
    lag = int(n ** (1 / 3))
    gamma = [np.mean((d[k:] - d.mean()) * (d[:n - k] - d.mean())) for k in range(lag + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    return d.mean() / np.sqrt(var / n)
