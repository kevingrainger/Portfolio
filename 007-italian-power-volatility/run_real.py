#!/usr/bin/env python3
#-------- run_real.py --------------------------------------------------------------
#-----------------------------------------------------------------------------
# The three stages of power.py on the real series (run fetch_real.py first):
# Italy North day-ahead price, 2018-2026, with Italian load, wind and solar, an
# eight-city ERA5 temperature and Dutch TTF gas.
#
#   Stage 1   what drives the price? XGBoost against a straight line, trained on past
#             years only, with log(price / gas) as the target; SHAP for the shapes.
#   Stage 2   what drives the volatility? GARCH-MIDAS with the monthly count of extreme
#             days (Italy-wide mean temperature above 28 C or below 3 C) setting the slow
#             level, against plain GARCH(1,1), by likelihood ratio.
#   Stage 3   does weather knowledge improve the volatility forecast out of sample? The
#             model is fitted on 2018-2022 and scored on 2023 onwards (QLIKE, Diebold-
#             Mariano), with the extreme-day count supplied three ways:
#               perfect      the count that actually happened (an upper bound: no
#                            forecast can know more)
#               climatology  the average count for that calendar month, 2018-2022
#               persistence  last month's count
#
# Returns. The daily change in log(price / gas), less its day-of-week mean, so gas
# moves and the weekly cycle are not counted as weather-driven volatility.
#
#   python run_real.py
#
# Output: results/real_stage1.csv, results/real_stage2.json, results/real_stage3.csv,
#         figures/fig5_real_price, fig6_real_shap, fig7_real_volatility

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figstyle                                   # noqa: E402
import power as pw                                # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
WINE, GREEN, SAND, GREY = "#882255", "#117733", "#DDCC77", "0.45"
SPLIT = "2023-01-01"


def load():
    df = pd.read_csv(os.path.join(HERE, "data", "real_daily.csv"), index_col=0, parse_dates=True)
    months = df.index.to_period("M")
    order = {m: i for i, m in enumerate(sorted(months.unique()))}
    month_idx = np.array([order[m] for m in months])
    extreme = df.groupby(months).extreme_month.first().astype(float)
    r = np.log(df.price / df.gas).diff()
    r = r - r.groupby(df.index.dayofweek).transform("mean")
    return df, month_idx, extreme, r.fillna(0).values


def stage1(df):
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    xgb = lambda: XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05)
    rows = []
    for target in ["log_price", "log_price_over_gas"]:
        for name, factory in [("linear", LinearRegression), ("XGBoost", xgb)]:
            _, rmse = pw.time_ordered_cv(df, factory, target=target, folds=6, min_train_years=3)
            rows.append(dict(target=target, model=name, rmse=rmse))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "real_stage1.csv"), index=False)
    print(out.pivot(index="model", columns="target", values="rmse").round(3))

    import shap
    X = df[pw.FEATURES]
    y = np.log(df.price) - df.log_gas
    model = xgb().fit(X, y)
    Xs = X.sample(2500, random_state=0)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    axes[0].scatter(Xs.temp, sv[:, pw.FEATURES.index("temp")], s=5, alpha=0.4, color=WINE, lw=0)
    axes[0].set_xlabel("Italy-wide daily mean temperature, C")
    axes[0].set_ylabel("effect on log(price / gas)")
    axes[0].set_title("Temperature, as XGBoost learned it", loc="left", fontsize=10)
    imp = np.abs(sv).mean(0)
    o = np.argsort(imp)
    axes[1].barh(np.array(pw.FEATURES)[o], imp[o], color=GREEN)
    axes[1].set_xlabel("mean |SHAP|, log(price / gas)")
    axes[1].set_title("What matters once gas is taken out", loc="left", fontsize=10)
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig6_real_shap"))
    return out


def stage2(df, month_idx, extreme, r):
    X = extreme.values
    p_midas, res_midas = pw.fit_garch_midas(r, month_idx, X)
    nll_null = pw.garch_midas_nll([p_midas["mu"], p_midas["alpha"], p_midas["beta"], np.log(np.var(r)), 0.0, 1.5],
                                  r, month_idx, X, 6)
    # the null (theta = 0) refitted properly: GARCH-MIDAS with no weather term
    from scipy.optimize import minimize
    f0 = lambda q: pw.garch_midas_nll([q[0], q[1], q[2], q[3], 0.0, 1.5], r, month_idx, X, 6)
    res0 = minimize(f0, [p_midas["mu"], p_midas["alpha"], p_midas["beta"], p_midas["m"]], method="Nelder-Mead",
                    options=dict(maxiter=8000, xatol=1e-8, fatol=1e-8))
    lr = 2 * (min(res0.fun, nll_null) - res_midas.fun)
    out = dict(params={k: float(v) for k, v in p_midas.items()}, LR=float(lr), p_value=float(chi2.sf(max(lr, 0), 2)),
               days=int(len(r)), start=str(df.index[0].date()), end=str(df.index[-1].date()))
    json.dump(out, open(os.path.join(RES, "real_stage2.json"), "w"), indent=1)
    print("stage 2:", json.dumps(out))
    var, tau = pw.garch_midas_variance(p_midas, r, month_idx, X)
    return var, tau, out


def stage3(df, month_idx, extreme, r):
    split = df.index < SPLIT
    realised = r ** 2 + 1e-8
    null = pw.fit_garch11(r[split])
    h0 = pw.garch11_variance(null, r)
    l0 = realised[~split] / h0[~split] - np.log(realised[~split] / h0[~split]) - 1
    months = extreme.index
    clim_by_month = extreme[months < pd.Period(SPLIT, "M")].groupby(lambda p: p.month).mean()
    inputs = {"perfect": extreme.values,
              "climatology": np.array([clim_by_month.loc[p.month] for p in months]),
              "persistence": np.r_[extreme.values[0], extreme.values[:-1]]}
    p, _ = pw.fit_garch_midas(r[split], month_idx[split], extreme.values)
    rows = []
    for name, X in inputs.items():
        v, _ = pw.garch_midas_variance(p, r, month_idx, X)
        la = realised[~split] / v[~split] - np.log(realised[~split] / v[~split]) - 1
        rows.append(dict(weather=name, qlike=la.mean(), qlike_garch=l0.mean(),
                         gain_pct=100 * (1 - la.mean() / l0.mean()), dm_t=pw.diebold_mariano(la, l0)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "real_stage3.csv"), index=False)
    print(out.round(4))
    return out


def fig_price(df, extreme):
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1]))
    axes[0].plot(df.index, df.price, color=WINE, lw=0.6, label="Italy North day-ahead, EUR/MWh")
    axes[0].plot(df.index, df.gas, color=GREEN, lw=1.0, label="TTF gas, EUR/MWh")
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper left")
    axes[1].bar(extreme.index.to_timestamp(), extreme.values, width=25, color=SAND)
    axes[1].set_ylabel("extreme-weather days\nper month")
    axes[0].set_title("Real data, 2018-2026: power follows gas; extreme days come in summers", loc="left")
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig5_real_price"))


def fig_vol(df, var, tau):
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    ax.plot(df.index, np.sqrt(var * 365), color="0.75", lw=0.5, label="total volatility")
    ax.plot(df.index, np.sqrt(tau * 365), color=WINE, lw=1.8, label="slow level set by extreme weather")
    ax.axvline(pd.Timestamp(SPLIT), color="0.5", lw=0.8, ls=":")
    ax.set_ylabel("annualised volatility of log(price / gas)")
    ax.legend(loc="upper left")
    ax.set_title("GARCH-MIDAS on the real series", loc="left")
    figstyle.save(fig, os.path.join(FIG, "fig7_real_volatility"))


def main():
    figstyle.apply()
    os.makedirs(RES, exist_ok=True)
    df, month_idx, extreme, r = load()
    fig_price(df, extreme)
    stage1(df)
    var, tau, _ = stage2(df, month_idx, extreme, r)
    fig_vol(df, var, tau)
    stage3(df, month_idx, extreme, r)


if __name__ == "__main__":
    main()
