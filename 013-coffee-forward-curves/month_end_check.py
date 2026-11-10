#!/usr/bin/env python3
#-------- month_end_check.py --------------------------------------------------------
#-----------------------------------------------------------------------------
# The World Bank Pink Sheet gives each month's average price, not its closing price.
# Returns on averaged prices pick up a spurious lag-1 autocorrelation of about 0.25 and
# lose about a third of their variance (Working 1960), which can blur or shift lead-lag
# results. Here the key arabica results are rerun on month-end prices of the ICE arabica
# front-month future (Yahoo KC=F, last close of each month). There's no free daily
# robusta futures series (ICE London), so robusta stays on the Pink Sheet.
#
#   1. lag-1 autocorrelation and variance of monthly returns, Pink Sheet against KC=F
#   2. the hazard-against-price lag scan of weather_signal.py, lags 0-6, on KC=F
#   3. the tail rule of tail_rule.py on KC=F's next-three-month change
#
# The front-month series jumps at each contract roll, which adds noise that the Pink
# Sheet doesn't have.
#
#   python month_end_check.py
#
# Output: results/month_end_check.json

import json
import os

import numpy as np
import pandas as pd

from coffee_data import WEIGHTS
from fetch_data import VIETNAM
from weather_signal import pink_sheet, brl, rain_heat, strip, corr_band
from tail_rule import block_ci, THRESH

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def kc_month_end():
    s = pd.read_csv(os.path.join(HERE, "data", "kc_front.csv"), index_col=0, parse_dates=True).squeeze().dropna()
    return s.resample("ME").last()


def lag_scan(ret, fx, hazards):
    df = pd.concat([ret.rename("ret"), fx.rename("brl"), hazards], axis=1).dropna()["2004-01":]
    res, _ = strip(df.ret.values, df.brl.values)
    rows = {}
    for h in hazards.columns:
        rows[h] = []
        for lag in range(7):
            r, band, n_eff = corr_band(df[h].shift(lag).values, res)
            rows[h].append(dict(lag=lag, r=round(r, 3), band=round(band, 3), significant=bool(abs(r) > band)))
    return rows, len(df)


def main():
    pink = np.log(pink_sheet()["2003-12":"2026-09"])
    kc = np.log(kc_month_end()["2003-12":"2026-09"])
    fx = np.log(brl().resample("ME").last()).diff()
    br3 = rain_heat("weather_era5.csv", WEIGHTS, 3)
    vn3 = rain_heat("weather_vietnam.csv", {k: v[2] for k, v in VIETNAM.items()}, 3)
    hazards = pd.DataFrame({"brazil_dry": br3.dry, "vietnam_dry": vn3.dry, "vietnam_heat": vn3.heat})

    out = {"returns": {}}
    for name, lp in [("arabica Pink Sheet (monthly average)", pink.arabica),
                     ("robusta Pink Sheet (monthly average)", pink.robusta),
                     ("arabica KC=F (month-end)", kc)]:
        r = lp.diff().dropna()["2004-01":]
        out["returns"][name] = dict(months=int(len(r)), lag1_autocorr=round(float(r.autocorr(1)), 3),
                                    sd=round(float(r.std()), 4))

    out["lags"] = {}
    for name, lp in [("arabica Pink Sheet", pink.arabica), ("arabica KC=F month-end", kc)]:
        out["lags"][name], n = lag_scan(lp.diff(), fx, hazards)
    out["lags_months"] = n

    # the tail rule on month-end arabica
    from seasonal_backtest import build
    df = build()["2008":"2026-07"]
    kcm = kc.reindex(df.index)
    next3 = {"Pink Sheet": df.arabica_next3_raw, "KC=F month-end": kcm.shift(-3) - kcm}
    flag = (df.vietnam_heat_fc > THRESH) | (df.vietnam_dry_fc > THRESH)
    out["tail_rule_vietnam_hot_or_dry"] = {}
    for name, r in next3.items():
        ok = r.notna() & df.vietnam_heat_fc.notna() & df.brazil_dry_fc.notna()
        f = ok & flag
        out["tail_rule_vietnam_hot_or_dry"][name] = dict(
            months=int(f.sum()), mean_change=round(float(r[f].mean()), 4),
            ci90=[round(v, 4) for v in block_ci(r[ok].values, flag[ok].values)],
            all_months=round(float(r[ok].mean()), 4))

    json.dump(out, open(os.path.join(RES, "month_end_check.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
