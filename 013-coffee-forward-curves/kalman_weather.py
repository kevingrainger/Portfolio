#!/usr/bin/env python3
#-------- kalman_weather.py ---------------------------------------------------------
#-----------------------------------------------------------------------------
# A weather-informed forward curve. The Kalman filter tracks two hidden states from each
# day's arabica futures curve, the spot price and the convenience yield (the benefit of
# holding physical coffee rather than a contract), with the state dynamics of Schwartz's
# (1997) two-factor model. A weather term beta x W(t) pushes the convenience yield.
#
# The first version used the frost warning, but no damaging frost fell in the 2024-2026
# window the curve covers. This version uses the weather that did happen there: the
# three-month heat in Vietnam and the three-month dryness in Brazil (ERA5, standardised),
# held constant through each month. A likelihood-ratio test says whether the weather term
# earns its place.
#
#   python kalman_weather.py
#
# Output: results/kalman_weather.json, results/kalman_states.csv

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2

from coffee_data import WEIGHTS
from fetch_data import VIETNAM
from run_analysis import curve_panel
from schwartz import NAMES, fit, kalman
from weather_signal import rain_heat

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def main():
    dates, logF, tau = curve_panel()
    br = rain_heat("weather_era5.csv", WEIGHTS, 3)
    vn = rain_heat("weather_vietnam.csv", {k: v[2] for k, v in VIETNAM.items()}, 3)
    daily = lambda s: s.reindex(pd.date_range(s.index.min(), dates[-1], freq="D")).ffill().reindex(dates).fillna(0).values
    p0, ll0 = fit(logF, tau)
    out = dict(days=len(dates), start=str(dates[0].date()), end=str(dates[-1].date()),
               no_weather=dict(zip(NAMES, p0.tolist())), loglik_no_weather=ll0, weather={})
    states = {"no_weather": kalman(p0, logF, tau, return_states=True)[1][:, 1]}
    for name, s in [("vietnam_heat", vn.heat), ("brazil_dry", br.dry)]:
        W = daily(s)
        p, ll = fit(logF, tau, W=W, start=p0)
        lr = 2 * (ll - ll0)
        out["weather"][name] = dict(beta=float(p[-1]), loglik=ll, LR=float(lr), p_value=float(chi2.sf(max(lr, 0), 1)),
                                    params=dict(zip(NAMES, p.tolist())))
        states[name] = kalman(p, logF, tau, W, return_states=True)[1][:, 1]
    pd.DataFrame(states, index=dates).to_csv(os.path.join(RES, "kalman_states.csv"))
    json.dump(out, open(os.path.join(RES, "kalman_weather.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "no_weather"}, indent=1, default=str))


if __name__ == "__main__":
    main()
