#!/usr/bin/env python3
#-------- recent_normal.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# Does the tail rule survive a moving normal? The forecasts are measured against the
# 1993-2016 hindcast normal, so in a warming climate the Vietnam heat forecast drifts
# upward and crosses the one-standard-deviation line more often (the 2026 forecasts sit
# 4-5 standard deviations above it). Here each forecast is measured instead against the
# same calendar month in the 15 years before it, using only what was known then, and the
# tail rule rerun: after a Vietnam hot-or-dry flag, the next three months' price change.
#
#   python recent_normal.py
#
# Output: results/recent_normal.json

import json
import os

import numpy as np
import pandas as pd

from seasonal_backtest import build

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


def trailing_z(s, years=15):
    out = pd.Series(np.nan, index=s.index)
    for t in s.index:
        past = s[(s.index.month == t.month) & (s.index < t) & (s.index >= t - pd.DateOffset(years=years))]
        if past.notna().sum() >= 10:
            out[t] = (s[t] - past.mean()) / past.std()
    return out


def rule(df, heat, dry, bean):
    r = df[f"{bean}_next3_raw"]
    ok = r.notna() & heat.notna() & dry.notna()
    f = ok & ((heat > 1) | (dry > 1))
    x = r[f].values
    boot = [rng.choice(x, len(x)).mean() for _ in range(4000)]
    return dict(months=int(f.sum()), share_flagged=float(f.sum() / ok.sum()), mean_change=float(x.mean()),
                ci90=[float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))], all_months=float(r[ok].mean()))


def main():
    df = build()
    heat_r, dry_r = trailing_z(df.vietnam_heat_fc), trailing_z(df.vietnam_dry_fc)
    d = df["2008":"2026-07"]
    out = {}
    for name, h, w in [("1993-2016 normal", d.vietnam_heat_fc, d.vietnam_dry_fc), ("trailing 15-year normal", heat_r["2008":"2026-07"], dry_r["2008":"2026-07"])]:
        out[name] = {bean: rule(d, h, w, bean) for bean in ["arabica", "robusta"]}
    out["vietnam_heat_2026_trailing"] = {str(k.date()): round(float(v), 2) for k, v in heat_r["2026"].items()}
    json.dump(out, open(os.path.join(HERE, "results", "recent_normal.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
