#!/usr/bin/env python3
# Does the GARCH-MIDAS estimator recover the planted parameters? Fit it to ten
# independently simulated datasets; an unbiased estimator scatters
# around the truth. Writes results/midas_recovery.csv.
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import power as pw


def one(seed):
    df, extreme = pw.simulate(seed=seed)
    months = df.index.to_period("M")
    order = {m: i for i, m in enumerate(sorted(months.unique()))}
    idx = np.array([order[m] for m in months])
    p, _ = pw.fit_garch_midas(df.shock.values, idx, extreme.values.astype(float))
    return dict(seed=seed, **p)


if __name__ == "__main__":
    with ProcessPoolExecutor(5) as ex:
        rows = list(ex.map(one, range(10)))
    out = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/midas_recovery.csv", index=False)
    print(out.round(3).to_string(index=False))
    print("\nmean:", out[["alpha", "beta", "m", "theta", "w"]].mean().round(3).to_dict())
    print("true:", {k: pw.TRUE[k] for k in ("garch_alpha", "garch_beta", "midas_m", "midas_theta", "midas_w")})
