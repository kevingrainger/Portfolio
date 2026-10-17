#!/usr/bin/env python3
#-------- check_mortality.py -------------------------------------------------------
#-----------------------------------------------------------------------------
# A second piece of evidence on the two reefs: how much coral had died by the AIMS voyage.
#
# The April 2016 surveys score bleaching, which both posterior modes fit. But AIMS also
# reported from RV Solander at Scott Reef, in its 5 April 2016 update, that mortality was
# already widespread, and later that coral cover at some sites had already fallen by up
# to half. The two modes disagree sharply about that. For 20 draws from each mode, the
# reef is run through the NOAA heat-stress record to 5 April 2016, and the share of the
# starting living coral (healthy + bleached) that has died is recorded.
#
#   python inference/check_mortality.py
#
# Output: inference/results_crw/mortality_check.json

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from coral_crw import CRWReef                    # noqa: E402
from coral_model import BLEACHED, HEALTHY        # noqa: E402
from simulate_bank_crw import forcing            # noqa: E402

DAY = "2016-04-05"


def living(r):
    return np.mean(np.isin(r.coral[r.reef], (HEALTHY, BLEACHED)))


def main():
    post = np.load(os.path.join(HERE, "results_crw", "observed.npz"))["post"]
    dates, T = forcing()
    stop = int(np.flatnonzero(np.asarray(dates) == np.datetime64(DAY))[0])
    rng = np.random.default_rng(3)
    out = {}
    for name, pool in [("sensitive", post[post[:, 0] < 8]), ("tolerant", post[post[:, 0] >= 8])]:
        loss = []
        for k, th in enumerate(pool[rng.choice(len(pool), 20, replace=False)]):
            r = CRWReef(mu_D=th[0], sigma_D=th[1], s=th[2], beta=th[3], seed=50 + k)
            r.step(T[0])
            start = living(r)
            for d in range(1, stop + 1):
                r.step(T[d])
            loss.append(1 - living(r) / start)
        loss = np.array(loss)
        out[name] = dict(median=float(np.median(loss)), p10=float(np.quantile(loss, 0.1)), p90=float(np.quantile(loss, 0.9)))
        print(f"{name}: living coral lost by {DAY}: median {out[name]['median']:.2f} "
              f"(10-90%: {out[name]['p10']:.2f} to {out[name]['p90']:.2f})")
    json.dump(out, open(os.path.join(HERE, "results_crw", "mortality_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
