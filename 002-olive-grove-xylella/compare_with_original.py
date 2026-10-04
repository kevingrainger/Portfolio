#!/usr/bin/env python3
#-------- compare_with_original.py --------------------------------------------
#-----------------------------------------------------------------------------
# Is olive_model.py the same model as the original OliveTree_SIR_model.py?
#
# Most fixes in olive_model only matter once trees turn symptomatic (around day
# 730), so the first 600 days are a like-for-like comparison - with one exception:
# the original starts with tender-sprout movement on day 0. olive_model can
# reproduce that (legacy_tender_start=True), so three versions are compared:
#
#   original       OliveTree_SIR_model.py
#   fast_legacy    olive_model.py with the original's day-0 behaviour - should match
#   fast_fixed     olive_model.py as it is used - shows what the day-0 slip did
#
# A stochastic model can only be checked statistically, never run for run.
#
# The original is slow - about five minutes per 600-day run, mostly in the tender
# season when every spittlebug is on a tree - so this takes about an hour. Writes results/original_vs_fast.json, which the notebook reads.

import io
import os
import sys
import json
import time
import random
import contextlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run_original(seeds, days=600):
    sys.modules.setdefault('seaborn', type(sys)('seaborn'))   # only used for a plot style
    import OliveTree_SIR_model as original
    out = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            g = original.OliveGrove(width=90, height=90, tree_spacing=5,
                                    vectors_per_hectare=1_000_000, sampling_rate=0.001)
            g.time_steps_per_day = 30
            g.initialize_vectors(1)
            for _ in range(days):
                for _ in range(30):
                    g.step()
        out.append(dict(seed=seed, infected=[1 - h['S'] for h in g.infection_history],
                        vectors=g.vector_infection_history, secs=time.time() - t))
        print(f"original seed {seed}: {time.time() - t:.0f}s", flush=True)
    return out


def run_fast(seeds, days=600, legacy=False):
    from olive_model import OliveGrove
    out = []
    for seed in seeds:
        t = time.time()
        g = OliveGrove(seed=seed, sampling_rate=0.001, time_steps_per_day=30,
                       legacy_tender_start=legacy).run(days)
        out.append(dict(seed=seed, infected=(1 - g.series('S')).tolist(),
                        vectors=g.series('vectors_infected').tolist(), secs=time.time() - t))
    return out


# The fast model is cheap, so its groups get 200 seeds: the first 40 run one at a
# time so their timings are a fair speed comparison, the rest in parallel.
def run_fast_many(seeds, days=600, legacy=False):
    from olive_model import run_ensemble
    runs = run_ensemble(days, seeds, legacy_tender_start=legacy)
    return [dict(seed=int(r['seed']), infected=(1 - r['S']).tolist(),
                 vectors=r['vectors'].tolist(), secs=None) for r in runs]


if __name__ == "__main__":
    results = dict(original=run_original(range(12)),
                   fast_legacy=run_fast(range(100, 140), legacy=True) + run_fast_many(range(140, 300), legacy=True),
                   fast_fixed=run_fast(range(100, 140), legacy=False) + run_fast_many(range(140, 300), legacy=False))
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', 'original_vs_fast.json'), 'w') as f:
        json.dump(results, f)
    print("written results/original_vs_fast.json")
