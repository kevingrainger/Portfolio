#!/usr/bin/env python3
#-------- fit_clumping.py -----------------------------------------------------------
#-----------------------------------------------------------------------------
# Uneven ground, fitted to how clumped the real vegetation is.
#
# clumping.py found the satellite vegetation about 40% more clumped than the lattice at the
# same density, at both sites and whatever the seed spread or seed rain. The lattice treats
# every cell alike; real ground isn't: water runs off slopes into hollows, and some soils
# hold it better. So each cell gets a fixed site quality Q (sahel_ca.quality_field: smoothed
# white noise, mean 1), b_i = b x Q_i, with two parameters:
#   strength   the standard deviation of Q (0 = uniform ground)
#   length     the smoothing length, in 30 m cells: the size of the good and bad patches
#
# For each pair, the lattice is run at Gourma and Fakara exactly as in the animation, and
# its clumping C = q(+|+)/rho compared with the satellite pictures' at the same density
# (clumping.py). The best pair minimises the squared log ratio over both sites. It is then
# checked against the Sahel-wide 1982-2021 test (joint_fit_1982.py metrics), so that
# matching the pattern doesn't cost the fit.
#
#   python fit_clumping.py
#
# Output: results/fit_clumping.json, results/fit_clumping.csv, figures/fig16_clumping.png

import json
import os
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
LENGTHS = [2, 4, 8]
DELTA, SEED_RAIN = 0.03, 0.03
SITES = ["gourma", "fakara"]


def site_run(args):
    site, strength, length = args
    import animate_landsat as al
    from clumping import clump
    from sahel_ca import Landscape, VEG, EMPTY, quality_field
    Q = quality_field(200, strength, length, seed=11)
    p3 = al.site_rain(site)
    land = Landscape(n=200, b=float(min(1, al.B0 * p3.loc[1981:1983].mean() / 400)), cover=0.5, seed=7, d=al.D0, f=0.9,
                     delta=DELTA, quality=Q)
    def seed():
        e = (land.grid == EMPTY) & (land.rng.random(land.grid.shape) < SEED_RAIN); land.grid[e] = VEG
    for _ in range(50):
        land.step(1); seed()
    cover, C = {}, {}
    for y, p in p3.items():
        land.b = float(min(1, al.B0 * p / 400)); land.step(1); seed()
        if y >= 1985:
            cover[int(y)] = land.cover(); C[int(y)] = clump(land.grid == VEG)
    return site, strength, length, cover, C


_Q = None


def _init(strength, length):
    global _Q
    from sahel_ca import quality_field
    _Q = quality_field(60, strength, length * 60 / 200, seed=11)       # same field, scaled to the 60 x 60 fit lattices


def fit_site(args):
    """joint_fit_1982.run_site with the quality field."""
    sid, p3, d, s_rain, seed_base = args
    from sahel_ca import Landscape, VEG, EMPTY
    covers = []
    for k in range(2):
        b = np.minimum(1.0, p3)
        land = Landscape(n=60, b=float(b[0]), cover=0.5, seed=seed_base + k, d=d, f=0.9, delta=DELTA, quality=_Q)
        for _ in range(50):
            land.step(1)
            e = (land.grid == EMPTY) & (land.rng.random(land.grid.shape) < s_rain); land.grid[e] = VEG
        out = []
        for bt in b:
            land.b = float(bt); land.step(1)
            e = (land.grid == EMPTY) & (land.rng.random(land.grid.shape) < s_rain); land.grid[e] = VEG
            out.append(land.cover())
        covers.append(out)
    return sid, np.mean(covers, axis=0)


def main():
    from clumping import satellite_clumping
    jobs = list(product(SITES, STRENGTHS, LENGTHS))
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        runs = pool.map(site_run, jobs)
    rows = []
    for site, st, L, cover, C in runs:
        s = satellite_clumping(site, cover)
        yrs = [y for y in s if y in C]
        rows.append(dict(site=site, strength=st, length=L, lattice_C=float(np.mean([C[y] for y in yrs])),
                         satellite_C=float(np.nanmean([s[y] for y in yrs])), cover_mean=float(np.mean(list(cover.values())))))
    g = pd.DataFrame(rows)
    g["log_ratio"] = np.log(g.lattice_C / g.satellite_C)
    score = g.groupby(["strength", "length"]).log_ratio.apply(lambda x: float((x ** 2).sum()))
    best = score.idxmin()
    print(g.round(3).to_string(index=False)); print("best", best, round(score.min(), 4))

    import joint_fit_1982 as jf
    t = pd.read_csv(os.path.join(HERE, "results", "covariates_sites.csv"))
    r = pd.read_csv(os.path.join(HERE, "data", "sahel_chirps_jjas.csv")).pivot(index="year", columns="id", values="rain_jjas")
    obs = pd.read_csv(os.path.join(HERE, "data", "sahel_avhrr_dry.csv")).pivot(index="year", columns="id", values="avhrr_dry")
    t = t[t.id.isin(r.columns) & t.id.isin(obs.columns)].copy()
    p3 = r.rolling(3, min_periods=1).mean()
    band = ((t.lon + 16) // 4).astype(int)
    test = list(t.id[band % 2 == 1])
    fits = {}
    for st, L in [(0.0, best[1]), best]:
        jobs = [(i, (0.45 * p3[i].loc[jf.YEARS] / 400).values, 0.3, SEED_RAIN, 1000 + int(i) * 10) for i in t.id]
        with Pool(max(1, os.cpu_count() - 1), initializer=_init, initargs=(st, L)) as pool:
            cov = pd.DataFrame(dict(pool.map(fit_site, jobs)), index=jf.YEARS)
        fits[f"strength {st}, length {L}"] = jf.metrics(cov, obs, test)
        print(st, L, {k: round(v, 2) for k, v in fits[f"strength {st}, length {L}"].items()}, flush=True)
    out = dict(best=dict(strength=best[0], length_cells=best[1], length_m=best[1] * 30), score=score.reset_index().to_dict(orient="records"),
               table=g.to_dict(orient="records"), sahel_fit=fits)
    json.dump(out, open(os.path.join(HERE, "results", "fit_clumping.json"), "w"), indent=1, default=float)
    g.to_csv(os.path.join(HERE, "results", "fit_clumping.csv"), index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw=dict(wspace=0.28))
    for ax, site in zip(axes, SITES):
        gs_ = g[g.site == site]
        for L, c, ls in zip(LENGTHS, [LIGHT, BLUE, NAVY], ["-", "-", "-"]):
            h = gs_[gs_.length == L].sort_values("strength")
            ax.plot(h.strength, h.lattice_C, "o" + ls, color=c, lw=1.6, ms=5, label=f"lattice, patches of {L * 30} m")
        sat = gs_.groupby("strength").satellite_C.mean()
        ax.plot(sat.index, sat.values, "s--", color=RED, lw=1.4, ms=6, label="satellite, at the same density")
        ax.axvline(best[0], color=GREY, lw=0.8, ls=":")
        ax.set_xlabel("strength of uneven ground (sd of site quality)"); ax.set_ylabel("clumping, q(+|+) / ρ")
        ax.set_title(f"{site.capitalize()}: clumping of vegetation", loc="left", fontsize=10.5)
        ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(os.path.join(HERE, "figures", "fig16_clumping.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
