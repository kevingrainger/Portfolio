#!/usr/bin/env python3
#-------- run_pinn.py ---------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Data-only and physics-informed U-Nets, trained on station readings. The emulator in
# run_unet.py learned from the physics surface, so it could never be an independent
# comparison; these learn from the thermometers themselves.
#
#   train     for each fold and each lambda in LAMBDAS: train on 2021 and June-July 2022
#             at the fold's training stations; score on August 2022 at the same
#             stations (an inner, temporal validation that never touches the fold's
#             own stations or 2023)
#   evaluate  lambda chosen on that inner validation; 2023 at the held-out stations,
#             alone and as the prior of the anchored physics
#   figures
#
#   python run_pinn.py [train] [evaluate] [figures]

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import run_unet as RU
from tmax1km import validation
from tmax1km.anchor import Anchor, Smoother
from tmax1km.pde import DaySystem

RES, WORK, PAD = RU.RES, RU.WORK, RU.PAD
LAMBDAS = [0.0, 0.03, 0.3, 3.0]
STEPS, SIZE, BATCH = 1200, 96, 8


def rate_map(D, prm):
    return RU.pad(prm.surface_rate(D.frac).astype(np.float32))


def arrays(days):
    """Standardised inputs, raw physics fields, station label maps for a set of days."""
    D = RU.ds()
    daily, static, mask = RU.inputs()
    scal = RU.standardise(daily, static, days)
    X = RU.stack(daily, static, scal, days)
    phys = np.zeros((len(days),) + PAD + (5,), np.float32)
    phys[..., 0] = daily[days, ..., 0]                                    # theta_E
    phys[..., 1] = daily[days, ..., 0] + daily[days, ..., 1]              # theta_s
    phys[..., 2] = daily[days, ..., 2]; phys[..., 3] = daily[days, ..., 3]
    Y = np.zeros((len(days),) + PAD, np.float32)
    have = np.zeros((len(days), len(D.stations)), bool)
    iy, ix = D.stations.iy.values, D.stations.ix.values
    for j, d in enumerate(days):
        y = D.y[d]
        ok = np.isfinite(y)
        Y[j, iy[ok], ix[ok]] = y[ok]
        have[j] = ok
    return X, phys, Y, have, scal, mask


def label_mask(have, keep):
    D = RU.ds()
    M = np.zeros((have.shape[0],) + PAD, np.float32)
    iy, ix = D.stations.iy.values, D.stations.ix.values
    for j in range(have.shape[0]):
        s = have[j] & keep
        M[j, iy[s], ix[s]] = 1.0
    return M


def train():
    import jax
    import optax
    from tmax1km import pinn, unet
    D = RU.ds()
    P, fold = RU.params(), RU.folds()
    days = D.years(2021, 2022)
    inner_va = days[(D.days[days].year == 2022) & (D.days[days].month == 8)]
    inner_tr = np.setdiff1d(days, inner_va)
    X, PHYS, Y, have, scal, mask = arrays(inner_tr)
    Xv, PHv, Yv, hv, _, _ = arrays(inner_va)
    Xv = RU.stack(RU.inputs()[0], RU.inputs()[1], scal, inner_va)
    land = mask
    log = []
    for k in range(5):
        prm = P[f"fold{k}"]
        PHYS[..., 4] = rate_map(D, prm)
        keep = fold != k
        M = label_mask(have, keep)
        centres = [(D.stations.iy.values[i], D.stations.ix.values[i]) for i in np.flatnonzero(keep)]
        for lam in LAMBDAS:
            path = os.path.join(WORK, f"pinn_fold{k}_lam{lam}.pkl")
            t0 = time.time()
            if os.path.exists(path):                         # resume: a finished network is only re-scored
                p = pickle.load(open(path, "rb"))["params"]
                dat = ph = np.nan
            else:
                rng = np.random.default_rng(100 + k)
                p = unet.init(jax.random.PRNGKey(k), X.shape[-1])
                opt = optax.adamw(optax.cosine_decay_schedule(2e-3, STEPS), weight_decay=1e-4)
                s = opt.init(p)
                step = pinn.make_step(opt, lam, prm.kappa, prm.tau_a)
                for it in range(STEPS):
                    b = pinn.station_crops(X, PHYS, Y, M, land, np.arange(len(inner_tr)), centres, SIZE, BATCH, rng)
                    p, s, l, (dat, ph) = step(p, s, *b)
            # inner validation: August 2022 at the training stations
            out = np.asarray(unet.predict(p, Xv))
            errs = []
            for j, d in enumerate(inner_va):
                ok = hv[j] & keep
                th = PHv[j, ..., 0] + out[j, ..., 0]
                errs += list(th[D.stations.iy.values[ok], D.stations.ix.values[ok]] - D.y[d][ok])
            rec = dict(fold=k, lam=lam, inner_val_rmse=validation.rmse(errs), last_data=float(dat), last_phys=float(ph),
                       seconds=time.time() - t0)
            log.append(rec); print(rec, flush=True)
            pickle.dump(dict(params=jax.device_get(p), scal=scal), open(path, "wb"))
            pd.DataFrame(log).to_csv(os.path.join(RES, "pinn_training.csv"), index=False, float_format="%.4f")


def name(lam):
    return f"L{lam:g}"


def evaluate():
    """Every lambda is scored. The inner validation sits at the training stations, where
    a physics penalty can only cost accuracy; its value is meant to show at stations the
    network never saw. So lambda is reported as a dial, not tuned on the test set."""
    from tmax1km import unet
    D = RU.ds()
    P, fold = RU.params(), RU.folds()
    daily, static, mask = RU.inputs()
    H, W = D.grid.shape
    sm = Smoother(D.grid, RU.L_Q)
    land = ~D.sea & D.grid.interior()
    nets = {(k, l): pickle.load(open(os.path.join(WORK, f"pinn_fold{k}_lam{l}.pkl"), "rb")) for k in range(5) for l in LAMBDAS}
    rows, truth_rows = [], []
    part = os.path.join(WORK, "pinn_eval"); os.makedirs(part, exist_ok=True)
    for d in D.years(2023):
        cache = os.path.join(part, f"{d}.pkl")
        if os.path.exists(cache):                          # resume: days already scored
            r_, t_ = pickle.load(open(cache, "rb")); rows += r_; truth_rows += t_
            continue
        day_rows, day_truth = [], []
        y, f = D.y[d], D.day(d)
        have = np.isfinite(y)
        truth = D.field("truth", d) + RU.GAMMA * D.elev if "truth" in D._maps else None
        for k in range(5):
            test, use = have & (fold == k), have & (fold != k)
            if not test.any():
                continue
            an = Anchor(DaySystem(D.grid, P[f"fold{k}"], D.frac, f), D.cells, sm)
            Sinv = np.linalg.inv(an.S[np.ix_(use, use)])
            rec = {int(i): dict(date=D.days[d], station=int(i), obs=y[i]) for i in np.flatnonzero(test)}
            tr = dict(date=D.days[d], fold=k)
            for l in LAMBDAS:
                net = nets[(k, l)]
                out = np.asarray(unet.predict(net["params"], RU.stack(daily, static, net["scal"], np.array([d]))))[0, :H, :W]
                field = f["theta_E"] + out[..., 0]
                at = field.ravel()[D.cells]
                anchored = at + an.S[:, use] @ (Sinv @ (y[use] - at[use]))
                n = name(l)
                for i in rec:
                    c = D.cells[i]
                    rec[i].update({n: at[i], f"{n}_lo": f["theta_E"].ravel()[c] + out[..., 1].ravel()[c],
                                   f"{n}_hi": f["theta_E"].ravel()[c] + out[..., 2].ravel()[c], f"{n}A": anchored[i]})
                if truth is not None:
                    err = field[land] - truth[land]
                    tr[n] = float(np.sqrt(np.mean(err ** 2)))
            day_rows += list(rec.values()); day_truth.append(tr)
        pickle.dump((day_rows, day_truth), open(cache, "wb"))
        rows += day_rows; truth_rows += day_truth
        print("evaluated", D.days[d].date(), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "pinn_predictions_2023.csv"), index=False, float_format="%.4f")
    pd.DataFrame(truth_rows).to_csv(os.path.join(RES, "pinn_truth_check.csv"), index=False, float_format="%.4f")
    summary()


def summary():
    base = pd.read_csv(os.path.join(RES, "predictions_2023.csv"), parse_dates=["date"])
    u = pd.read_csv(os.path.join(RES, "unet_predictions_2023.csv"), parse_dates=["date"])
    p = pd.read_csv(os.path.join(RES, "pinn_predictions_2023.csv"), parse_dates=["date"])
    m = base.merge(u[["date", "station", "U", "M4"]], on=["date", "station"]).merge(p.drop(columns="obs"), on=["date", "station"])
    hot = m.obs >= m.obs.quantile(0.95)
    far = m.anchor_km >= 60
    models = ["B0", "M1", "M2", "M3", "U", "M4"] + [name(l) for l in LAMBDAS] + [name(l) + "A" for l in LAMBDAS]
    out = []
    for nm in models:
        e = m[nm] - m.obs
        row = dict(model=nm, rmse=validation.rmse(e), bias=float(e.mean()), rmse_hottest_5pct=validation.rmse(e[hot]),
                   rmse_far_60km=validation.rmse(e[far]))
        for lab, lo, hi in validation.ELEVATION_BANDS:
            sel = (m.elevation >= lo) & (m.elevation < hi)
            row[lab] = validation.rmse(e[sel])
        if nm in [name(l) for l in LAMBDAS]:
            row["coverage_90"] = float(((m.obs >= m[f"{nm}_lo"]) & (m.obs <= m[f"{nm}_hi"])).mean())
        out.append(row)
    tab = pd.DataFrame(out)
    tab.to_csv(os.path.join(RES, "pinn_skill_table.csv"), index=False, float_format="%.3f")
    truth = pd.read_csv(os.path.join(RES, "pinn_truth_check.csv"))
    tu = pd.read_csv(os.path.join(RES, "unet_truth_check.csv"))
    tr = pd.read_csv(os.path.join(RES, "pinn_training.csv"))
    info = dict(n=int(len(m)), n_far=int(far.sum()),
                truth_rmse={**dict(B0=float(tu.B0.mean()), M2=float(tu.M2.mean()), U=float(tu.U.mean())),
                            **{name(l): float(truth[name(l)].mean()) for l in LAMBDAS}},
                inner_val_rmse={name(l): float(tr[tr.lam == l].inner_val_rmse.mean()) for l in LAMBDAS})
    json.dump(info, open(os.path.join(RES, "pinn_summary.json"), "w"), indent=2)
    print(tab.round(3).to_string(index=False)); print(json.dumps(info, indent=1))


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import figstyle
    from tmax1km import plots, unet
    figstyle.apply()
    tab = pd.read_csv(os.path.join(RES, "pinn_skill_table.csv")).set_index("model")
    info = json.load(open(os.path.join(RES, "pinn_summary.json")))
    lams = LAMBDAS
    xs = np.arange(len(lams))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(wspace=0.3))
    a1.plot(xs, [info["inner_val_rmse"][name(l)] for l in lams], "s--", color="#555555", mfc="white", mew=1.3, label="training stations, Aug 2022")
    a1.plot(xs, [tab.loc[name(l), "rmse"] for l in lams], "o-", color="#0072B2", mfc="white", mew=1.3, label="held-out stations, 2023")
    a1.plot(xs, [tab.loc[name(l), "rmse_far_60km"] for l in lams], "^-", color="#009E73", mfc="white", mew=1.3, label="held-out, > 60 km from an anchor")
    a1.plot(xs, [info["truth_rmse"][name(l)] for l in lams], "D-", color="#CC79A7", mfc="white", mew=1.3, label="every land cell (hidden truth)")
    a1.set_xticks(xs); a1.set_xticklabels(["0\n(data only)"] + [f"{l:g}" for l in lams[1:]])
    a1.set_xlabel(r"weight of the heat budget in the loss, $\lambda$"); a1.set_ylabel("RMSE (K)")
    a1.set_title("(a) Turning up the physics", loc="left")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8.5)
    models = ["B0", "M1", "U", name(lams[0]), name(lams[-1]), "M2", name(lams[-1]) + "A"]
    labels = ["ERA5-Land", "physics only", "U-Net emulator", "U-Net, data only", f"U-Net, physics-informed (λ = {lams[-1]:g})",
              "physics, anchored", "physics, anchored, PINN prior"]
    cols = ["#9a9a9a", "#a8d5ba", "#56B4E9", "#E69F00", "#0072B2", "#3f9b7a", "#CC79A7"]
    groups = ["rmse", "rmse_far_60km", "> 800 m", "rmse_hottest_5pct"]
    glab = ["all", "> 60 km from\nan anchor", "> 800 m", "hottest 5%\nof days"]
    w = 0.8 / len(models)
    for j, (m_, lab, c) in enumerate(zip(models, labels, cols)):
        a2.bar(np.arange(len(groups)) + (j - len(models) / 2 + 0.5) * w, tab.loc[m_, groups].values.astype(float), w,
               color=c, label=lab, edgecolor="white", linewidth=0.8)
    a2.set_xticks(np.arange(len(groups))); a2.set_xticklabels(glab)
    a2.set_ylabel("RMSE at held-out stations, 2023 (K)"); a2.grid(axis="x", visible=False)
    a2.set_title("(b) Where it matters", loc="left")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8)
    figstyle.watermark(fig, "Simulated test data")
    figstyle.save(fig, os.path.join(HERE, "figures", "fig8_physics_informed"))
    plt.close(fig)

    # the hottest 2023 day: data-only and physics-informed fields side by side (3D)
    D = RU.ds()
    fold = RU.folds()
    daily, static, mask = RU.inputs()
    H, W = D.grid.shape
    days23 = D.years(2023)
    d = int(days23[np.nanargmax(np.nanmean(D.tmax[days23], axis=1))])
    y, f = D.y[d], D.day(d)
    have = np.isfinite(y)
    k = int(np.argmax([np.sum(have & (fold == j)) for j in range(5)]))
    land = ~D.sea
    fields = []
    for l in (lams[0], lams[-1]):
        net = pickle.load(open(os.path.join(WORK, f"pinn_fold{k}_lam{l}.pkl"), "rb"))
        out = np.asarray(unet.predict(net["params"], RU.stack(daily, static, net["scal"], np.array([d]))))[0, :H, :W]
        fields.append(f["theta_E"] + out[..., 0])
    lo, hi = np.nanpercentile(fields[1][land], [1, 99.5])
    norm = Normalize(lo - 1, hi + 1)
    test = have & (fold == k)
    cells = (D.stations.ix.values, D.stations.iy.values, test)
    fig = plt.figure(figsize=(14, 5.0))
    fig.subplots_adjust(left=0.0, right=0.97, top=1.02, bottom=0.0, wspace=0.02)
    for j, (field, title) in enumerate([(fields[0], "(a) U-Net trained on thermometers only"),
                                        (fields[1], f"(b) The same, with the heat budget in its loss (λ = {lams[-1]:g})")]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        plots.temperature_surface(ax, D.grid, np.where(land, field, np.nan), cells, y, plots.TEMP, norm, title,
                                  stride=2, zlim=(norm.vmin, norm.vmax + 2), hide=~land)
        ax.view_init(elev=28, azim=-62)
    fig.text(0.5, 0.02, f"{D.days[d].date()}, hottest 2023 day at the stations. Pins: the {test.sum()} stations of held-out fold {k}; "
             "a pin's head is the reading.", ha="center", fontsize=9)
    figstyle.watermark(fig, "Simulated test data")
    figstyle.save(fig, os.path.join(HERE, "figures", "fig9_pinn_surfaces"))
    plt.close(fig)


if __name__ == "__main__":
    for w in sys.argv[1:]:
        globals()[w]()
