#!/usr/bin/env python3
#-------- run_unet.py ---------------------------------------------------------------------
#-----------------------------------------------------------------------------
# The neural-network addition: a U-Net trained from scratch, scored exactly as the other
# models are (2023, stations in held-out 50 km blocks), plus the U-Net used as the prior
# of the physics anchoring (M4), the way XGBoost is used in M3.
#
#   targets   for every 2021-2022 day and every fold, the anchored physics surface (M2)
#             built WITHOUT that fold's stations: the dense training target
#   train     one U-Net per fold, on random 96 km windows of 2021-2022
#   evaluate  2023: U-Net alone (U), U-Net anchored (M4), at held-out stations; error
#             against the simulation's hidden truth at every land cell; interval coverage
#   figures
#
#   python run_unet.py [targets] [train] [evaluate] [figures]

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
from tmax1km import validation
from tmax1km.anchor import Anchor, Smoother
from tmax1km.data import Dataset
from tmax1km.grid import CLASSES, GAMMA
from tmax1km.pde import DaySystem, Params

RES = os.path.join(HERE, "results")
WORK = os.path.join(HERE, "data", "cache", "unet")
os.makedirs(WORK, exist_ok=True)
L_Q = json.load(open(os.path.join(RES, "run.json")))["l_q_km"]
PAD = (248, 464)                        # the 245 x 460 grid padded to a multiple of 4
STATIC = ["elevation", "slope", "tpi", "coast"]

_D = None


def ds():
    global _D
    if _D is None:
        _D = Dataset(verbose=True)
    return _D


def params():
    df = pd.read_csv(os.path.join(RES, "m1_params.csv")).set_index("model")
    return {m: Params.unpack(df.loc[m, Params.names()].values.astype(float)) for m in df.index}


def folds():
    D = ds()
    return validation.spatial_folds(D.stations, D.grid)[0]


def pad(a):
    out = np.zeros(PAD + a.shape[2:], a.dtype)
    out[:a.shape[0], :a.shape[1]] = a
    return out


#-------- inputs: 4 daily channels and 10 static ones ------------------------------------
def inputs():
    path = os.path.join(WORK, "inputs.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["daily"], z["static"], z["mask"]
    D = ds()
    n = len(D.days)
    daily = np.zeros((n,) + PAD + (4,), np.float32)
    for d in range(n):
        f = D.day(d)
        daily[d] = pad(np.stack([f["theta_E"], f["theta_s"] - f["theta_E"], f["u"], f["v"]], -1))
    static = pad(np.stack([D.features[k] for k in STATIC] + [D.frac[i] for i in range(len(CLASSES))], -1)).astype(np.float32)
    mask = pad((~D.sea & D.grid.interior()).astype(np.float32))
    np.savez(path, daily=daily, static=static, mask=mask)
    return daily, static, mask


#-------- targets ---------------------------------------------------------------------------
def targets():
    D = ds()
    P, fold = params(), folds()
    train_days = D.years(2021, 2022)
    sm = Smoother(D.grid, L_Q)
    T = np.lib.format.open_memmap(os.path.join(WORK, "targets.npy"), "w+", np.float32, (5, len(train_days)) + PAD)
    t0 = time.time()
    for j, d in enumerate(train_days):
        y, f = D.y[d], D.day(d)
        have = np.isfinite(y)
        for k in range(5):
            an = Anchor(DaySystem(D.grid, P[f"fold{k}"], D.frac, f), D.cells, sm)
            theta, _ = an.solve(y, have & (fold != k))
            T[k, j] = pad((theta - f["theta_E"]).astype(np.float32))
        if j % 10 == 0:
            print(f"targets day {j}/{len(train_days)}  {time.time() - t0:.0f}s", flush=True)
    T.flush()
    np.save(os.path.join(WORK, "train_days.npy"), train_days)


#-------- training ---------------------------------------------------------------------------
def standardise(daily, static, days):
    mu_d = daily[days].reshape(-1, 4).mean(0); sd_d = daily[days].reshape(-1, 4).std(0) + 1e-6
    mu_s = static.reshape(-1, static.shape[-1]).mean(0); sd_s = static.reshape(-1, static.shape[-1]).std(0) + 1e-6
    return mu_d, sd_d, mu_s, sd_s


def stack(daily, static, scal, days):
    mu_d, sd_d, mu_s, sd_s = scal
    s = (static - mu_s) / sd_s
    return np.concatenate([(daily[days] - mu_d) / sd_d, np.broadcast_to(s, (len(days),) + s.shape)], -1).astype(np.float32)


def train(steps=1500, size=96, batch=8, seed=0):
    import jax
    import optax
    from tmax1km import unet
    daily, static, mask = inputs()
    T = np.load(os.path.join(WORK, "targets.npy"), mmap_mode="r")
    train_days = np.load(os.path.join(WORK, "train_days.npy"))
    scal = standardise(daily, static, train_days)
    X = stack(daily, static, scal, train_days)                       # (days, H, W, 14)
    log = []
    for k in range(5):
        rng = np.random.default_rng(seed + k)
        Y = np.asarray(T[k])
        # validate on the last 20 training days (late 2022), train on the rest
        tr, va = np.arange(len(train_days) - 20), np.arange(len(train_days) - 20, len(train_days), 4)
        p = unet.init(jax.random.PRNGKey(seed + k), X.shape[-1])
        sched = optax.cosine_decay_schedule(2e-3, steps)
        opt = optax.adam(sched)
        s = opt.init(p)
        step = unet.make_step(opt)
        t0 = time.time()
        for it in range(steps):
            xb, yb, mb = unet.random_crops(X, Y, mask, tr, size, batch, rng)
            p, s, l, (mse, pin) = step(p, s, xb, yb, mb)
            if it % 500 == 0 or it == steps - 1:
                out = np.asarray(unet.predict(p, X[va]))
                m = mask.astype(bool)
                vrmse = float(np.sqrt(np.mean((out[..., 0] - Y[va])[:, m] ** 2)))
                log.append(dict(fold=k, step=it, train_loss=float(l), val_rmse_K=vrmse, seconds=time.time() - t0))
                print(log[-1], flush=True)
        pickle.dump(dict(params=jax.device_get(p), scal=scal), open(os.path.join(WORK, f"unet_fold{k}.pkl"), "wb"))
        if k == 0:
            print("parameters:", unet.n_params(p))
    pd.DataFrame(log).to_csv(os.path.join(RES, "unet_training.csv"), index=False, float_format="%.4f")


#-------- evaluation ---------------------------------------------------------------------------
def evaluate():
    from tmax1km import unet
    D = ds()
    P, fold = params(), folds()
    daily, static, mask = inputs()
    test_days = D.years(2023)
    sm = Smoother(D.grid, L_Q)
    land = (~D.sea & D.grid.interior())
    H, W = D.grid.shape
    rows, truth_rows = [], []
    nets = [pickle.load(open(os.path.join(WORK, f"unet_fold{k}.pkl"), "rb")) for k in range(5)]
    for d in test_days:
        y, f = D.y[d], D.day(d)
        have = np.isfinite(y)
        truth = D.field("truth", d) + GAMMA * D.elev if "truth" in D._maps else None
        for k in range(5):
            test, use = have & (fold == k), have & (fold != k)
            if not test.any():
                continue
            x = stack(daily, static, nets[k]["scal"], np.array([d]))
            out = np.asarray(unet.predict(nets[k]["params"], x))[0, :H, :W]
            field = f["theta_E"] + out[..., 0]
            lo, hi = f["theta_E"] + out[..., 1], f["theta_E"] + out[..., 2]
            an = Anchor(DaySystem(D.grid, P[f"fold{k}"], D.frac, f), D.cells, sm)
            # M4: the U-Net field as the prior, corrected by the same anchoring as M2 and M3
            u_at = field.ravel()[D.cells]
            lam = np.linalg.solve(an.S[np.ix_(use, use)], y[use] - u_at[use])
            m4 = u_at + an.S[:, use] @ lam
            m2 = an.at_stations(y, use)
            for i in np.flatnonzero(test):
                c = D.cells[i]
                rows.append(dict(date=D.days[d], station=int(i), U=field.ravel()[c], U_lo=lo.ravel()[c], U_hi=hi.ravel()[c],
                                 M4=m4[i], M2_check=m2[i], obs=y[i]))
            if truth is not None:
                m2_field, _ = an.solve(y, use, check=False)
                truth_rows.append(dict(date=D.days[d], fold=k,
                                       U=float(np.sqrt(np.mean((field[land] - truth[land]) ** 2))),
                                       M2=float(np.sqrt(np.mean((m2_field[land] - truth[land]) ** 2))),
                                       B0=float(np.sqrt(np.mean((f["theta_E"][land] - truth[land]) ** 2)))))
        print("evaluated", D.days[d].date(), flush=True)
    pred = pd.DataFrame(rows)
    pred.to_csv(os.path.join(RES, "unet_predictions_2023.csv"), index=False, float_format="%.4f")
    pd.DataFrame(truth_rows).to_csv(os.path.join(RES, "unet_truth_check.csv"), index=False, float_format="%.4f")
    summary()


def conformal():
    """Conformalised quantile regression (Romano, Patterson & Candes 2019). The U-Net's
    interval was trained against the smooth physics surface, not against thermometers,
    so it is too narrow at stations. On 2022, each fold's net is scored at that fold's
    stations (never anchors in its targets); the 90th percentile of how far readings
    fall outside the interval widens it for 2023."""
    from tmax1km import unet
    D = ds()
    fold = folds()
    daily, static, mask = inputs()
    H, W = D.grid.shape
    nets = [pickle.load(open(os.path.join(WORK, f"unet_fold{k}.pkl"), "rb")) for k in range(5)]
    E = {k: [] for k in range(5)}
    for d in D.years(2022):
        y = D.y[d]
        have = np.isfinite(y)
        for k in range(5):
            test = have & (fold == k)
            if not test.any():
                continue
            out = np.asarray(unet.predict(nets[k]["params"], stack(daily, static, nets[k]["scal"], np.array([d]))))[0, :H, :W]
            tE = D.field("theta_E", d).ravel()[D.cells[test]]
            lo, hi = tE + out[..., 1].ravel()[D.cells[test]], tE + out[..., 2].ravel()[D.cells[test]]
            E[k] += list(np.maximum(lo - y[test], y[test] - hi))
    Q = {k: float(np.quantile(E[k], 0.9)) for k in range(5)}
    json.dump(Q, open(os.path.join(RES, "unet_conformal.json"), "w"), indent=2)
    print("conformal widening by fold (K):", Q)


def summary():
    D = ds()
    st = D.stations
    fold_of = folds()
    u = pd.read_csv(os.path.join(RES, "unet_predictions_2023.csv"), parse_dates=["date"])
    base = pd.read_csv(os.path.join(RES, "predictions_2023.csv"), parse_dates=["date"])
    base["station"] = base["station"].astype(int)
    m = base.merge(u[["date", "station", "U", "U_lo", "U_hi", "M4", "M2_check"]], on=["date", "station"])
    assert np.allclose(m.M2, m.M2_check, atol=1e-3), "M2 recomputed differs from the pipeline's"
    hot = m.obs >= m.obs.quantile(0.95)
    out = []
    for name in ["B0", "B1", "B3", "M1", "M2", "M3", "U", "M4"]:
        e = m[name] - m.obs
        row = dict(model=name, rmse=validation.rmse(e), bias=float(e.mean()), rmse_hottest_5pct=validation.rmse(e[hot]))
        for lab, lo, hi in validation.ELEVATION_BANDS:
            sel = (m.elevation >= lo) & (m.elevation < hi)
            row[lab] = validation.rmse(e[sel])
        out.append(row)
    tab = pd.DataFrame(out)
    tab.to_csv(os.path.join(RES, "unet_skill_table.csv"), index=False, float_format="%.3f")
    cover = float(((m.obs >= m.U_lo) & (m.obs <= m.U_hi)).mean())
    cover_cqr, width = None, float((m.U_hi - m.U_lo).mean())
    if os.path.exists(os.path.join(RES, "unet_conformal.json")):
        Q = {int(k): v for k, v in json.load(open(os.path.join(RES, "unet_conformal.json"))).items()}
        q = m.station.map(lambda i: Q[int(fold_of[i])])
        cover_cqr = float(((m.obs >= m.U_lo - q) & (m.obs <= m.U_hi + q)).mean())
        width_cqr = float((m.U_hi - m.U_lo + 2 * q).mean())
    truth = pd.read_csv(os.path.join(RES, "unet_truth_check.csv"))
    info = dict(coverage_90=cover, mean_width_K=width, coverage_90_conformal=cover_cqr,
                mean_width_conformal_K=width_cqr if cover_cqr is not None else None,
                n=int(len(m)), truth_rmse=truth[["B0", "M2", "U"]].mean().to_dict())
    json.dump(info, open(os.path.join(RES, "unet_summary.json"), "w"), indent=2)
    print(tab.round(3).to_string(index=False)); print(info)


#-------- figures ------------------------------------------------------------------------------
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    sys.path.insert(0, os.path.dirname(HERE))
    import figstyle
    from tmax1km import plots, unet
    figstyle.apply()
    D = ds()
    P, fold = params(), folds()
    daily, static, mask = inputs()
    H, W = D.grid.shape
    st = D.stations
    # the hottest 2023 day at the stations, and the fold with the most held-out stations reporting
    days23 = D.years(2023)
    d = int(days23[np.nanargmax(np.nanmean(D.tmax[days23], axis=1))])
    y, f = D.y[d], D.day(d)
    have = np.isfinite(y)
    k = int(np.argmax([np.sum(have & (fold == j)) for j in range(5)]))
    test, use = have & (fold == k), have & (fold != k)
    net = pickle.load(open(os.path.join(WORK, f"unet_fold{k}.pkl"), "rb"))
    out = np.asarray(unet.predict(net["params"], stack(daily, static, net["scal"], np.array([d]))))[0, :H, :W]
    u_field = f["theta_E"] + out[..., 0]
    an = Anchor(DaySystem(D.grid, P[f"fold{k}"], D.frac, f), D.cells, Smoother(D.grid, L_Q))
    m2_field, _ = an.solve(y, use, check=False)
    land = ~D.sea
    lo, hi = np.nanpercentile(m2_field[land], [1, 99.5])
    norm = Normalize(lo - 1, hi + 1)
    cells = (st.ix.values, st.iy.values, test)
    fig = plt.figure(figsize=(14, 5.0))
    fig.subplots_adjust(left=0.0, right=0.97, top=1.02, bottom=0.0, wspace=0.02)
    for j, (field, title) in enumerate([(m2_field, "(a) Physics surface, anchored on the other folds' stations"),
                                        (u_field, "(b) U-Net, from the coarse fields alone")]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        plots.temperature_surface(ax, D.grid, np.where(land, field, np.nan), cells, y, plots.TEMP, norm, title,
                                  stride=2, zlim=(norm.vmin, norm.vmax + 2), hide=~land)
        ax.view_init(elev=28, azim=-62)
    fig.text(0.5, 0.02, f"{D.days[d].date()}, hottest 2023 day at the stations. Pins: the {test.sum()} stations of held-out fold {k}, "
             "which neither surface was given; a pin's head is the reading.", ha="center", fontsize=9)
    figstyle.watermark(fig, "Simulated test data")
    figstyle.save(fig, os.path.join(HERE, "figures", "fig6_unet_surfaces"))
    plt.close(fig)

    # skill by elevation band and on the hottest days
    tab = pd.read_csv(os.path.join(RES, "unet_skill_table.csv"))
    info = json.load(open(os.path.join(RES, "unet_summary.json")))
    models = ["B0", "M2", "M3", "U", "M4"]
    names = {"B0": "ERA5-Land", "M2": "physics, anchored", "M3": "physics + XGBoost prior", "U": "U-Net alone", "M4": "physics + U-Net prior"}
    cols = {"B0": "#9a9a9a", "M2": "#3f9b7a", "M3": "#14532d", "U": "#0072B2", "M4": "#CC79A7"}
    groups = ["rmse", "< 200 m", "200-800 m", "> 800 m", "rmse_hottest_5pct"]
    labels = ["all", "< 200 m", "200-800 m", "> 800 m", "hottest 5%\nof days"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.8 / len(models)
    t = tab.set_index("model")
    for j, m in enumerate(models):
        ax.bar(np.arange(len(groups)) + (j - len(models) / 2 + 0.5) * width, t.loc[m, groups].values.astype(float), width,
               color=cols[m], label=names[m], edgecolor="white", linewidth=1)
    ax.set_xticks(np.arange(len(groups))); ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE at held-out stations, 2023 (K)"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, fontsize=8.5)
    ax.set_title(f"Five models, identical test data; the U-Net's 90% interval holds {100 * info['coverage_90']:.0f}% of readings, "
                 f"{100 * info['coverage_90_conformal']:.0f}% once conformalised", loc="left", fontsize=10)
    figstyle.watermark(fig, "Simulated test data")
    figstyle.save(fig, os.path.join(HERE, "figures", "fig7_unet_skill"))
    plt.close(fig)

    # interactive: error at each held-out station for M2, U-Net and M4
    import folium
    from make_maps import base_map
    u = pd.read_csv(os.path.join(RES, "unet_predictions_2023.csv"))
    base = pd.read_csv(os.path.join(RES, "predictions_2023.csv"))
    mrg = base.merge(u[["date", "station", "U", "M4"]], on=["date", "station"])
    per = mrg.groupby("station").apply(lambda g: pd.Series({m: validation.rmse(g[m] - g.obs) for m in ["M2", "U", "M4"]}))
    m = base_map(D)
    cmap = plots.TEMP
    nrm = Normalize(0.3, 1.6)
    from matplotlib.colors import to_hex
    for key in ["M2", "U", "M4"]:
        layer = folium.FeatureGroup(names[key], show=(key == "U"))
        for i, r in per.iterrows():
            s_ = st.iloc[int(i)]
            folium.CircleMarker([s_.latitude, s_.longitude], radius=7, color="#1b1b1f", weight=1, fill=True, fill_opacity=1,
                                fill_color=to_hex(cmap(nrm(r[key]))),
                                tooltip=f"{s_['name']} ({s_.elevation:.0f} m): {names[key]} {r[key]:.2f} K; "
                                        f"physics {r.M2:.2f}, U-Net {r.U:.2f}, physics + U-Net {r.M4:.2f}").add_to(layer)
        layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(os.path.join(HERE, "maps", "unet_errors.html"))


if __name__ == "__main__":
    for w in sys.argv[1:]:
        globals()[w]()
