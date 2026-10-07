#!/usr/bin/env python3
#-------- run_pipeline.py --------------------------------------------------------------
#-----------------------------------------------------------------------------
# Steps 4-9: everything that takes minutes to hours. The notebook reads what this writes
# to results/, so its figures and tables never depend on a long run in a notebook cell.
#
#   python run_pipeline.py            run whatever has not been run yet
#   python run_pipeline.py --force    run everything again
#
#   fit       M1 parameters on about 40 days of 2021-2022: one fit on all stations, one
#             per spatial fold, and one with the wind switched off
#   smooth    choose the smoothness length l_q by blocked hold-out; collect the q fields
#             XGBoost learns from
#   learn     train XGBoost on q, once per fold (never seeing that fold's stations)
#   evaluate  2023, held-out blocks only: B0-B3 and M1-M3 at every held-out station
#   maps      the final product on every day with all stations as anchors, asserting
#             the surface equals the stations to 1e-6 K
#
# Each stage is cached. Two worker processes are used throughout.

import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tmax1km import ml, validation
from tmax1km.anchor import Anchor, Smoother, footprints
from tmax1km.data import Dataset
from tmax1km.grid import CLASSES, GAMMA
from tmax1km.pde import DaySystem, Params, length_scales, misfit

FORCE = "--force" in sys.argv
SMOKE = "--smoke" in sys.argv                          # a few days and iterations, to check the plumbing
RESULTS = os.path.join(HERE, "results_smoke" if SMOKE else "results")
WORKERS = 2
N_FIT_DAYS = 4 if SMOKE else 40
LQ_GRID = [10.0, 25.0] if SMOKE else [10.0, 25.0, 60.0, 120.0, 250.0]   # km
ITERATIONS = (2, 1, 1) if SMOKE else (30, 10, 12)      # all stations, each fold, no wind
N_SAMPLE = 2500                                       # cells drawn per day and fold for XGBoost
ALL = 5                                               # index of the "all stations" model, after folds 0-4

_DS, _SMOOTH, _MODELS = None, {}, {}


def ds():
    global _DS
    if _DS is None:
        _DS = Dataset(verbose=False)
    return _DS


def smoother(l_q):
    if l_q not in _SMOOTH:
        _SMOOTH[l_q] = Smoother(ds().grid, l_q)
    return _SMOOTH[l_q]


def q_model(k):
    if k not in _MODELS:
        m = ml.new_model()
        m.load_model(os.path.join(RESULTS, f"xgb_q_{'all' if k == ALL else 'fold' + str(k)}.json"))
        _MODELS[k] = m
    return _MODELS[k]


def calm(fields):
    return {**fields, "u": np.zeros_like(fields["u"]), "v": np.zeros_like(fields["v"])}


def run_parallel(fn, tasks, name, chunk=8):
    """Map fn over tasks in worker processes. Finished chunks are kept on disk, so an
    interrupted run picks up where it stopped."""
    import pickle
    tmp = os.path.join(HERE, "data", "cache", "run_smoke" if SMOKE else "run")
    os.makedirs(tmp, exist_ok=True)
    out = []
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("spawn")) as pool:
        for i in range(0, len(tasks), chunk):
            path = os.path.join(tmp, f"{name}_{i:04d}.pkl")
            if os.path.exists(path) and not FORCE:
                part = pickle.load(open(path, "rb"))
            else:
                part = list(pool.map(fn, tasks[i:i + chunk]))
                pickle.dump(part, open(path + ".tmp", "wb"))
                os.replace(path + ".tmp", path)
                print(f"  {name}: {min(i + chunk, len(tasks))}/{len(tasks)}", flush=True)
            out += part
    return out


#-------- stage: fit ------------------------------------------------------------------------
def _misfit_chunk(args):
    vec, days, use, wind = args
    D = ds()
    fields = [D.day(d) if wind else calm(D.day(d)) for d in days]
    y = [np.where(use, D.y[d], np.nan) for d in days]
    n = sum(int(np.isfinite(v).sum()) for v in y)
    J, g = misfit(vec, D.grid, D.frac, fields, D.cells, y)
    return J * n, g * n, n


class Objective:
    """Misfit and gradient summed over worker processes, with a log of every evaluation."""

    def __init__(self, pool, days, use, wind=True, label=""):
        self.pool, self.use, self.wind, self.label = pool, use, wind, label
        self.chunks = [list(c) for c in np.array_split(days, WORKERS)]
        self.calls = 0

    def __call__(self, vec):
        out = list(self.pool.map(_misfit_chunk, [(vec, c, self.use, self.wind) for c in self.chunks]))
        n = sum(o[2] for o in out)
        J, g = sum(o[0] for o in out) / n, sum(o[1] for o in out) / n
        self.calls += 1
        print(f"  {self.label} eval {self.calls:3d}  rmse {np.sqrt(J):.4f} K", flush=True)
        return J, g


def stage_fit(D, fold, fit_days):
    out = os.path.join(RESULTS, "m1_params.csv")
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) and not FORCE else []
    finished = {r["model"]: np.array([r[n] for n in Params.names()]) for r in rows}
    every = np.ones(len(D.stations), bool)
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("spawn")) as pool:
        def fit(label, use, start, maxiter, wind=True):
            if label in finished:
                return finished[label]
            # the best point so far is written after every evaluation, so a stopped fit resumes from it
            ckpt = os.path.join(RESULTS, f"_fit_{label}.json")
            state = json.load(open(ckpt)) if os.path.exists(ckpt) and not FORCE else dict(x=list(start), best=None, calls=0)
            obj = Objective(pool, fit_days, use, wind, label)
            obj.calls = state["calls"]

            def wrapped(vec):
                J, g = obj(vec)
                if state["best"] is None or J < state["best"]:
                    state.update(x=list(vec), best=float(J))
                state["calls"] = obj.calls
                json.dump(state, open(ckpt, "w"))
                return J, g

            left = max(1, maxiter - state["calls"])
            res = minimize(wrapped, np.array(state["x"]), jac=True, method="L-BFGS-B", bounds=Params.bounds(),
                           options=dict(maxiter=left, maxfun=left + 5))
            rows.append(dict(model=label, **dict(zip(Params.names(), res.x)), train_rmse=np.sqrt(res.fun), evaluations=obj.calls))
            pd.DataFrame(rows).to_csv(out, index=False)
            os.remove(ckpt)
            return res.x

        start = fit("all", every, Params().pack(), ITERATIONS[0])
        for k in range(5):
            fit(f"fold{k}", fold != k, start, ITERATIONS[1])
        fit("all_no_wind", every, start, ITERATIONS[2], wind=False)


def load_params():
    df = pd.read_csv(os.path.join(RESULTS, "m1_params.csv")).set_index("model")
    return {m: df.loc[m, Params.names()].values.astype(float) for m in df.index}


#-------- stage: smooth -----------------------------------------------------------------------
def _smooth_day(args):
    """One training day: for every fold (and for all stations), anchor on the stations
    outside the fold at each candidate l_q, score the held-out ones, and sample q."""
    d, vecs, fold, seed = args
    D = ds()
    y, fields = D.y[d], D.day(d)
    have = np.isfinite(y)
    inner = D.grid.interior().ravel()
    rng = np.random.default_rng(seed + d)
    errors, samples = [], []
    for k in range(6):
        sysd = DaySystem(D.grid, Params.unpack(vecs[k]), D.frac, fields)
        G = footprints(sysd, D.cells)
        theta0 = sysd.theta()
        use = have & (fold != k)
        test = have & (fold == k)
        weight = np.abs(G[use]).sum(0)
        cells = ml.sample_by_footprint(weight, N_SAMPLE, rng, inner)
        for l_q in LQ_GRID:
            sm = smoother(l_q)
            WG = np.column_stack([sm.solve(g) for g in G[use]])
            lam = np.linalg.solve(G[use] @ WG, y[use] - theta0[D.cells[use]])
            if test.any():
                pred = theta0[D.cells[test]] + (G[test] @ WG) @ lam
                errors += [(d, k, l_q, int(i), float(e)) for i, e in zip(np.flatnonzero(test), pred - y[test])]
            samples.append((k, l_q, cells, (WG[cells] @ lam).astype(np.float32)))
    return errors, [(d,) + s for s in samples]


def stage_smooth(D, fold, fit_days, params):
    vecs = [params[f"fold{k}"] for k in range(5)] + [params["all"]]
    out = run_parallel(_smooth_day, [(int(d), vecs, fold, 11) for d in fit_days], "smooth", 4)
    err = pd.DataFrame([e for o in out for e in o[0]], columns=["day", "fold", "l_q", "station", "error"])
    err.to_csv(os.path.join(RESULTS, "lq_search.csv"), index=False)
    score = err.groupby("l_q").error.apply(validation.rmse)
    best = float(score.idxmin())
    print("held-out RMSE by l_q (km):", score.round(3).to_dict(), "-> chosen", best)
    X, q, key = [], [], []
    for o in out:
        for d, k, l_q, cells, vals in o[1]:
            if l_q == best:
                X.append(ml.cell_features(D, d, cells)); q.append(vals); key.append(np.full(len(cells), k))
    np.savez_compressed(os.path.join(RESULTS, "q_training.npz"), X=np.vstack(X).astype(np.float32), q=np.concatenate(q),
                        fold=np.concatenate(key), l_q=best)
    return best


#-------- stage: learn ------------------------------------------------------------------------
def stage_learn():
    t = np.load(os.path.join(RESULTS, "q_training.npz"))
    report = []
    for k in range(6):
        m = t["fold"] == k
        model = ml.new_model(n_jobs=WORKERS).fit(t["X"][m], t["q"][m])
        name = "all" if k == ALL else f"fold{k}"
        model.save_model(os.path.join(RESULTS, f"xgb_q_{name}.json"))
        r2 = 1 - np.mean((model.predict(t["X"][m]) - t["q"][m]) ** 2) / np.var(t["q"][m])
        report.append(dict(model=name, rows=int(m.sum()), q_sd=float(t["q"][m].std()), train_r2=float(r2),
                           **dict(zip(ml.FEATURES, model.feature_importances_))))
    pd.DataFrame(report).to_csv(os.path.join(RESULTS, "xgb_q_report.csv"), index=False)


#-------- stage: evaluate ---------------------------------------------------------------------
def _evaluate_day(args):
    """One 2023 day: predictions at held-out stations from M1, M2 and M3 (per-fold
    parameters), plus M1 and M2 with and without wind using the all-station parameters."""
    d, vecs, fold, l_q = args
    D = ds()
    y, fields = D.y[d], D.day(d)
    have = np.isfinite(y)
    sm = smoother(l_q)
    rows = {int(i): dict(day=d, station=int(i)) for i in np.flatnonzero(have)}
    for k in range(5):
        test, use = have & (fold == k), have & (fold != k)
        if not test.any():
            continue
        sysd = DaySystem(D.grid, Params.unpack(vecs[f"fold{k}"]), D.frac, fields)
        an = Anchor(sysd, D.cells, sm)
        m2 = an.at_stations(y, use)
        q_hat = ml.predict_field(q_model(k), D, d, fields).ravel()
        theta0_hat = sysd.theta(q_hat)
        lam = np.linalg.solve(an.S[np.ix_(use, use)], y[use] - theta0_hat[D.cells[use]])
        m3 = theta0_hat[D.cells] + an.S[:, use] @ lam
        for i in np.flatnonzero(test):
            rows[int(i)].update(M1=an.theta0[D.cells[i]], M2=m2[i], M3=m3[i], M3_prior=theta0_hat[D.cells[i]])
    for label, vec, f in (("wind", vecs["all"], fields), ("calm", vecs["all_no_wind"], calm(fields))):
        sysd = DaySystem(D.grid, Params.unpack(vec), D.frac, f)
        an = Anchor(sysd, D.cells, sm)
        for k in range(5):
            test, use = have & (fold == k), have & (fold != k)
            if test.any():
                m2 = an.at_stations(y, use)
                for i in np.flatnonzero(test):
                    rows[int(i)].update({f"M1_{label}": an.theta0[D.cells[i]], f"M2_{label}": m2[i]})
    return list(rows.values())


def station_table(D, days, use=None):
    """One row per station-day with the covariates the station-level baselines use."""
    rows = []
    for d in days:
        f = D.day(d)
        X = ml.cell_features(D, d, D.cells, f)
        r = D.y[d] - f["theta_E"].ravel()[D.cells]
        ok = np.isfinite(r) if use is None else np.isfinite(r) & use
        for i in np.flatnonzero(ok):
            rows.append((d, i, r[i]) + tuple(X[i]))
    return pd.DataFrame(rows, columns=["day", "station", "residual"] + ml.FEATURES)


def stage_evaluate(D, fold, params, l_q):
    test_days = D.years(2023)[:4] if SMOKE else D.years(2023)
    out = run_parallel(_evaluate_day, [(int(d), params, fold, l_q) for d in test_days], "evaluate")
    pred = pd.DataFrame([r for o in out for r in o])

    # baselines that work on station tables
    train = station_table(D, D.years(2021, 2022))
    test = station_table(D, test_days)
    xy = np.column_stack([D.stations.x.values, D.stations.y.values])
    kcols = ["elevation", "coast", "built-up", "tree cover", "water", "lst_excess", "wind"]
    b1, b3 = np.full(len(test), np.nan), np.full(len(test), np.nan)
    variogram = []
    for k in range(5):
        tr = train[fold[train.station.values] != k]
        rk = validation.RegressionKriging().fit(tr[kcols].values, tr.residual.values, xy[tr.station.values], tr.day.values)
        variogram.append(dict(fold=k, nugget=rk.nugget, sill=rk.sill, range_km=rk.range_km))
        xgb = ml.new_model(n_jobs=WORKERS).fit(tr[ml.FEATURES].values, tr.residual.values)
        te_k = np.flatnonzero(fold[test.station.values] == k)
        b3[te_k] = xgb.predict(test.iloc[te_k][ml.FEATURES].values)
        for d, g in test.iloc[te_k].groupby("day"):
            anchors = test[(test.day == d) & (fold[test.station.values] != k)]
            b1[g.index.values] = rk.predict(anchors[kcols].values, anchors.residual.values, xy[anchors.station.values],
                                            g[kcols].values, xy[g.station.values])
    pd.DataFrame(variogram).to_csv(os.path.join(RESULTS, "b1_variogram.csv"), index=False)
    test["B1"], test["B3"] = b1, b3
    pred = pred.merge(test[["day", "station", "residual", "B1", "B3"]], on=["day", "station"])
    theta_E = np.array([D.field("theta_E", d).ravel()[D.cells[s]] for d, s in zip(pred.day, pred.station)])
    pred["obs"] = D.y[pred.day.values, pred.station.values]
    pred["B0"] = theta_E
    pred["B1"] += theta_E
    pred["B3"] += theta_E
    if D.has_eobs:
        pred["B2"] = [D.field("theta_eobs", d).ravel()[D.cells[s]] for d, s in zip(pred.day, pred.station)]
    st = D.stations
    pred["date"] = D.days[pred.day.values]
    pred["name"] = st.name.values[pred.station.values]
    pred["elevation"] = st.elevation.values[pred.station.values]
    pred["fold"] = fold[pred.station.values]
    pred["anchor_km"] = validation.nearest_anchor_km(st, fold)[pred.station.values]
    speed = {int(d): float(np.hypot(D.field("u", d), D.field("v", d)).mean()) for d in test_days}
    pred["wind_kmh"] = pred.day.map(speed).values
    cols = ["date", "day", "station", "name", "fold", "elevation", "anchor_km", "wind_kmh", "obs"] + \
        [c for c in ["B0", "B1", "B2", "B3", "M1", "M2", "M3", "M3_prior", "M1_wind", "M2_wind", "M1_calm", "M2_calm"] if c in pred]
    pred[cols].drop(columns="day").to_csv(os.path.join(RESULTS, "predictions_2023.csv"), index=False, float_format="%.4f")


#-------- stage: maps -------------------------------------------------------------------------
def _map_day(args):
    """The final product for one day, anchored on every station: fields for M1, M2, M3.
    Returns station-level checks, error against the hidden truth (placeholder only) and,
    for the days picked out for figures, the fields themselves."""
    d, vec, l_q, keep, foot_stations = args
    D = ds()
    y, fields = D.y[d], D.day(d)
    have = np.isfinite(y)
    sysd = DaySystem(D.grid, Params.unpack(vec), D.frac, fields)
    an = Anchor(sysd, D.cells, smoother(l_q))
    m1 = an.theta0.reshape(D.grid.shape)
    m2, q2 = an.solve(y, have)                                            # asserts |H theta - y| < 1e-6
    q_hat = ml.predict_field(q_model(ALL), D, d, fields)
    an.q_hat, an.theta0 = q_hat.ravel(), sysd.theta(q_hat)
    m3, q3 = an.solve(y, have)
    gap = float(np.abs(m3.ravel()[D.cells[have]] - y[have]).max())
    out = dict(day=d, gap=gap, q2=q2.astype(np.float32), q3=q3.astype(np.float32), q_hat=q_hat.astype(np.float32),
               t3=(m3 - GAMMA * D.elev).astype(np.float32), weight=np.abs(an.G[have]).sum(0).astype(np.float32))
    if "truth" in D._maps:
        truth = D.field("truth", d) + GAMMA * D.elev
        land = ~D.sea & D.grid.interior()
        cand = dict(B0=fields["theta_E"], M1=m1, M2=m2, M3=m3)
        if D.has_eobs:
            cand["B2"] = D.field("theta_eobs", d)
        out["truth_mse"] = {k: float(np.mean((v[land] - truth[land]) ** 2)) for k, v in cand.items()}
        far = land & (out["weight"].reshape(D.grid.shape) < np.quantile(out["weight"], 0.5))
        out["truth_mse_far"] = {k: float(np.mean((v[far] - truth[far]) ** 2)) for k, v in cand.items()}
    if keep:
        out["fields"] = dict(theta_E=fields["theta_E"], theta_s=fields["theta_s"], M1=m1, M2=m2, M3=m3, u=fields["u"], v=fields["v"])
        out["footprints"] = an.G[foot_stations].reshape(len(foot_stations), *D.grid.shape)
    return out


def stage_maps(D, params, l_q):
    july22 = np.flatnonzero((D.days.year == 2022) & (D.days.month == 7))
    hot = int(july22[np.nanargmax(np.nanmean(D.tmax[july22], axis=1))])               # hottest July 2022 day at the stations
    speed = np.array([np.hypot(D.field("u", d), D.field("v", d)).mean() for d in range(len(D.days))])
    windy, still = int(np.argmax(speed)), int(np.argmin(speed))
    low = D.stations.elevation.values < 300
    centre = np.hypot(D.stations.x - D.grid.x.mean(), D.stations.y - D.grid.y.mean()).values
    foot = [int(i) for i in np.argsort(np.where(low, centre, np.inf))[:6]]            # six lowland stations nearest the centre
    keep = {hot, windy, still}
    todo = sorted(keep) if SMOKE else range(len(D.days))
    out = run_parallel(_map_day, [(d, params["all"], l_q, d in keep, foot) for d in todo], "maps")
    by_day = {o["day"]: o for o in out}
    worst = max(o["gap"] for o in out)
    assert worst < 1e-6
    print(f"anchors reproduced on all {len(out)} days, worst gap {worst:.1e} K")
    save = dict(mean_q_m2=np.mean([o["q2"] for o in out], 0), mean_q_m3=np.mean([o["q3"] for o in out], 0),
                mean_q_hat=np.mean([o["q_hat"] for o in out], 0), mean_tmax=np.mean([o["t3"] for o in out], 0),
                mean_weight=np.mean([o["weight"] for o in out], 0).reshape(D.grid.shape),
                days=np.array([hot, windy, still]), speed=speed, footprint_stations=np.array(foot), worst_gap=worst)
    for name, d in (("hot", hot), ("windy", windy), ("still", still)):
        o = by_day[d]
        for k, v in o["fields"].items():
            save[f"{name}_{k}"] = v.astype(np.float32)
        save[f"{name}_footprints"] = o["footprints"].astype(np.float32)
        save[f"{name}_q3"] = o["q3"]
    np.savez_compressed(os.path.join(RESULTS, "maps.npz"), **save)
    if "truth_mse" in out[0]:
        rows = []
        for o in out:
            for scope, key in (("all land cells", "truth_mse"), ("land cells far from stations", "truth_mse_far")):
                rows.append(dict(date=D.days[o["day"]], scope=scope, **o[key]))
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "truth_check.csv"), index=False, float_format="%.5f")
    pd.DataFrame(dict(date=D.days[[o["day"] for o in out]], gap=[o["gap"] for o in out])).to_csv(os.path.join(RESULTS, "anchor_gap.csv"), index=False)


#-------- Step 9 summaries ----------------------------------------------------------------------
def stage_summary(D, params, fold):
    p = Params.unpack(params["all"])
    maps = np.load(os.path.join(RESULTS, "maps.npz"))
    weight = maps["mean_weight"]
    share = [(weight * D.frac[i]).sum() / weight.sum() for i in range(len(CLASSES))]
    per_fold = np.array([Params.unpack(params[f"fold{k}"]).tau_c for k in range(5)])
    tab = pd.DataFrame(dict(land_cover=CLASSES, tau_h=p.tau_c, tau_fold_min=per_fold.min(0), tau_fold_max=per_fold.max(0),
                            ratio_to_tau_a=p.tau_c / p.tau_a, footprint_share=share, area_share=D.frac.mean((1, 2))))
    tab.to_csv(os.path.join(RESULTS, "tau_by_land_cover.csv"), index=False)
    speed = maps["speed"]
    land = ~D.sea
    rows = []
    for label, s in (("median day", np.median(speed)), ("calmest day", speed.min()), ("windiest day", speed.max())):
        L = length_scales(p, D.frac, s)
        rows.append(dict(day=label, wind_kmh=s, tau_h=np.median(L["tau_h"][land]), advective_km=np.median(L["advective_km"][land]),
                         diffusive_km=np.median(L["diffusive_km"][land]),
                         numerical_diffusion_m2s=s * D.grid.dx / 2 / 3.6e-3))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "length_scales.csv"), index=False)


if __name__ == "__main__":
    force = FORCE
    os.makedirs(RESULTS, exist_ok=True)
    D = Dataset()
    fold, block = validation.spatial_folds(D.stations, D.grid)
    pd.DataFrame(dict(id=D.stations.id, name=D.stations.name, block=block, fold=fold,
                      anchor_km=validation.nearest_anchor_km(D.stations, fold))).to_csv(os.path.join(RESULTS, "folds.csv"), index=False)
    train_days = D.years(2021, 2022)
    fit_days = np.sort(np.random.default_rng(0).choice(train_days, N_FIT_DAYS, replace=False))
    need = lambda name: force or not os.path.exists(os.path.join(RESULTS, name))
    clock = time.time()

    def done(stage):
        print(f"[{stage}] finished at {(time.time() - clock) / 60:.1f} min", flush=True)

    if need("m1_params.csv") or len(pd.read_csv(os.path.join(RESULTS, "m1_params.csv"))) < 7:
        stage_fit(D, fold, fit_days); done("fit")
    params = load_params()
    if need("q_training.npz"):
        stage_smooth(D, fold, fit_days, params); done("smooth")
    l_q = float(np.load(os.path.join(RESULTS, "q_training.npz"))["l_q"])
    if need("xgb_q_all.json"):
        stage_learn(); done("learn")
    if need("predictions_2023.csv"):
        stage_evaluate(D, fold, params, l_q); done("evaluate")
    if need("maps.npz"):
        stage_maps(D, params, l_q); done("maps")
    stage_summary(D, params, fold)
    json.dump(dict(placeholder=bool(D.placeholder), l_q_km=l_q, n_stations=int(len(D.stations)), n_fit_days=N_FIT_DAYS,
                   n_days=int(len(D.days))), open(os.path.join(RESULTS, "run.json"), "w"), indent=1)
    done("all")
