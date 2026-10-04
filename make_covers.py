#!/usr/bin/env python3
#-------- make_covers.py --------------------------------------------------------------
#-----------------------------------------------------------------------------
# The cover image for each project - the one graphic on the website card and at the
# top of each README. Kept separate from the notebooks' figures so re-running a
# notebook never overwrites a cover. All covers are 2:1, ~2600 px wide.
#
#   python make_covers.py              all of them
#   python make_covers.py coral        one (coral | corrlib | seismic | options | power)

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import figstyle

figstyle.apply()
W, H, DPI = 13.0, 6.5, 200                              # 2600 x 1300 px


def out(folder, name="cover"):
    return os.path.join(ROOT, folder, "figures", name)


#-------- 001 coral: the reef through a heatwave and back (animation) -----------------
def coral():
    sys.path.insert(0, os.path.join(ROOT, "001-coral-reef-tipping-points"))
    import coral_model as cm
    from matplotlib.animation import FuncAnimation, PillowWriter

    reef = cm.Reef(seed=4, neighbourhood="nn1", grid=110)
    base = reef.thresh - 1.0
    schedule = np.concatenate([np.linspace(0, 4.4, 330), np.full(120, 4.4), np.linspace(4.4, 0, 330), np.zeros(150)])
    frames, cover, temp = [], [], []
    for i, a in enumerate(schedule):
        reef.step(base + a)
        cover.append(reef.cover()); temp.append(a)
        if i % 6 == 0:
            frames.append((reef.image(), i))

    fig = plt.figure(figsize=(12, 6), dpi=110)
    fig.patch.set_facecolor("#0b1d33")
    ax_reef = fig.add_axes([0.02, 0.06, 0.46, 0.86])
    ax_path = fig.add_axes([0.56, 0.16, 0.40, 0.70])
    ax_reef.axis("off")
    im = ax_reef.imshow(frames[0][0], interpolation="nearest")
    title = fig.text(0.25, 0.94, "", ha="center", color="white", fontsize=15)
    ax_path.set_facecolor("#0b1d33")
    for s in ax_path.spines.values():
        s.set_color("#8899aa")
    ax_path.tick_params(colors="#c8d2dc")
    ax_path.set_xlim(-0.1, 4.6); ax_path.set_ylim(0, 1)
    ax_path.set_xlabel("water temperature above baseline (°C)", color="#c8d2dc")
    ax_path.set_ylabel("healthy coral cover", color="#c8d2dc")
    ax_path.grid(color="#2a3d55")
    ax_path.set_title("the reef's path: up one way, down another", color="white", fontsize=12)
    warm, = ax_path.plot([], [], color="#ff7a59", lw=2.5, label="warming")
    cool, = ax_path.plot([], [], color="#5ec8ff", lw=2.5, label="cooling back")
    dot, = ax_path.plot([], [], "o", color="white", ms=8)
    ax_path.legend(facecolor="#0b1d33", edgecolor="#2a3d55", labelcolor="white", loc="lower left")
    turn = 450

    def draw(k):
        img, i = frames[k]
        im.set_data(img)
        phase = "heating" if i < 330 else "heatwave" if i < 450 else "cooling" if i < 780 else "recovery?"
        title.set_text(f"day {i:3d}   +{temp[i]:.1f} °C   ({phase})")
        warm.set_data(temp[:min(i, turn) + 1], cover[:min(i, turn) + 1])
        if i > turn:
            cool.set_data(temp[turn:i + 1], cover[turn:i + 1])
        dot.set_data([temp[i]], [cover[i]])
        return im, warm, cool, dot, title

    anim = FuncAnimation(fig, draw, frames=len(frames))
    path = out("001-coral-reef-tipping-points") + ".gif"
    anim.save(path, writer=PillowWriter(fps=12))
    draw(len(frames) - 1)
    fig.savefig(out("001-coral-reef-tipping-points") + ".png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("coral cover:", path, os.path.getsize(path) // 1024, "KB")


#-------- 003 corrlib: a clustered correlation heatmap ---------------------------------
def corrlib():
    sys.path.insert(0, os.path.join(ROOT, "003-corrlib-correlation-toolbox"))
    import yfinance as yf
    from corrlib import Correlator
    from corrlib.plotting import _linkage
    from scipy.cluster.hierarchy import dendrogram, leaves_list

    sectors = {"Energy": ["XOM", "CVX", "COP", "SLB", "OXY"], "Banks": ["JPM", "BAC", "GS", "MS", "C"],
               "Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"], "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "UNH"],
               "Staples": ["PG", "KO", "PEP", "WMT", "COST"], "Industrials": ["CAT", "BA", "GE", "HON", "UNP"],
               "Utilities": ["NEE", "DUK", "SO", "D", "AEP"], "Property": ["AMT", "PLD", "SPG", "O"]}
    tickers = [t for v in sectors.values() for t in v]
    cache = os.path.join(ROOT, "003-corrlib-correlation-toolbox", "data_stock_closes.csv")
    if os.path.exists(cache):
        raw = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        raw = yf.download(tickers, start="2015-01-01", end="2025-01-01", progress=False, auto_adjust=False)["Close"]
        raw.to_csv(cache)
    raw = raw[[t for t in tickers if t in raw]].dropna(axis=1)
    returns = np.log(raw).diff().dropna()
    market = returns.mean(axis=1).values
    c = Correlator(measure="spearman", remove=[market], clean="nonlinear_shrinkage")
    C = c.fit(returns)
    names = list(returns.columns)
    Z = _linkage(C)
    order = leaves_list(Z)
    colour_of = {t: s for s, ts in sectors.items() for t in ts}
    palette = dict(zip(sectors, figstyle.PALETTE))

    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax_d = fig.add_axes([0.06, 0.80, 0.50, 0.16])
    ax_s = fig.add_axes([0.06, 0.775, 0.50, 0.02])
    ax_h = fig.add_axes([0.06, 0.07, 0.50, 0.70])
    ax_c = fig.add_axes([0.575, 0.07, 0.012, 0.70])
    dendrogram(Z, ax=ax_d, color_threshold=0, above_threshold_color="0.35", no_labels=True)
    ax_d.axis("off")
    ax_s.imshow([[list(palette).index(colour_of[names[i]]) for i in order]], aspect="auto",
                cmap=matplotlib.colors.ListedColormap(list(palette.values())), vmin=0, vmax=len(palette) - 1)
    ax_s.axis("off")
    im = ax_h.imshow(C[np.ix_(order, order)], cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax_h.set_xticks(range(len(order))); ax_h.set_yticks(range(len(order)))
    ax_h.set_xticklabels([names[i] for i in order], rotation=90, fontsize=6.5)
    ax_h.set_yticklabels([names[i] for i in order], fontsize=6.5)
    ax_h.grid(False)
    fig.colorbar(im, cax=ax_c, label="correlation, market removed")

    ax_t = fig.add_axes([0.66, 0.07, 0.32, 0.86]); ax_t.axis("off")
    ax_t.text(0, 0.97, "Forty stocks, ten years.", fontsize=19, weight="bold", va="top")
    ax_t.text(0, 0.86, "Take out the market -\nthe one thing every\nstock shares - clean\nthe noise with random-\nmatrix theory, and\nthe sectors appear.",
              fontsize=13.5, va="top", linespacing=1.5)
    for k, (s, col) in enumerate(palette.items()):
        ax_t.add_patch(plt.Rectangle((0, 0.38 - k * 0.045), 0.05, 0.03, color=col, transform=ax_t.transAxes))
        ax_t.text(0.08, 0.395 - k * 0.045, s, fontsize=11, va="center")
    figstyle.save(fig, out("003-corrlib-correlation-toolbox"))
    plt.close(fig)
    print("corrlib cover done")


#-------- 004 seismic: before and after ----------------------------------------------------
def seismic():
    sys.path.insert(0, os.path.join(ROOT, "004-seismic-denoiser"))
    sys.path.insert(0, os.path.join(ROOT, "003-corrlib-correlation-toolbox"))
    import seismic as s
    from corrlib import array_stacking as stack

    X, clean, delays, x_km, exposure = s.synthetic_array(n_sensors=24, n_samples=4200, arrival=3600, road_strength=4.0, seed=3)
    A = s.align_known(X, delays)
    true = s.align_known(clean, delays)[0]
    avg, _ = stack.simple(A)
    opt, w = stack.mvdr(A, clean="nonlinear_shrinkage", noise_window=(1200, 3400))
    t = np.arange(X.shape[1]) / s.FS
    win = (t > 33.5) & (t < 38.5)

    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [2, 12], btype="band", fs=s.FS, output="sos")
    bp = lambda x: sosfiltfilt(sos, x)
    single = A[np.argsort(A[:, 1200:3400].std(1))[12]]          # a typical sensor

    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax_raw = fig.add_axes([0.04, 0.10, 0.43, 0.80])
    cmap = plt.get_cmap("magma")
    win_raw = (t > 33.5) & (t < 39.6)                          # wide enough for the last arrival
    for k in range(24):
        ax_raw.plot(t[win_raw], X[k, win_raw] / 5 + k, color=cmap(0.15 + 0.7 * exposure[k] / exposure.max()), lw=0.7)
        ax_raw.plot((3600 + delays[k]) / s.FS, k, marker="v", color=figstyle.PALETTE[2], ms=5)
    ax_raw.set_yticks([]); ax_raw.set_xlabel("time (s)")
    ax_raw.set_title("Before: 24 sensors. The wave (green marks) is in there somewhere", loc="left", fontsize=11.5)
    ax_raw.grid(False)

    panels = [("one sensor", single, "0.55"), ("plain average", avg, figstyle.PALETTE[1]),
              ("optimal stack, cleaned covariance", opt, figstyle.PALETTE[0])]
    lim = 1.15 * np.abs(bp(single)[win]).max()
    for k, (label, trace, col) in enumerate(panels):
        ax = fig.add_axes([0.53, 0.66 - k * 0.28, 0.44, 0.22])
        ax.plot(t[win], bp(true)[win], color="k", lw=1.3, ls="--", label="true wave")
        ax.plot(t[win], bp(trace)[win], color=col, lw=1.2, label=label)
        ax.set_ylim(-lim, lim); ax.set_yticks([])
        ax.set_title(f"After: {label}", loc="left", fontsize=11)
        if k == 0:
            ax.legend(loc="upper right", fontsize=8.5)
        if k < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (s)")
    fig.text(0.53, 0.955, "Same 2-12 Hz filter on every panel; the stack does the rest", fontsize=10, color="0.35")
    figstyle.save(fig, out("004-seismic-denoiser"))
    plt.close(fig)
    print("seismic cover done")


#-------- 005 options: the Monte Carlo ------------------------------------------------------
def options():
    softs = pd.read_csv(os.path.join(ROOT, "005-option-pricer", "data", "softs_futures.csv"), index_col=0, parse_dates=True)
    r = np.log(softs["coffee"]).diff().dropna()
    z = r / r.groupby(r.index.year).transform("std")
    seasonal = z.groupby(z.index.month).std(); seasonal /= seasonal.mean()
    dates = pd.bdate_range("2025-04-01", "2025-09-30")
    level = r[:"2025-03-31"].iloc[-252:].std() * np.sqrt(252)
    sig = level * seasonal.loc[dates[:-1].month].values
    F0 = float(softs["coffee"][:"2025-03-31"].iloc[-1]); dt = 1 / 252
    rng = np.random.default_rng(3)
    n_paths = 400
    Zs = rng.standard_normal((n_paths, len(sig)))
    paths = F0 * np.exp(np.cumsum(-0.5 * sig ** 2 * dt + sig * np.sqrt(dt) * Zs, axis=1))
    paths = np.hstack([np.full((n_paths, 1), F0), paths])
    fix = dates >= "2025-07-01"
    averages = paths[:, fix].mean(axis=1)
    payoff = np.maximum(averages - F0, 0)

    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0.06, 0.12, 0.62, 0.78])
    norm = matplotlib.colors.Normalize(np.percentile(averages, 5), np.percentile(averages, 95))
    cmap = plt.get_cmap("viridis")
    for p, a in zip(paths[:250], averages[:250]):
        ax.plot(dates, p, color=cmap(norm(a)), lw=0.6, alpha=0.55)
    ax.axvspan(pd.Timestamp("2025-07-01"), dates[-1], color="0.85", zorder=0)
    ax.text(pd.Timestamp("2025-07-04"), ax.get_ylim()[1] * 0.97, "averaging window:\nthe option pays on the\naverage price here",
            va="top", fontsize=10)
    ax.axhline(F0, color="k", ls="--", lw=1)
    ax.text(dates[2], F0 * 1.01, "strike", fontsize=9)
    ax.set_ylabel("coffee futures (US cents/lb)")
    ax.set_title("Monte Carlo: 250 of 400 simulated coffee price paths, coloured by their average", loc="left")

    axh = fig.add_axes([0.72, 0.12, 0.25, 0.78])
    axh.hist(averages, bins=40, orientation="horizontal", color=figstyle.PALETTE[0], alpha=0.8)
    axh.axhline(F0, color="k", ls="--", lw=1)
    axh.set_ylim(ax.get_ylim())
    axh.set_xlabel("paths"); axh.set_yticklabels([])
    axh.set_title(f"average price at expiry\n{np.mean(payoff > 0):.0%} of paths pay out", fontsize=11)
    fig.autofmt_xdate()
    figstyle.save(fig, out("005-option-pricer"))
    plt.close(fig)
    print("options cover done")


#-------- 006 Italian power: the price, projected forward with the weather -------------------
def power():
    sys.path.insert(0, os.path.join(ROOT, "006-italian-power-volatility"))
    import power as pw
    from sklearn.linear_model import LinearRegression

    df, extreme = pw.placeholder()
    feats = lambda d: np.column_stack([d.temp, (d.temp - 18) ** 2, d.log_gas, d.load, d.renewables, d.weekend])
    y = np.log(df.price.values)
    model = LinearRegression().fit(feats(df), y)
    resid = y - model.predict(feats(df))
    phi = np.corrcoef(resid[1:], resid[:-1])[0, 1]
    innov_sd = np.std(resid[1:] - phi * resid[:-1])

    horizon = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=270, freq="D")
    doy = horizon.dayofyear.values
    hist = df.copy(); hist["doy"] = hist.index.dayofyear
    clim_mean = hist.groupby("doy").temp.mean().reindex(range(1, 367)).interpolate().values
    rng = np.random.default_rng(1)
    n = 300
    sims = np.empty((n, len(horizon)))
    for k in range(n):
        anomaly = np.zeros(len(horizon)); e = rng.normal(0, 2.2, len(horizon))
        for t in range(1, len(horizon)):
            anomaly[t] = 0.8 * anomaly[t - 1] + e[t]
        heat = np.zeros(len(horizon))
        summer = np.flatnonzero(np.isin(horizon.month, [6, 7, 8]))
        for _ in range(rng.integers(1, 4)):
            c = rng.choice(summer); heat[max(0, c - 4):c + 5] += rng.uniform(4, 8)
        temp = clim_mean[doy - 1] + anomaly + heat
        future = pd.DataFrame(dict(temp=temp, log_gas=np.full(len(horizon), df.log_gas.iloc[-30:].mean()),
                                   load=1 + 0.08 * np.cos(2 * np.pi * (doy - 15) / 365.25 * 2) + 0.004 * np.maximum(temp - 24, 0) ** 2,
                                   renewables=0.25 + 0.12 * np.sin(2 * np.pi * (doy - 80) / 365.25),
                                   weekend=(horizon.dayofweek >= 5).astype(int)), index=horizon)
        rr = np.zeros(len(horizon)); rr_prev = resid[-1]
        shocks = rng.normal(0, innov_sd, len(horizon))
        for t in range(len(horizon)):
            rr_prev = phi * rr_prev + shocks[t]; rr[t] = rr_prev
        sims[k] = np.exp(model.predict(feats(future)) + rr)

    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0.06, 0.12, 0.90, 0.78])
    recent = df.loc["2022-07-01":]
    ax.plot(recent.index, recent.price, color="0.25", lw=0.8, label="day-ahead price")
    for lo, hi, a in [(5, 95, 0.15), (25, 75, 0.3)]:
        ax.fill_between(horizon, np.percentile(sims, lo, 0), np.percentile(sims, hi, 0), color=figstyle.PALETTE[1], alpha=a, lw=0,
                        label=f"projection, {lo}-{hi}% of weather scenarios")
    ax.plot(horizon, np.median(sims, 0), color=figstyle.PALETTE[1], lw=2, label="projection, median")
    ax.plot(horizon, sims[1], color=figstyle.PALETTE[1], lw=0.5, alpha=0.7, label="one weather scenario")
    top = np.percentile(recent.price, 99.5) * 1.15
    ax.set_ylim(0, top)
    ax.axvline(horizon[0], color="k", lw=0.8, ls=":")
    ax.text(horizon[4], top * 0.04, "forecast starts", fontsize=9)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Italian power prices, projected nine months ahead from 300 weather scenarios (gas held flat)", loc="left")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    figstyle.watermark(fig)
    figstyle.save(fig, out("006-italian-power-volatility"))
    plt.close(fig)
    print("power cover done")


JOBS = dict(coral=coral, corrlib=corrlib, seismic=seismic, options=options, power=power)

if __name__ == "__main__":
    for name in (sys.argv[1:] or JOBS):
        JOBS[name]()
