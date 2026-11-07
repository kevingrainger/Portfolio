#!/usr/bin/env python3
#-------- ships_cfar.py -------------------------------------------------------------
#-----------------------------------------------------------------------------
# Counting ships at San Antonio from radar with a CFAR detector.
#
# A fixed brightness threshold failed in ships_radar.py: rough storm seas are bright
# everywhere and hide ships, calm seas are dark and let noise through. CFAR (constant
# false alarm rate) sets the threshold from the sea around each pixel instead. For every
# pixel the background is a ring around it (here from 250 m to 500 m out), with a guard
# zone in between so a ship doesn't count as its own background. In decibels, a pixel is
# a detection when it stands more than K standard deviations above its ring's mean; K = 5
# would give about one false alarm in three million pixels if the sea were Gaussian in dB.
# Neighbouring detections are joined into objects, which must be:
#   at least 20 pixels (2,000 m2) and at most 4,000 (a large bulk carrier is about 3,000),
#   bright at its peak, above 0 dB: a steel hull is a corner reflector, while the brightest
#   sea clutter in these passes peaks below -1.5 dB (CFAR alone let rough sea through),
#   not stretched into a line (length / width below 6: port reflectors leave streaks
#   along the radar's flight direction), and
#   on water at least 300 m from land, so harbour walls, cranes and breakers don't count.
#   The land mask is fixed, from all passes (land is where the ground is bright in most of
#   them): in a storm the sea can be as bright as land, and a mask made from that pass alone
#   hid the ships.
#
# Sentinel-1 RTC (VV) from Microsoft Planetary Computer, every pass January-August 2026,
# a 16 x 16 km box over the port and its anchorage. Passes are cached in data/s1_sanantonio/.
# The detector is checked against the by-eye counts on 12, 16, 22 and 24 July, and the
# series against daily port calls at San Antonio from IMF PortWatch.
#
#   python ships_cfar.py
#
# Output: results/ships_cfar.csv, figures/fig11_ships_cfar.png

import json
import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

import verify_smelters as vs

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "s1_sanantonio")
LAT, LON = -33.575, -71.66
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
STORM = ("2026-07-15", "2026-07-21")
BY_EYE = {"2026-07-12": 4, "2026-07-16": 2, "2026-07-22": 1, "2026-07-24": 7}
K, GUARD, OUTER = 5.0, 25, 50                       # pixels of 10 m
PEAK_DB = 0.0
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"


def passes():
    os.makedirs(CACHE, exist_ok=True)
    body = {"collections": ["sentinel-1-rtc"], "intersects": {"type": "Point", "coordinates": [LON, LAT]},
            "datetime": "2026-01-01T00:00:00Z/2026-08-31T23:59:59Z", "limit": 200}
    feats = sorted(vs.post_json(vs.STAC, body)["features"], key=lambda f: f["properties"]["datetime"])
    sas = None
    out = []
    for f in feats:
        day = f["properties"]["datetime"][:10]
        path = os.path.join(CACHE, f"{day}_{f['id'][-6:]}.npy")
        if not os.path.exists(path):
            sas = sas or json.loads(urllib.request.urlopen(TOKEN, timeout=60).read())["token"]
            vs.HALF = 8000
            a = vs.read_box(f["assets"]["vv"]["href"], sas, LAT, LON)[0].astype(np.float32)
            np.save(path, a)
        out.append((day, path))
    return out


def ring_stats(x):
    """Mean and standard deviation of x over the ring between GUARD and OUTER pixels."""
    def box_sum(v, n):
        return ndimage.uniform_filter(v, 2 * n + 1, mode="reflect") * (2 * n + 1) ** 2
    n_out, n_in = (2 * OUTER + 1) ** 2, (2 * GUARD + 1) ** 2
    s1 = box_sum(x, OUTER) - box_sum(x, GUARD)
    s2 = box_sum(x * x, OUTER) - box_sum(x * x, GUARD)
    m = s1 / (n_out - n_in)
    return m, np.sqrt(np.maximum(s2 / (n_out - n_in) - m * m, 1e-6))


def static_land(paths):
    """Land is where the ground is bright in most passes: a single pass can't decide, since a
    storm sea can be as bright as land."""
    votes = None
    for p in paths:
        a = np.load(p)
        db = 10 * np.log10(np.maximum(np.where(np.isfinite(a) & (a > 0), a, 1e-5), 1e-5))
        v = (ndimage.uniform_filter((db > -14).astype(float), 25) > 0.5).astype(np.int16)
        votes = v if votes is None else votes + v
    return ndimage.binary_dilation(votes > 0.6 * len(paths), iterations=30)


LAND = None


def detect(a):
    valid = np.isfinite(a) & (a > 0)
    db = 10 * np.log10(np.maximum(np.where(valid, a, 1e-5), 1e-5))
    land = LAND | ndimage.binary_dilation(~valid, iterations=30)
    filled = np.where(land, np.median(db[~land]) if (~land).any() else -25, db)       # land doesn't enter the sea's ring
    m, s = ring_stats(filled)
    hit = ((filled - m) / s > K) & ~land
    lab, n = ndimage.label(hit)
    ships = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        obj = lab[sl] == i
        size = obj.sum()
        if not 20 <= size <= 4000:
            continue
        if db[sl][obj].max() < PEAK_DB:                     # a steel hull is a strong reflector
            continue
        yy, xx = np.nonzero(obj)
        if size >= 8:
            ev = np.linalg.eigvalsh(np.cov(np.vstack([yy, xx])))
            if np.sqrt(ev[-1] / max(ev[0], 0.25)) > 6:
                continue
        ships.append((sl[0].start + yy.mean(), sl[1].start + xx.mean(), int(size)))
    return db, land, ships, float(np.median(db[~land])) if (~land).any() else np.nan


def main():
    global LAND
    rows, frames = [], {}
    ps = passes()
    LAND = static_land([p for _, p in ps])
    for day, path in ps:
        a = np.load(path)
        if (np.isfinite(a) & (a > 0)).mean() < 0.9:
            continue
        db, land, ships, sea = detect(a)
        rows.append(dict(date=day, ships=len(ships), sea_db=sea, storm=STORM[0] <= day <= STORM[1], by_eye=BY_EYE.get(day)))
        frames[day] = (db, land, ships)
    df = pd.DataFrame(rows).drop_duplicates("date")
    df.to_csv(os.path.join(HERE, "results", "ships_cfar.csv"), index=False)
    print(df.to_string(index=False))

    p = pd.read_csv(os.path.join(HERE, "data", "ports_daily.csv"), parse_dates=["date"])
    sa = p[p.portname.str.contains("San Antonio", case=False, na=False) & (p.country.str.contains("Chile", na=False) if "country" in p else True)] \
        if "portname" in p else None
    calls = sa.groupby("date").portcalls.sum().rolling(7, center=True).mean() if sa is not None and len(sa) else None
    d = df.set_index(pd.to_datetime(df.date))
    out = dict(passes=len(df), mean_ships=float(df.ships.mean()), storm_passes=df[df.storm].ships.tolist(),
               by_eye_check={k: dict(by_eye=v, cfar=int(df.set_index("date").ships.get(k, -1))) for k, v in BY_EYE.items()},
               corr_ships_sea_roughness=float(np.corrcoef(df.ships, df.sea_db)[0, 1]))
    if calls is not None:
        j = pd.concat([d.ships, calls.rename("calls")], axis=1, join="inner").dropna()
        out["corr_ships_portcalls_7d"] = float(j.corr().iloc[0, 1]) if len(j) > 5 else None
        out["n_matched"] = len(j)
    json.dump(out, open(os.path.join(HERE, "results", "ships_cfar.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 9.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.12, wspace=0.06)
    for k, day in enumerate(BY_EYE):
        ax = fig.add_subplot(gs[0, k])
        if day not in frames:
            ax.axis("off"); continue
        db, land, ships = frames[day]
        ax.imshow(np.clip(db, -24, -4), cmap="gray")
        ax.imshow(np.ma.masked_where(~land, land), cmap="Blues", alpha=0.18, vmin=0, vmax=1.5)
        for y, x, s in ships:
            ax.add_patch(plt.Circle((x, y), 28, fill=False, color=RED, lw=1.4))
        ax.set_xticks([]); ax.set_yticks([])
        storm = STORM[0] <= day <= STORM[1]
        ax.set_title(f"{day}{' (storm)' if storm else ''}\nCFAR {len(ships)}, by eye {BY_EYE[day]}", loc="left", fontsize=10,
                     color=RED if storm else "k")
        h, w = db.shape
        ax.plot([0.06 * w, 0.06 * w + 200], [0.94 * h] * 2, color="w", lw=3)
        ax.text(0.06 * w + 100, 0.905 * h, "2 km", color="w", ha="center", fontsize=9)
    ax = fig.add_subplot(gs[1, :])
    ax.axvspan(pd.Timestamp(STORM[0]), pd.Timestamp(STORM[1]), color=RED, alpha=0.12, lw=0, label="July storm")
    ax.plot(d.index, d.ships, "-o", color=NAVY, lw=1.3, ms=5, label="ships found by CFAR, each radar pass")
    e = d.dropna(subset=["by_eye"])
    ax.plot(e.index, e.by_eye, "D", color=RED, ms=7, mfc="none", mew=1.5, label="counted by eye")
    ax.set_ylabel("ships on the water")
    if calls is not None:
        ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
        ax2.plot(calls.loc["2026-01-01":"2026-08-31"].index, calls.loc["2026-01-01":"2026-08-31"], color=BLUE, lw=1.2, alpha=0.8, label="port calls a day, 7-day mean (PortWatch)")
        ax2.set_ylabel("port calls a day"); ax2.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1, -0.1))
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3)
    ax.set_title("Ships in the San Antonio box, Sentinel-1 radar, 2026", loc="left")
    fig.savefig(os.path.join(HERE, "figures", "fig11_ships_cfar.png"), dpi=160, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
