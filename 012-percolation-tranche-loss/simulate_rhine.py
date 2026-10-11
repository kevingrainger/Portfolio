#!/usr/bin/env python3
#-------- simulate_rhine.py ------------------------------------------------------------
#-----------------------------------------------------------------------------
# Case study A: the Rhine low water.
#   calib    which pairs (direct severity, transmissibility) reproduce a 1% fall in
#            German industrial output in a low-water month, and how far each sits
#            from the percolation threshold
#   futures  the same past, different futures: output loss against the Kaub level for
#            economies that all match 2018
#   maps     static map of disruption probability, and an interactive folium map
#
#   python simulate_rhine.py [calib] [futures] [maps]

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as MplPolygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import figstyle
import rhine as R

figstyle.apply()
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
COL = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#555555"]
INK = "#222222"
H_MONTH = 55.0        # representative Kaub level of a month spent below 78 cm
RHINE_NAMES = {"Rhine", "Rhein", "Rhin", "Waal", "Nederrijn", "Lek"}

ECON = R.Economy(N=4000, seed=1)
TC = ECON.threshold()


def calib():
    sev = np.round(np.linspace(0.005, 0.065, 13), 4)
    rows = []
    for s in sev:
        T = R.calibrate_T(ECON, s, kaub_cm=H_MONTH)
        direct = R.mean_loss(ECON, H_MONTH, 0.0, s, 80, np.random.default_rng(1))
        rows.append(dict(severity=float(s), T_star=T, ratio=T / TC, direct=direct, amplification=0.01 / direct))
        print(rows[-1])
    json.dump(dict(Tc=TC, kaub_month_cm=H_MONTH, rows=rows), open(os.path.join(RES, "rhine_calibration.json"), "w"), indent=2)


def load_calib():
    return json.load(open(os.path.join(RES, "rhine_calibration.json")))


def pick_three(cal):
    rows = [r for r in cal["rows"] if np.isfinite(r["T_star"])]
    picks = []
    for target in (0.3, 0.55, 0.8):
        picks.append(min(rows, key=lambda r: abs(r["ratio"] - target)))
    return picks


def futures(runs=150):
    cal = load_calib()
    picks = pick_three(cal)
    h = np.arange(20, 151, 5.0)
    level_curves = [[R.mean_loss(ECON, x, p["T_star"], p["severity"], runs, np.random.default_rng(2)) for x in h]
                    for p in picks]
    x = np.linspace(0, 1.6, 33)
    T_curves = [[R.mean_loss(ECON, 25.0, xx * TC, p["severity"], runs, np.random.default_rng(3)) for xx in x]
                for p in picks]
    json.dump(dict(h=h.tolist(), picks=picks, level_curves=level_curves, T_over_Tc=x.tolist(), T_curves=T_curves),
              open(os.path.join(RES, "rhine_futures.json"), "w"), indent=2)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw=dict(wspace=0.3, width_ratios=[1.35, 1]))
    for k, (p, L) in enumerate(zip(picks, T_curves)):
        a1.plot(x, 100 * np.array(L), color=COL[k], ls=["-", "--", "-."][k],
                label=f"direct share {p['direct'] / 0.01:.0%} of the 2018 loss")
        a1.plot([p["ratio"]], [100 * np.interp(p["ratio"], x, L)], "o", color=COL[k], ms=6, mfc="white", mew=1.5)
    a1.axvline(1, color=INK, lw=0.9, ls=(0, (2, 2)))
    sec = a1.secondary_xaxis("top"); sec.set_xticks([1]); sec.set_xticklabels([r"$T_c$"]); sec.tick_params(length=0)
    a1.set_xlabel(r"transmissibility  $T/T_c$  (share of inputs that cannot be replaced)")
    a1.set_ylabel("German industrial output lost (%)")
    a1.set_xlim(0, 1.6); a1.set_ylim(0, None)
    a1.set_title("(a) At the record low of 25 cm; circles mark 2018", loc="left")
    a1.legend(loc="upper left", fontsize=8.5)

    rows = [r for r in cal["rows"] if np.isfinite(r["T_star"])]
    a2.plot([r["direct"] / 0.01 * 100 for r in rows], [r["ratio"] for r in rows], "o-", ms=4.5, mfc="white",
            mew=1.3, color=COL[0])
    a2.axhline(1, color=INK, lw=0.9, ls=(0, (2, 2)))
    a2.set_ylim(0, 1.1); a2.set_xlim(0, 100)
    a2.set_xlabel("share of the 1% loss that is direct (%)"); a2.set_ylabel(r"distance to threshold  $T^*/T_c$")
    a2.set_title("(b) One number fixes a line, not a point", loc="left")
    figstyle.save(fig, os.path.join(FIG, "fig5_rhine_futures"))
    plt.close(fig)


def basemap(ax):
    gj = json.load(open(os.path.join(RES, "rhine_basemap.geojson")))
    polys, rivers, rhine = [], [], []
    for f in gj["features"]:
        g, kind = f["geometry"], f["properties"]["kind"]
        if kind == "country":
            parts = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            for poly in parts:
                polys.append(MplPolygon(np.array(poly[0]), closed=True))
        else:
            lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
            (rhine if f["properties"]["name"] in RHINE_NAMES else rivers).extend(np.array(l) for l in lines)
    ax.add_collection(PatchCollection(polys, facecolor="#f4f3ef", edgecolor="#b9b6ad", linewidth=0.6))
    ax.add_collection(LineCollection(rivers, colors="#9ec3dc", linewidths=0.6))
    ax.add_collection(LineCollection(rhine, colors="#3b7bb5", linewidths=2.0))
    ax.set_xlim(3.3, 14.6); ax.set_ylim(47.2, 54.2)
    ax.set_aspect(1 / np.cos(np.radians(50.5)))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def disruption_probability(kaub, T, sev, runs=200, seed=4):
    rng = np.random.default_rng(seed)
    acc = np.zeros(ECON.N)
    for _ in range(runs):
        acc += ECON.disrupted(kaub, T, rng, sev)
    return acc / runs


def maps():
    cal = load_calib()
    picks = pick_three(cal)
    near = picks[-1]
    tip = dict(near, T_star=1.25 * TC)
    scen = [("(a) 25 cm, network as calibrated to 2018", 25.0, near),
            (r"(b) 25 cm, road and rail also strained ($T = 1.25\,T_c$)", 25.0, tip)]
    probs = [disruption_probability(h, p["T_star"], p["severity"]) for _, h, p in scen]
    import seaborn as sns
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.2), gridspec_kw=dict(wspace=0.04))
    order = np.argsort(probs[1])
    for ax, (title, h, p), pr in zip(axes, scen, probs):
        basemap(ax)
        sc = ax.scatter(ECON.lon[order], ECON.lat[order], c=pr[order], s=7 + 30 * np.sqrt(ECON.output[order] / ECON.output.max()),
                        cmap=cmap, vmin=0, vmax=0.4, edgecolors="white", linewidths=0.2, zorder=3)
        ax.plot(*R.KAUB, marker="v", color=INK, ms=8, zorder=4)
        ax.annotate("Kaub", R.KAUB, xytext=(-34, -4), textcoords="offset points", fontsize=9, color=INK)
        lost = np.sum(pr * ECON.output * ECON.germany) / np.sum(ECON.output * ECON.germany)
        ax.set_title(f"{title}\n{lost:.1%} of German industrial output lost", loc="left", fontsize=10)
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("probability a firm loses the month's output")
    figstyle.save(fig, os.path.join(FIG, "fig4_rhine_map"))
    plt.close(fig)

    # interactive version for the website
    import folium
    m = folium.Map(location=[50.6, 8.2], zoom_start=6, tiles=None, control_scale=True, min_zoom=5, max_zoom=11)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles &copy; Esri, HERE, Garmin, OpenStreetMap contributors", name="light grey base", max_zoom=11).add_to(m)
    gj = json.load(open(os.path.join(RES, "rhine_basemap.geojson")))
    folium.GeoJson({"type": "FeatureCollection", "features": [f for f in gj["features"] if f["properties"]["name"] in RHINE_NAMES]},
                   style_function=lambda f: dict(color="#3b7bb5", weight=3)).add_to(m)
    for (title, h, p), pr in zip(scen, probs):
        layer = folium.FeatureGroup(name=title[4:].replace("$T = 1.25\\,T_c$", "T = 1.25 Tc"), show=(p is tip))
        for i in np.argsort(pr):
            c = matplotlib.colors.to_hex(cmap(min(pr[i] / 0.4, 1.0)))
            folium.CircleMarker([ECON.lat[i], ECON.lon[i]], radius=2 + 6 * np.sqrt(ECON.output[i] / ECON.output.max()),
                                color=c, fill=True, fill_opacity=0.85, weight=0,
                                tooltip=f"{R.SECTORS[ECON.sector[i]]}, near {R.HUBS[ECON.hub[i]][0]}: {pr[i]:.0%}").add_to(layer)
        layer.add_to(m)
    folium.Marker([R.KAUB[1], R.KAUB[0]], tooltip="Kaub gauge").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    os.makedirs(os.path.join(HERE, "maps"), exist_ok=True)
    m.save(os.path.join(HERE, "maps", "rhine_cascade.html"))


if __name__ == "__main__":
    for w in sys.argv[1:] or ["calib", "futures", "maps"]:
        globals()[w]()
        print(w, "done")
