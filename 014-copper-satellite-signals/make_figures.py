#!/usr/bin/env python3
#-------- make_figures.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# The three figures of 014 (run fetch_ports.py, verify_smelters.py, analyse_ports.py first).
#
#   fig1_validation   Peru: ships at the copper ports against official copper exports
#   fig2_grasberg     Grasberg's port before and after the September 2025 mudslide
#   fig3_smelters     four smelters in shortwave infrared: furnaces glow

import json
import os
import sys
import urllib.request

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figstyle                                                  # noqa: E402
import verify_smelters as vs                                     # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
GREEN, WINE, GREY = "#117733", "#882255", "0.55"
GALLERY = ["Chuquicamata (Codelco)", "Balkhash (Kazakhmys)", "Hamburg (Aurubis)", "Guangxi Jinchuan (Fangchenggang)"]


def fig1_validation():
    v = pd.read_csv(os.path.join(RES, "validation_peru.csv"), index_col=0, parse_dates=True)
    s = json.load(open(os.path.join(RES, "validation_stats.json")))
    idx = 100 * v / v.loc["2019"].mean()
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(idx.index, idx.official_kt, color="0.15", lw=2, label="official copper exports (BCRP)")
    ax.plot(idx.index, idx.ships_copper_ports_kt, color=GREEN, label=f"ships, copper ports (r = {s['corr_level_copper']:.2f})")
    ax.plot(idx.index, idx.ships_placebo_ports_kt, color=GREY, lw=1, label=f"ships, placebo ports (r = {s['corr_level_placebo']:.2f})")
    ax.set_ylabel("monthly volume, 2019 average = 100")
    ax.set_title("Peru: dry-bulk ships at the copper ports track official copper exports", loc="left")
    ax.legend(loc="upper left", ncol=1)
    ax.set_ylim(0, None)
    figstyle.save(fig, os.path.join(FIG, "fig1_validation"))


def fig2_grasberg():
    a = pd.read_csv(os.path.join(RES, "amamapare.csv"), index_col=0, parse_dates=True)
    hg = pd.read_csv(os.path.join(HERE, "data", "hg_front.csv"), index_col=0, parse_dates=True).squeeze()
    hg = hg["2025-06-01":"2026-01-31"]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.2), sharex=True)
    roll = a.export_dry_bulk.rolling(14, min_periods=7).mean() / 1e3
    axes[0].plot(roll.index, roll.values, color=WINE)
    axes[0].set_ylabel("dry-bulk exports,\nthousand t/day (14-day mean)")
    axes[1].plot(hg.index, hg.values, color="0.2")
    axes[1].set_ylabel("COMEX copper, US$/lb")
    for ax in axes:
        ax.axvline(pd.Timestamp("2025-09-08"), color="0.6", lw=0.8, ls=":")
    axes[0].annotate("mudslide at the Grasberg Block Cave,\n8 September 2025", (pd.Timestamp("2025-09-08"), roll.max() * 0.85),
                     xytext=(8, 0), textcoords="offset points", fontsize=8.5)
    axes[0].set_title("Amamapare, Grasberg's port in Papua: the ships stopped", loc="left")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    figstyle.save(fig, os.path.join(FIG, "fig2_grasberg"))


def fig3_smelters():
    sm = pd.read_csv(os.path.join(HERE, "smelters.csv")).set_index("name")
    token = json.loads(urllib.request.urlopen(vs.TOKEN, timeout=60).read())["token"]
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 3.0))
    old = vs.HALF
    vs.HALF = 1500
    for ax, name in zip(axes, GALLERY):
        r = sm.loc[name]
        sc = vs.scenes(r.lat, r.lon)[0]
        off = 1000 if sc["properties"].get("s2:processing_baseline", "00.00") >= "04.00" else 0
        bands = [np.clip((vs.read_box(sc["assets"][b]["href"], token, r.lat, r.lon)[0] - off) / 1e4, 0, None)
                 for b in ("B12", "B11", "B8A")]
        rgb = np.dstack(bands)
        rgb = np.clip(rgb / np.percentile(rgb, 99.5), 0, 1) ** 0.8
        ax.imshow(rgb)
        hot, _, _ = vs.hot_mask(sc, token, r.lat, r.lon)
        ax.contour(hot, levels=[0.5], colors=["#DDCC77"], linewidths=0.8)
        ax.set_title(f"{name.split(' (')[0]}, {r.country}\n{sc['properties']['datetime'][:10]}", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    vs.HALF = old
    fig.suptitle("Smelters in shortwave infrared (Sentinel-2 bands 12, 11, 8A; 3 x 3 km): furnaces glow, hot pixels outlined",
                 x=0.01, y=1.0, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    figstyle.save(fig, os.path.join(FIG, "fig3_smelters"))


if __name__ == "__main__":
    figstyle.apply()
    fig1_validation()
    fig2_grasberg()
    fig3_smelters()
