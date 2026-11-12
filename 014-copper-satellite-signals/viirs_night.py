#!/usr/bin/env python3
#-------- viirs_night.py ------------------------------------------------------------
#-----------------------------------------------------------------------------
# Can night-time satellite data see the Chilean smelters through the July 2026 storm,
# when Sentinel-2 was blind at Caletones for 39 days?
#
# Two daily VIIRS products (Suomi NPP, 1:30 pm and 1:30 am passes), from Earth Engine:
#   heat     VNP14A1 thermal anomalies, 1 km: the maximum fire radiative power (MW) in a
#            3 x 3 km box round each smelter, day or night. A smelter's furnaces and
#            slag show as a persistent hot spot; a zero is "no anomaly", which also
#            happens under thick cloud, so cloud is read from the night-lights product
#   lights   VNP46A2 Black Marble night lights, 500 m: the mean radiance in a 2 x 2 km
#            box (nW/cm2/sr), kept only where the pixel is flagged cloud-free
# For Caletones and Chagres, 1 May - 31 October 2026. Compared with the Sentinel-2 record
# (event_chile_2026.py): last clear pass before the storm 11 July, next 19 August.
#
#   python viirs_night.py      (Earth Engine account; EE_PROJECT)
#
# Output: data/viirs_chile_2026.csv, results/viirs_night.json, figures/fig12_viirs_night.png

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SITES = {"Caletones (Codelco)": (-34.1065, -70.4577), "Chagres (Anglo American)": (-32.8032, -70.9529)}
STORM = ("2026-07-15", "2026-07-21")
S2_GAP = ("2026-07-11", "2026-08-19")
START, END = "2026-05-01", "2026-11-01"
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"


def fetch():
    path = os.path.join(HERE, "data", "viirs_chile_2026.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    import ee
    ee.Initialize(project=os.environ.get("EE_PROJECT", "zinc-interface-468219-a4"))
    rows = []
    for name, (lat, lon) in SITES.items():
        pt = ee.Geometry.Point([lon, lat])
        heat = ee.ImageCollection("NASA/VIIRS/002/VNP14A1").filterDate(START, END).select(["MaxFRP", "FireMask"])
        def h(im):
            v = im.reduceRegion(ee.Reducer.max(), pt.buffer(1500).bounds(), 1000)
            return ee.Feature(None, {"date": im.date().format("YYYY-MM-dd"), "frp": v.get("MaxFRP"), "mask": v.get("FireMask")})
        for f in ee.FeatureCollection(heat.map(h)).getInfo()["features"]:
            rows.append(dict(site=name, kind="heat", **f["properties"]))
        nl = ee.ImageCollection("NASA/VIIRS/002/VNP46A2").filterDate(START, END)
        def n(im):
            clear = im.select("QF_Cloud_Mask").bitwiseAnd(0b11000000).rightShift(6).eq(0)        # confident clear
            good = im.select("Mandatory_Quality_Flag").lte(1)
            v = im.select("DNB_BRDF_Corrected_NTL").updateMask(clear.And(good)).reduceRegion(ee.Reducer.mean(), pt.buffer(1000).bounds(), 500)
            c = clear.reduceRegion(ee.Reducer.mean(), pt.buffer(1000).bounds(), 500)
            return ee.Feature(None, {"date": im.date().format("YYYY-MM-dd"), "ntl": v.get("DNB_BRDF_Corrected_NTL"), "clear": c.get("QF_Cloud_Mask")})
        for f in ee.FeatureCollection(nl.map(n)).getInfo()["features"]:
            rows.append(dict(site=name, kind="lights", **f["properties"]))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df.date)
    df.to_csv(path, index=False)
    return df


def main():
    df = fetch()
    out = {}
    plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(SITES), 1, figsize=(13, 7.6), sharex=True, gridspec_kw=dict(hspace=0.25))
    for ax, name in zip(axes, SITES):
        h = df[(df.site == name) & (df.kind == "heat")].set_index("date").sort_index()
        l = df[(df.site == name) & (df.kind == "lights")].set_index("date").sort_index()
        h["frp"] = pd.to_numeric(h.frp, errors="coerce").fillna(0) / 10           # MaxFRP is stored in 0.1 MW
        l["ntl"] = pd.to_numeric(l.ntl, errors="coerce")
        gap = (h.index >= S2_GAP[0]) & (h.index <= S2_GAP[1])
        lg = (l.index >= S2_GAP[0]) & (l.index <= S2_GAP[1])
        pre = (l.index >= "2026-05-15") & (l.index < STORM[0])
        post = (l.index > "2026-08-19") & (l.index < "2026-10-15")
        storm_heat = h.frp[STORM[0]:STORM[1]]
        out_storm = {str(t.date()): float(v) for t, v in storm_heat[storm_heat > 0].items()}
        out[name] = dict(heat_in_storm=out_storm, heat_days_detected=int((h.frp > 0).sum()), heat_days=len(h),
                         heat_detected_in_s2_gap=int((h.frp[gap] > 0).sum()), days_in_s2_gap=int(gap.sum()),
                         clear_nights_in_s2_gap=int(l.ntl[lg].notna().sum()),
                         lights_median_before=float(l.ntl[pre].median()), lights_median_after=float(l.ntl[post].median()),
                         lights_median_in_gap=float(l.ntl[lg].median()) if l.ntl[lg].notna().any() else None,
                         clear_nights_before=int(l.ntl[pre].notna().sum()), clear_nights_after=int(l.ntl[post].notna().sum()))
        ax.axvspan(pd.Timestamp(S2_GAP[0]), pd.Timestamp(S2_GAP[1]), color=GREY, alpha=0.12, lw=0, label="no clear Sentinel-2 pass" if name.startswith("Cal") else None)
        ax.axvspan(pd.Timestamp(STORM[0]), pd.Timestamp(STORM[1]), color=RED, alpha=0.15, lw=0, label="storm")
        ax.plot(l.index, l.ntl, "o", color=NAVY, ms=3.5, label="night lights, cloud-free nights (VNP46A2)")
        ax.plot(l.ntl.rolling("15D", min_periods=3).median(), color=NAVY, lw=1.4, alpha=0.8)
        ax.set_ylabel("night radiance, nW/cm²/sr"); ax.set_ylim(0, float(l.ntl.quantile(0.95)) * 1.4)
        ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
        hd = h[h.frp > 0]
        ax2.vlines(hd.index, 0, hd.frp, color=RED, lw=1.2, alpha=0.8)
        ax2.plot(hd.index, hd.frp, "^", color=RED, ms=4, label="thermal anomaly, fire radiative power (VNP14A1)")
        ax2.set_ylabel("MW"); ax2.set_ylim(0, max(1, hd.frp.max() * 1.3) if len(hd) else 1)
        ax.set_title(f"{name.split(' (')[0]}, VIIRS night lights and thermal anomalies, 2026", loc="left")
        if name.startswith("Cal"):
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    json.dump(out, open(os.path.join(HERE, "results", "viirs_night.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    fig.savefig(os.path.join(HERE, "figures", "fig12_viirs_night.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
