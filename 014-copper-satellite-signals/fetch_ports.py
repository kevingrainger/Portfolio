#!/usr/bin/env python3
#-------- fetch_ports.py -----------------------------------------------------------
#-----------------------------------------------------------------------------
# Daily ship traffic at the ports copper leaves from and arrives at, from IMF PortWatch.
#
# PortWatch turns AIS, the position beacon every large ship carries, into daily port
# calls and estimated cargo tonnes by ship type (Cerdeiro et al. 2020; Arslanalp,
# Koepke & Verschuur 2021). It is free, updated weekly, and needs no account. It does
# not say what a ship carries, so dry-bulk tonnes at a copper port is a proxy for
# copper concentrate, and the validation against official export statistics decides
# whether the proxy is any good.
#
#   python fetch_ports.py
#
# Output: data/ports_daily.csv, one row per port and day, 2019-.

import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BASE = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/Daily_Ports_Data/FeatureServer/0/query"

# Copper export ports, by PortWatch id (Angamos is inside PortWatch's Mejillones). Which
# mines ship through which port is to be checked against company reports before any
# port is credited to a mine in the write-up; the ones marked (check) are least certain.
EXPORT_PORTS = {
    "port56":   "Antofagasta",                 # Chile: Antofagasta region mines
    "port728":  "Mejillones",                  # Chile: Angamos terminal, Codelco north
    "port1031": "Puerto Caldera",              # Chile: Punta Padrones, Candelaria
    "port1034": "Puerto Chanaral",             # Chile: Barquito, Codelco El Salvador
    "port90":   "Bahia Quintero (Ventanas)",   # Chile: Ventanas smelter
    "port1058": "Puerto San Antonio",          # Chile: central-zone cathode (check)
    "port2107": "Guayacan",                    # Chile: Coquimbo region (check)
    "port1064": "Punta Lobitos",               # Peru: Huarmey, Antamina's concentrate port
    "port88":   "Matarani",                    # Peru: southern mines, Las Bambas
    "port1047": "Ilo",                         # Peru: Southern Copper's Ilo smelter
    "port1045": "Callao",                      # Peru: central concentrate (check)
}
# Grasberg's port in Papua, Indonesia: the September 2025 mudslide case
CASE_PORTS = {"port41": "Amamapare"}
# Placebos: Chilean and Peruvian ports that ship forestry, salmon, fishmeal and
# phosphate, not copper. A signal that "works" as well from these is noise.
PLACEBO_PORTS = {
    "port92":   "San Vicente",                 # Chile: forestry and pulp
    "port263":  "Coronel",                     # Chile: forestry
    "port652":  "Lirquen",                     # Chile: forestry
    "port1053": "Puerto Montt",                # Chile: salmon
    "port873":  "Paita",                       # Peru: fish and fruit
    "port1026": "Bayovar",                     # Peru: phosphate
    "port1040": "Chimbote",                    # Peru: fishmeal
}
ROLES = {**{p: "copper" for p in EXPORT_PORTS}, **{p: "case" for p in CASE_PORTS},
         **{p: "placebo" for p in PLACEBO_PORTS}}

FIELDS = ["date", "portid", "portname", "ISO3", "portcalls", "portcalls_dry_bulk",
          "portcalls_general_cargo", "export", "export_dry_bulk", "export_general_cargo",
          "import", "import_dry_bulk"]


def query(where, offset):
    q = dict(where=where, outFields=",".join(FIELDS), returnGeometry="false", f="json",
             orderByFields="date", resultOffset=offset, resultRecordCount=2000)
    for k in range(4):
        try:
            with urllib.request.urlopen(BASE + "?" + urllib.parse.urlencode(q), timeout=120) as r:
                d = json.load(r)
            if "error" in d:
                raise RuntimeError(d["error"])
            return [f["attributes"] for f in d["features"]], d.get("exceededTransferLimit", False)
        except Exception:
            if k == 3:
                raise
            time.sleep(5 * (k + 1))


def fetch_port(portid):
    rows, offset = [], 0
    while True:
        got, more = query(f"portid='{portid}'", offset)
        rows += got
        offset += len(got)
        if not more or not got:
            return rows


def main():
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "ports_daily.csv")
    frames = []
    for pid, name in {**EXPORT_PORTS, **CASE_PORTS, **PLACEBO_PORTS}.items():
        df = pd.DataFrame(fetch_port(pid))
        df["date"] = pd.to_datetime(df["date"])
        df["role"] = ROLES[pid]
        frames.append(df)
        print(f"{name}: {len(df)} days, {df['date'].min().date()} to {df['date'].max().date()}")
    pd.concat(frames).sort_values(["portid", "date"]).to_csv(path, index=False)


if __name__ == "__main__":
    main()
