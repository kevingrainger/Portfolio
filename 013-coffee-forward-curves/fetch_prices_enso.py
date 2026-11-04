#!/usr/bin/env python3
#-------- fetch_prices_enso.py ----------------------------------------------------------
#-----------------------------------------------------------------------------
# Both beans and El Nino:
#   prices   World Bank Pink Sheet, monthly arabica and robusta (US$/kg), from 1960
#   ENSO     NOAA CPC Oceanic Nino Index (three-month running SST anomaly, Nino 3.4)
# The weather for the two origins compared (Brazil's arabica belt and Vietnam's robusta
# highlands) comes from fetch_data.py.
#
#   python fetch_prices_enso.py
#
# Output: data/wb_pinksheet_monthly.xlsx, data/oni.txt

import os
import urllib.request


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

PINK_SHEET = ("https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/"
              "related/CMO-Historical-Data-Monthly.xlsx")
ONI = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"



def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())



def main():
    os.makedirs(DATA, exist_ok=True)
    download(PINK_SHEET, os.path.join(DATA, "wb_pinksheet_monthly.xlsx"))
    download(ONI, os.path.join(DATA, "oni.txt"))
    print("prices and ONI downloaded")


if __name__ == "__main__":
    main()
