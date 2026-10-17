# 014 data

| File | What | Source | Licence | In the repo? |
|---|---|---|---|---|
| `ports_daily.csv` | daily port calls and estimated cargo by ship type, 19 ports, 2019– | [IMF PortWatch](https://portwatch.imf.org), ArcGIS service `Daily_Ports_Data` | IMF PortWatch terms: free reuse with attribution | yes |
| `bcrp_copper_exports.csv` | Peru's monthly copper export volume, thousand tonnes | [BCRP series PN38783BM](https://estadisticas.bcrp.gob.pe/estadisticas/series/mensuales/resultados/PN38783BM/html) | public statistics | yes |
| `hg_front.csv` | COMEX copper, front month | Yahoo Finance | not redistributable | **no**: `analyse_ports.py` downloads it |
| `../smelters.csv` | smelter hot spots found in Sentinel-2 | made by `verify_smelters.py` | CC BY 4.0 (this repo) | yes |

Rebuild from any machine, no accounts needed:

```
pip install pandas numpy yfinance rasterio
python fetch_ports.py          # PortWatch, about 5 minutes
python analyse_ports.py        # also fetches BCRP and the copper price
python verify_smelters.py      # Sentinel-2 via Microsoft Planetary Computer, about 10 minutes
```

The smelter time series (`smelter_heat.py`) needs a free non-commercial Google Earth Engine sign-in; see the top of that script.
