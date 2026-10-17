# 015 data

The only input is daily front-month futures closes for COMEX copper and gold, CBOT wheat and corn, and ICE coffee and cocoa, from Yahoo Finance (`front_months.csv`). Yahoo's terms don't allow redistribution, so the file isn't in the repo. `backtest.py` downloads it on first run:

```
pip install yfinance pandas numpy scipy matplotlib
python backtest.py        # downloads data/front_months.csv if missing, then runs (about 1 minute)
python monte_carlo.py     # no data needed, about 10 minutes
python make_figures.py
```

Yahoo revises and extends its series, so a later download can differ slightly from the one behind the saved results (downloaded 12 October 2026).
