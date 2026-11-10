#-------- make_oxo_gap_figure.py --------------------------------------------------
# What oxidising the metal does to each chain's HOMO-LUMO gap: bare chain against the
# same chain with one oxygen bound to the metal, as a dumbbell per chain.
#
#   python make_oxo_gap_figure.py
#
# Reads data/oxo_spin.csv. The bare vanadium strong and fully fused chains were not run.

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import figstyle

figstyle.apply()
d = pd.read_csv(HERE / "data" / "oxo_spin.csv")
LINKS = [("weak", "weak"), ("strong", "strong"), ("extra-strong", "fully fused"), ("DPP", "DPP")]
COL = {"Fe": "#c0392b", "V": "#1f6fb2"}

fig, ax = plt.subplots(figsize=(7.0, 4.0))
rows = []
for m in ["Fe", "V"]:
    for key, name in LINKS:
        rows.append((m, key, name))
for y, (m, key, name) in enumerate(rows):
    b = d[(d.metal == m) & (d.linking == key) & (d.oxo == 0)]
    o = d[(d.metal == m) & (d.linking == key) & (d.oxo == 1)]
    go = o.gap_eV.iloc[0]
    if len(b):
        gb = b.gap_eV.iloc[0]
        ax.plot([gb, go], [y, y], color="#bbbbbb", lw=2, zorder=1)
        ax.scatter([gb], [y], s=55, facecolor="white", edgecolor=COL[m], lw=1.6, zorder=3, label="bare" if y == 0 else None)
        ax.annotate(f"{gb:.2f}", (gb, y), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=8, color="#555555")
    else:
        ax.annotate("bare not run", (go, y), xytext=(-12, 0), textcoords="offset points", ha="right", va="center", fontsize=7.5, color="#888888")
    ax.scatter([go], [y], s=55, color=COL[m], zorder=3, label="with O on the metal" if y == 0 else None)
    ax.annotate(f"{go:.2f}", (go, y), xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color=COL[m])
ax.set_yticks(range(len(rows)), [f"{'Fe' if m == 'Fe' else 'V'}, {name}" for m, _, name in rows])
ax.invert_yaxis()
ax.axhline(3.5, color="#dddddd", lw=0.8)
ax.set_xlim(0, 0.75)
ax.set_xlabel("HOMO-LUMO gap (eV)")
ax.set_title("Chain gap, bare and with oxygen on the metal", loc="left")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="#555555", mew=1.6, ms=7, label="bare (no oxygen)"),
                   Line2D([], [], marker="o", ls="", color="#555555", ms=7, label="with O on the metal")],
          loc="upper right")
figstyle.save(fig, str(HERE / "figures" / "fig17_oxo_gap_dumbbell"))
print("wrote figures/fig17_oxo_gap_dumbbell.png")
