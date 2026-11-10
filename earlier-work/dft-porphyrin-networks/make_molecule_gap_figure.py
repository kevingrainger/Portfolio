#-------- make_molecule_gap_figure.py ----------------------------------------------
# Single molecules, every metal and arm structure: the HOMO-LUMO gap (top) and the
# Mulliken spin on the metal and on the ring (bottom), drawn as dot plots.
#
#   python make_molecule_gap_figure.py
#
# Reads data/molecule_gaps.csv and data/molecule_spin.csv (written from the FHI-aims
# runs). Zinc is run without spin; the other metals are spin-polarised (collinear).

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import figstyle

figstyle.apply()
gaps = pd.read_csv(HERE / "data" / "molecule_gaps.csv")
spin = pd.read_csv(HERE / "data" / "molecule_spin.csv")

METALS = ["Zn", "Fe", "Mn", "Co", "V"]
ARMS = [("P", "plain ring", "o"), ("DPP", "two fused arms (DPP)", "s"), ("TPP", "four fused arms (TPP)", "D")]
ARM_C = {"P": "#7f7f7f", "DPP": "#0072B2", "TPP": "#D55E00"}
METAL_C, RING_C = "#1b7837", "#8172b3"

fig = plt.figure(figsize=(9.0, 6.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=0.45, wspace=0.12)

# --- top: gap by metal, one marker per arm structure, log scale (0.04 to 3.3 eV) ---
ax = fig.add_subplot(gs[0, :])
off = {"P": -0.26, "DPP": 0.0, "TPP": 0.26}
lab = {"P": ((-7, -3), "right"), "DPP": ((0, 8), "center"), "TPP": ((7, -3), "left")}
for arm, label, mk in ARMS:
    d = gaps[gaps.ligand == arm].set_index("metal")
    xs = [i + off[arm] for i, m in enumerate(METALS) if m in d.index]
    ys = [d.loc[m, "gap_eV"] for m in METALS if m in d.index]
    ax.scatter(xs, ys, marker=mk, s=46, color=ARM_C[arm], label=label, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=lab[arm][0], ha=lab[arm][1], fontsize=7.5, color=ARM_C[arm])
ax.set_yscale("log")
ax.set_ylim(0.025, 30)
ax.set_xticks(range(len(METALS)), METALS)
ax.set_xlim(-0.6, len(METALS) - 0.4)
ax.set_ylabel("HOMO-LUMO gap (eV, log scale)")
ax.set_title("HOMO-LUMO gap by metal and arms", loc="left")
ax.legend(ncol=3, loc="upper right")
ax.text(0, 0.03, "no spin", ha="center", fontsize=7.5, color="#555555")
for i, m in enumerate(METALS):
    if m == "Co":
        ax.text(i, 0.03, "DPP run did not finish", ha="center", fontsize=7.5, color="#555555")

# --- bottom: spin on the metal and on the ring, one panel per arm structure ---
mags = ["Fe", "Mn", "Co", "V"]
for j, (arm, label, _) in enumerate(ARMS):
    a = fig.add_subplot(gs[1, j], sharey=None if j == 0 else fig.axes[1])
    d = spin[spin.ligand == arm].set_index("metal")
    for i, m in enumerate(mags):
        if m not in d.index:
            a.text(i, 0.15, "did not\nfinish", ha="center", fontsize=7, color="#555555")
            continue
        ms, rs = d.loc[m, "metal_spin"], d.loc[m, "ring_spin"]
        a.plot([i, i], [ms, rs], color="#bbbbbb", lw=1.2, zorder=1)
        a.scatter([i], [ms], s=40, color=METAL_C, zorder=3, label="metal" if i == 0 else None)
        a.scatter([i], [rs], s=40, color=RING_C, zorder=3, label="ring" if i == 0 else None)
    a.axhline(0, color="#333333", lw=0.7)
    a.set_xticks(range(len(mags)), mags)
    a.set_xlim(-0.6, len(mags) - 0.4)
    a.set_title(label, loc="left", fontsize=10)
    if j == 0:
        a.set_ylabel("spin moment (μB)")
        a.legend(loc="lower left")
    else:
        a.tick_params(labelleft=False)
fig.axes[1].set_ylim(-2.7, 4.0)

figstyle.save(fig, str(HERE / "figures" / "fig16_molecule_gap_and_spin"))
print("wrote figures/fig16_molecule_gap_and_spin.png")
