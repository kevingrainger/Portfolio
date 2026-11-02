#-------- make_figures.py -------------------------------------------------------
# Figures for the capstone (DFT study of metal-porphyrin networks), redrawn from the
# original FHI-aims outputs. The raw runs (several GB) are not in this repo; point
# THESIS_DATA at the folder holding Zinc/, Molecules Analysis/ and Final Bands/.
#
#   THESIS_DATA=~/Documents python make_figures.py
#
# Writes small extracted tables to data/ so the figures can be rebuilt without the raw runs.

import os, re, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import figstyle

RAW = Path(os.environ.get("THESIS_DATA", Path.home() / "Documents"))
OUT, DATA = HERE / "figures", HERE / "data"


def gap_from_output(path):
    """HOMO, LUMO and gap (eV) from the last SCF cycle of an FHI-aims output.aims."""
    txt = Path(path).read_text(errors="ignore")
    homo = re.findall(r"Highest occupied state \(VBM\) at\s+(-?[\d.]+) eV", txt)
    lumo = re.findall(r"Lowest unoccupied state \(CBM\) at\s+(-?[\d.]+) eV", txt)
    gap = re.findall(r"Overall HOMO-LUMO gap:\s+([\d.]+) eV", txt)
    if not gap:
        return None
    return float(homo[-1]), float(lumo[-1]), float(gap[-1])


def read_dos(folder):
    """Total DOS if FHI-aims wrote it, otherwise the sum of the species-projected DOS."""
    f = folder / "KS_DOS_total.dat"
    if f.exists():
        d = np.loadtxt(f, comments="#")
        return d[:, 0], d[:, 1]
    parts = sorted(folder.glob("*_l_proj_dos.dat"))
    E = np.loadtxt(parts[0], comments="#")[:, 0]
    return E, sum(np.loadtxt(p, comments="#")[:, 1] for p in parts)


figstyle.apply()

# 1. HOMO-LUMO gaps of the single molecules, by metal and ligand.
rows = []
for metal, base in [("Zn", RAW / "Zinc"), ("Co", RAW / "Molecules Analysis/Mols Spin"),
                    ("Fe", RAW / "Molecules Analysis/Mols Spin"), ("Mn", RAW / "Molecules Analysis/Mols Spin"),
                    ("V", RAW / "Molecules Analysis/Mols Spin")]:
    prefix = "Vn" if metal == "V" else metal
    for lig in ["P", "TPP", "DPP"]:
        f = base / f"{prefix}{lig}" / "output.aims"
        if f.exists() and (g := gap_from_output(f)):
            rows.append((metal, lig, *g))
DATA.mkdir(exist_ok=True)
with open(DATA / "molecule_gaps.csv", "w") as fh:
    fh.write("metal,ligand,homo_eV,lumo_eV,gap_eV\n")
    for r in rows:
        fh.write(",".join(map(str, r)) + "\n")

metals = [m for m in ["Zn", "Co", "Fe", "Mn", "V"] if any(r[0] == m for r in rows)]
ligs = ["P", "TPP", "DPP"]
fig, ax = plt.subplots(figsize=(7, 3.8))
w = 0.26
cols = {"P": "#4c72b0", "TPP": "#55a868", "DPP": "#8172b3"}
for k, lig in enumerate(ligs):
    vals = [next((r[4] for r in rows if r[0] == m and r[1] == lig), np.nan) for m in metals]
    ax.bar(np.arange(len(metals)) + (k - 1) * w, vals, w, label=lig, color=cols[lig])
ax.set_xticks(range(len(metals)), metals)
ax.set_ylabel("HOMO–LUMO gap (eV)")
ax.set_title("Single-molecule gaps from the FHI-aims runs", loc="left")
ax.legend(title="ligand", frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
figstyle.save(fig, str(OUT / "fig1_molecule_gaps"))

# 2. Density of states near the Fermi level for the three zinc molecules, one panel each.
# ZnDPP has no total DOS file; its curve is the sum of species-projected DOS, which FHI-aims
# wrote with narrower broadening, so the panels are compared by peak positions, not shapes.
fig, axes = plt.subplots(3, 1, figsize=(7, 5.4), sharex=True)
for ax, (lig, c) in zip(axes, cols.items()):
    E, D = read_dos(RAW / "Zinc" / f"Zn{lig}")
    m = (E > -4) & (E < 4)
    ax.fill_between(E[m], D[m] / D[m].max(), color=c, alpha=0.35, lw=0)
    ax.plot(E[m], D[m] / D[m].max(), color=c, lw=1.1)
    ax.axvline(0, color="0.4", lw=0.8, ls=":")
    ax.set_yticks([])
    ax.set_ylabel(f"Zn{lig}", rotation=0, ha="right", va="center")
axes[0].set_title("Zinc porphyrin: density of states near the chemical potential", loc="left")
axes[-1].set_xlabel("energy relative to the chemical potential (eV)")
figstyle.save(fig, str(OUT / "fig2_zinc_dos"))
print(len(rows), "molecules;", "figures in", OUT)
