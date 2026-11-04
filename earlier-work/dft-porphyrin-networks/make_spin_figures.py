#-------- make_spin_figures.py --------------------------------------------------
# What one oxygen on the metal does to the iron and vanadium chains: where the
# spin sits (Mulliken, by element) and the spin-resolved density of states,
# bare chain against the same chain with an oxygen bound to the metal.
#
#   CORMAC=~/OneDrive/Desktop/"For Cormac" python make_spin_figures.py
#
# Writes data/oxo_spin.csv so the bar chart can be rebuilt without the raw runs.

import os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import figstyle

SIMS = Path(os.environ.get("CORMAC", Path.home() / "OneDrive/Desktop/For Cormac")) / "Porphyrin Simulations (Zn,V,Cu,Mn,Co)"
MOLS = Path(os.environ.get("THESIS_DATA", Path.home() / "Documents")) / "Molecules Analysis/Mols Spin"
OUT, DATA = HERE / "figures", HERE / "data"
FE, V = "Iron Porphyrin simulations", "Vanadium Porphyrin Simulations"
# (metal, linking) -> (bare chain, chain with O on the metal). The thesis calls the
# three-bond TPP chain "strong"; there are no spin-polarised bare V strong/extra runs.
RUNS = {("Fe", "weak"): (f"{FE}/Iron Chains/Fe Collinear/Weak", f"{FE}/Fe Oxygen Chain/Weak"),
        ("Fe", "strong"): (f"{FE}/Iron Chains/Fe Collinear/TPP", f"{FE}/Fe Oxygen Chain/Strong"),
        ("Fe", "extra-strong"): (f"{FE}/Iron Chains/Fe Collinear/Extra", f"{FE}/Fe Oxygen Chain/Extra"),
        ("Fe", "DPP"): (f"{FE}/Iron Chains/Fe Collinear/DPP", f"{FE}/Fe Oxygen Chain/DPP"),
        ("V", "weak"): (f"{V}/Vanadium Chains/Newest Vanadium chains/Weak", f"{V}/Vanadium Oxygen Chain/Weak"),
        ("V", "strong"): (None, f"{V}/Vanadium Oxygen Chain/Strong"),
        ("V", "extra-strong"): (None, f"{V}/Vanadium Oxygen Chain/Extra"),
        ("V", "DPP"): (f"{V}/Vanadium Chains/V Collinear Chain sims/DPP", f"{V}/Vanadium Oxygen Chain/DPP")}


def output_file(folder):
    return max(folder.glob("output*.aims"), key=lambda p: p.stat().st_mtime)


def spin_by_element(folder):
    """Mulliken spin summed by element, total moment and gap from the last SCF of a run."""
    txt = output_file(folder).read_text(errors="ignore")
    geo = next(folder / n for n in ("geometry.in.next_step", "geometry.in") if (folder / n).exists())
    sp = re.findall(r"^\s*atom\s+\S+\s+\S+\s+\S+\s+(\w+)", geo.read_text(), re.M)
    block = txt.split("Summary of the per-atom spin analysis:")[-1].split("Summary")[0]
    out = {}
    for i, s in re.findall(r"^\s*\|\s+(\d+)\s+(-?\d+\.\d+)", block, re.M):
        e = sp[int(i) - 1]
        out[e] = out.get(e, 0.0) + float(s)
    gap = re.findall(r"(?:Overall|ESTIMATED overall) HOMO-LUMO gap:\s+([\d.]+) eV", txt)
    out["gap"] = float(gap[-1]) if gap else np.nan
    return out


def pdos(folder, el):
    """Spin-up and spin-down DOS of one element against energy relative to the chemical potential."""
    up = np.loadtxt(folder / f"{el}_l_proj_dos_spin_up.dat.no_soc", comments="#")
    dn = np.loadtxt(folder / f"{el}_l_proj_dos_spin_dn.dat.no_soc", comments="#")
    return up[:, 0], up[:, 1], dn[:, 1]


if __name__ == "__main__":
    figstyle.apply()
    rows = []
    for (metal, link), pair in RUNS.items():
        for oxo, rel in zip((0, 1), pair):
            if rel is None:
                continue
            s = spin_by_element(SIMS / rel)
            rows.append(dict(metal=metal, linking=link, oxo=oxo, metal_spin=s.get(metal, 0), O_spin=s.get("O", 0),
                             backbone_spin=s.get("C", 0) + s.get("N", 0) + s.get("H", 0), gap_eV=s["gap"]))
    tab = pd.DataFrame(rows)
    DATA.mkdir(exist_ok=True); tab.round(3).to_csv(DATA / "oxo_spin.csv", index=False)
    print(tab.round(2).to_string(index=False))

    # 1. Where the spin sits, bare against oxo, for each linking.
    links = ["weak", "strong", "extra-strong", "DPP"]
    col = {"metal_spin": "#1b7837", "O_spin": "#7f7f7f", "backbone_spin": "#8172b3"}
    nice = {"metal_spin": "metal", "O_spin": "oxygen", "backbone_spin": "carbon backbone"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, metal in zip(axes, ["Fe", "V"]):
        for i, link in enumerate(links):
            for j, oxo in enumerate((0, 1)):
                r = tab[(tab.metal == metal) & (tab.linking == link) & (tab.oxo == oxo)]
                x = i + (j - 0.5) * 0.38
                if r.empty:
                    ax.text(x, 0.1, "no run", rotation=90, ha="center", va="bottom", fontsize=7, color="0.5")
                    continue
                pos = neg = 0.0
                for k in col:
                    v = float(r[k].iloc[0])
                    base = pos if v >= 0 else neg
                    ax.bar(x, v, 0.34, bottom=base, color=col[k], alpha=1 if oxo else 0.55,
                           edgecolor="white", lw=0.5, label=nice[k] if (i, j) == (0, 1) else None)
                    pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
                ax.text(x, -2.55, "O" if oxo else "bare", ha="center", fontsize=7, color="0.3")
        ax.axhline(0, color="0.3", lw=0.7)
        ax.set_xticks(range(len(links)), links); ax.tick_params(axis="x", pad=14)
        ax.set_title(f"{metal} chains: bare (faded) and with O on the metal", loc="left")
        ax.set_ylim(-2.8, 5.4)
    axes[0].set_ylabel(r"spin moment per unit cell ($\mu_B$)")
    axes[1].legend(loc="upper right")
    figstyle.save(fig, str(OUT / "fig8_oxo_spin_by_element"))

    # 2. Spin-resolved DOS near the chemical potential, bare against oxo (spin up above, down below).
    cases = [("V", "weak"), ("Fe", "extra-strong")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex=True)
    for c, (metal, link) in enumerate(cases):
        for r, oxo in enumerate((0, 1)):
            ax = axes[r, c]; f = SIMS / RUNS[(metal, link)][oxo]
            E, cu, cd = pdos(f, "C")
            m = (E > -2.5) & (E < 2.5)
            ax.fill_between(E[m], cu[m], color="#8172b3", alpha=0.35, lw=0)
            ax.fill_between(E[m], -cd[m], color="#8172b3", alpha=0.35, lw=0, label="carbon")
            for el, cl in [(metal, "#1b7837")] + ([("O", "#3a3a3a")] if oxo else []):
                E2, u, d = pdos(f, el); m2 = (E2 > -2.5) & (E2 < 2.5)
                ax.plot(E2[m2], u[m2], color=cl, lw=1.1, label=el)
                ax.plot(E2[m2], -d[m2], color=cl, lw=1.1)
            ax.axvline(0, color="0.4", lw=0.8, ls=":"); ax.axhline(0, color="0.3", lw=0.6)
            ax.set_yticks([])
            ax.set_title(f"{metal} {link}, {'with O on the metal' if oxo else 'bare'}", loc="left", fontsize=10)
            ax.text(-2.4, ax.get_ylim()[1] * 0.8, "spin up", fontsize=7, color="0.4")
            ax.text(-2.4, ax.get_ylim()[0] * 0.85, "spin down", fontsize=7, color="0.4")
            ax.legend(loc="upper right", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("energy relative to the chemical potential (eV)")
    figstyle.save(fig, str(OUT / "fig9_oxo_spin_dos"))

    # 3. Single molecules, every magnetic metal: spin on the metal against spin on the ring.
    # Folder names as saved (vanadium is "Vn"); the CoDPP run never finished, so it has no output.
    mol = []
    for metal in ["Fe", "Mn", "Co", "V"]:
        for lig in ["P", "DPP", "TPP"]:
            f = MOLS / f"{'Vn' if metal == 'V' else metal}{lig}"
            if not any(f.glob("output*.aims")):
                continue
            s = spin_by_element(f)
            mol.append(dict(metal=metal, ligand=lig, metal_spin=s.get(metal, 0),
                            ring_spin=s.get("C", 0) + s.get("N", 0) + s.get("H", 0), gap_eV=s["gap"]))
    mt = pd.DataFrame(mol); mt.round(3).to_csv(DATA / "molecule_spin.csv", index=False)
    print(mt.round(2).to_string(index=False))
    metals = ["Fe", "Mn", "Co", "V"]
    names = {"P": "plain ring (P)", "DPP": "two fused arms (DPP)", "TPP": "four fused arms (TPP)"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, lig in zip(axes, ["P", "DPP", "TPP"]):
        for i, metal in enumerate(metals):
            r = mt[(mt.metal == metal) & (mt.ligand == lig)]
            if r.empty:
                ax.text(i, 0.15, "run did not finish", rotation=90, ha="center", va="bottom", fontsize=7, color="0.5")
                continue
            ax.bar(i - 0.19, r.metal_spin.iloc[0], 0.36, color="#1b7837", label="metal" if i == 0 else None)
            ax.bar(i + 0.19, r.ring_spin.iloc[0], 0.36, color="#8172b3", label="ring" if i == 0 else None)
        ax.axhline(0, color="0.3", lw=0.7)
        ax.set_xticks(range(len(metals)), metals)
        ax.set_title(names[lig], loc="left")
    axes[0].set_ylabel(r"spin moment ($\mu_B$)")
    axes[0].legend(loc="lower left")
    figstyle.save(fig, str(OUT / "fig10_molecule_spin_by_metal"))
    print("figures in", OUT)
