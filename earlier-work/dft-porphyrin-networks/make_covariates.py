#!/usr/bin/env python3
#-------- make_covariates.py --------------------------------------------------------
# The capstone's design variables side by side: the metal, the fused arms (ligand), the
# bridge between units and an oxygen on the metal, against the spin and the gap they give.
#   top left      single molecules: HOMO-LUMO gap by metal and arms (colour, log scale)
#   top right     chains: HOMO-LUMO gap by bridge, zinc, iron and vanadium, bare and with O
#                 on the metal (colour), with the backbone spin written in each cell
#   bottom left   single molecules: spin on the porphyrin ring (colour), total moment in cell
#   bottom right  every chain: backbone spin against the gap, coloured by metal, bare or oxo
# Zinc chain gaps (no spin) are from the README: 0.31 weak, 0.05 strong, 0.78 fully fused.
#
#   python make_covariates.py
#
# Output: figures/fig11_covariates.png

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, LIGHT, RED, GREY = "#0b2545", "#1f6fb2", "#8cc0e8", "#c0392b", "#8a8a8a"
plt.rcParams.update({"font.family": "serif", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})


def main():
    mol = pd.read_csv(os.path.join(HERE, "data", "molecule_spin.csv"))
    oxo = pd.read_csv(os.path.join(HERE, "data", "oxo_spin.csv"))
    norm = colors.TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)
    gaps = pd.read_csv(os.path.join(HERE, "data", "molecule_gaps.csv"))
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw=dict(width_ratios=[1, 1.3], wspace=0.35, hspace=0.45))
    gnorm = colors.LogNorm(vmin=0.03, vmax=3.5)

    ax = axs[0, 0]
    gm = ["Zn", "Fe", "Mn", "Co", "V"]
    G = np.full((5, 3), np.nan)
    for i, m in enumerate(gm):
        for j, l in enumerate(["P", "DPP", "TPP"]):
            r = gaps[(gaps.metal == m) & (gaps.ligand == l)]
            if len(r):
                G[i, j] = r.gap_eV.iloc[0]
                ax.text(j, i, f"{G[i, j]:.2f} eV", ha="center", va="center", fontsize=9.5, color="white" if G[i, j] < 0.15 else "k")
            else:
                ax.text(j, i, "did not finish", ha="center", va="center", fontsize=8, color=GREY)
    imG = ax.imshow(G, cmap="YlGnBu_r", norm=gnorm, aspect="auto")
    ax.set_xticks(range(3), ["plain", "2 arms (DPP)", "4 arms (TPP)"], fontsize=8.5, rotation=20)
    ax.set_yticks(range(5), ["Zn (no spin)", "Fe", "Mn", "Co", "V"])
    ax.set_title("Molecules: HOMO-LUMO gap", loc="left")
    ax.spines[:].set_visible(False)
    fig.colorbar(imG, ax=ax, shrink=0.85, label="gap (eV, log scale)")

    ax = axs[0, 1]
    crow = [("Zn", None), ("Fe", 0), ("Fe", 1), ("V", 0), ("V", 1)]
    links = ["weak", "strong", "extra-strong", "DPP"]
    ZN = {"weak": 0.31, "strong": 0.05, "extra-strong": 0.78}
    C = np.full((5, 4), np.nan)
    for i, (m, o) in enumerate(crow):
        for j, l in enumerate(links):
            if m == "Zn":
                if l in ZN:
                    C[i, j] = ZN[l]
                    ax.text(j, i, f"{ZN[l]:.2f} eV", ha="center", va="center", fontsize=8.5, color="white" if ZN[l] < 0.15 else "k")
                else:
                    ax.text(j, i, "not run", ha="center", va="center", fontsize=8, color=GREY)
                continue
            r = oxo[(oxo.metal == m) & (oxo.oxo == o) & (oxo.linking == l)]
            if len(r):
                C[i, j] = r.gap_eV.iloc[0]
                ax.text(j, i, f"{C[i, j]:.2f} eV\n{(r.backbone_spin.iloc[0] if abs(r.backbone_spin.iloc[0]) >= 0.05 else 0):+.1f} μB".replace("+0.0", "0.0"), ha="center", va="center", fontsize=8.5,
                        color="white" if C[i, j] < 0.15 else "k")
            else:
                ax.text(j, i, "not run", ha="center", va="center", fontsize=8, color=GREY)
    imC = ax.imshow(C, cmap="YlGnBu_r", norm=gnorm, aspect="auto")
    ax.set_xticks(range(4), ["weak", "strong", "fully fused", "DPP"], fontsize=9)
    ax.set_yticks(range(5), ["Zn (no spin)", "Fe bare", "Fe=O", "V bare", "V=O"])
    ax.set_title("Chains: HOMO-LUMO gap (backbone spin below)", loc="left")
    ax.spines[:].set_visible(False)
    fig.colorbar(imC, ax=ax, shrink=0.85, label="gap (eV, log scale)")

    ax = axs[1, 0]
    metals, ligs = ["Fe", "Mn", "Co", "V"], ["P", "DPP", "TPP"]
    M = np.full((len(metals), len(ligs)), np.nan)
    for i, m in enumerate(metals):
        for j, l in enumerate(ligs):
            r = mol[(mol.metal == m) & (mol.ligand == l)]
            if len(r):
                M[i, j] = r.ring_spin.iloc[0]
                tot = r.metal_spin.iloc[0] + r.ring_spin.iloc[0]
                ax.text(j, i, f"{(tot if abs(tot) >= 0.05 else 0):+.1f}".replace("+0.0", "0.0"), ha="center", va="center", fontsize=9.5, color="k")
            else:
                ax.text(j, i, "did not finish", ha="center", va="center", fontsize=8, color=GREY)
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(3), ["plain", "2 arms (DPP)", "4 arms (TPP)"], fontsize=8.5, rotation=20)
    ax.set_yticks(range(4), metals); ax.set_title("Molecules: ring spin (total in cell)", loc="left")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="spin on the ring (μB)")

    ax = axs[1, 1]
    for (m, o), g in oxo.groupby(["metal", "oxo"]):
        ax.scatter(g.backbone_spin, g.gap_eV, s=70, marker="o" if o == 0 else "D",
                   facecolor=(RED if m == "Fe" else BLUE) if o == 1 else "white",
                   edgecolor=RED if m == "Fe" else BLUE, lw=1.6, label=f"{m}{'=O' if o else ' bare'}")
        for _, r in g.iterrows():
            ax.annotate(r.linking.replace("extra-strong", "fused"), (r.backbone_spin, r.gap_eV), xytext=(5, 3),
                        textcoords="offset points", fontsize=7.5, color=GREY)
    ax.axvline(0, color="0.85", lw=0.8)
    ax.set_xlabel("spin on the carbon backbone (μB)"); ax.set_ylabel("gap (eV)")
    ax.set_title("Chains: backbone spin vs gap", loc="left"); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.savefig(os.path.join(HERE, "figures", "fig11_covariates.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("written")


if __name__ == "__main__":
    main()
