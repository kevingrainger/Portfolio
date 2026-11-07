#!/usr/bin/env python3
#-------- make_thesis_orbitals.py ---------------------------------------------------
#-----------------------------------------------------------------------------
# Orbital pictures from the thesis itself. The FHI-aims runs can't be repeated, but the
# thesis PDF holds the original isosurface renders (red and blue: the two signs of the
# wavefunction). This pulls them out by page and position and lays them out as figures:
#
#   fig13_fe_dz2_top_and_side   the Fe-TPP HOMO from above and side-on: the iron dz2
#                               orbital standing out of the plane of the ring
#   fig14_orbital_atlas         HOMO-1, HOMO, LUMO and LUMO+1 for Zn-TPP, Fe-TPP, Fe-DPP,
#                               V-TPP and V-DPP (thesis figures 2.1-2.6)
#   fig15_chain_orbitals        the frontier orbitals of the periodic chains (thesis
#                               figures 3.2, 3.4, 3.6 and 3.8)
#
# The extracted renders are kept in figures/thesis/ so the figures rebuild without the PDF.
#
#   python make_thesis_orbitals.py ["path/to/Capstone_Script.pdf"]
#
# Output: figures/thesis/*.png, figures/fig13-15

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "figures", "thesis")

# (page, top-left y, x) of each render in the PDF, in points, named
MOLECULES = {  # page, then positions of HOMO-1, HOMO, LUMO, LUMO+1
    "Zn-TPP": (23, [(474, 298), (474, 410), (557, 298), (557, 410)]),
    "Fe-TPP": (25, [(73, 298), (73, 410), (156, 298), (156, 410)]),
    "Fe-DPP": (24, [(437, 298), (437, 410), (520, 298), (520, 410)]),
    "V-TPP": (26, [(73, 298), (73, 410), (156, 298), (156, 410)]),
    "V-DPP": (25, [(424, 298), (424, 410), (507, 298), (507, 410)]),
}
CHAINS = {  # page, positions in reading order (a)-(d)
    "Zn chain, strong coupling": (28, [(342, 73), (342, 307), (464, 73), (464, 307)]),
    "Fe-DPP chain": (29, [(254, 100), (254, 334), (366, 100), (366, 334)]),
    "V chain, weak coupling": (30, [(116, 73), (116, 307), (253, 73), (253, 307)]),
    "V-extra chain, strong coupling": (31, [(73, 73), (73, 307), (209, 73), (209, 307)]),
}
SIDE = (24, (145, 185))
LEVELS = ["HOMO-1", "HOMO", "LUMO", "LUMO+1"]


def extract(pdf):
    import pymupdf
    os.makedirs(SRC, exist_ok=True)
    d = pymupdf.open(pdf)

    def grab(page, pos, name):
        p = d[page - 1]
        for im in p.get_images(full=True):
            for r in p.get_image_rects(im[0]):
                if abs(r.y0 - pos[0]) < 3 and abs(r.x0 - pos[1]) < 3:
                    img = Image.open(__import__("io").BytesIO(d.extract_image(im[0])["image"])).convert("RGB")
                    img.thumbnail((800, 800))
                    img.save(os.path.join(SRC, name + ".png"))
                    return
        raise KeyError((page, pos))

    grab(*SIDE, "fe_tpp_homo_side")
    for mol, (page, pos) in MOLECULES.items():
        for lev, xy in zip(LEVELS, pos):
            grab(page, xy, f"{mol}_{lev}".replace("+", "p").replace("-", "_"))
    for ch, (page, pos) in CHAINS.items():
        for k, xy in enumerate(pos):
            grab(page, xy, f"chain_{ch.split(',')[0].replace(' ', '_').replace('-', '_')}_{'abcd'[k]}")


def load(name):
    im = np.asarray(Image.open(os.path.join(SRC, name + ".png")).convert("RGB"))
    mask = (im < 245).any(axis=2)                       # crop the white margin
    ys, xs = np.where(mask)
    if len(ys):
        pad = 10
        im = im[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
    return im


def main():
    if len(sys.argv) > 1:
        extract(sys.argv[1])
    plt.rcParams.update({"font.family": "serif", "font.size": 11})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1, 1.9], wspace=0.05))
    axes[0].imshow(load("Fe_TPP_HOMO")); axes[0].set_title("Fe-TPP HOMO, from above", loc="left")
    axes[1].imshow(load("fe_tpp_homo_side")); axes[1].set_title("The same orbital side-on: iron dz² out of the ring's plane", loc="left")
    for ax in axes:
        ax.axis("off")
    fig.savefig(os.path.join(HERE, "figures", "fig13_fe_dz2_top_and_side.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(len(MOLECULES), 4, figsize=(10, 2.7 * len(MOLECULES)), gridspec_kw=dict(wspace=0.02, hspace=0.12))
    for i, mol in enumerate(MOLECULES):
        for j, lev in enumerate(LEVELS):
            ax = axes[i, j]
            ax.imshow(load(f"{mol}_{lev}".replace("+", "p").replace("-", "_")))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(lev, fontsize=12)
            if j == 0:
                ax.set_ylabel(mol, fontsize=12, rotation=0, ha="right", va="center", labelpad=12)
    fig.savefig(os.path.join(HERE, "figures", "fig14_orbital_atlas.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    labs = ["(a)", "(b)", "(c)", "(d)"]
    fig, axes = plt.subplots(len(CHAINS), 2, figsize=(13, 2.2 * len(CHAINS)), gridspec_kw=dict(wspace=0.03, hspace=0.35))
    for i, ch in enumerate(CHAINS):
        base = f"chain_{ch.split(',')[0].replace(' ', '_').replace('-', '_')}"
        for j, k in enumerate("bc"):                     # (b) HOMO and (c) LUMO in the thesis panels
            ax = axes[i, j]
            ax.imshow(load(f"{base}_{k}")); ax.axis("off")
            ax.set_title(f"{ch}: {'HOMO' if k == 'b' else 'LUMO'}", loc="left", fontsize=11)
    fig.savefig(os.path.join(HERE, "figures", "fig15_chain_orbitals.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
