#-------- make_orbitals.py ------------------------------------------------------
# Orbital pictures for the capstone, drawn straight from the FHI-aims cube files.
# Each picture is the wavefunction on a plane 0.8 Å above the molecule (where the
# pi lobes are), red for one sign and blue for the other, with the atoms on top.
# Periodic chains are tiled three cells wide so the repeat is visible.
#
#   CORMAC=~/OneDrive/Desktop/"For Cormac" python make_orbitals.py
#
# The cube files (100 MB each) are not in this repo.

import os, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
SIMS = Path(os.environ.get("CORMAC", Path.home() / "OneDrive/Desktop/For Cormac")) / "Porphyrin Simulations (Zn,V,Cu,Mn,Co)"
OUT = HERE / "figures"
BOHR = 0.529177
ATOM_STYLE = {"C": ("#3a3a3a", 9), "N": ("#6a3d9a", 14), "H": ("#bdbdbd", 4), "O": ("white", 18),
              "Zn": ("#1b7837", 40), "Fe": ("#1b7837", 40), "V": ("#1b7837", 40)}


def read_cube(path):
    """Grid values (nx, ny, nz), origin and step (Å), and atoms as (Z, x, y, z) in Å."""
    with open(path) as fh:
        fh.readline(); fh.readline()
        n, *o = fh.readline().split()
        n, o = abs(int(n)), np.array(o[:3], float) * BOHR
        axes = [fh.readline().split() for _ in range(3)]
        shape = [int(a[0]) for a in axes]
        step = np.array([float(axes[i][i + 1]) for i in range(3)]) * BOHR
        atoms = np.array([fh.readline().split() for _ in range(n)], float)
        vals = np.fromstring(fh.read(), sep=" ")
    atoms = np.c_[atoms[:, 0], atoms[:, 2:5] * BOHR]
    return vals.reshape(shape), o, step, atoms


def occupations(folder):
    """State number -> (occupation, eigenvalue in eV) from the last table in output.aims."""
    txt = (folder / "output.aims").read_text(errors="ignore")
    block = txt.split("State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]")[-1]
    rows = re.findall(r"^\s+(\d+)\s+([\d.]+)\s+-?[\d.]+\s+(-?[\d.]+)\s*$", block[:200000], re.M)
    return {int(s): (float(o), float(e)) for s, o, e in rows}


def labels(folder):
    """Name each cube (HOMO, LUMO+1, ...) from the states requested in control.in."""
    src = folder / "control.in" if (folder / "control.in").exists() else folder / "output.aims"   # aims echoes control.in
    states = [int(s) for s in re.findall(r"^\s*output\s+cube\s+eigenstate\s+(\d+)", src.read_text(errors="ignore"), re.M)]
    occ = occupations(folder)
    homo = max(s for s, (o, e) in occ.items() if o > 0.5)
    name = lambda s: "HOMO" if s == homo else "LUMO" if s == homo + 1 else (f"HOMO−{homo - s}" if s < homo else f"LUMO+{s - homo - 1}")
    return {f"cube_{i + 2:03d}": (name(s), occ.get(s, (np.nan, np.nan))[1]) for i, s in enumerate(states)}


def lattice_a(folder):
    g = next((folder / n for n in ("geometry.in", "geometry.in.next_step") if (folder / n).exists()), None)
    m = g and re.search(r"^\s*lattice_vector\s+(\S+)", g.read_text(), re.M)
    return float(m.group(1)) if m else None


def draw(ax, cube, tile=1, a=None, height=0.8, lim=None, rotate=False):
    """Signed orbital on the plane `height` Å above the mean carbon plane, atoms and bonds on top."""
    v, o, d, atoms = read_cube(cube)
    if rotate:                                                 # a ribbon running along y, turned to lie along x
        v, o, d = v.transpose(1, 0, 2), o[[1, 0, 2]], d[[1, 0, 2]]
        atoms = atoms[:, [0, 2, 1, 3]]
    if a:                                                      # fold atoms into the cube's own cell before tiling
        atoms[:, 1] = o[0] + (atoms[:, 1] - o[0]) % a
    zc = atoms[atoms[:, 0] == 6, 3].mean() + height
    k = (zc - o[2]) / d[2]; k0 = int(np.floor(k)); w = k - k0
    plane = (1 - w) * v[:, :, k0] + w * v[:, :, k0 + 1]
    x = o[0] + d[0] * np.arange(v.shape[0]); y = o[1] + d[1] * np.arange(v.shape[1])
    lim = lim or np.percentile(np.abs(plane), 99.7)
    norm = TwoSlopeNorm(0, -lim, lim)
    shifts = [i * a for i in range(tile)] if a else [0]
    if a:                                                      # one cell of grid per tile (some cubes span two)
        cell = x < x[0] + a
        x, plane = x[cell], plane[cell]
    for s in shifts:
        ax.pcolormesh(x + s, y, plane.T, cmap="RdBu_r", norm=norm, shading="auto", rasterized=True)
    pos = np.vstack([atoms[:, 1:] + [s, 0, 0] for s in shifts]); Z = np.tile(atoms[:, 0], len(shifts))
    x0, x1 = x[0], x[-1] + shifts[-1]
    keep = (pos[:, 0] >= x0) & (pos[:, 0] <= x1) if a else np.ones(len(pos), bool)
    pos, Z = pos[keep], Z[keep]
    sym = {1: "H", 6: "C", 7: "N", 8: "O", 23: "V", 26: "Fe", 30: "Zn"}
    el = np.array([sym.get(int(z), "C") for z in Z])
    for i in range(len(pos)):                                   # bonds by distance
        dd = np.linalg.norm(pos[i + 1:, :2] - pos[i, :2], axis=1)
        for j in np.where(dd < (1.25 if "H" in (el[i],) else 1.75))[0] + i + 1:
            if "H" in (el[i], el[j]) and dd[j - i - 1] > 1.2:
                continue
            ax.plot(pos[[i, j], 0], pos[[i, j], 1], color="#2b2b2b", lw=0.5, alpha=0.7, zorder=3)
    for e, (c, s) in ATOM_STYLE.items():
        m = el == e
        ax.scatter(pos[m, 0], pos[m, 1], s=s, c=c, edgecolors="#222" if e in ("O", "Zn", "Fe", "V") else "none",
                   linewidths=0.6, zorder=4)
    ax.set_aspect("equal"); ax.axis("off")
    pad = 1.5
    ax.set_xlim(x0, x1) if a else ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
    return lim


def panel_set(entries, fname, title, tile=3, ncols=2, width=11, rotate=False, height=1.9):
    """entries: list of (row label, folder, [cube names]). One row per structure."""
    nrow = len(entries)
    fig, axes = plt.subplots(nrow, ncols, figsize=(width, height * nrow + 0.6), squeeze=False)
    for r, (rlab, folder, cubes) in enumerate(entries):
        lab = labels(folder); a = lattice_a(folder) if tile > 1 else None
        for c, cb in enumerate(cubes):
            ax = axes[r, c]
            draw(ax, folder / f"{cb}_first_order_density.cube", tile=tile if a else 1, a=a, rotate=rotate)
            name, e = lab[cb]
            ax.set_title(f"{rlab}: {name}  ({e:.2f} eV)", loc="left", fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"]})
    zc = SIMS / "Zinc porphyrin Simulations/Zinc Chains/Older Sims/Very old"
    panel_set([("weak", zc / "ZnTPP_weak_correct/Revised", ["cube_003", "cube_004"]),
               ("strong", zc / "ZnTPP_strong_correct/Revised", ["cube_003", "cube_004"]),
               ("extra-strong", zc / "ZnTPP_extra_strong/Revised", ["cube_003", "cube_004"])],
              "fig3_zinc_chain_orbitals", "Zinc chains: the frontier orbitals as the linking tightens")
    # The molecule runs saved one state below and one above the frontier; the "full cube files" folders
    # have four cubes but no record of which states they are, so they are not used.
    zm = SIMS / "Zinc porphyrin Simulations/Zinc Molecules"
    panel_set([("ZnP", zm / "ZnP", ["cube_002", "cube_003"]), ("ZnTPP", zm / "ZnTPP_1", ["cube_002", "cube_003"]),
               ("ZnDPP", zm / "ZnDPP", ["cube_003", "cube_002"])],
              "fig4_zinc_molecule_orbitals", "The three zinc molecules: plain ring, four fused arms, two fused arms",
              tile=1, width=7, height=2.6)
    vc = SIMS / "Vanadium Porphyrin Simulations/Vanadium Chains/Vanadium No Spin chain sims"
    panel_set([("V weak", vc / "Weak", ["cube_003", "cube_004"]),
               ("V strong", vc / "Strong", ["cube_003", "cube_004"]),
               ("V DPP", vc / "DPP", ["cube_003", "cube_004"])],
              "fig6_vanadium_chain_orbitals", "Vanadium chains: the metal's d orbital joins the frontier")
    panel_set([("Fe=O in a graphene ribbon", SIMS / "GNR with Iron Porphyrin oxide/GNR", ["cube_002", "cube_003"])],
              "fig7_feo_ribbon_orbitals", "An Fe=O porphyrin set into a graphene nanoribbon", tile=1, ncols=2,
              width=12, rotate=True, height=1.6)
