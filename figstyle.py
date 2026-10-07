#-------- figstyle.py -----------------------------------------------------------
#-----------------------------------------------------------------------------
# One figure style for every project, so the portfolio reads as one body of work.
#
#   import figstyle
#   figstyle.apply()
#   ...
#   figstyle.save(fig, "figures/threshold")      # writes threshold.png and threshold.pdf
#
# PNG for READMEs and the website; vector PDF for a LaTeX note or print. Every
# saved figure is one call, so nothing shown in a notebook goes unsaved.

import os

import matplotlib.pyplot as plt

# A restrained palette that stays distinguishable in greyscale and for the common
# forms of colour blindness (Okabe & Ito, 2008)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def apply():
    plt.rcParams.update({
        "figure.figsize": (7.0, 4.2),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def watermark(fig, text="Simulated test data"):
    """Mark a figure made from simulated data with a small footnote, so it is never
    mistaken for a measurement."""
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=7, color="0.45", zorder=100)


def save(fig, path_without_extension, formats=("png", "pdf")):
    folder = os.path.dirname(path_without_extension)
    if folder:
        os.makedirs(folder, exist_ok=True)
    for ext in formats:
        fig.savefig(f"{path_without_extension}.{ext}")
    return [f"{path_without_extension}.{ext}" for ext in formats]
