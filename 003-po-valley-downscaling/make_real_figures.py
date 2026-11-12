#-------- make_real_figures.py -----------------------------------------------------------
# Redraws the map and skill figures from the saved real-data run (results_real/), and the
# simulated skill figure as a dot plot. Nothing is re-fitted: the maps come from
# results_real/maps.npz and the skill from results_real/predictions_2023.csv.
#
#   python make_real_figures.py [--extras] [--emulator]   (needs data/real/, written by download_data.py)

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import figstyle
from tmax1km import plots, validation
from tmax1km.data import Dataset

OUT = os.path.join(HERE, "figures", "real")
NAMES = {"B0": "ERA5-Land", "B1": "regression-kriging", "B2": "E-OBS", "B3": "XGBoost on residuals",
         "M1": "physics only", "M2": "physics, anchored", "M3": "physics, anchored, XGBoost prior"}


def skill(res, models, out, watermark):
    pred = validation.add_bands(pd.read_csv(os.path.join(HERE, res, "predictions_2023.csv"), parse_dates=["date"]))
    models = [m for m in models if m in pred]
    by_elev = validation.skill(pred, models, "elevation band")
    by_dist = validation.skill(pred, models, "anchor distance")
    fig = plots.skill_bars(by_elev, by_dist, models, NAMES)
    if watermark:
        figstyle.watermark(fig, "Simulated test data")
    figstyle.save(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    figstyle.apply()
    os.makedirs(OUT, exist_ok=True)
    ds = Dataset(verbose=False)
    assert not ds.simulated, "data/real/ is needed for the real figures"
    maps = np.load(os.path.join(HERE, "results_real", "maps.npz"))

    # anchored without the XGBoost prior (M2): on the real data the prior made things worse
    fig = plots.cover(ds, maps, key="hot_M2")
    fig.savefig(os.path.join(OUT, "cover_real.png"), dpi=200, bbox_inches=fig.bbox_inches)
    plt.close(fig)
    fig = plots.pde_surfaces(ds, maps, key="hot_M2")
    figstyle.save(fig, os.path.join(OUT, "hero_real"), formats=("png",))
    plt.close(fig)
    fig = plots.downscaling(ds, maps, key="hot_M2", label="Physics, anchored to stations (1 km)")
    figstyle.save(fig, os.path.join(OUT, "fig2_heatwave_day_real"), formats=("png",))
    plt.close(fig)
    fig = plots.footprints(ds, maps)
    figstyle.save(fig, os.path.join(OUT, "fig4_footprints_real"), formats=("png",))
    plt.close(fig)
    fig = plots.missing_physics(ds, maps, key="mean_q_m2")
    figstyle.save(fig, os.path.join(OUT, "fig5_missing_physics_real"), formats=("png",))
    plt.close(fig)

    skill("results_real", ["B0", "B1", "M1", "M2"], os.path.join(OUT, "fig3_skill_real"), False)
    skill("results", ["B0", "B1", "B2", "M1", "M2"], os.path.join(HERE, "figures", "fig3_skill"), True)
    print("wrote figures/real/*_real.png and figures/fig3_skill")


def extras():
    """Station map on the real 38-station network, and the step-0 recovery check with plain titles."""
    import time
    from tmax1km import step0
    ds = Dataset(verbose=False)
    fig = plots.stations(ds)
    figstyle.save(fig, os.path.join(OUT, "fig1_stations_real"), formats=("png",))
    plt.close(fig)
    t = time.time()
    static = np.load(os.path.join(HERE, "data", "estimated", "static_1km.npz"))
    complete = step0.run(static, n_days=20, q_size=0.0)
    missing = step0.run(static, n_days=20, q_size=0.3)
    fig = plots.recovery([step0.recovery_table(complete), step0.recovery_table(missing)],
                         ["Physics complete (q = 0)", "Physics missing (q unknown)"])
    fig.tight_layout()
    figstyle.save(fig, os.path.join(HERE, "figures", "fig0_step0_recovery"))
    plt.close(fig)
    print(f"step 0 redrawn in {time.time() - t:.0f} s, anchors missed by {missing['anchor_gap']:.1e} K, "
          f"q correlation {missing['q_correlation']:.2f}")


if __name__ == "__main__" and "--extras" in sys.argv:
    extras()


def emulator():
    """Emulator (U) against physics only (M1) and anchored (M2): all days and the hottest 5%, simulated and real."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True)
    rows = [("M1", "physics only"), ("M2", "physics, anchored"), ("U", "U-Net emulator")]
    cols = {"rmse": ("#0072B2", "all days"), "rmse_hottest_5pct": ("#D55E00", "hottest 5% of days")}
    for ax, (res, title) in zip(axes, (("results", "Simulated check"), ("results_real", "Real data"))):
        t = pd.read_csv(os.path.join(HERE, res, "unet_skill_table.csv")).set_index("model")
        y = np.arange(len(rows))[::-1]
        for (m, _), yi in zip(rows, y):
            ax.plot(t.loc[m, ["rmse", "rmse_hottest_5pct"]], [yi, yi], color="0.8", lw=2, zorder=1)
        for c, (col, lab) in cols.items():
            ax.plot([t.loc[m, c] for m, _ in rows], y, "o", color=col, ms=8, label=lab, zorder=2)
        ax.set_yticks(y); ax.set_yticklabels([r[1] for r in rows])
        ax.set_xlabel("RMSE at held-out stations, 2023 (K)"); ax.set_title(title, loc="left")
    axes[1].legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    figstyle.save(fig, os.path.join(OUT, "fig7_emulator_real"), formats=("png",))
    plt.close(fig)


if __name__ == "__main__" and "--emulator" in sys.argv:
    figstyle.apply()
    emulator()
