#!/usr/bin/env python3
#-------- make_bs_surface.py ------------------------------------------------------------
# A standard Black-Scholes call surface from the SOR solver in pde_numba.py:
# left, option value over stock price and time to expiry, coloured by gamma (where the
# value curves fastest), with the payoff at expiry drawn in; right, the same solution as
# value curves at several times to expiry, closing in on the payoff.
#   python make_bs_surface.py

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pde_numba as pn

K, T, r, sigma = 100.0, 1.0, 0.05, 0.25
S, tau, V, _, _ = pn.black_scholes_pde(K, T, r, sigma, kind="call", american=False)

keep = (S >= 40) & (S <= 170)
S, V = S[keep], V[:, keep]
dS = S[1] - S[0]
gamma = np.gradient(np.gradient(V, dS, axis=1), dS, axis=1)
gamma[0] = 0.0                                      # the payoff's kink, not a real value

SS, TT = np.meshgrid(S, tau)
norm = colors.PowerNorm(0.5, vmin=0, vmax=np.percentile(gamma[5:], 99.5))
face = cm.RdYlBu_r(norm(np.clip(gamma, 0, None)))

plt.rcParams.update({"font.family": "serif", "font.size": 11})
fig = plt.figure(figsize=(13, 5.6))

ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot_surface(SS, TT, V, facecolors=face, rstride=2, cstride=2, linewidth=0,
                antialiased=True, shade=False)
ax.plot(S, np.zeros_like(S), np.maximum(S - K, 0), color="#0b2545", lw=2.2, label="payoff at expiry")
ax.set_xlabel("stock price")
ax.set_ylabel("time to expiry (years)")
ax.set_zlabel("call value", labelpad=6)
ax.set_box_aspect(None, zoom=0.92)
ax.view_init(elev=24, azim=-128)
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
ax.set_title("Call value, coloured by gamma", loc="left")
ax.legend(loc="upper left", frameon=False)
sm = cm.ScalarMappable(norm=norm, cmap="RdYlBu_r")
fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, label="gamma")

ax2 = fig.add_subplot(1, 2, 2)
blues = plt.cm.Blues(np.linspace(0.35, 1.0, 5))
for colour, t_show in zip(blues[::-1], [1.0, 0.5, 0.25, 0.1, 0.02]):
    n = int(round(t_show / (tau[1] - tau[0])))
    ax2.plot(S, V[n], color=colour, lw=1.8, label=f"{t_show:g} year{'' if t_show == 1 else 's'} to expiry")
ax2.plot(S, np.maximum(S - K, 0), color="#c0392b", lw=2, ls="--", label="payoff at expiry")
ax2.axvline(K, color="0.6", lw=0.8)
ax2.text(K + 1, 62, "strike", color="0.4")
ax2.set_xlabel("stock price")
ax2.set_ylabel("call value")
ax2.set_title("The same solution, slice by slice", loc="left")
ax2.legend(frameon=False)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = os.path.join(HERE, "figures", "fig5_black_scholes_surface.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
print("wrote", out)
