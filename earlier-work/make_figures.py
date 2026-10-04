#!/usr/bin/env python3
#-------- make_figures.py -------------------------------------------------------------
#-----------------------------------------------------------------------------
# Figures for the earlier (college) projects, re-plotted from their own outputs:
#   pde-solvers-cpp        Laplace by SOR (Numba port of the C++), SOR cost against
#                          omega and grid size, and Black-Scholes by Crank-Nicolson + SOR
#   higher-order-odes-cpp  RK4 against the analytic solution, and its h^4 convergence
#   potts-monte-carlo-cpp  magnetisation and its variance through the phase transition
#
#   python make_figures.py              all
#   python make_figures.py pde          one (pde | odes | potts)

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "pde-solvers-cpp"))
import figstyle

figstyle.apply()


def fig_path(folder, name):
    return os.path.join(HERE, folder, "figures", name)


#-------- PDEs --------------------------------------------------------------------------
def pde():
    import pde_numba as pn
    F = "pde-solvers-cpp"

    #Laplace on the C++ grid, solved at the optimal omega
    N = 253
    phi, bnd = pn.laplace_problem(N)
    t0 = time.perf_counter()
    sweeps = pn.sor_laplace(phi, bnd, pn.optimal_omega(N))
    print(f"Laplace N={N}: {sweeps} sweeps, {time.perf_counter() - t0:.2f} s, dphi/dy = {pn.derivative_value(phi):.5f}")
    x = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    fig = plt.figure(figsize=(13, 6.5), dpi=200)
    ax = fig.add_axes([0.0, 0.05, 0.55, 0.9], projection="3d")
    ax.plot_surface(X, Y, phi, cmap="viridis", rstride=2, cstride=2, linewidth=0, antialiased=True)
    ax.view_init(elev=32, azim=-128)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("φ")
    ax.set_title("Laplace's equation by successive over-relaxation", loc="left")
    ax2 = fig.add_axes([0.6, 0.12, 0.36, 0.72])
    cs = ax2.contourf(X, Y, phi, levels=30, cmap="viridis")
    ax2.contour(X, Y, phi, levels=15, colors="w", linewidths=0.4)
    fig.colorbar(cs, ax=ax2, label="φ")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_aspect("equal"); ax2.grid(False)
    ax2.set_title("Box held at φ = 1, line at φ = 0", loc="left")
    figstyle.save(fig, fig_path(F, "fig1_laplace_surface"))
    figstyle.save(fig, fig_path(F, "cover"))
    plt.close(fig)

    #SOR cost against omega and grid size: the parameter surface
    Ns = np.arange(21, 122, 10)
    omegas = np.linspace(1.0, 1.98, 30)
    cost = np.empty((len(Ns), len(omegas)))
    for a, n in enumerate(Ns):
        for b, w in enumerate(omegas):
            p, bd = pn.laplace_problem(int(n))
            cost[a, b] = pn.sor_laplace(p, bd, w, 1e-6)
    W, NN = np.meshgrid(omegas, Ns)
    fig = plt.figure(figsize=(13, 6.5), dpi=200)
    ax = fig.add_axes([0.0, 0.05, 0.55, 0.9], projection="3d")
    ax.plot_surface(W, NN, np.log10(cost), cmap="magma", linewidth=0, antialiased=True, alpha=0.9)
    best = np.array([omegas[np.argmin(c)] for c in cost])
    ax.plot(best, Ns, np.log10(cost.min(1)), "c-", lw=2.5, label="fastest ω")
    ax.view_init(elev=28, azim=-60)
    ax.set_xlabel("ω"); ax.set_ylabel("grid size N"); ax.set_zlabel("log10 sweeps")
    ax.set_title("SOR cost surface", loc="left"); ax.legend()
    ax2 = fig.add_axes([0.62, 0.14, 0.35, 0.72])
    for n, c, col in zip(Ns[::3], cost[::3], plt.cm.viridis(np.linspace(0, 0.9, len(Ns[::3])))):
        ax2.semilogy(omegas, c, color=col, label=f"N = {n}")
    ax2.set_xlabel("relaxation factor ω"); ax2.set_ylabel("sweeps to converge")
    ax2.set_title("Over-relaxing pays, until it doesn't", loc="left"); ax2.legend(fontsize=9)
    figstyle.save(fig, fig_path(F, "fig2_sor_parameter_surface"))
    plt.close(fig)
    print("measured best omega vs theory:", [(int(n), round(b, 3), round(pn.optimal_omega(n), 3)) for n, b in zip(Ns[::5], best[::5])])

    #Black-Scholes: European call and American put surfaces, early-exercise boundary
    K, T, r, sig = 100.0, 1.0, 0.05, 0.2
    S, t, Vc, _, _ = pn.black_scholes_pde(K, T, r, sig, kind="call", american=False)
    S, t, Vp, bnd_S, _ = pn.black_scholes_pde(K, T, r, sig, kind="put", american=True)
    S, t, Ve, _, _ = pn.black_scholes_pde(K, T, r, sig, kind="put", american=False)
    keep = (S >= 50) & (S <= 150)
    SS, TT = np.meshgrid(S[keep], t, indexing="xy")
    fig = plt.figure(figsize=(13, 6.5), dpi=200)
    for k, (V, title, cmap) in enumerate([(Vc, "European call", "viridis"), (Vp, "American put", "plasma")]):
        ax = fig.add_axes([0.0 + k * 0.5, 0.05, 0.5, 0.88], projection="3d")
        ax.plot_surface(SS, TT, V[:, keep], cmap=cmap, rstride=3, cstride=2, linewidth=0, antialiased=True)
        if k == 1:
            ok = np.isfinite(bnd_S) & (bnd_S >= 50)
            ax.plot(bnd_S[ok], t[ok], K - bnd_S[ok] + 0.3, "c-", lw=3, label="early-exercise boundary", zorder=10)
            ax.legend(loc="upper right")
        ax.view_init(elev=30, azim=-60 if k == 0 else -120)
        ax.set_xlabel("stock price S"); ax.set_ylabel("time to expiry"); ax.set_zlabel("option value")
        ax.set_title(f"{title}: Crank-Nicolson + {'projected ' if k else ''}SOR", loc="left")
    figstyle.save(fig, fig_path(F, "fig3_black_scholes_surfaces"))
    plt.close(fig)

    #validation and the boundary
    exact_c = pn.black_scholes_closed_form(S, K, T, r, sig, "call")
    exact_p = pn.black_scholes_closed_form(S, K, T, r, sig, "put")
    near = (S > 50) & (S < 150)
    print(f"max error vs closed form, S in 50-150: call {np.abs(Vc[-1] - exact_c)[near].max():.4f}, put {np.abs(Ve[-1] - exact_p)[near].max():.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    ax = axes[0]
    ax.plot(S, exact_p, "k-", lw=3, alpha=0.3, label="European put, closed form")
    ax.plot(S, Ve[-1], "--", color=figstyle.PALETTE[0], label="European put, PDE")
    ax.plot(S, Vp[-1], color=figstyle.PALETTE[1], label="American put, PDE")
    ax.plot(S, np.maximum(K - S, 0), ":", color="0.4", label="exercise value")
    ax.set_xlim(40, 160); ax.set_ylim(0, 45)
    ax.set_xlabel("stock price S"); ax.set_ylabel("value today"); ax.legend()
    ax.set_title("Checked against Black-Scholes; the American premium", loc="left")
    ax = axes[1]
    ax.plot(t, bnd_S, color=figstyle.PALETTE[1], lw=2)
    ax.set_xlabel("time to expiry (years)"); ax.set_ylabel("exercise if S below")
    ax.set_title("When to exercise an American put", loc="left")
    fig.tight_layout()
    figstyle.save(fig, fig_path(F, "fig4_black_scholes_check"))
    plt.close(fig)
    print("pde done")


#-------- ODEs --------------------------------------------------------------------------
def read_blocks(path):
    blocks, cur = {}, None
    for line in open(path, encoding="utf8", errors="ignore"):
        line = line.strip()
        if line.startswith("(Q"):
            cur = line
            blocks[cur] = []
        elif cur and line and (line[0].isdigit() or line[0] == "-"):
            blocks[cur].append([float(v) for v in line.split()])
    return {k: np.array(v) for k, v in blocks.items()}


def odes():
    F = "higher-order-odes-cpp"
    B = read_blocks(os.path.join(HERE, F, "Collated Data.txt"))
    q1, conv, q2 = B["(Q1)(i)"], B["(Q1)(ii)"], B["(Q2)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=200)
    ax = axes[0]
    ax.plot(q1[:, 0], q1[:, 2], "k-", lw=4, alpha=0.25, label="analytic")
    ax.plot(q1[:, 0], q1[:, 1], "o", color=figstyle.PALETTE[0], label="RK4, h = 0.1")
    ax.set_xlabel("t"); ax.set_ylabel("y"); ax.legend()
    ax.set_title("Fourth-order Runge-Kutta", loc="left")
    ax = axes[1]
    h, err = conv[:, 0], conv[:, 4]
    slope = np.polyfit(np.log10(h), np.log10(err), 1)[0]
    ax.loglog(h, err, "o", color=figstyle.PALETTE[1], label=f"error at t = 1 (slope {slope:.2f})")
    ax.loglog(h, err[0] * (h / h[0]) ** 4, "k--", lw=1, label="h⁴")
    ax.set_xlabel("step size h"); ax.set_ylabel("|RK4 - analytic|"); ax.legend()
    ax.set_title("Error falls as h⁴, as it should", loc="left")
    ax = axes[2]
    ax.plot(q2[:, 0], q2[:, 1], color=figstyle.PALETTE[2])
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("A higher-order ODE, solved as a system", loc="left")
    fig.tight_layout()
    figstyle.save(fig, fig_path(F, "fig1_rk4"))
    plt.close(fig)
    print(f"odes done (convergence slope {slope:.2f})")


#-------- Potts --------------------------------------------------------------------------
def potts():
    F = "potts-monte-carlo-cpp"
    d = pd.read_csv(os.path.join(HERE, F, "mc_results.csv"), skipinitialspace=True)
    beta_c = np.log(1 + np.sqrt(3))                       # exact, 2D 3-state Potts
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=200)
    for ax, col, lab, c in [(axes[0], "M", "magnetisation ⟨M⟩", 0), (axes[1], "Variance", "fluctuations, var(M)", 1)]:
        ax.plot(d["Beta"], d[col], "o-", color=figstyle.PALETTE[c], ms=4)
        ax.axvline(beta_c, color="0.4", ls="--", lw=1)
        ax.text(beta_c + 0.01, ax.get_ylim()[1] * 0.92, f"exact β_c = {beta_c:.3f}", fontsize=9, color="0.3")
        ax.set_xlabel("inverse temperature β"); ax.set_ylabel(lab)
    axes[0].set_title("3-state Potts model, 24 × 24: order appears", loc="left")
    axes[1].set_title("Fluctuations peak at the transition", loc="left")
    fig.tight_layout()
    figstyle.save(fig, fig_path(F, "fig1_transition"))
    plt.close(fig)
    print(f"potts done (variance peaks at beta = {d.loc[d['Variance'].idxmax(), 'Beta']:.2f})")


JOBS = dict(pde=pde, odes=odes, potts=potts)
if __name__ == "__main__":
    for name in (sys.argv[1:] or JOBS):
        JOBS[name]()
