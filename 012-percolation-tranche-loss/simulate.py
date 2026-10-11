#!/usr/bin/env python3
#-------- simulate.py ---------------------------------------------------------------
#-----------------------------------------------------------------------------
# Every number and figure in the note:
#   fig1  giant failure cluster S over mean degree and transmissibility (3D)
#   fig2  tranche expected loss against transmissibility, single-seed model
#   fig3  the network-factor model against Vasicek at equal mean loss
#   fig4  simulation against theory: percolation and the Watts cascade window
#   cover an animated cascade (website card)
#
#   python simulate.py            all          python simulate.py fig3     one

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import figstyle
import percolation as P

figstyle.apply()
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

# tranche colours: fixed order, with line styles as a second encoding
TR_COL = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#555555"]
TR_LS = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
INK = "#222222"

# The calibration used throughout (stated once, in the note's Table 1)
C_MEAN = 4.0           # mean number of trading partners
P_IDIO = 0.01          # own-cause default probability per year
LGD = 0.60             # loss given default
T_BAR, BETA = 0.05, 0.40   # median transmissibility and its loading on the economy
RHO = 0.20             # Vasicek asset correlation for the comparison
TRANCHES = [(0.00, 0.03, "Equity 0–3%"), (0.03, 0.07, "Mezzanine 3–7%"),
            (0.07, 0.15, "Senior 7–15%"), (0.15, 0.30, "Super senior 15–30%"),
            (0.30, 1.00, "Top 30–100%")]


#-------- fig1: the percolation surface -------------------------------------------
def fig1():
    c = np.linspace(0.5, 8, 90)
    T = np.linspace(0.0, 1.0, 90)
    S = np.zeros((len(T), len(c)))
    for j, cj in enumerate(c):
        d = P.Degrees.poisson(cj)
        for i, t in enumerate(T):
            S[i, j] = P.giant_component(d, t)
    C, TT = np.meshgrid(c, T)

    fig = plt.figure(figsize=(8.2, 6.0))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(C, TT, S, cmap=sns.color_palette("crest", as_cmap=True),
                    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=True)
    cc = np.linspace(1.0, 8, 200)                       # critical line  c T = 1
    ax.plot(cc, 1 / cc, np.full_like(cc, 0.012), color=INK, lw=2.4, zorder=10, label=r"threshold  $cT = 1$")
    ax.contour(C, TT, S, levels=[0.05, 0.25, 0.5, 0.75], offset=0, colors="0.55", linewidths=0.7)
    ax.set_xlabel(r"mean trading partners $c$")
    ax.set_ylabel(r"transmissibility $T$")
    ax.set_zlabel(r"giant failure cluster $S$")
    ax.view_init(elev=30, azim=-58)
    ax.set_box_aspect((1.2, 1.0, 0.7))
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor("0.85")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.92))
    ax.set_title("Bond percolation on a Poisson network", pad=4)
    figstyle.save(fig, os.path.join(FIG, "fig1_percolation_surface"))
    figstyle.save(fig, os.path.join(FIG, "hero"), formats=("png",))
    plt.close(fig)


#-------- fig2: the senior tranche switches on --------------------------------------
def fig2():
    d = P.Degrees.poisson(C_MEAN)
    lam = 2.0                                            # expected own-cause failures
    T = np.linspace(0.15, 0.6, 600)
    S = np.array([P.giant_component(d, t) for t in T])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(wspace=0.28))
    for k, (a, b, name) in enumerate(TRANCHES[:4]):
        el = P.percolation_tranche_el(S, lam, LGD, a, b)
        hit = (1 - np.exp(-lam * S)) * (LGD * S > a)
        a1.plot(T, el, color=TR_COL[k], ls=TR_LS[k], label=name)
        a2.plot(T, hit, color=TR_COL[k], ls=TR_LS[k], label=name)
    for ax in (a1, a2):
        ax.axvline(d.Tc, color=INK, lw=0.9, ls=(0, (2, 2)))
        ax.set_xlabel(r"transmissibility $T$")
        ax.set_xlim(T[0], T[-1]); ax.set_ylim(0, 1)
        sec = ax.secondary_xaxis("top")
        sec.set_xticks([d.Tc]); sec.set_xticklabels([r"$T_c$"])
        sec.tick_params(length=0)
    a1.set_ylabel("expected tranche loss")
    a2.set_ylabel("probability the tranche is touched")
    a1.set_title("(a) Expected loss", loc="left")
    a2.set_title("(b) Probability of any loss", loc="left")
    a2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    figstyle.save(fig, os.path.join(FIG, "fig2_tranche_switch"))
    plt.close(fig)


#-------- fig3: equal mean loss, different tails ------------------------------------
def fig3():
    d = P.Degrees.poisson(C_MEAN)
    grid = P.network_loss_grid(d, P_IDIO, LGD, T_BAR, BETA, n=6001)
    z, w, T, L = grid
    mean_loss = float((L * w).sum())
    p_vas = mean_loss / LGD
    p_above = float((w * (T > d.Tc)).sum())

    # densities: network loss by change of variable through L(Z), which is increasing
    x = np.linspace(1e-4, 0.6, 3000)
    dLdz = np.gradient(L, z)
    ok = dLdz > 1e-10
    net_pdf = np.interp(x, L[ok], (np.exp(-z[ok] ** 2 / 2) / np.sqrt(2 * np.pi)) / dLdz[ok],
                        left=0, right=0)
    vas_pdf = P.vasicek_loss_pdf(x / LGD, p_vas, RHO) / LGD

    rows = []
    for (a, b, name) in TRANCHES:
        ne = P.network_tranche_el(grid, a, b)
        ve = P.vasicek_tranche_el(p_vas, RHO, LGD, a, b)
        rows.append(dict(tranche=name, attach=a, detach=b, network=ne, vasicek=ve,
                         ratio=ne / ve if ve > 0 else float("inf")))
    sens = {}
    for rho in (0.10, 0.20, 0.30):
        sens[str(rho)] = [P.network_tranche_el(grid, a, b) / max(P.vasicek_tranche_el(p_vas, rho, LGD, a, b), 1e-300)
                          for a, b, _ in TRANCHES]
    json.dump(dict(mean_loss=mean_loss, p_vasicek=p_vas, p_above_threshold=p_above, Tc=d.Tc,
                   calibration=dict(c=C_MEAN, p=P_IDIO, lgd=LGD, T_bar=T_BAR, beta=BETA, rho=RHO),
                   tranches=rows, ratio_by_rho=sens),
              open(os.path.join(RES, "equal_mean_comparison.json"), "w"), indent=2)
    def fmt(x):
        if x > 1e4:
            return r"$>10^4$"
        return f"{x:.1f}" if x < 10 else f"{x:,.0f}".replace(",", "{,}")

    def pct(x):
        return f"{100 * x:.2g}\\%" if x >= 1e-4 else f"{100 * x:.1e}\\%"
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Tranche & Network & Vasicek ($\rho=0.2$) & \multicolumn{3}{c}{Ratio, $\rho = 0.1,\ 0.2,\ 0.3$} \\", r"\midrule"]
    for k, r in enumerate(rows):
        name = r["tranche"].replace("%", r"\%").replace("–", "--")
        lines.append(f"{name} & {pct(r['network'])} & {pct(r['vasicek'])} & "
                     + " & ".join(fmt(sens[q][k]) for q in ("0.1", "0.2", "0.3")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(RES, "tranche_table.tex"), "w").write("\n".join(lines) + "\n")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(wspace=0.3, width_ratios=[1.25, 1]))
    a1.plot(x, vas_pdf, color="#555555", ls="--", label=rf"Vasicek, $\rho = {RHO}$")
    a1.plot(x, net_pdf, color=TR_COL[0], label="network factor model")
    a1.set_yscale("log"); a1.set_ylim(1e-2, 1e3); a1.set_xlim(0, 0.45)
    for a, b, _ in TRANCHES[1:]:
        a1.axvline(a, color="0.75", lw=0.7)
    a1.set_xlabel("portfolio loss in one year")
    a1.set_ylabel("probability density")
    a1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a1.set_title(f"(a) Loss distributions, both with mean {mean_loss:.1%}", loc="left")
    a1.legend(loc="upper right")

    ratio = np.array([r["ratio"] for r in rows])
    names = [r["tranche"].replace(" ", "\n", 1) for r in rows]
    ypos = np.arange(len(rows))[::-1]
    a2.barh(ypos, ratio, color=TR_COL, height=0.62, edgecolor="white", linewidth=2)
    a2.set_xscale("log"); a2.axvline(1, color=INK, lw=0.9)
    a2.set_yticks(ypos); a2.set_yticklabels(names)
    a2.set_xlabel("expected loss, network ÷ Vasicek")
    a2.set_title("(b) Where the risk moves", loc="left")
    a2.grid(axis="y", visible=False)
    for yy, r in zip(ypos, ratio):
        a2.text(max(r, 1) * 1.25, yy, f"{r:,.1f}×" if r < 10 else f"{r:,.0f}×", va="center", fontsize=8.5, color=INK)
    a2.set_xlim(0.1, max(ratio) * 12)
    figstyle.save(fig, os.path.join(FIG, "fig3_equal_mean"))
    plt.close(fig)
    return rows, mean_loss, p_above


#-------- fig4: simulation against theory -------------------------------------------
def fig4(N=10_000, trials=12, seed=7):
    rng = np.random.default_rng(seed)
    out = {}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(wspace=0.28))
    nets = [("Poisson, $c = 4$", P.Degrees.poisson(4.0), TR_COL[0], "o"),
            (r"power law, $\gamma = 2.5$", P.Degrees.power_law(2.5, kmin=1, kmax=100), TR_COL[2], "s")]
    for name, d, col, mk in nets:
        Tf = np.linspace(0, 1, 400)
        a1.plot(Tf, [P.giant_component(d, t) for t in Tf], color=col, label=name + " (theory)")
        Ts = np.linspace(0.05, 1, 20)
        sim = np.array([[P.largest_cluster(d, N, t, rng) for _ in range(trials)] for t in Ts])
        a1.errorbar(Ts, sim.mean(1), yerr=sim.std(1), fmt=mk, ms=4.5, color=col, mfc="white",
                    mew=1.3, capsize=0, lw=1, label=name + f" (simulated, $N = 10^4$)")
        out[name] = dict(T=Ts.tolist(), mean=sim.mean(1).tolist(), sd=sim.std(1).tolist(), Tc=d.Tc)
    a1.set_xlabel(r"transmissibility $T$"); a1.set_ylabel("largest failure cluster")
    a1.set_title("(a) Percolation: simulation against theory", loc="left")
    a1.legend(loc="upper left", fontsize=8)
    a1.set_xlim(0, 1); a1.set_ylim(0, 1)

    phi = 0.18
    zs = np.arange(0.5, 9.01, 0.5)
    p_global, size_global = [], []
    for z in zs:
        d = P.Degrees.poisson(z)
        sizes = []
        for r in range(4):
            adj = P.adjacency(P.configuration_model(d, N, rng), N)
            sizes += [P.watts_cascade(adj, N, phi, rng) for _ in range(100)]
        sizes = np.array(sizes)
        big = sizes > 0.05
        p_global.append(big.mean())
        size_global.append(sizes[big].mean() if big.any() else 0.0)
    cond = np.array([P.watts_condition(P.Degrees.poisson(z), phi) for z in np.linspace(0.3, 9.5, 600)])
    zz = np.linspace(0.3, 9.5, 600)
    inside = cond > 1
    lo, hi = zz[inside].min(), zz[inside].max()
    a2.axvspan(lo, hi, color="0.93", lw=0, label="analytic cascade window")
    a2.plot(zs, p_global, "o-", color=TR_COL[0], ms=4.5, mfc="white", mew=1.3, label="P(global cascade)")
    a2.plot(zs, size_global, "s--", color=TR_COL[1], ms=4.5, mfc="white", mew=1.3, label="size of a global cascade")
    a2.set_xlabel(r"mean trading partners $z$"); a2.set_ylabel("fraction")
    a2.set_ylim(0, 1.05); a2.set_xlim(0, 9.5)
    a2.set_title(rf"(b) Watts threshold cascade, $\phi = {phi}$", loc="left")
    a2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    out["watts"] = dict(z=zs.tolist(), p_global=p_global, size_global=size_global, window=[lo, hi], phi=phi)
    json.dump(out, open(os.path.join(RES, "simulation_check.json"), "w"), indent=2)
    figstyle.save(fig, os.path.join(FIG, "fig4_simulation_check"))
    plt.close(fig)


#-------- cover: a cascade spreading through a supply network (GIF) -----------------
def cover(n=420, seed=11):
    import networkx as nx
    from matplotlib.animation import FuncAnimation, PillowWriter
    rng = np.random.default_rng(seed)
    G = nx.random_geometric_graph(n, 0.085, seed=seed)    # spatial layout reads better
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    pos = nx.get_node_attributes(G, "pos")
    nodes = list(G.nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    xy = np.array([pos[v] for v in nodes])
    T = 0.62
    kept = [(u, v) for u, v in G.edges if rng.random() < T]
    H = nx.Graph(); H.add_nodes_from(nodes); H.add_edges_from(kept)
    s0 = nodes[int(np.argmin(((xy - [0.5, 0.5]) ** 2).sum(1)))]
    gen = nx.single_source_shortest_path_length(H, s0)
    gmax = max(gen.values())
    frames = list(range(gmax + 1)) + [gmax] * 12
    curve = [sum(1 for g in gen.values() if g <= k) / len(nodes) for k in range(gmax + 1)]

    fig = plt.figure(figsize=(12, 6), dpi=110)
    ax = fig.add_axes([0.02, 0.04, 0.48, 0.90]); ax.axis("off")
    ax2 = fig.add_axes([0.58, 0.18, 0.38, 0.62])
    seg = np.array([[pos[u], pos[v]] for u, v in G.edges])
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(seg, colors="0.85", linewidths=0.6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=16, c=["#b8c4d6"] * len(nodes), edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    title = fig.text(0.26, 0.955, "", ha="center", fontsize=13, color=INK)
    ax2.set_xlim(0, gmax); ax2.set_ylim(0, 1)
    ax2.set_xlabel("steps after the first failure"); ax2.set_ylabel("share of firms failed")
    ax2.set_title("one supplier fails; above the threshold the cascade spans the network", fontsize=10.5)
    line, = ax2.plot([], [], color="#0072B2", lw=2.2)
    cmap = sns.color_palette("flare", as_cmap=True)

    def draw(k):
        cols = []
        for v in nodes:
            g = gen.get(v)
            cols.append(cmap(min(g / max(gmax, 1), 1.0)) if (g is not None and g <= k) else "#b8c4d6")
        sc.set_color(cols); sc.set_edgecolor("white")
        line.set_data(range(k + 1), curve[:k + 1])
        title.set_text(f"step {k:2d}    failed {curve[k]:.0%}")
        return sc, line, title

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(os.path.join(FIG, "cover.gif"), writer=PillowWriter(fps=6))
    draw(gmax); fig.savefig(os.path.join(FIG, "cover.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "cover"]
    for w in which:
        print(w, globals()[w]() or "")
