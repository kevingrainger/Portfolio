#!/usr/bin/env python3
#-------- corrlib : plotting.py ----------------------------------------------
#-----------------------------------------------------------------------------
# Seeing the matrix.
#
# A correlation heatmap in whatever order the data arrived looks like static,
# even when there is strong structure in it. Re-order the rows so that things
# that move together sit next to each other, and the structure appears as
# squares down the diagonal - sectors, regions, groups of stations.
#
# The ordering comes from hierarchical clustering, with the distance between two
# variables taken as
#
#       d = sqrt(2 (1 - rho))
#
# which is a proper distance - zero for perfectly correlated, 2 for perfectly
# opposite - so clustering on it is well defined (Mantegna 1999).
#
# Every function takes an optional ax, so the figures compose; summary() puts
# the four standard panels together for a fitted Correlator.

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram as _dendrogram
from scipy.spatial.distance import squareform

from . import random_matrix_cleaning as rmt


def _linkage(C, method="average"):
    C = np.asarray(C, dtype=float)
    D = np.sqrt(np.clip(2 * (1 - C), 0, None))
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)                                 # exact symmetry, squareform insists
    return linkage(squareform(D, checks=False), method=method)


# Row order that puts similar variables next to each other.
def cluster_order(C, method="average"):
    return leaves_list(_linkage(C, method))


#-------- Heatmap --------------------------------------------------------------
#-----------------------------------------------------------------------------
# order="cluster" re-orders by clustering, order=None keeps the input order, or
# pass an explicit order (e.g. from another matrix, so two heatmaps line up).
def heatmap(C, labels=None, order="cluster", ax=None, title=None, vmax=1.0):
    C = np.asarray(C, dtype=float)
    if isinstance(order, str) and order == "cluster":
        order = cluster_order(C)
    elif order is None:
        order = np.arange(C.shape[0])
    order = np.asarray(order)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C[np.ix_(order, order)], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    if labels is not None and len(labels) <= 60:
        names = [labels[i] for i in order]
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    ax.figure.colorbar(im, ax=ax, shrink=0.75)
    return ax


#-------- Dendrogram -----------------------------------------------------------
#-----------------------------------------------------------------------------
# The cluster tree itself - which groups merge first, and at what distance.
def dendrogram(C, labels=None, ax=None, method="average"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    _dendrogram(_linkage(C, method), labels=labels, ax=ax, leaf_rotation=90,
                leaf_font_size=7, color_threshold=None)
    ax.set_ylabel('distance  sqrt(2(1 - rho))')
    ax.set_title('Which variables group together', fontsize=10)
    return ax


#-------- Spectrum -------------------------------------------------------------
#-----------------------------------------------------------------------------
# Eigenvalues against the noise band. Anything past the edge is structure.
def spectrum(C, n_observations, n_removed=0, ax=None, title="Eigenvalues vs pure noise"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    rmt.plot_spectrum(C, n_observations, title=title, n_removed=n_removed, ax=ax)
    return ax


#-------- Side by side ---------------------------------------------------------
#-----------------------------------------------------------------------------
# Several matrices, one shared ordering (from the first) and one colour scale,
# so differences are differences in the matrix and not in the layout.
def compare(matrices, titles, labels=None, order="cluster"):
    first = np.asarray(matrices[0], dtype=float)
    if isinstance(order, str) and order == "cluster":
        order = cluster_order(first)
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.2 * len(matrices), 4.6))
    axes = np.atleast_1d(axes)
    for ax, M, title in zip(axes, matrices, titles):
        heatmap(M, labels=labels, order=order, ax=ax,
                title=f"{title}\neffective rank {rmt.effective_rank(M):.1f}")
    fig.tight_layout()
    return fig


#-------- Everything for a fitted Correlator -----------------------------------
#-----------------------------------------------------------------------------
def summary(correlator):
    c = correlator
    order = cluster_order(c.measured_)
    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])

    heatmap(c.measured_, c.labels_, order=order, ax=fig.add_subplot(grid[0, 0]),
            title=f"measured ({c.measure}), nothing removed")
    heatmap(c.matrix_, c.labels_, order=order, ax=fig.add_subplot(grid[0, 1]),
            title=f"final: removed {c.describe_removal()}, cleaned with {c.clean}")
    spectrum(c.residual_, c.n_effective_, n_removed=c.n_removed_,
             ax=fig.add_subplot(grid[1, 0]), title="after removal, before cleaning")
    dendrogram(c.cleaned_, c.labels_, ax=fig.add_subplot(grid[1, 1]))
    fig.tight_layout()
    return fig
