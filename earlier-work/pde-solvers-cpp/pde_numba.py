#!/usr/bin/env python3
#-------- pde_numba.py ----------------------------------------------------------------
#-----------------------------------------------------------------------------
# Python + Numba versions of the C++ PDE solvers in this folder, for plotting.
#
#   laplace      a line-for-line port of "Dirichlet Boundary conditions Successive
#                Over Relaxtion Method.cpp": Laplace's equation on the unit square,
#                phi = x on the top edge, phi = y on the right, 0 on the others, a
#                box held at phi = 1 and a line held at phi = 0 inside the domain.
#   black-scholes  new: the Black-Scholes PDE solved the same way - Crank-Nicolson in
#                time, SOR for each time step's linear system, and projected SOR
#                (never below the exercise value) for American options.
#
# Numba compiles the loops to machine code, so the Python runs at C++-like speed.

import numpy as np
from numba import njit


#-------- Laplace, as in the C++ -----------------------------------------------------
def laplace_problem(N=253):
    phi = np.zeros((N, N))
    boundary = np.zeros((N, N), dtype=np.bool_)
    s = np.arange(N) / (N - 1)
    phi[:, N - 1] = s; boundary[:, N - 1] = True        # top side = x
    phi[N - 1, :] = s; boundary[N - 1, :] = True        # right side = y
    phi[:, 0] = 0.0; boundary[:, 0] = True              # bottom = 0
    phi[0, :] = 0.0; boundary[0, :] = True              # left = 0
    bx1, bx2, by1, by2 = 2 * (N - 1) // 10, 4 * (N - 1) // 10, 7 * (N - 1) // 10, 9 * (N - 1) // 10
    for j in range(by1, by2 + 1):                       # box A, phi = 1
        phi[bx1, j] = phi[bx2, j] = 1.0
        boundary[bx1, j] = boundary[bx2, j] = True
    for i in range(bx1, bx2 + 1):
        phi[i, by1] = phi[i, by2] = 1.0
        boundary[i, by1] = boundary[i, by2] = True
    lx, ly1, ly2 = 8 * (N - 1) // 10, 1 * (N - 1) // 10, 6 * (N - 1) // 10
    for j in range(ly1, ly2 + 1):                       # line B, phi = 0
        phi[lx, j] = 0.0
        boundary[lx, j] = True
    return phi, boundary


@njit(cache=True)
def sor_laplace(phi, boundary, omega, tol=1e-7, max_iter=1_000_000):
    """In place. Returns the number of sweeps to converge (max change < tol)."""
    N = phi.shape[0]
    iterations = 0
    delta_max = 1.0
    while delta_max >= tol and iterations < max_iter:
        delta_max = 0.0
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                if not boundary[i, j]:
                    new = (1 - omega) * phi[i, j] + omega * 0.25 * (phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1])
                    d = abs(new - phi[i, j])
                    if d > delta_max:
                        delta_max = d
                    phi[i, j] = new
        iterations += 1
    return iterations


def derivative_value(phi):
    """d(phi)/dy at (2/5, 1/2), central difference - as the C++ prints."""
    N = phi.shape[0]
    i, j = 2 * (N - 1) // 5, (N - 1) // 2
    return (phi[i, j + 1] - phi[i, j - 1]) * (N - 1) / 2


def optimal_omega(N):
    """Theoretical optimum for SOR on an N x N Laplace grid."""
    return 2 / (1 + np.sin(np.pi / (N - 1)))


#-------- Black-Scholes, the same way -------------------------------------------------
# V(S, t) for an option on S. Stepping backwards from expiry, Crank-Nicolson turns
# each time step into a tridiagonal system, solved here by SOR - the same iteration
# as the Laplace solver. For an American option, after each SOR update the value is
# projected up to the exercise value: you can always exercise, so it is never worth
# less. That projection traces out the early-exercise boundary for free.
@njit(cache=True)
def _cn_sor(V, payoff, a, b, c, omega, tol, american, lower_bc_value, upper_bc_value):
    #Crank-Nicolson:  -a x[i-1] + (1+b) x[i] - c x[i+1]  =  a V[i-1] + (1-b) V[i] + c V[i+1]
    M = V.shape[0]
    rhs = np.empty(M)
    for i in range(1, M - 1):
        rhs[i] = a[i] * V[i - 1] + (1 - b[i]) * V[i] + c[i] * V[i + 1]
    x = V.copy()
    x[0] = lower_bc_value                   #boundary values at the new time level
    x[M - 1] = upper_bc_value
    err = 1.0
    sweeps = 0
    while err > tol and sweeps < 100_000:
        err = 0.0
        for i in range(1, M - 1):
            gs = (rhs[i] + a[i] * x[i - 1] + c[i] * x[i + 1]) / (1 + b[i])
            new = x[i] + omega * (gs - x[i])
            if american and new < payoff[i]:
                new = payoff[i]
            d = abs(new - x[i])
            if d > err:
                err = d
            x[i] = new
        sweeps += 1
    return x, sweeps


def black_scholes_pde(K=100.0, T=1.0, r=0.05, sigma=0.2, kind="put", american=True,
                      S_max=300.0, M=301, steps=200, omega=1.2, tol=1e-8):
    """Returns S grid, time grid (time to expiry), V[t, S], early-exercise boundary, total sweeps."""
    S = np.linspace(0, S_max, M)
    dt = T / steps
    i = np.arange(M, dtype=float)
    #discretised operator, scaled by dt/2 for Crank-Nicolson
    a = 0.25 * dt * (sigma ** 2 * i ** 2 - r * i)
    b = 0.5 * dt * (sigma ** 2 * i ** 2 + r)
    c = 0.25 * dt * (sigma ** 2 * i ** 2 + r * i)
    payoff = np.maximum(K - S, 0) if kind == "put" else np.maximum(S - K, 0)
    V = payoff.copy()
    grid = np.empty((steps + 1, M)); grid[0] = V
    boundary_S, total = [np.nan], 0
    for n in range(1, steps + 1):
        tau = n * dt
        upper = 0.0 if kind == "put" else S_max - K * np.exp(-r * tau)
        lower = (K if american else K * np.exp(-r * tau)) if kind == "put" else 0.0
        V, sweeps = _cn_sor(V, payoff, a, b, c, omega, tol, american, lower, upper)
        total += sweeps
        grid[n] = V
        if american and kind == "put":
            exercised = np.flatnonzero((V <= payoff + 1e-9) & (S < K))
            boundary_S.append(S[exercised.max()] if len(exercised) else np.nan)
    return S, np.linspace(0, T, steps + 1), grid, np.array(boundary_S), total


def black_scholes_closed_form(S, K, T, r, sigma, kind="call"):
    from scipy.stats import norm
    S = np.asarray(S, dtype=float)
    with np.errstate(divide="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call if kind == "call" else call - S + K * np.exp(-r * T)
