"""Finite-difference solution of the Black–Scholes PDE.

With x = ln S and tau = T - t (time to expiry), the Black–Scholes PDE

    dV/dt + 1/2 sigma^2 S^2 d2V/dS2 + (r - q) S dV/dS - r V = 0

becomes a constant-coefficient convection–diffusion equation

    dV/dtau = 1/2 sigma^2 d2V/dx2 + (r - q - 1/2 sigma^2) dV/dx - r V,

with the payoff as the initial condition at tau = 0. Working in x keeps the
coefficients constant, so the grid is uniform and the same tridiagonal matrix is
used at every step.

Time stepping uses the theta-scheme:

    theta = 0    explicit Euler    (first order in time, conditionally stable)
    theta = 1    implicit Euler    (first order in time, unconditionally stable)
    theta = 1/2  Crank–Nicolson    (second order in time, unconditionally stable)

Crank–Nicolson can produce spurious oscillations near the kink in the payoff,
so by default the first few steps use implicit Euler ("Rannacher smoothing"),
which damps the high-frequency error and restores second-order convergence.
"""

import numpy as np
from scipy.linalg import solve_banded


def price(S, K, T, r, sigma, q=0.0, kind="call", n_space=400, n_time=400,
          theta=0.5, n_std=5.0, rannacher_steps=2):
    """Price a European option by solving the Black–Scholes PDE on a grid.

    n_space         number of intervals in x = ln S
    n_time          number of time steps
    n_std           half-width of the grid in standard deviations of ln S_T
    rannacher_steps implicit Euler steps at the start (only used if theta < 1)
    """
    S_grid, V, _ = solve(S, K, T, r, sigma, q, kind, n_space, n_time, theta, n_std,
                         rannacher_steps)
    return float(V[len(V) // 2])


def solve(S, K, T, r, sigma, q=0.0, kind="call", n_space=400, n_time=400,
          theta=0.5, n_std=5.0, rannacher_steps=2, keep_history=False):
    """Run the scheme and return (S_grid, V at tau = T, history).

    The spot S sits on the middle node of S_grid. If keep_history is True,
    history is a list holding the solution after every time step; otherwise None.
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")

    # Uniform grid in x centred on ln S, so the spot lies exactly on a node.
    # An even number of intervals is needed for the centre to be a node.
    n_space += n_space % 2
    x0 = np.log(S)
    half_width = n_std * sigma * np.sqrt(T)
    x = np.linspace(x0 - half_width, x0 + half_width, n_space + 1)
    dx = x[1] - x[0]
    dtau = T / n_time
    S_grid = np.exp(x)

    V = _cell_averaged_payoff(x, dx, K, kind)

    # Coefficients of the spatial operator L V_j = a V_{j-1} + b V_j + c V_{j+1}.
    diff = 0.5 * sigma**2 / dx**2
    conv = (r - q - 0.5 * sigma**2) / (2 * dx)
    a = diff - conv
    b = -2 * diff - r
    c = diff + conv

    def boundary(tau):
        """Dirichlet values at the lowest and highest grid points."""
        far_S = S_grid[[0, -1]] * np.exp(-q * tau)
        disc_K = K * np.exp(-r * tau)
        if kind == "call":
            return 0.0, far_S[1] - disc_K
        return disc_K - far_S[0], 0.0

    m = n_space - 1  # number of interior nodes

    def step(V, tau_old, th):
        tau_new = tau_old + dtau
        # Banded form of (I - th*dtau*L) for solve_banded.
        ab = np.zeros((3, m))
        ab[0, 1:] = -th * dtau * c
        ab[1, :] = 1 - th * dtau * b
        ab[2, :-1] = -th * dtau * a

        # Right-hand side (I + (1-th)*dtau*L) V_old on the interior.
        LV = a * V[:-2] + b * V[1:-1] + c * V[2:]
        rhs = V[1:-1] + (1 - th) * dtau * LV

        lo_new, hi_new = boundary(tau_new)
        rhs[0] += th * dtau * a * lo_new
        rhs[-1] += th * dtau * c * hi_new

        V_new = np.empty_like(V)
        V_new[1:-1] = solve_banded((1, 1), ab, rhs)
        V_new[0], V_new[-1] = lo_new, hi_new
        return V_new

    history = [V] if keep_history else None
    tau = 0.0
    for n in range(n_time):
        th = 1.0 if (0 < theta < 1 and n < rannacher_steps) else theta
        V = step(V, tau, th)
        tau += dtau
        if keep_history:
            history.append(V)

    return S_grid, V, history


def _cell_averaged_payoff(x, dx, K, kind):
    """Payoff averaged over [x_j - dx/2, x_j + dx/2] instead of sampled at x_j.

    Sampling the payoff pointwise makes the error depend on where the strike
    happens to fall between two nodes, so convergence is erratic. Averaging over
    each cell (done exactly here, since e^x integrates in closed form) removes
    that dependence and gives clean second-order convergence.
    """
    lo, hi = x - dx / 2, x + dx / 2
    k = np.log(K)
    if kind == "call":
        a = np.clip(k, lo, hi)  # payoff is positive on [a, hi]
        integral = (np.exp(hi) - np.exp(a)) - K * (hi - a)
    else:
        b = np.clip(k, lo, hi)  # payoff is positive on [lo, b]
        integral = K * (b - lo) - (np.exp(b) - np.exp(lo))
    return integral / dx
