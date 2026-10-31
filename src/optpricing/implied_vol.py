"""Implied volatility: invert the Black–Scholes formula for sigma.

Given a market price P, the implied volatility is the sigma that solves

    f(sigma) = BS(sigma) - P = 0.

The Black–Scholes price is strictly increasing in sigma (vega > 0), so the root is
unique when it exists. Newton's method uses f'(sigma) = vega and converges
quadratically near the root. Far from the money vega is tiny and Newton steps
can overshoot, so we keep a bracket [lo, hi] and fall back to bisection whenever
a Newton step would leave it. This is a safeguarded Newton method.
"""

import numpy as np

from . import black_scholes as bs


def implied_vol(P, S, K, T, r, q=0.0, kind="call", tol=1e-8, max_iter=100,
                lo=1e-4, hi=5.0):
    """Implied volatility for a single option, or NaN if the price is out of bounds."""
    # No-arbitrage bounds: a price outside them has no implied volatility.
    disc_S, disc_K = S * np.exp(-q * T), K * np.exp(-r * T)
    if kind == "call":
        lower, upper = max(disc_S - disc_K, 0.0), disc_S
    else:
        lower, upper = max(disc_K - disc_S, 0.0), disc_K
    if not (lower < P < upper):
        return np.nan

    def f(sig):
        return bs.price(S, K, T, r, sig, q, kind) - P

    if f(lo) > 0 or f(hi) < 0:
        return np.nan

    # Starting guess from Brenner–Subrahmanyam's at-the-money approximation.
    sigma = np.clip(np.sqrt(2 * np.pi / T) * P / S, lo, hi)
    for _ in range(max_iter):
        err = f(sigma)
        if abs(err) < tol:
            return float(sigma)
        if err > 0:
            hi = sigma
        else:
            lo = sigma
        v = bs.vega(S, K, T, r, sigma, q)
        newton = sigma - err / v if v > 1e-12 else np.nan
        sigma = newton if lo < newton < hi else 0.5 * (lo + hi)
    return float(sigma)
