"""Closed-form Black–Scholes prices and Greeks for European options.

All functions accept NumPy arrays and broadcast, so a whole option chain can be
priced in one call. Notation follows the README:

    S      spot price
    K      strike
    T      time to expiry in years
    r      continuously compounded risk-free rate
    sigma  volatility (annualised)
    q      continuous dividend yield
    kind   "call" or "put"
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def price(S, K, T, r, sigma, q=0.0, kind="call"):
    """Black–Scholes price of a European call or put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    if kind == "call":
        return disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    if kind == "put":
        return disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def delta(S, K, T, r, sigma, q=0.0, kind="call"):
    """dV/dS."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return -np.exp(-q * T) * norm.cdf(-d1)


def gamma(S, K, T, r, sigma, q=0.0):
    """d^2V/dS^2 (identical for calls and puts)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """dV/dsigma (identical for calls and puts). Used by Newton's method for implied vol."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, q=0.0, kind="call"):
    """dV/dt (calendar time), i.e. minus the derivative with respect to T."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    decay = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if kind == "call":
        return (decay - r * K * np.exp(-r * T) * norm.cdf(d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1))
    return (decay + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1))
