"""Gatheral's SVI ("stochastic volatility inspired") parameterisation of a smile.

For one expiry, the total implied variance w(k) = sigma_imp(k)^2 * T as a function
of log-moneyness k = ln(K/F) is modelled as

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + s^2))

Five parameters: a (level), b (slope of the wings), rho (skew, -1 < rho < 1),
m (horizontal shift) and s (curvature at the minimum). The function is a
hyperbola, so the wings are asymptotically linear in k, which matches Lee's
moment formula for how fast implied variance can grow in the tails.

We fit it by robust non-linear least squares (a soft-L1 loss, so a few stale
trades don't drag the curve), weighting points by 1 / (implied vol), which
makes the fit roughly uniform in volatility rather than variance units.
"""

import numpy as np
from scipy.optimize import least_squares


def total_variance(k, a, b, rho, m, s):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))


def fit(k, iv, T):
    """Fit SVI to one expiry. Returns (params, rmse in vol points)."""
    k, iv = np.asarray(k), np.asarray(iv)
    w = iv**2 * T

    def residuals(p):
        return (total_variance(k, *p) - w) / (2 * iv * T)  # ≈ error in vol units

    x0 = [0.5 * w.min(), 0.1, -0.5, 0.0, 0.1]
    lower = [-1.0, 1e-6, -0.999, -1.0, 1e-4]
    upper = [1.0, 5.0, 0.999, 1.0, 2.0]
    res = least_squares(residuals, x0, bounds=(lower, upper), loss="soft_l1", f_scale=0.01)

    params = res.x
    fitted_iv = np.sqrt(np.maximum(total_variance(k, *params), 0) / T)
    rmse = float(np.sqrt(np.mean((fitted_iv - iv) ** 2)))
    return params, rmse


def implied_vol(k, T, params):
    return np.sqrt(np.maximum(total_variance(k, *params), 0) / T)
