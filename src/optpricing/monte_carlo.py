"""Monte Carlo pricing of European options under geometric Brownian motion.

Because a European payoff depends only on S_T, there is no need to simulate a
whole path: under the risk-neutral measure

    S_T = S_0 exp((r - q - sigma^2 / 2) T + sigma sqrt(T) Z),   Z ~ N(0, 1)

is exact. The price is the discounted average payoff, and the central limit
theorem gives a standard error that shrinks like 1 / sqrt(N).

Two variance-reduction techniques are available:

* Antithetic variates: pair every Z with -Z. The pair is negatively
  correlated, so its average has lower variance than two independent draws.
* Control variate: the discounted terminal price e^{-rT} S_T has known mean
  S_0 e^{-qT}. Its correlation with the payoff lets us cancel part of the noise.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self):
        half = 1.96 * self.std_error
        return self.price - half, self.price + half


def price(S, K, T, r, sigma, q=0.0, kind="call", n_paths=100_000,
          antithetic=False, control_variate=False, seed=None):
    """Monte Carlo estimate of a European option price with its standard error."""
    rng = np.random.default_rng(seed)

    if antithetic:
        half = rng.standard_normal(n_paths // 2)
        Z = np.concatenate([half, -half])
    else:
        Z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma**2) * T
    S_T = S * np.exp(drift + sigma * np.sqrt(T) * Z)

    if kind == "call":
        payoff = np.maximum(S_T - K, 0.0)
    elif kind == "put":
        payoff = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")

    discount = np.exp(-r * T)
    Y = discount * payoff

    if control_variate:
        X = discount * S_T
        X_mean = S * np.exp(-q * T)
        cov = np.cov(Y, X)
        beta = cov[0, 1] / cov[1, 1]
        Y = Y - beta * (X - X_mean)

    if antithetic:
        # Antithetic pairs are not independent, so average each pair first and
        # compute the standard error over the independent pair means.
        m = len(Y) // 2
        samples = 0.5 * (Y[:m] + Y[m:])
    else:
        samples = Y

    est = samples.mean()
    se = samples.std(ddof=1) / np.sqrt(len(samples))
    return MCResult(price=float(est), std_error=float(se), n_paths=len(Z))
