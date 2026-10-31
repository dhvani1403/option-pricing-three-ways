"""Checks that the three pricing methods are correct and agree with each other."""

import numpy as np
import pytest

from optpricing import black_scholes as bs
from optpricing import finite_difference as fd
from optpricing import implied_vol as iv
from optpricing import monte_carlo as mc

HULL = dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20)
BASE = dict(S=100.0, K=100.0, T=1.0, r=0.04, sigma=0.20, q=0.01)


# Black–Scholes formula --------------------------------------------------------

def test_textbook_values():
    # Worked example in Hull, "Options, Futures, and Other Derivatives": c = 4.76, p = 0.81.
    assert bs.price(**HULL, kind="call") == pytest.approx(4.76, abs=5e-3)
    assert bs.price(**HULL, kind="put") == pytest.approx(0.81, abs=5e-3)


@pytest.mark.parametrize("K", [70.0, 100.0, 130.0])
@pytest.mark.parametrize("T", [0.1, 1.0, 3.0])
def test_put_call_parity(K, T):
    p = dict(BASE, K=K, T=T)
    lhs = bs.price(**p, kind="call") - bs.price(**p, kind="put")
    rhs = p["S"] * np.exp(-p["q"] * T) - K * np.exp(-p["r"] * T)
    assert lhs == pytest.approx(rhs, abs=1e-12)


def test_price_bounds():
    K = np.linspace(50, 150, 21)
    c = bs.price(**dict(BASE, K=K), kind="call")
    disc_S = BASE["S"] * np.exp(-BASE["q"] * BASE["T"])
    assert np.all(c <= disc_S)
    assert np.all(c >= np.maximum(disc_S - K * np.exp(-BASE["r"] * BASE["T"]), 0))
    assert np.all(np.diff(c) < 0)  # calls get cheaper as the strike rises


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    h = 1e-4
    up, dn = dict(BASE, S=BASE["S"] + h), dict(BASE, S=BASE["S"] - h)
    V, Vu, Vd = (bs.price(**p, kind=kind) for p in (BASE, up, dn))
    assert bs.delta(**BASE, kind=kind) == pytest.approx((Vu - Vd) / (2 * h), rel=1e-6)
    assert bs.gamma(**BASE) == pytest.approx((Vu - 2 * V + Vd) / h**2, rel=1e-3)

    vu = bs.price(**dict(BASE, sigma=BASE["sigma"] + h), kind=kind)
    vd = bs.price(**dict(BASE, sigma=BASE["sigma"] - h), kind=kind)
    assert bs.vega(**BASE) == pytest.approx((vu - vd) / (2 * h), rel=1e-6)

    tu = bs.price(**dict(BASE, T=BASE["T"] + h), kind=kind)
    td = bs.price(**dict(BASE, T=BASE["T"] - h), kind=kind)
    assert bs.theta(**BASE, kind=kind) == pytest.approx(-(tu - td) / (2 * h), rel=1e-6)


def test_black_scholes_pde_is_satisfied():
    """theta + 1/2 sigma^2 S^2 gamma + (r - q) S delta - r V = 0."""
    s, r, q = BASE["S"], BASE["r"], BASE["q"]
    for kind in ("call", "put"):
        residual = (bs.theta(**BASE, kind=kind)
                    + 0.5 * BASE["sigma"] ** 2 * s**2 * bs.gamma(**BASE)
                    + (r - q) * s * bs.delta(**BASE, kind=kind)
                    - r * bs.price(**BASE, kind=kind))
        assert abs(residual) < 1e-10


# Monte Carlo ------------------------------------------------------------------

@pytest.mark.parametrize("opts", [{}, {"antithetic": True},
                                  {"antithetic": True, "control_variate": True}])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_monte_carlo_within_four_standard_errors(opts, kind):
    exact = bs.price(**BASE, kind=kind)
    res = mc.price(**BASE, kind=kind, n_paths=200_000, seed=42, **opts)
    assert abs(res.price - exact) < 4 * res.std_error


def test_variance_reduction_reduces_standard_error():
    plain = mc.price(**BASE, n_paths=200_000, seed=1)
    anti = mc.price(**BASE, n_paths=200_000, seed=1, antithetic=True)
    both = mc.price(**BASE, n_paths=200_000, seed=1, antithetic=True, control_variate=True)
    assert both.std_error < anti.std_error < plain.std_error


def test_ci_coverage():
    """About 95% of 95% confidence intervals should contain the true price."""
    exact = bs.price(**BASE)
    hits = 0
    for seed in range(400):
        lo, hi = mc.price(**BASE, n_paths=2_000, seed=seed).ci95
        hits += lo <= exact <= hi
    assert 0.92 <= hits / 400 <= 0.98


# Finite differences -----------------------------------------------------------

@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_crank_nicolson_matches_formula(kind, K):
    p = dict(BASE, K=K)
    assert fd.price(**p, kind=kind) == pytest.approx(bs.price(**p, kind=kind), abs=1e-3)


def test_crank_nicolson_is_second_order():
    exact = bs.price(**BASE)
    e1 = abs(fd.price(**BASE, n_space=200, n_time=200) - exact)
    e2 = abs(fd.price(**BASE, n_space=400, n_time=400) - exact)
    assert 3.5 < e1 / e2 < 4.5  # halving the step should cut the error by ~4


def test_implicit_euler_is_first_order():
    exact = bs.price(**BASE)
    e1 = abs(fd.price(**BASE, n_space=200, n_time=200, theta=1.0) - exact)
    e2 = abs(fd.price(**BASE, n_space=400, n_time=400, theta=1.0) - exact)
    assert 1.7 < e1 / e2 < 2.3


def test_odd_grid_is_handled():
    exact = bs.price(**BASE)
    assert fd.price(**BASE, n_space=201, n_time=200) == pytest.approx(exact, abs=1e-3)


# Implied volatility -----------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6, 1.5])
@pytest.mark.parametrize("K", [60.0, 100.0, 160.0])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_vol_round_trip(sigma, K, kind):
    p = dict(BASE, K=K, sigma=sigma)
    P = float(bs.price(**p, kind=kind))
    # Implied vol is only identifiable from the time value (price above its no-arbitrage
    # lower bound). Deep in or out of the money at low vol, that is below rounding error.
    fwd = BASE["S"] * np.exp(-BASE["q"] * BASE["T"]) - K * np.exp(-BASE["r"] * BASE["T"])
    lower_bound = max(fwd, 0.0) if kind == "call" else max(-fwd, 0.0)
    if P - lower_bound < 1e-6:
        pytest.skip("no time value, so implied vol is not identifiable")
    got = iv.implied_vol(P, BASE["S"], K, BASE["T"], BASE["r"], BASE["q"], kind)
    assert got == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_rejects_arbitrage_prices():
    # A call can never be worth more than the (dividend-discounted) stock.
    assert np.isnan(iv.implied_vol(200.0, 100.0, 100.0, 1.0, 0.04, 0.01, "call"))
    # Nor less than its intrinsic forward value.
    assert np.isnan(iv.implied_vol(0.0, 100.0, 50.0, 1.0, 0.04, 0.01, "call"))
