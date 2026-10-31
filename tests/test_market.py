"""Checks for the market-data pipeline and the SVI fit, using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from optpricing import black_scholes as bs
from optpricing import market, svi


def test_tbill_conversion():
    # A 4% discount yield on a 91-day bill is a little above 4% continuously compounded.
    r = market.tbill_to_continuous(0.04)
    assert 0.040 < r < 0.041


def test_svi_recovers_known_parameters():
    true = (0.01, 0.1, -0.6, 0.02, 0.15)
    T = 0.5
    k = np.linspace(-0.4, 0.2, 40)
    vol = svi.implied_vol(k, T, true)
    params, rmse = svi.fit(k, vol, T)
    assert rmse < 1e-4
    np.testing.assert_allclose(svi.implied_vol(k, T, params), vol, atol=1e-4)


def _synthetic_chain(S=5000.0, r=0.04, q=0.012, sigma=0.18):
    """A clean chain priced by Black–Scholes, in the same format as fetch_data.py."""
    rows = []
    for expiry in ("2026-12-18", "2027-03-19", "2027-06-17"):
        T = (pd.Timestamp(expiry + " 21:00", tz="UTC")
             - pd.Timestamp("2026-10-30 20:00", tz="UTC")).total_seconds() / (365 * 86400)
        for K in np.arange(4000, 6001, 50.0):
            for kind in ("call", "put"):
                rows.append({
                    "contractSymbol": f"SPXW{expiry[2:4]}{expiry[5:7]}{expiry[8:]}"
                                      f"{kind[0].upper()}{int(K * 1000):08d}",
                    "lastTradeDate": "2026-10-30 19:30:00+00:00",
                    "strike": K, "lastPrice": float(bs.price(S, K, T, r, sigma, q, kind)),
                    "kind": kind, "expiry": expiry, "spot": S,
                    "spot_date": "2026-10-30", "tbill_yield": 0.0,
                })
    return pd.DataFrame(rows)


def test_pipeline_recovers_dividend_yield_and_flat_vol():
    S, r, q, sigma = 5000.0, 0.04, 0.012, 0.18
    chain = market.prepare_chain(_synthetic_chain(S, r, q, sigma))
    forwards = market.parity_forwards(chain, r)
    carry = market.pooled_carry(forwards, S)
    assert r - carry == pytest.approx(q, abs=1e-6)

    ivs = market.implied_vols(chain, r, r - carry)
    assert len(ivs) > 50
    np.testing.assert_allclose(ivs["iv"], sigma, atol=1e-5)


def test_stale_trades_are_flagged():
    raw = _synthetic_chain()
    raw.loc[:9, "lastTradeDate"] = "2026-10-29 15:00:00+00:00"
    chain = market.prepare_chain(raw)
    assert (~chain["fresh"]).sum() == 10
