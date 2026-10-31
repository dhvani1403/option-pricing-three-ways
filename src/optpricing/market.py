"""Clean a raw option-chain snapshot and turn it into implied volatilities.

Steps (each explained in the README, section 6):

1. Time to expiry. SPXW options settle at the 4:00 pm ET close on the expiry
   date; standard monthly SPX options settle at the 9:30 am ET open.
2. Prices. The snapshot was taken over a weekend, when the quoted bid and ask
   are zero, so we use the last traded price and keep only trades from the final
   hour of the session, so that the price is close to the closing spot.
3. Interest rate. The 13-week T-bill yield is quoted on a discount basis; we
   convert it to a continuously compounded rate.
4. Forward and dividend yield. Put–call parity, C - P = e^{-rT}(F - K), gives
   one forward estimate per strike. We pool these across expiries to estimate a
   single carry rate r - q, which is far less noisy than per-expiry estimates.
5. Implied volatility of out-of-the-money options only (puts below the forward,
   calls above), because these are the more actively traded side.
"""

import numpy as np
import pandas as pd

from .implied_vol import implied_vol

VALUATION_CLOSE_UTC = "20:00"  # 4:00 pm ET during US daylight saving time
LAST_HOUR_START_UTC = "19:00"


def tbill_to_continuous(discount_yield, days=91):
    """Convert a T-bill discount yield to a continuously compounded rate."""
    price = 1 - discount_yield * days / 360
    return -np.log(price) / (days / 365)


def prepare_chain(raw):
    """Add root, settlement time, time to expiry and a freshness filter."""
    d = raw.copy()
    d["lastTradeDate"] = pd.to_datetime(d["lastTradeDate"], utc=True)
    d["root"] = d["contractSymbol"].str.extract(r"^([A-Z]+)\d")[0]

    spot_date = d["spot_date"].iloc[0]
    valuation = pd.Timestamp(f"{spot_date} {VALUATION_CLOSE_UTC}", tz="UTC")
    # Expiry dates are after the end of daylight saving (Nov 1), so ET = UTC-5.
    settle_utc = np.where(d["root"] == "SPXW", " 21:00", " 14:30")
    d["settlement"] = pd.to_datetime(d["expiry"] + settle_utc, utc=True)
    d["T"] = (d["settlement"] - valuation).dt.total_seconds() / (365 * 24 * 3600)

    last_hour = pd.Timestamp(f"{spot_date} {LAST_HOUR_START_UTC}", tz="UTC")
    d["fresh"] = d["lastTradeDate"] >= last_hour

    # Where both roots exist for an expiry, keep the PM-settled weekly (SPXW).
    has_w = d.groupby("expiry")["root"].transform(lambda s: (s == "SPXW").any())
    d = d[~(has_w & (d["root"] == "SPX"))]
    return d.reset_index(drop=True)


def parity_forwards(d, r, band=0.05, min_pairs=3):
    """Per-expiry forward estimates from put–call parity on matched strikes.

    Returns a DataFrame with expiry, T, the median forward, and the number of pairs.
    """
    S = d["spot"].iloc[0]
    fresh = d[d["fresh"]]
    rows = []
    for expiry, g in fresh.groupby("expiry"):
        calls = g[g["kind"] == "call"].set_index("strike")["lastPrice"]
        puts = g[g["kind"] == "put"].set_index("strike")["lastPrice"]
        K = calls.index.intersection(puts.index)
        K = K[np.abs(np.log(K / S)) < band]
        if len(K) < min_pairs:
            continue
        T = g["T"].iloc[0]
        F_each = K + np.exp(r * T) * (calls[K] - puts[K])
        rows.append({"expiry": expiry, "T": T, "F": float(np.median(F_each)),
                     "n_pairs": len(K)})
    return pd.DataFrame(rows)


def pooled_carry(forwards, S, min_T=14 / 365):
    """Least-squares fit of ln(F/S) = (r - q) T through the origin.

    Very short expiries are excluded: there, ln(F/S) is tiny and dominated by the
    mismatch between the option trade times and the closing spot.
    """
    f = forwards[forwards["T"] >= min_T]
    y = np.log(f["F"] / S).to_numpy()
    T = f["T"].to_numpy()
    return float(T @ y / (T @ T))


def implied_vols(d, r, q, min_T=7 / 365, min_price=0.5, k_range=(-0.35, 0.15)):
    """Implied volatility for fresh out-of-the-money options.

    Returns one row per option with log-moneyness k = ln(K / F) and the implied vol.
    """
    S = d["spot"].iloc[0]
    x = d[d["fresh"] & (d["T"] >= min_T) & (d["lastPrice"] >= min_price)].copy()
    x["F"] = S * np.exp((r - q) * x["T"])
    x["k"] = np.log(x["strike"] / x["F"])
    otm = ((x["kind"] == "put") & (x["k"] < 0)) | ((x["kind"] == "call") & (x["k"] >= 0))
    x = x[otm & x["k"].between(*k_range)]

    x["iv"] = [implied_vol(P, S, K, T, r, q, kind)
               for P, K, T, kind in zip(x["lastPrice"], x["strike"], x["T"], x["kind"])]
    x = x.dropna(subset=["iv"])
    return x[["expiry", "root", "kind", "strike", "T", "F", "k", "lastPrice", "iv"]] \
        .sort_values(["T", "strike"]).reset_index(drop=True)
