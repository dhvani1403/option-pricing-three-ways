"""Download a snapshot of the S&P 500 index (SPX) option chain from Yahoo Finance.

SPX options are European-style and cash-settled, which is what the Black–Scholes
model assumes. (Options on the SPY ETF are American, so they would not fit.)

The raw chain is saved to data/ so every later result can be reproduced without
a network connection.

Usage:  python scripts/fetch_data.py [--max-days 400]
"""

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="^SPX")
    parser.add_argument("--max-days", type=int, default=400,
                        help="skip expiries more than this many days away")
    args = parser.parse_args()

    ticker = yf.Ticker(args.ticker)
    hist = ticker.history(period="5d")
    spot_date = hist.index[-1].date()
    spot = float(hist["Close"].iloc[-1])

    # 13-week T-bill yield, as a reference rate to compare with the implied one.
    tbill = float(yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]) / 100

    frames = []
    for expiry in ticker.options:
        days = (dt.date.fromisoformat(expiry) - spot_date).days
        if days <= 0 or days > args.max_days:
            continue
        chain = ticker.option_chain(expiry)
        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            df = df.copy()
            df["kind"] = kind
            df["expiry"] = expiry
            frames.append(df)

    chain = pd.concat(frames, ignore_index=True)
    chain["spot"] = spot
    chain["spot_date"] = spot_date.isoformat()
    chain["tbill_yield"] = tbill

    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / f"spx_options_{spot_date.isoformat()}.csv"
    chain.to_csv(out, index=False)
    print(f"Saved {len(chain)} contracts across {chain['expiry'].nunique()} expiries "
          f"(spot {spot:.2f} on {spot_date}) to {out.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
