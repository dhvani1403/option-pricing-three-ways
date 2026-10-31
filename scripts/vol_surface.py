"""Part 2: implied volatility from real S&P 500 (SPX) option prices.

Reads the snapshot saved by fetch_data.py, backs out implied volatilities with the
safeguarded Newton solver, fits an SVI smile to each expiry and plots the surface.

Produces:
  figures/smiles.png            implied vol vs log-moneyness, data and SVI fits
  figures/vol_surface.png       3D implied-volatility surface
  figures/term_structure.png    at-the-money vol and skew vs expiry
  results/implied_vols.csv      every implied vol computed
  results/market_summary.md     numbers quoted in the README

Usage:  python scripts/vol_surface.py [data/spx_options_YYYY-MM-DD.csv]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import plot_style as style  # noqa: E402
from optpricing import market, svi  # noqa: E402

MIN_POINTS = 15  # an expiry needs this many quotes, on both sides of the money, to fit SVI


def latest_snapshot():
    files = sorted((ROOT / "data").glob("spx_options_*.csv"))
    if not files:
        sys.exit("No data found. Run  python scripts/fetch_data.py  first.")
    return files[-1]


def fit_all(ivs):
    fits = []
    for expiry, g in ivs.groupby("expiry"):
        if len(g) < MIN_POINTS or (g["k"] < 0).sum() < 5 or (g["k"] > 0).sum() < 5:
            continue
        T = g["T"].iloc[0]
        params, rmse = svi.fit(g["k"], g["iv"], T)
        atm = float(svi.implied_vol(0.0, T, params))
        # Skew: d(sigma)/dk at k = 0, by central difference on the fitted curve.
        h = 1e-3
        skew = float((svi.implied_vol(h, T, params) - svi.implied_vol(-h, T, params)) / (2 * h))
        fits.append({"expiry": expiry, "T": T, "n": len(g), "rmse": rmse, "atm_vol": atm,
                     "skew": skew, "params": params})
    return pd.DataFrame(fits).sort_values("T").reset_index(drop=True)


def skew_power_law(fits):
    """Least-squares fit of log(-skew) = log c - alpha log T."""
    slope, intercept = np.polyfit(np.log(fits["T"]), np.log(-fits["skew"]), 1)
    return -slope, intercept


def calendar_violations(fits, k_grid=np.linspace(-0.2, 0.08, 29)):
    """Fraction of (k, adjacent expiry pair) checks where total variance decreases.

    With no calendar arbitrage, w(k, T) must be non-decreasing in T at fixed k.
    """
    W = np.array([svi.total_variance(k_grid, *p) for p in fits["params"]])
    return float(np.mean(np.diff(W, axis=0) < 0))


def smiles_figure(ivs, fits, spot_date):
    # Pick about six expiries spread across the term structure.
    picks = fits.iloc[np.unique(np.linspace(0, len(fits) - 1, 6).round().astype(int))]
    norm = Normalize(vmin=-0.4, vmax=1.0)
    cmap = plt.get_cmap(style.SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, row in enumerate(picks.itertuples()):
        g = ivs[ivs["expiry"] == row.expiry]
        color = cmap(norm(i / max(len(picks) - 1, 1)))
        ax.scatter(g["k"], 100 * g["iv"], s=10, color=color, alpha=0.55, lw=0)
        k_fine = np.linspace(g["k"].min(), g["k"].max(), 200)
        days = int(round(row.T * 365))
        ax.plot(k_fine, 100 * svi.implied_vol(k_fine, row.T, row.params), color=color,
                label=f"{row.expiry}  ({days} days)")
    ax.axvline(0, color=style.INK_2, lw=1, ls=":")
    ax.set_xlabel("Log-moneyness  k = ln(K / F)")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"S&P 500 volatility smiles, {spot_date}: dots = market, lines = SVI fit")
    ax.legend(title="Expiry", fontsize=9, title_fontsize=9)
    fig.tight_layout()
    style.save(fig, "smiles.png")


def surface_figure(fits, spot_date):
    # Interpolate total variance linearly in T between fitted expiries; total
    # variance (not vol) is the quantity that should grow with T.
    # Start at one month: the shortest expiries have narrow strike ranges, so their
    # SVI wings would be extrapolated well outside the data.
    fits = fits[fits["T"] >= 1 / 12]
    k_grid = np.linspace(-0.25, 0.10, 60)
    T_grid = np.linspace(fits["T"].min(), fits["T"].max(), 60)
    W = np.array([svi.total_variance(k_grid, *p) for p in fits["params"]])
    W_interp = np.array([[np.interp(T, fits["T"], W[:, j]) for j in range(len(k_grid))]
                         for T in T_grid])
    IV = np.sqrt(np.maximum(W_interp, 0) / T_grid[:, None])
    KK, TT = np.meshgrid(k_grid, T_grid)

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(KK, TT * 12, 100 * IV, cmap=style.SEQUENTIAL, linewidth=0,
                           antialiased=True, rstride=1, cstride=1, alpha=0.95,
                           vmin=5, vmax=100 * IV.max())
    ax.set_xlabel("Log-moneyness k", labelpad=8)
    ax.set_ylabel("Months to expiry", labelpad=8)
    ax.set_zlabel("Implied vol (%)", labelpad=6)
    ax.view_init(elev=24, azim=-60)
    ax.set_title(f"S&P 500 implied volatility surface, {spot_date}", pad=12)
    fig.colorbar(surf, shrink=0.55, pad=0.08, label="Implied volatility (%)")
    style.save(fig, "vol_surface.png")


def term_structure_figure(fits):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    months = fits["T"] * 12
    a1.plot(months, 100 * fits["atm_vol"], "o-", color=style.BLUE)
    a1.set_xlabel("Months to expiry")
    a1.set_ylabel("At-the-money implied vol (%)")
    a1.set_title("ATM volatility term structure")
    alpha, log_c = skew_power_law(fits)
    a2.loglog(months, -fits["skew"], "o", color=style.ORANGE, label="fitted SVI skew")
    a2.loglog(months, np.exp(log_c) * fits["T"] ** (-alpha), color=style.INK_2, lw=1.2,
              ls="--", label=f"power law  ∝ T^(−{alpha:.2f})")
    a2.set_xlabel("Months to expiry")
    a2.set_ylabel("−dσ/dk at the money")
    a2.set_title("Skew decays like a power of maturity")
    for axis in (a2.xaxis, a2.yaxis):
        axis.set_major_formatter(FormatStrFormatter("%g"))
        axis.set_minor_formatter(FormatStrFormatter("%g"))
    a2.tick_params(which="minor", labelsize=8)
    a2.legend()
    fig.tight_layout()
    style.save(fig, "term_structure.png")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_snapshot()
    style.apply()
    raw = pd.read_csv(path)
    chain = market.prepare_chain(raw)
    S = float(chain["spot"].iloc[0])
    spot_date = chain["spot_date"].iloc[0]

    r = float(market.tbill_to_continuous(chain["tbill_yield"].iloc[0]))
    forwards = market.parity_forwards(chain, r)
    carry = market.pooled_carry(forwards, S)
    q = r - carry
    print(f"Spot {S:.2f} on {spot_date};  r = {r:.4%},  r - q = {carry:.4%},  q = {q:.4%}")

    ivs = market.implied_vols(chain, r, q)
    fits = fit_all(ivs)
    print(f"{len(ivs)} implied vols across {ivs['expiry'].nunique()} expiries; "
          f"SVI fitted to {len(fits)} expiries, median RMSE {100 * fits['rmse'].median():.2f} vol pts")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    ivs.to_csv(out / "implied_vols.csv", index=False)
    summary = [
        f"- Snapshot: {spot_date} close, SPX = {S:,.2f}",
        f"- Contracts in raw chain: {len(raw):,}; after cleaning: {len(ivs):,} implied vols "
        f"across {ivs['expiry'].nunique()} expiries",
        f"- Risk-free rate (13-week T-bill, continuous): {r:.3%}",
        f"- Carry r − q from put–call parity ({len(forwards)} expiries): {carry:.3%}, "
        f"so implied dividend yield q = {q:.3%}",
        f"- SVI fitted to {len(fits)} expiries; median fit error "
        f"{100 * fits['rmse'].median():.2f} vol points, worst {100 * fits['rmse'].max():.2f}",
        f"- ATM vol ranges {100 * fits['atm_vol'].min():.1f}%–{100 * fits['atm_vol'].max():.1f}%; "
        f"ATM skew dσ/dk ranges {fits['skew'].min():.2f} to {fits['skew'].max():.2f}; "
        f"power-law fit |skew| ∝ T^(−{skew_power_law(fits)[0]:.2f})",
        f"- Calendar-arbitrage check: total variance decreases with T in "
        f"{calendar_violations(fits):.1%} of adjacent-expiry comparisons "
        f"({calendar_violations(fits[fits['T'] >= 1 / 12]):.1%} beyond one month)",
    ]
    (out / "market_summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    fits.drop(columns="params").to_csv(out / "svi_fits.csv", index=False)

    smiles_figure(ivs, fits, spot_date)
    surface_figure(fits, spot_date)
    term_structure_figure(fits)


if __name__ == "__main__":
    main()
