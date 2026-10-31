"""Part 1: show that the three pricing methods agree, and measure how fast they converge.

Produces:
  figures/three_methods.png        prices across strikes from all three methods
  figures/mc_convergence.png       Monte Carlo RMSE vs number of paths
  figures/fd_convergence.png       finite-difference error vs grid size
  figures/explicit_stability.png   explicit Euler just inside / outside its stability limit
  results/agreement_table.md       table used in the README

Usage:  python scripts/convergence.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import plot_style as style  # noqa: E402
from optpricing import black_scholes as bs  # noqa: E402
from optpricing import finite_difference as fd  # noqa: E402
from optpricing import monte_carlo as mc  # noqa: E402

# Benchmark contract: a one-year at-the-money call.
BASE = dict(S=100.0, K=100.0, T=1.0, r=0.04, sigma=0.20, q=0.01)


def agreement_table():
    """Price a grid of options three ways and write a Markdown table."""
    rows = []
    header = ("| Type | K | T | Black–Scholes | Monte Carlo (± 1 s.e.) | PDE (Crank–Nicolson) "
              "| MC error / s.e. | PDE error |")
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    for kind in ("call", "put"):
        for K in (80, 100, 120):
            for T in (0.25, 1.0):
                p = dict(BASE, K=K, T=T)
                exact = float(bs.price(**p, kind=kind))
                m = mc.price(**p, kind=kind, n_paths=1_000_000, antithetic=True,
                             control_variate=True, seed=K * 10 + int(T * 4))
                f = fd.price(**p, kind=kind, n_space=800, n_time=800)
                rows.append(f"| {kind} | {K} | {T:g} | {exact:.4f} | {m.price:.4f} ± "
                            f"{m.std_error:.4f} | {f:.4f} | {(m.price - exact) / m.std_error:+.2f} "
                            f"| {f - exact:+.1e} |")
    out = ROOT / "results" / "agreement_table.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join([header, sep] + rows) + "\n")
    print(f"  wrote {out.relative_to(ROOT)}")


def three_methods_figure():
    strikes = np.linspace(60, 140, 17)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, kind in zip(axes, ("call", "put")):
        K_fine = np.linspace(60, 140, 200)
        ax.plot(K_fine, bs.price(**dict(BASE, K=K_fine), kind=kind), color=style.BLUE,
                label="Black–Scholes formula", zorder=1)
        fd_prices = [fd.price(**dict(BASE, K=K), kind=kind) for K in strikes]
        ax.plot(strikes, fd_prices, "s", color=style.ORANGE, mfc="white", mew=2,
                markersize=9, label="PDE (Crank–Nicolson)", zorder=2)
        mc_res = [mc.price(**dict(BASE, K=K), kind=kind, n_paths=20_000, seed=i)
                  for i, K in enumerate(strikes)]
        ax.errorbar(strikes, [m.price for m in mc_res],
                    yerr=[1.96 * m.std_error for m in mc_res], fmt="o", color=style.AQUA,
                    markersize=4, capsize=2, label="Monte Carlo, 20k paths (95% CI)", zorder=3)
        ax.set_title(f"European {kind}, S = 100, T = 1, σ = 20%")
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Option price")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    style.save(fig, "three_methods.png")


def mc_convergence_figure(n_repeats=200):
    """RMSE over many independent runs, for three estimators."""
    exact = float(bs.price(**BASE))
    Ns = np.unique(np.logspace(2, 5.5, 15).astype(int) // 2 * 2)
    variants = [
        ("Plain Monte Carlo", dict(), style.BLUE),
        ("Antithetic variates", dict(antithetic=True), style.ORANGE),
        ("Antithetic + control variate", dict(antithetic=True, control_variate=True), style.AQUA),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    rng = np.random.default_rng(0)
    for label, opts, color in variants:
        rmse = []
        for N in Ns:
            seeds = rng.integers(0, 2**31, n_repeats)
            errs = [mc.price(**BASE, n_paths=N, seed=s, **opts).price - exact for s in seeds]
            rmse.append(np.sqrt(np.mean(np.square(errs))))
        rmse = np.array(rmse)
        slope = np.polyfit(np.log(Ns), np.log(rmse), 1)[0]
        ax.loglog(Ns, rmse, "o-", color=color, label=f"{label} (slope {slope:.2f})")
        print(f"  MC {label}: fitted slope {slope:.3f}, RMSE at N={Ns[-1]}: {rmse[-1]:.2e}")
    ref = rmse[0] * 3 * np.sqrt(Ns[0] / Ns)
    ax.loglog(Ns, ref, "--", color=style.INK_2, lw=1.2, label=r"reference $\propto N^{-1/2}$")
    ax.set_xlabel("Number of simulated paths N")
    ax.set_ylabel("Root-mean-square error")
    ax.set_title("Monte Carlo error shrinks like 1/√N")
    ax.legend(fontsize=9)
    fig.tight_layout()
    style.save(fig, "mc_convergence.png")


def fd_convergence_figure():
    exact = float(bs.price(**BASE))
    Ns = np.array([25, 50, 100, 200, 400, 800, 1600]) // 2 * 2
    schemes = [("Implicit Euler (θ = 1)", 1.0, style.BLUE),
               ("Crank–Nicolson (θ = ½)", 0.5, style.ORANGE)]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, th, color in schemes:
        errs = np.array([abs(fd.price(**BASE, n_space=N, n_time=N, theta=th) - exact)
                         for N in Ns])
        slope = np.polyfit(np.log(Ns[1:]), np.log(errs[1:]), 1)[0]
        ax.loglog(Ns, errs, "o-", color=color, label=f"{label} (slope {slope:.2f})")
        print(f"  FD {label}: fitted slope {slope:.3f}, error at N={Ns[-1]}: {errs[-1]:.2e}")
    ax.set_xlabel("Grid points N (in both space and time)")
    ax.set_ylabel("|PDE price − exact price|")
    ax.set_title("Crank–Nicolson is second order, implicit Euler first")
    ax.legend()
    fig.tight_layout()
    style.save(fig, "fd_convergence.png")


def explicit_stability_figure():
    """Explicit Euler needs sigma^2 dt / dx^2 <= 1 (roughly). Show both sides of it."""
    n_space = 200
    half_width = 5 * BASE["sigma"] * np.sqrt(BASE["T"])
    dx = 2 * half_width / n_space
    exact = float(bs.price(**BASE))

    fig, (ax_growth, ax_profile) = plt.subplots(1, 2, figsize=(11, 4.2))
    for target, color in ((0.95, style.BLUE), (1.05, style.ORANGE)):
        n_time = int(round(BASE["sigma"] ** 2 * BASE["T"] / (target * dx**2)))
        lam = BASE["sigma"] ** 2 * BASE["T"] / n_time / dx**2
        S_grid, V, hist = fd.solve(**BASE, n_space=n_space, n_time=n_time, theta=0.0,
                                   keep_history=True)
        # Size of the error relative to the exact solution, step by step.
        tau = np.linspace(0, BASE["T"], n_time + 1)
        errs = [np.max(np.abs(h - bs.price(**dict(BASE, S=S_grid, T=max(t, 1e-12)))))
                for h, t in zip(hist, tau)]
        label = f"σ²Δt/Δx² = {lam:.2f}"
        ax_growth.semilogy(np.arange(n_time + 1), errs, color=color, label=label)
        print(f"  explicit, ratio {lam:.3f}: price {V[n_space // 2]:.4g} (exact {exact:.4f})")

        # Profile at the first step where the unstable solution is visibly wrong.
        bad = next((i for i, e in enumerate(errs) if e > 2.0), len(hist) - 1)
        window = (S_grid > 70) & (S_grid < 140)
        ax_profile.plot(S_grid[window], hist[bad][window], color=color,
                        lw=1.5, label=f"{label}, after {bad} steps")
    ax_profile.plot(S_grid[window], np.maximum(S_grid[window] - BASE["K"], 0),
                    color=style.INK_2, lw=1, ls="--", label="payoff")

    ax_growth.axhline(1, color=style.INK_2, lw=1, ls=":")
    ax_growth.set_xlabel("Time step")
    ax_growth.set_ylabel("max |numerical − exact| over the grid")
    ax_growth.set_title("Error stays bounded below the limit, explodes above it")
    ax_growth.legend()
    ax_profile.set_xlabel("Spot price S")
    ax_profile.set_ylabel("Numerical option value")
    ax_profile.set_title("The unstable mode is a grid-scale zig-zag")
    ax_profile.legend(fontsize=9)
    fig.tight_layout()
    style.save(fig, "explicit_stability.png")


def main():
    style.apply()
    print("Agreement table")
    agreement_table()
    print("Figures")
    three_methods_figure()
    mc_convergence_figure()
    fd_convergence_figure()
    explicit_stability_figure()


if __name__ == "__main__":
    main()
