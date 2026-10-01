#!/usr/bin/env python3
"""
kappa_linear_survey.py — linear theory of the ion-cyclotron instability against kappa
====================================================================================
What the kappa index does to the parallel ion-cyclotron instability beyond
the single point the runs sample, with the kinetic solver of linear_theory.py
(no simulation data, no compute allocation):

* ``kappa_growth_spectra``   gamma(k) and omega_r(k) for kappa = 2, 3, 5, 8 and
  the Maxwellian at the parameters of a run, in the two conventions of the
  literature: the same temperature (the comparison of these runs: beta is
  kappa-independent) and the same Maxwellian core (Lazar et al. 2015:
  T_kappa = kappa/(kappa - 3/2) T_core, a hotter plasma).
* ``kappa_growth_vs_anisotropy``  maximum growth rate against T_perp/T_par at
  the beta of the run, and its ratio to the Maxwellian. Why: Lazar et al.
  (2013, A&A 554, A64) found for the electron whistler-cyclotron instability
  that suprathermals raise the growth near threshold and lower it far from it;
  this locates that switch for the ions and places the runs on it.
* ``kappa_thresholds``  anisotropy thresholds A(beta_par) at gamma_max =
  1e-3 and 1e-2 Omega_ci for each kappa, with a fit A = 1 + a / beta^b, the
  bi-Maxwellian contour of Hellinger et al. (2006), and the (beta, A)
  trajectories of the runs when their analysis roots are given.

Both species carry the same kappa, as in the PSC loader; the electrons are
isotropic, so their tail does not drive the mode. Damped kappa roots are out of
reach of the solver (linear_theory.py): a point without an unstable root counts
as gamma = 0.

Usage:
    python kappa_linear_survey.py --outdir OUT [--case mirror_bimaxwellian_moderate]
        [--runs RUN_MAXW RUN_K5 RUN_K3] [--jobs 8]
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

os.environ.pop("PSC_ANALYSIS_DATA_DIR", None)
import plot_style as ps  # noqa: E402

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import psc_units  # noqa: E402
from linear_theory import ParallelDispersion  # noqa: E402

KAPPAS = (None, 8.0, 5.0, 3.0, 2.0)
COLORS = {None: "#000000", 8.0: "#56B4E9", 5.0: "#D55E00", 3.0: "#009E73", 2.0: "#CC79A7"}
K_GRID = np.linspace(0.03, 1.8, 46)
A_GRID = np.array([1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0])
BETA_GRID = np.logspace(np.log10(0.3), np.log10(20.0), 9)
LEVELS = (1e-3, 1e-2)
#: Growth rates below this are marginal roots, not resolved by the solver.
GAMMA_FLOOR = 3e-4


def label(kappa) -> str:
    return "Maxwellian" if kappa is None else rf"$\kappa={kappa:g}$"


def spectrum(args) -> tuple:
    """(k, gamma, omega_r) of the ion-cyclotron branch; gamma = 0 where no unstable root."""
    beta, anis, kappa, beta_e, mass_ratio = args
    disp = ParallelDispersion(beta, anis, beta_e, 1.0, mass_ratio, 1.0 / psc_units.VA_OVER_C, kappa=kappa)
    rows = disp.scan(K_GRID, "plus")
    gamma = np.array([r["gamma_over_Omegai"] for r in rows], dtype=float)
    omega = np.array([r["omega_r_over_Omegai"] for r in rows], dtype=float)
    ok = np.isfinite(gamma) & (omega > 0) & (omega < 1) & (gamma > 0)
    return np.where(ok, gamma, 0.0), np.where(ok, omega, np.nan)


def core_beta(beta: float, kappa) -> float:
    """beta of the kappa plasma whose Maxwellian core has ``beta`` (Lazar et al. 2015)."""
    return beta if kappa is None else beta * kappa / (kappa - 1.5)


def run_all(tasks: list, jobs: int) -> list:
    if jobs <= 1:
        return [spectrum(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(spectrum, tasks, chunksize=2))


def fit_threshold(beta: np.ndarray, anis: np.ndarray) -> tuple[float, float]:
    """A = 1 + a / beta^b by least squares in log space."""
    ok = np.isfinite(anis) & (anis > 1.0)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.log(beta[ok]), np.log(anis[ok] - 1.0), 1)
    return float(np.exp(intercept)), float(-slope)


def threshold_curve(gmax: np.ndarray, level: float) -> np.ndarray:
    """A at which gamma_max crosses ``level`` for each beta (log-linear interpolation in A)."""
    out = np.full(gmax.shape[0], np.nan)
    for i, row in enumerate(gmax):
        above = np.flatnonzero(row >= level)
        if above.size == 0 or above[0] == 0:
            continue
        j = above[0]
        lo, hi = max(row[j - 1], level * 1e-3), row[j]
        frac = (np.log(level) - np.log(lo)) / (np.log(hi) - np.log(lo))
        out[i] = A_GRID[j - 1] + frac * (A_GRID[j] - A_GRID[j - 1])
    return out


def run_trajectory(root: Path) -> tuple | None:
    files = sorted((root / "01_anisotropy").glob("*anisotropy_evolution.csv"))
    if not files:
        return None
    with files[0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
    return (np.array([f(r, "beta_parallel_global") for r in rows]),
            np.array([f(r, "anisotropy_global") for r in rows]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--case", default="mirror_bimaxwellian_moderate")
    parser.add_argument("--runs", nargs="*", type=Path, default=[],
                        help="analysis roots whose (beta, A) trajectories are drawn on the thresholds")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    profile = psc_units._PROFILES[args.case]
    beta0, anis0 = profile["beta_i_par"], profile["Ti_perp_over_Ti_par"]
    beta_e, mr = profile["beta_e_par"], profile["mass_ratio"]

    # One batch of solver calls: spectra (two conventions), anisotropy sweep, (beta, A) grid.
    tasks, keys = [], []
    for kappa in KAPPAS:
        tasks.append((beta0, anis0, kappa, beta_e, mr)); keys.append(("same_T", kappa))
        tasks.append((core_beta(beta0, kappa), anis0, kappa, beta_e, mr)); keys.append(("same_core", kappa))
        for i, beta in enumerate(BETA_GRID):
            for j, anis in enumerate(A_GRID):
                tasks.append((float(beta), float(anis), kappa, beta_e, mr)); keys.append(("grid", kappa, i, j))
        for j, anis in enumerate(A_GRID):
            tasks.append((beta0, float(anis), kappa, beta_e, mr)); keys.append(("sweep", kappa, j))
    # The scans take minutes; they are kept so a change to a figure does not repeat them.
    cache = args.outdir / "kappa_linear_survey_scans.npz"
    signature = np.array(repr(tasks))
    if cache.exists() and str(np.load(cache)["signature"]) == str(signature):
        stored = np.load(cache)
        values = list(zip(stored["gamma"], stored["omega"]))
        print(f"{len(tasks)} dispersion scans read from {cache.name}")
    else:
        print(f"{len(tasks)} dispersion scans on {args.jobs} processes")
        values = run_all(tasks, args.jobs)
        np.savez_compressed(cache, signature=signature, gamma=np.array([v[0] for v in values]),
                            omega=np.array([v[1] for v in values]))
    results = dict(zip(keys, values))

    # ── Figure 1: spectra ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), sharex=True, gridspec_kw={"hspace": 0.1})
    rows = []
    for col, (conv, title) in enumerate((("same_T", "same temperature (these runs)"),
                                         ("same_core", "same Maxwellian core (hotter with the tail)"))):
        for kappa in KAPPAS:
            gamma, omega = results[(conv, kappa)]
            grow = gamma > 0
            c = ps.c(COLORS[kappa])
            axes[0, col].plot(K_GRID[grow], gamma[grow], color=c, lw=1.7, label=label(kappa))
            axes[1, col].plot(K_GRID[grow], omega[grow], color=c, lw=1.7)
            j = int(np.argmax(gamma))
            rows.append({"convention": conv, "kappa": kappa or "inf", "gamma_max": gamma[j],
                         "k_at_max_di": K_GRID[j], "omega_r_at_max": omega[j]})
        axes[0, col].set_title(title, fontsize=11.5)
        axes[1, col].set_xlabel(r"$k_\parallel d_i$")
    axes[0, 0].set_ylabel(r"$\gamma/\Omega_{ci}$")
    axes[1, 0].set_ylabel(r"$\omega_r/\Omega_{ci}$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.03),
               ncol=len(KAPPAS), frameon=False)
    fig.suptitle(rf"Ion-cyclotron instability, $\beta_{{i\parallel}}={beta0:g}$, "
                 rf"$T_\perp/T_\parallel={anis0:g}$", fontsize=13)
    ps.save(fig, args.outdir / "kappa_growth_spectra.png")
    with (args.outdir / "kappa_growth_spectra.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    # ── Figure 2: anisotropy sweep ───────────────────────────────────────────
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.4, 7.2), sharex=True,
                                      gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.08})
    sweep = {k: np.array([results[("sweep", k, j)][0].max() for j in range(A_GRID.size)]) for k in KAPPAS}
    sweep_rows = []
    for kappa in KAPPAS:
        c = ps.c(COLORS[kappa])
        g = sweep[kappa]
        # Below GAMMA_FLOOR a root is marginal: its value is solver noise, and a
        # ratio of two such numbers means nothing.
        shown = g > GAMMA_FLOOR
        top.plot(A_GRID[shown], g[shown], "o-", ms=3.5, color=c, lw=1.6, label=label(kappa))
        if kappa is not None:
            both = shown & (sweep[None] > GAMMA_FLOOR)
            bottom.plot(A_GRID[both], g[both] / sweep[None][both], "o-", ms=3.5, color=c, lw=1.6)
        sweep_rows += [{"kappa": kappa or "inf", "A": a, "gamma_max": v} for a, v in zip(A_GRID, g)]
    for ax in (top, bottom):
        ax.axvline(anis0, color=ps.MUTED_CLR, lw=0.9, ls="--")
    top.set_yscale("log")
    top.set_ylabel(r"$\gamma_{\max}/\Omega_{ci}$")
    top.set_title(rf"Maximum growth rate against the anisotropy, $\beta_{{i\parallel}}={beta0:g}$ "
                  "(same temperature)", fontsize=12)
    bottom.axhline(1.0, color=ps.MUTED_CLR, lw=0.8)
    bottom.set_ylabel(r"$\gamma_{\max}$ / Maxwellian")
    bottom.set_xlabel(r"$T_{i\perp}/T_{i\parallel}$")
    handles, labels = top.get_legend_handles_labels()
    handles.append(Line2D([], [], color=ps.MUTED_CLR, lw=0.9, ls="--", label="anisotropy of the runs"))
    fig.legend(handles, labels + ["anisotropy of the runs"], loc="upper center", bbox_to_anchor=(0.5, 0.03),
               ncol=3, frameon=False)
    ps.save(fig, args.outdir / "kappa_growth_vs_anisotropy.png")
    with (args.outdir / "kappa_growth_vs_anisotropy.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0])); writer.writeheader(); writer.writerows(sweep_rows)

    # ── Figure 3: thresholds ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(11.2, 4.9), sharey=True, gridspec_kw={"wspace": 0.06})
    beta_fine = np.logspace(np.log10(BETA_GRID[0]), np.log10(BETA_GRID[-1]), 100)
    fit_rows = []
    for ax, level in zip(axes, LEVELS):
        for kappa in KAPPAS:
            gmax = np.array([[results[("grid", kappa, i, j)][0].max() for j in range(A_GRID.size)]
                             for i in range(BETA_GRID.size)])
            curve = threshold_curve(gmax, level)
            a, b = fit_threshold(BETA_GRID, curve)
            c = ps.c(COLORS[kappa])
            ax.plot(BETA_GRID, curve, "o", ms=4, color=c)
            if np.isfinite(a):
                ax.plot(beta_fine, 1.0 + a / beta_fine ** b, "-", color=c, lw=1.5, label=label(kappa))
            fit_rows.append({"gamma_level": level, "kappa": kappa or "inf", "a": a, "b": b,
                             **{f"A_at_beta_{bb:.3g}": v for bb, v in zip(BETA_GRID, curve)}})
        if level == 1e-3:    # Hellinger et al. (2006), proton cyclotron, bi-Maxwellian
            ax.plot(beta_fine, 1.0 + 0.43 / beta_fine ** 0.42, ":", color=ps.MUTED_CLR, lw=1.6,
                    label="Hellinger et al. (2006)")
        for k, root in enumerate(args.runs):
            traj = run_trajectory(root)
            if traj is not None:
                ax.plot(*traj, color=ps.MUTED_CLR, lw=1.0, alpha=0.8,
                        label="runs (trajectories)" if k == 0 else None)
        ax.set_xscale("log")
        ps.plain_log_axis(ax, "x")
        ax.set_xlabel(r"$\beta_{i\parallel}$")
        ax.set_title(rf"$\gamma_{{\max}} = 10^{{{int(np.log10(level))}}}\,\Omega_{{ci}}$", fontsize=12)
        ax.set_ylim(1.0, 3.0)
    axes[0].set_ylabel(r"$T_{i\perp}/T_{i\parallel}$ at threshold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False)
    fig.suptitle(r"Ion-cyclotron anisotropy thresholds against $\kappa$ (same temperature); "
                 r"lines: $A = 1 + a/\beta^{b}$", fontsize=12.5, y=1.02)
    ps.save(fig, args.outdir / "kappa_thresholds.png")
    keys_out = list(dict.fromkeys(k for r in fit_rows for k in r))
    with (args.outdir / "kappa_thresholds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys_out); writer.writeheader(); writer.writerows(fit_rows)
    for r in fit_rows:
        print(f"  gamma = {r['gamma_level']:g}  kappa = {r['kappa']}:  A = 1 + {r['a']:.3f} / beta^{r['b']:.3f}")
    print(f"written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
