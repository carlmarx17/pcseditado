#!/usr/bin/env python3
"""
trajectory_linear_theory.py — does the bi-kappa description survive saturation?
===============================================================================
From the analysis products of the runs (no raw data).

The comparison of a simulation with linear theory is usually made once, at the
initial state. Here it is made at every time: the kinetic dispersion relation
(linear_theory.ParallelDispersion) is solved at the wavenumber of the dominant
mode for a bi-Maxwellian / bi-kappa plasma that has the ion beta, the ion
anisotropy, the electron beta and anisotropy and the kappa index *measured at
that time*, and the root is compared with the growth rate the mode actually has
then (local slope of ln|dB_k|).

Why: quasi-linear models of these instabilities evolve the moments and keep the
shape of the distribution (bi-Maxwellian, or bi-kappa with a fixed kappa). If
that closure holds, the mode grows at the instantaneous linear rate and stops
when the rate reaches zero, i.e. when the anisotropy reaches the marginal value
of that mode. The two departures this script measures are the gap between the
two rates in the non-linear stage and the anisotropy left above the marginal
value when the mode stops growing.

* ``trajectory_linear_theory.png``  top: measured and instantaneous linear
  growth rate of the dominant mode; bottom: the measured ion anisotropy and the
  marginal anisotropy of that mode (gamma = 0 at the measured betas and kappa).
* ``trajectory_linear_theory.csv``  the series of the figure, with the linear
  rate also for the initial kappa (how much the evolution of kappa matters).
* ``trajectory_saturation.csv``     one row per run: time at which the mode
  stops growing, the linear rate, the anisotropy and its excess over the
  marginal value there, and the fluctuation energy. The excess is given for
  the box moments and for the particles of the prt window in the local-field
  frame (interpolated between snapshots): the two differ by a few 0.01 once
  the wave is large, which bounds how far the excess can be read.

Limits: the solver gives both species the same kappa (as the particle loader
does), while the measured index is the ions'; a damped bi-kappa root cannot be
evaluated (Z_kappa is integrated for Im(omega) > 0), so a stable state is a gap
in the linear curve; before the v6c products the amplitude is that of the
k-shell of the mode, not of the single Fourier mode.

Usage:
    python trajectory_linear_theory.py RUN_MAXW RUN_K5 RUN_K3 --outdir OUT
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

os.environ.pop("PSC_ANALYSIS_DATA_DIR", None)
import plot_style as ps  # noqa: E402

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from linear_theory import ParallelDispersion  # noqa: E402

SERIES = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

#: A measured 1/kappa below this is a Maxwellian for the solver.
MAXWELLIAN_INVERSE_KAPPA = 0.02
#: Anisotropy from which the marginal value is approached (always unstable here).
A_START = 2.2
#: A root is the ion-cyclotron (L-mode) one if 0 < omega_r < Omega_ci.
GAMMA_FLOOR = 1e-3


def c_over_va(profile: dict) -> float:
    """c / v_A of a run profile.

    The profile key ``vA_over_c`` is the historical name of the C++ input: it
    is B0 in code units (Omega_ce/omega_pe), not v_A/c. With n0 = 1, m_e = 1:
    v_A/c = B0 / sqrt(m_i/m_e), as in psc_units.VA.
    """
    return float(np.sqrt(profile["mass_ratio"]) / profile["vA_over_c"])


def _is_ion_cyclotron(root: complex) -> bool:
    return bool(np.isfinite(root) and 0.0 < np.real(root) < 1.0)


def linear_root(state: dict, kappa: float | None, k: float, seed: complex | None = None,
                a_i: float | None = None) -> complex:
    """L-mode root (omega_r + i gamma, in Omega_ci) at k d_i for a measured state.

    ``state``: beta_i, a_i, beta_e, a_e, mass_ratio, c_over_va. The previous
    root is tried as seed (the state changes slowly); otherwise the root is
    dragged up from long wavelengths, and a root on another branch is refused.
    """
    disp = ParallelDispersion(state["beta_i"], state["a_i"] if a_i is None else a_i, state["beta_e"],
                              state["a_e"], state["mass_ratio"], state["c_over_va"], kappa=kappa)
    if seed is not None and np.isfinite(seed):
        root = disp.solve(k, "plus", seed)
        if _is_ion_cyclotron(root):
            return root
    row = disp.scan(np.linspace(0.05, k, 12), "plus")[-1]
    root = complex(row["omega_r_over_Omegai"], row["gamma_over_Omegai"])
    return root if _is_ion_cyclotron(root) else complex(np.nan, np.nan)


def marginal_anisotropy(state: dict, kappa: float | None, k: float, step: float = 0.04,
                        seed: complex | None = None) -> tuple[float, complex]:
    """Ion anisotropy at which the mode k is marginal (gamma = 0), other parameters fixed.

    At fixed k the growth rate changes sign at a finite anisotropy (where
    omega_r = Omega_ci (A - 1)/A), so it is followed down from A_START in steps
    and the zero is interpolated; for a bi-kappa, whose damped side cannot be
    evaluated, it is extrapolated from the last two growing points. Returns the
    anisotropy and the root at A_START (seed for the next time).
    """
    first = linear_root(state, kappa, k, seed, a_i=A_START)
    points, root, a = [], first, A_START
    while a > 1.0 and np.isfinite(root):
        gamma = float(np.imag(root))
        points.append((a, gamma))
        if gamma < GAMMA_FLOOR:
            break
        a -= step
        root = ParallelDispersion(state["beta_i"], a, state["beta_e"], state["a_e"], state["mass_ratio"],
                                  state["c_over_va"], kappa=kappa).solve(k, "plus", root)
        if not _is_ion_cyclotron(root):
            break
    growing = [p for p in points if p[1] >= GAMMA_FLOOR]
    if len(growing) < 2:
        return float("nan"), first
    below = [p for p in points if p[1] < GAMMA_FLOOR]
    (a1, g1), (a0, g0) = (growing[-1], below[0]) if below else (growing[-2], growing[-1])
    if g1 == g0:
        return float("nan"), first
    return float(a1 - g1 * (a1 - a0) / (g1 - g0)), first


def windowed_growth(t, amp, centres, width: float):
    """Local slope of ln(amp) in windows of ``width`` around ``centres``, with its fit error."""
    t, amp = np.asarray(t, float), np.asarray(amp, float)
    gamma, err = np.full(len(centres), np.nan), np.full(len(centres), np.nan)
    for i, centre in enumerate(centres):
        sel = (np.abs(t - centre) <= 0.5 * width) & (amp > 0) & np.isfinite(amp)
        if sel.sum() < 6:
            continue
        coeff, cov = np.polyfit(t[sel], np.log(amp[sel]), 1, cov=True)
        gamma[i], err[i] = coeff[0], float(np.sqrt(cov[0, 0]))
    return gamma, err


def zero_crossing(t, y) -> float:
    """First time y goes from positive to non-positive, linearly interpolated."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    for i in range(1, len(t)):
        if np.isfinite(y[i - 1]) and np.isfinite(y[i]) and y[i - 1] > 0 >= y[i]:
            return float(t[i - 1] + y[i - 1] * (t[i] - t[i - 1]) / (y[i - 1] - y[i]))
    return float("nan")


def load(root: Path) -> dict:
    """The mode amplitude, the moments, kappa(t) and the fluctuation energy of one run."""
    import kappa_dynamics as kd
    import paper_figures as pf

    run = pf.load_run(root)
    kappa_run = kd.load_run(root) if run["kappa"] is not None else None
    if kappa_run is not None:
        series = kappa_run["pop"]["all"]
        run["kappa_t"], run["inverse_kappa"] = series["t"], series["inv"]
    run["field"] = kd.load_field(root)
    # Independent estimate of the anisotropy: the particles of the prt window
    # in the frame of the local field (the moments are those of the whole box).
    summary = root / "03_particles" / "vdf_hole_vs_peak_summary.csv"
    rows = [r for r in pf._rows(summary) if r["population"] == "all"] if summary.exists() else []
    run["a_particles"] = (np.array([pf._f(r["omega_ci_t"]) for r in rows]),
                          np.array([pf._f(r["A_local_b"]) for r in rows]))
    return run


def kappa_at(run: dict, t: float) -> float | None:
    if run["kappa"] is None:
        return None
    inv = float(np.interp(t, run["kappa_t"], run["inverse_kappa"]))
    return None if inv < MAXWELLIAN_INVERSE_KAPPA else 1.0 / inv


def trajectory(run: dict, dt: float, width: float) -> dict:
    """Measured and instantaneous linear growth rate, and the marginal anisotropy, in time."""
    a, profile = run["aniso"], run["profile"]
    t_start = run["t_lin"][0] if np.isfinite(run["t_lin"][0]) else run["t_settle"]
    t_end = float(np.nanmax(run["kt_t"]))
    centres = np.arange(t_start + 0.5 * width, t_end - 0.5 * width + 1e-9, dt)
    measured, measured_err = windowed_growth(run["kt_t"], run["kt_amp"], centres, width)
    out = {key: np.full(len(centres), np.nan) for key in (
        "gamma_linear", "gamma_linear_kappa0", "omega_r", "a_marginal", "a_i", "beta_i", "beta_e", "kappa")}
    seed = seed0 = seed_marginal = None
    for i, t in enumerate(centres):
        j = int(np.argmin(np.abs(a["omega_ci_t"] - t)))
        state = {"beta_i": float(a["beta_parallel_global"][j]), "a_i": float(a["anisotropy_global"][j]),
                 "beta_e": float(a["beta_e_parallel_global"][j]), "a_e": float(a["anisotropy_e_global"][j]),
                 "mass_ratio": float(profile["mass_ratio"]), "c_over_va": c_over_va(profile)}
        kappa = kappa_at(run, t)
        root = linear_root(state, kappa, run["k_mode"], seed)
        root0 = root if run["kappa"] is None else linear_root(state, run["kappa"], run["k_mode"], seed0)
        seed, seed0 = (root if np.isfinite(root) else seed), (root0 if np.isfinite(root0) else seed0)
        out["a_marginal"][i], seed_marginal = marginal_anisotropy(state, kappa, run["k_mode"], seed=seed_marginal)
        out["gamma_linear"][i], out["omega_r"][i] = np.imag(root), np.real(root)
        out["gamma_linear_kappa0"][i] = np.imag(root0)
        out["a_i"][i], out["beta_i"][i], out["beta_e"][i] = state["a_i"], state["beta_i"], state["beta_e"]
        out["kappa"][i] = np.inf if kappa is None else kappa
    out.update({"t": centres, "gamma_measured": measured, "gamma_measured_err": measured_err})
    return out


def saturation(run: dict, tr: dict) -> dict:
    """State of the run when its dominant mode stops growing."""
    t_sat = zero_crossing(tr["t"], tr["gamma_measured"])
    if not np.isfinite(t_sat):
        t_sat = run["t_peak"]
    at = lambda key: float(np.interp(t_sat, tr["t"], tr[key]))  # noqa: E731
    field = run["field"]
    energy = float(np.interp(t_sat, field["t"], field["W"])) if field else float("nan")
    tp, ap = run.get("a_particles", (np.array([]), np.array([])))
    a_particles = float(np.interp(t_sat, tp, ap)) if tp.size >= 2 else float("nan")
    return {"t_saturation": t_sat, "gamma_linear": at("gamma_linear"), "a_i": at("a_i"),
            "a_marginal": at("a_marginal"), "excess_anisotropy": at("a_i") - at("a_marginal"),
            "a_particles_local": a_particles, "excess_anisotropy_particles": a_particles - at("a_marginal"),
            "kappa": at("kappa") if np.all(np.isfinite(tr["kappa"])) else float("inf"),
            "fluctuation_energy": energy, "a_final": float(run["aniso"]["anisotropy_global"][-1])}


def plot(runs, tracks, sats, path: Path, width: float) -> None:
    fig, axes = plt.subplots(2, len(runs), figsize=(4.1 * len(runs), 6.0), sharex=True, sharey="row",
                             squeeze=False, layout="constrained")
    theory = ps.MUTED_CLR
    for i, (run, tr, sat) in enumerate(zip(runs, tracks, sats)):
        col, top, bottom = ps.c(SERIES[i]), axes[0, i], axes[1, i]
        top.fill_between(tr["t"], tr["gamma_measured"] - tr["gamma_measured_err"],
                         tr["gamma_measured"] + tr["gamma_measured_err"], color=col, alpha=0.25, lw=0)
        top.plot(tr["t"], tr["gamma_measured"], color=col, lw=1.8)
        top.plot(tr["t"], tr["gamma_linear"], "--", color=theory, lw=1.5)
        top.axhline(0.0, color=theory, lw=0.6)
        a = run["aniso"]
        bottom.plot(a["omega_ci_t"], a["anisotropy_global"], color=col, lw=1.8)
        bottom.plot(tr["t"], tr["a_marginal"], "--", color=theory, lw=1.5)
        for ax in (top, bottom):
            ax.axvline(sat["t_saturation"], color=theory, lw=0.9, ls=":")
        top.set_title(run["label"], fontsize=12)
        bottom.set_xlabel(r"$t\,\Omega_{ci}$")
    axes[0, 0].set_ylabel(r"$\gamma/\Omega_{ci}$, mode $k_\parallel d_i=" + f"{runs[0]['k_mode']:.2f}$")
    axes[1, 0].set_ylabel(r"$T_{i\perp}/T_{i\parallel}$")
    # One legend for the six panels, below them; the method goes in the labels
    # so that no free note competes with it for the bottom margin.
    handles = [Line2D([], [], color=ps.c(SERIES[0]), lw=1.8,
                      label=rf"measured (run colour); rate: slope of $\ln|\delta B_k|$ in {width:g} "
                            r"$\Omega_{ci}^{-1}$ windows, from the start of the linear phase"),
               Line2D([], [], color=theory, lw=1.5, ls="--",
                      label=r"linear theory with the moments and $\kappa$ measured at that time "
                            "(bottom: anisotropy at which the mode is marginal)"),
               Line2D([], [], color=theory, lw=0.9, ls=":", label="mode stops growing")]
    fig.legend(handles=handles, loc="outside lower center", ncol=1, frameon=False, fontsize=10)
    fig.suptitle("Instantaneous linear theory along the runs", fontsize=13)
    ps.save(fig, path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--dt", type=float, default=5.0, help="spacing of the evaluation times, in 1/Omega_ci")
    p.add_argument("--window", type=float, default=15.0, help="width of the growth-fit window, in 1/Omega_ci")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    runs = [load(root) for root in args.runs]
    tracks = [trajectory(run, args.dt, args.window) for run in runs]
    sats = [saturation(run, tr) for run, tr in zip(runs, tracks)]
    plot(runs, tracks, sats, args.outdir / "trajectory_linear_theory.png", args.window)

    keys = ("t", "gamma_measured", "gamma_measured_err", "gamma_linear", "gamma_linear_kappa0", "omega_r",
            "a_i", "a_marginal", "beta_i", "beta_e", "kappa")
    with open(args.outdir / "trajectory_linear_theory.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run",) + keys)
        for run, tr in zip(runs, tracks):
            for i in range(len(tr["t"])):
                writer.writerow([run["name"]] + [f"{tr[key][i]:.6g}" for key in keys])
    with open(args.outdir / "trajectory_saturation.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run"] + list(sats[0]))
        writer.writeheader()
        for run, sat in zip(runs, sats):
            writer.writerow({"run": run["name"], **{key: f"{value:.6g}" for key, value in sat.items()}})
            print(f"{run['name']}: stops growing at t Omega_ci = {sat['t_saturation']:.1f}; linear rate there "
                  f"{sat['gamma_linear']:+.4f}; A = {sat['a_i']:.3f}, marginal {sat['a_marginal']:.3f}, "
                  f"excess {sat['excess_anisotropy']:+.3f} (particles, local frame: "
                  f"{sat['excess_anisotropy_particles']:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
