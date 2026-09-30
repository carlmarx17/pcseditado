#!/usr/bin/env python3
"""
paper_figures.py — publication figures of a controlled series, from analysis products
=====================================================================================
Builds the figures and the summary table of one controlled series (runs that
differ only in the initial distribution) from the CSV/JSON products the
pipeline already wrote, so they can be regenerated on a laptop without the
raw run data:

* ``mode_amplitude``      amplitude of the first transverse k-shell (the
                          box-fundamental ion-cyclotron mode) against time,
                          with the linear-theory growth rate over the fitted
                          linear phase. Why: the domain rms of dB is dominated
                          by the particle noise of every k (~2e-2 B0) and hides
                          the growth; the mode amplitude rises from the noise
                          by two decades.
* ``growth_rate_vs_kappa`` measured gamma against 1/kappa with the parallel
                          kinetic dispersion relation at the same k. Why: the
                          single most direct test of the paper's claim, that
                          suprathermal tails change the growth rate.
* ``brazil_trajectories`` global (beta_i||, A_i) trajectories of the series
                          with the mirror threshold of the measured electrons
                          (Hellinger 2007) and the IC contour (Hellinger 2006).
* ``resonance``           reduced f(v||) of the three distributions at the same
                          T|| with the cyclotron-resonant velocity of the mode.
                          Why: explains the trend -- the resonance sits on the
                          shoulder of the distribution, where a kappa plasma of
                          equal temperature has fewer ions.
* ``kappa_local``         1/kappa_eff from the local-field, kurtosis-based
                          estimator of vdf_spatial.py. Why: the global-B0 fit of
                          kappa_evolution.py reads the wave-tilted, resonantly
                          distorted distribution at saturation as a "tail"
                          (kappa ~ 13 in the bi-Maxwellian run while the local
                          estimator gives a Maxwellian); only the local one is
                          quoted.
* ``series_summary.csv``  the numbers quoted in the text.

Usage::

    python paper_figures.py ../analysis_results/v6b/mirror_bimaxwellian_moderate \\
        ../analysis_results/v6b/mirror_bikappa5_moderate \\
        ../analysis_results/v6b/mirror_bikappa3_moderate \\
        --outdir ../analysis_results/v6b/paper_figures/mirror_moderate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.pop("PSC_ANALYSIS_DATA_DIR", None)
import plot_style as ps  # noqa: E402

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402

import psc_units  # noqa: E402
from growth_fit import reference_growth_row  # noqa: E402
from plasma_physics import mirror_threshold_electrons  # noqa: E402

#: Okabe-Ito, fixed order: Maxwellian, kappa 5, kappa 3.
SERIES = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
#: 1/kappa grid of the theory curve (0 = Maxwellian).
INV_KAPPA_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.3333, 0.36)


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_run(root: Path) -> dict:
    """Products of one run directory (the run's analysis root)."""
    name = root.name
    profile = psc_units._PROFILES[name]
    phys = root / "09_physical_diagnostics"
    growth_rows = _rows(phys / "growth_rate_summary.csv")
    ref = reference_growth_row(growth_rows) or {}
    phase = json.loads((phys / "linear_phase.json").read_text())
    modes = _rows(phys / "mode_growth_table.csv")
    dominant = next((m for m in modes if m.get("dominant") == "1"), modes[0])
    kappa = profile.get("kappa")
    run = {
        "name": name, "root": root, "profile": profile, "kappa": kappa,
        "label": "bi-Maxwellian" if kappa is None else rf"bi-$\kappa$, $\kappa={kappa:g}$",
        "gamma": _f(ref.get("gamma")), "gamma_err": _f(ref.get("gamma_err")),
        "fit_ok": str(ref.get("fit_ok")) in ("1", "True", "true"),
        "t_lin": (_f(ref.get("linear_phase_start")), _f(ref.get("linear_phase_end"))),
        "k_mode": abs(_f(phase["mode"]["k_parallel_di"])),
        "k_perp_mode": abs(_f(phase["mode"]["k_perp_di"])),
        "peak_amplitude": _f(dominant.get("max_amplitude_over_B0")),
        "t_peak": _f(dominant.get("t_max_amplitude")),
        "compressibility": _f(dominant.get("compressibility")),
    }
    kt = _rows(root / "04_spectra" / next(p.name for p in (root / "04_spectra").glob("field_energy_kt_*.csv")))
    shells = np.unique([_f(r["k"]) for r in kt])
    shell = shells[np.argmin(np.abs(shells - np.hypot(run["k_mode"], run["k_perp_mode"])))]
    sel = [r for r in kt if _f(r["k"]) == shell]
    run["kt_shell"] = float(shell)
    run["kt_t"] = np.array([_f(r["omega_ci_t"]) for r in sel])
    run["kt_amp"] = np.sqrt(np.clip([_f(r["E_perp"]) for r in sel], 0, None))
    run["amp_label"] = rf"$|\delta\hat{{B}}_\perp|$, $|k|d_i\simeq{shell:.2f}$ shell (arb. units)"
    # Exact amplitude of the dominant mode when physical_diagnostics wrote its
    # history (pipeline >= this revision); the k-shell energy otherwise.
    history = phys / "mode_amplitude_timeseries.csv"
    tag = (f"amp_over_B0_kpar{phase['mode']['k_parallel_di']:.3f}"
           f"_kperp{phase['mode']['k_perp_di']:+.3f}")
    if history.exists():
        rows = _rows(history)
        if rows and tag in rows[0]:
            run["kt_t"] = np.array([_f(r["omega_ci_t"]) for r in rows])
            run["kt_amp"] = np.array([_f(r[tag]) for r in rows])
            run["amp_label"] = rf"$|\delta\hat{{\mathbf{{B}}}}(k_\parallel d_i={run['k_mode']:.2f})|/B_0$"
    aniso = _rows(next((root / "01_anisotropy").glob("*anisotropy_evolution.csv")))
    run["aniso"] = {key: np.array([_f(r.get(key)) for r in aniso]) for key in (
        "omega_ci_t", "anisotropy_global", "beta_parallel_global",
        "beta_e_parallel_global", "anisotropy_e_global")}
    summary = root / "03_particles" / "vdf_hole_vs_peak_summary.csv"
    local = [r for r in _rows(summary) if r["population"] == "all"] if summary.exists() else []
    run["kappa_local"] = (np.array([_f(r["omega_ci_t"]) for r in local]),
                          np.array([_f(r["kappa_eff"], float("inf")) for r in local]),
                          np.array([_f(r["kappa_eff_error"]) for r in local]))
    return run


def theory(run: dict, kappa: float | None) -> dict:
    """Parallel (L-mode) kinetic root at the run's mode wavenumber."""
    from linear_theory import ParallelDispersion
    p = run["profile"]
    disp = ParallelDispersion(p["beta_i_par"], p["Ti_perp_over_Ti_par"], p["beta_e_par"],
                              p["Te_perp_over_Te_par"], p["mass_ratio"],
                              1.0 / psc_units.VA_OVER_C, kappa=kappa)
    # The L-mode ion-cyclotron root has 0 < omega_r < Omega_ci. The default
    # seed usually lands on it; otherwise drag the root up from long
    # wavelengths, and refuse a root on another branch rather than plot it.
    def ok(row):
        return np.isfinite(row["gamma_over_Omegai"]) and 0.0 < row["omega_r_over_Omegai"] < 1.0
    row = disp.scan([run["k_mode"]], "plus")[-1]
    if not ok(row):
        row = disp.scan(np.linspace(0.05, run["k_mode"], 30), "plus")[-1]
    if not ok(row):
        raise RuntimeError(f"no ion-cyclotron root at k d_i = {run['k_mode']:.3f}, kappa = {kappa}")
    return {"gamma": row["gamma_over_Omegai"], "omega_r": row["omega_r_over_Omegai"]}


def half_relaxation_time(run: dict) -> float:
    t, a = run["aniso"]["omega_ci_t"], run["aniso"]["anisotropy_global"]
    target = a[0] - 0.5 * (a[0] - a[-1])
    below = np.flatnonzero(a <= target)
    return float(t[below[0]]) if below.size else float("nan")


def plot_mode_amplitude(runs, outdir):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for i, run in enumerate(runs):
        col = ps.c(SERIES[i])
        keep = ps.measured_fluctuation(run["kt_t"]) & (run["kt_amp"] > 0)
        t, amp = run["kt_t"][keep], run["kt_amp"][keep]
        ax.plot(t, amp, color=col, lw=1.6, label=run["label"])
        t0, t1 = run["t_lin"]
        if np.isfinite(t0) and np.isfinite(t1):
            a0 = float(np.interp(t0, t, amp))
            tt = np.linspace(t0, t1, 50)
            ax.plot(tt, a0 * np.exp(run["theory"]["gamma"] * (tt - t0)) * 1.8, "--",
                    color=col, lw=1.2, alpha=0.9)
    ax.plot([], [], "--", color=ps.MUTED_CLR, label=r"linear-theory $\gamma$ (shifted $\times1.8$ for visibility)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t\,\Omega_{ci}$")
    ax.set_ylabel(runs[0]["amp_label"])
    ax.set_title(r"Growth of the ion-cyclotron mode ($k_\parallel d_i = 0.31$)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    ps.plain_log_axis(ax, "y")
    ps.save(fig, outdir / "mode_amplitude.png")


def plot_growth_vs_kappa(runs, curve, outdir):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    inv = np.array([c[0] for c in curve])
    ax.plot(inv, [c[1] for c in curve], "-", color=ps.MUTED_CLR, lw=1.8,
            label=rf"linear theory, $k_\parallel d_i={runs[0]['k_mode']:.3f}$")
    for i, run in enumerate(runs):
        x = 0.0 if run["kappa"] is None else 1.0 / run["kappa"]
        ax.errorbar(x, run["gamma"], yerr=run["gamma_err"], fmt="o", ms=7, capsize=4,
                    color=ps.c(SERIES[i]), label=f"PIC, {run['label']}")
    ax.set_xlabel(r"$1/\kappa$  (0 = Maxwellian)")
    ax.set_ylabel(r"$\gamma/\Omega_{ci}$")
    ax.set_title("Ion-cyclotron growth rate: PIC vs linear theory")
    ax.set_xlim(-0.02, max(inv) + 0.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    ps.save(fig, outdir / "growth_rate_vs_kappa.png")


def ic_contour(beta):
    """Hellinger et al. (2006) IC contour for gamma_max = 1e-3 Omega_ci."""
    return 1.0 + 0.43 / np.asarray(beta) ** 0.42


def plot_brazil(runs, outdir):
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    b_all = np.concatenate([r["aniso"]["beta_parallel_global"] for r in runs])
    beta = np.linspace(0.9 * np.nanmin(b_all), 1.1 * np.nanmax(b_all), 200)
    ref = runs[0]["aniso"]
    for idx, ls, when in ((0, "--", "initial"), (-1, "-", "final")):
        be, ae = ref["beta_e_parallel_global"][idx], ref["anisotropy_e_global"][idx]
        if np.isfinite(be) and np.isfinite(ae):
            ax.plot(beta, mirror_threshold_electrons(beta, be, ae), ls, color=ps.c("#CC79A7"),
                    lw=1.6, label=rf"mirror, {when} electrons ($\beta_{{e\parallel}}={be:.1f}$)")
    ax.plot(beta, ic_contour(beta), ":", color=ps.c("#56B4E9"), lw=1.8,
            label=r"IC, $\gamma_{\max}=10^{-3}\Omega_{ci}$")
    for i, run in enumerate(runs):
        a = run["aniso"]
        col = ps.c(SERIES[i])
        ax.plot(a["beta_parallel_global"], a["anisotropy_global"], color=col, lw=1.8, label=run["label"])
        marks = [np.argmin(np.abs(a["omega_ci_t"] - tm)) for tm in (0, 50, 100, 150)]
        ax.plot(a["beta_parallel_global"][marks], a["anisotropy_global"][marks], "o",
                color=col, ms=5)
    # Time labels on the bi-Maxwellian trajectory, placed on the side the
    # other trajectories do not occupy (they cross near t Omega_ci ~ 100).
    last = runs[0]["aniso"]
    offsets = {0: (-10, -8, "right"), 50: (10, -6, "left"), 100: (-10, -10, "right"),
               150: (-10, -10, "right")}
    for tm, (dx, dy, ha) in offsets.items():
        j = int(np.argmin(np.abs(last["omega_ci_t"] - tm)))
        ax.annotate(rf"{tm}", (last["beta_parallel_global"][j], last["anisotropy_global"][j]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=10,
                    color=ps.MUTED_CLR, va="top", ha=ha)
    ax.set_xlabel(r"$\beta_{i\parallel}$")
    ax.set_ylabel(r"$A_i=T_{i\perp}/T_{i\parallel}$")
    ax.set_title(r"Global ion trajectories (labels: $t\,\Omega_{ci}$ of the markers)")
    ax.set_ylim(1.0, 2.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=10)
    ps.save(fig, outdir / "brazil_trajectories.png")


def reduced_pdf(x, kappa):
    """1D marginal of the (bi-)kappa of unit variance; Gaussian for kappa=None."""
    from scipy.special import gammaln
    if kappa is None:
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    nu = 2.0 * kappa - 1.0                     # Student-t degrees of freedom
    s = np.sqrt((nu - 2.0) / nu)               # scale giving unit variance
    z = x / s
    logc = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
    return np.exp(logc - (nu + 1) / 2 * np.log1p(z ** 2 / nu)) / s


def plot_resonance(runs, outdir):
    p = runs[0]["profile"]
    vth = math.sqrt(p["beta_i_par"] / 2.0)     # sqrt(T||/m_i) in v_A
    x = np.linspace(0, 4.5, 400)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for i, run in enumerate(runs):
        ax.plot(x * vth, reduced_pdf(x, run["kappa"]) / vth, color=ps.c(SERIES[i]), lw=1.8,
                label=run["label"])
        vres = abs(run["resonance_v"])
        ax.axvline(vres, color=ps.c(SERIES[i]), lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 0.5)
    ax.set_xlabel(r"$|v_\parallel|/v_A$")
    ax.set_ylabel(r"$f(v_\parallel)\,v_A$  (same $T_\parallel$)")
    ax.set_title("Reduced ion distributions at the cyclotron resonance")
    ratios = ", ".join(rf"{r['label']}: {r['f_res_ratio']:.2f}" for r in runs[1:])
    ax.text(0.98, 0.95, "dotted: $|v_{\\rm res}|=|\\omega_r-\\Omega_{ci}|/k_\\parallel$ of each run\n"
            rf"$f(v_{{\rm res}})/f_{{\rm Maxw}}(v_{{\rm res}})$ — {ratios}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=ps.MUTED_CLR)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    ps.plain_log_axis(ax, "y")
    ps.save(fig, outdir / "resonance.png")


def plot_kappa_local(runs, outdir):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for i, run in enumerate(runs):
        t, kap, err = run["kappa_local"]
        if t.size == 0:
            continue
        inv = np.where(np.isfinite(kap) & (kap > 0), 1.0 / kap, 0.0)
        inv_err = np.where(np.isfinite(err) & np.isfinite(kap) & (kap > 0), err / kap ** 2, 0.0)
        ax.errorbar(t, inv, yerr=inv_err, fmt="o-", ms=5, capsize=3, lw=1.6,
                    color=ps.c(SERIES[i]), label=run["label"])
        t_end = run["t_lin"][1]
        if np.isfinite(t_end):
            ax.axvline(t_end, color=ps.c(SERIES[i]), lw=0.9, ls=":")
    ax.axhline(0.0, color=ps.MUTED_CLR, lw=0.8, ls="--")
    ax.set_xlabel(r"$t\,\Omega_{ci}$")
    ax.set_ylabel(r"$1/\kappa_{\rm eff}$  (0 = Maxwellian)")
    ax.set_title("Ion suprathermal index, local-field frame (dotted: end of linear phase)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    ps.save(fig, outdir / "kappa_local.png")


def write_summary(runs, outdir):
    fields = ["run", "kappa", "k_par_di", "gamma_pic", "gamma_pic_err", "gamma_theory",
              "relative_difference_pct", "omega_r_theory", "v_res_over_vA",
              "f_at_v_res_over_maxwellian", "peak_dB_over_B0", "t_peak", "t_half_relaxation",
              "A_i_final", "beta_e_final", "compressibility"]
    with (outdir / "series_summary.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for run in runs:
            th = run["theory"]
            w.writerow({
                "run": run["name"], "kappa": run["kappa"] or "inf", "k_par_di": f"{run['k_mode']:.4f}",
                "gamma_pic": f"{run['gamma']:.4f}", "gamma_pic_err": f"{run['gamma_err']:.4f}",
                "gamma_theory": f"{th['gamma']:.4f}",
                "relative_difference_pct": f"{100 * (run['gamma'] / th['gamma'] - 1):+.1f}",
                "omega_r_theory": f"{th['omega_r']:.3f}", "v_res_over_vA": f"{run['resonance_v']:.3f}",
                "f_at_v_res_over_maxwellian": f"{run['f_res_ratio']:.3f}",
                "peak_dB_over_B0": f"{run['peak_amplitude']:.3f}", "t_peak": f"{run['t_peak']:.1f}",
                "t_half_relaxation": f"{half_relaxation_time(run):.1f}",
                "A_i_final": f"{run['aniso']['anisotropy_global'][-1]:.3f}",
                "beta_e_final": f"{run['aniso']['beta_e_parallel_global'][-1]:.2f}",
                "compressibility": f"{run['compressibility']:.2g}",
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run analysis roots, Maxwellian first")
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    runs = [load_run(r) for r in args.runs]
    vth = math.sqrt(runs[0]["profile"]["beta_i_par"] / 2.0)
    for run in runs:
        run["theory"] = theory(run, run["kappa"])
        # Cyclotron resonance of an L-mode: omega - k v|| = Omega_ci.
        run["resonance_v"] = (run["theory"]["omega_r"] - 1.0) / run["k_mode"]
        x = abs(run["resonance_v"]) / vth
        run["f_res_ratio"] = float(reduced_pdf(x, run["kappa"]) / reduced_pdf(x, None))
    curve = []
    for inv in INV_KAPPA_GRID:
        root = theory(runs[0], None if inv == 0 else 1.0 / inv)
        curve.append((inv, root["gamma"]))
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_mode_amplitude(runs, args.outdir)
    plot_growth_vs_kappa(runs, curve, args.outdir)
    plot_brazil(runs, args.outdir)
    plot_resonance(runs, args.outdir)
    plot_kappa_local(runs, args.outdir)
    write_summary(runs, args.outdir)
    print(f"Paper figures written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
