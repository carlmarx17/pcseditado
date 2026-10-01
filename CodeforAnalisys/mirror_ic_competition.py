#!/usr/bin/env python3
"""
mirror_ic_competition.py — why the ion-cyclotron mode wins over the mirror mode in these runs
============================================================================================
From the analysis products of the runs (no raw data):

* ``mirror_drive.png``  the distance from the mirror threshold through each
  run, Gamma = sum_s beta_perp,s (A_s - 1) - 1 - (A_i - A_e)^2 / [2 (1/beta_par,i
  + 1/beta_par,e)] (Hellinger 2007, eq. 32, measured ions and electrons), and
  the cold-electron value, next to the anisotropy. Why: the mirror mode can
  only grow while Gamma > 0; the ion-cyclotron wave removes the anisotropy and
  the electrons heat, and both lower Gamma.
* ``mirror_ic_growth.csv``  the growth rates side by side: ion-cyclotron from
  kinetic theory and from the PIC mode fit, the near-threshold mirror estimate
  gamma_m = Omega_ci Gamma^2 / [4 sqrt(3 pi) beta_perp A^(3/2) Pi^(1/2)]
  (Hellinger 2007, eq. 17; valid for Gamma << 1 and cold electrons, so only an
  order of magnitude here), the oblique modes the box holds, and what they did.

Usage:
    python mirror_ic_competition.py RUN_MAXW RUN_K5 RUN_K3 --outdir OUT
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

import psc_units  # noqa: E402
from growth_fit import reference_growth_row  # noqa: E402

SERIES = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def mirror_gamma(beta_par_i, a_i, beta_par_e=0.0, a_e=1.0):
    """Distance from the mirror threshold; cold electrons for beta_par_e = 0."""
    beta_par_i, a_i = np.asarray(beta_par_i, float), np.asarray(a_i, float)
    beta_par_e, a_e = np.asarray(beta_par_e, float), np.asarray(a_e, float)
    drive = beta_par_i * a_i * (a_i - 1.0) + beta_par_e * a_e * (a_e - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        electric = np.where(beta_par_e > 0, (a_i - a_e) ** 2 / (2.0 * (1.0 / beta_par_i + 1.0 / beta_par_e)), 0.0)
    return drive - 1.0 - electric


def mirror_growth_near_threshold(beta_par_i: float, a_i: float) -> float:
    """gamma_m / Omega_ci of Hellinger (2007), eq. (17): cold electrons, Gamma << 1."""
    beta_perp = beta_par_i * a_i
    gamma = beta_perp * (a_i - 1.0) - 1.0
    if gamma <= 0:
        return 0.0
    pi = 1.0 + 0.5 * (beta_perp - beta_par_i)
    return float(gamma ** 2 / (4.0 * np.sqrt(3.0 * np.pi) * beta_perp * a_i ** 1.5 * np.sqrt(pi)))


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load(root: Path) -> dict:
    profile = psc_units._PROFILES[root.name]
    an = _rows(next((root / "01_anisotropy").glob("*anisotropy_evolution.csv")))
    col = lambda k: np.array([_f(r.get(k)) for r in an])
    phys = root / "09_physical_diagnostics"
    ref = reference_growth_row(_rows(phys / "growth_rate_summary.csv")) or {}
    modes = _rows(phys / "mode_growth_table.csv")
    oblique = [m for m in modes if _f(m["theta_kB_deg"]) >= 40.0]
    kappa = profile.get("kappa")
    return {"name": root.name, "profile": profile,
            "label": "bi-Maxwellian" if kappa is None else rf"bi-$\kappa$, $\kappa_0={kappa:g}$",
            "t": col("omega_ci_t"), "A": col("anisotropy_global"), "beta": col("beta_parallel_global"),
            "beta_e": col("beta_e_parallel_global"), "A_e": col("anisotropy_e_global"),
            "gamma_ic": _f(ref.get("gamma")), "gamma_ic_err": _f(ref.get("gamma_err")),
            "t_lin_end": _f(ref.get("linear_phase_end")),
            "peak_ic": max((_f(m["max_amplitude_over_B0"]) for m in modes if _f(m["theta_kB_deg"]) < 20.0), default=np.nan),
            "peak_oblique": max((_f(m["max_amplitude_over_B0"]) for m in oblique), default=np.nan),
            "oblique_accepted": sum(str(m.get("fit_ok")) in ("1", "True") for m in oblique),
            "n_oblique": len(oblique)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    runs = [load(r) for r in args.runs]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True,
                                      gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.08})
    for i, run in enumerate(runs):
        c = ps.c(SERIES[i % len(SERIES)])
        top.plot(run["t"], run["A"], color=c, lw=1.7, label=run["label"])
        hot = mirror_gamma(run["beta"], run["A"], run["beta_e"], run["A_e"])
        cold = mirror_gamma(run["beta"], run["A"])
        bottom.plot(run["t"], hot, color=c, lw=1.7)
        bottom.plot(run["t"], cold, ":", color=c, lw=1.3)
        if np.isfinite(run["t_lin_end"]):
            for ax in (top, bottom):
                ax.axvline(run["t_lin_end"], color=c, lw=0.9, ls=(0, (5, 2, 1, 2)), alpha=0.8)
    bottom.axhline(0.0, color=ps.MUTED_CLR, lw=1.0, ls="--")
    top.set_ylabel(r"$T_{i\perp}/T_{i\parallel}$")
    bottom.set_ylabel(r"mirror drive $\Gamma$")
    bottom.set_xlabel(r"$t\,\Omega_{ci}$")
    top.set_title(r"Distance from the mirror threshold through the runs ($\Gamma > 0$: unstable)", fontsize=12)
    handles, labels = top.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    extra = [Line2D([], [], color=ps.MUTED_CLR, lw=1.7, label="measured electrons"),
             Line2D([], [], color=ps.MUTED_CLR, lw=1.3, ls=":", label="cold electrons"),
             Line2D([], [], color=ps.MUTED_CLR, lw=0.9, ls=(0, (5, 2, 1, 2)), label="end of linear phase"),
             Line2D([], [], color=ps.MUTED_CLR, lw=1.0, ls="--", label=r"mirror threshold, $\Gamma=0$")]
    fig.legend(handles + extra, labels + [h.get_label() for h in extra], loc="upper center",
               bbox_to_anchor=(0.5, 0.03), ncol=4, frameon=False, fontsize=9.5)
    ps.save(fig, args.outdir / "mirror_drive.png")

    rows = []
    for run in runs:
        p = run["profile"]
        b0, a0 = p["beta_i_par"], p["Ti_perp_over_Ti_par"]
        hot = mirror_gamma(run["beta"], run["A"], run["beta_e"], run["A_e"])
        below = np.flatnonzero(hot < 0)
        rho_i = np.sqrt(b0 * a0 / 2.0)                      # thermal gyroradius in d_i
        rows.append({
            "run": run["name"], "gamma_ic_pic": run["gamma_ic"], "gamma_ic_pic_err": run["gamma_ic_err"],
            "mirror_Gamma_initial_cold": float(mirror_gamma(b0, a0)),
            "mirror_Gamma_initial_measured_electrons": float(hot[0]),
            "mirror_Gamma_final": float(hot[-1]),
            "t_mirror_stable": float(run["t"][below[0]]) if below.size else float("nan"),
            "gamma_mirror_near_threshold_estimate": mirror_growth_near_threshold(b0, a0),
            "rho_i_over_di": rho_i, "box_over_rho_i": p["domain_di"] / rho_i,
            "k_perp_rho_i_of_first_box_mode": 2 * np.pi / p["domain_di"] * rho_i,
            "peak_dB_ion_cyclotron": run["peak_ic"], "peak_dB_oblique": run["peak_oblique"],
            "oblique_modes_followed": run["n_oblique"], "oblique_modes_with_accepted_growth": run["oblique_accepted"]})
    with (args.outdir / "mirror_ic_growth.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
