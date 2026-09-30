#!/usr/bin/env python3
"""
compare_physical_cases.py
=========================
Compare integrated diagnostics across PSC cases, e.g. bi-Maxwellian vs
bi-Kappa twins of the same regime.

Every comparison follows the species that drives the instability (read from
the analysis manifests): ions for mirror/firehose, electrons for whistler.
The magnetic amplitude is the vector fluctuation <|dB|^2>^1/2/B0, gamma is
the reference row of growth_rate_summary.csv (the dominant Fourier mode of dB)
quoted with its error, the energy panel uses the global DiagEnergies budget
when it exists, and the heat flux the third-moment table of
heat_flux_analysis.py. The run parameters must match (``validate_comparison``)
unless --allow-parameter-mismatch is given; realizations of one case are
compared with convergence_study.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from analysis_contract import strict_dumps
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plot_style as ps
from growth_fit import BRANCH_SERIES, branch_growth_rows, reference_growth_row
from analysis_contract import gap_segments

ps.apply()

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

DARK_BG = ps.c("#0d1117")
PANEL_BG = ps.c("#161b22")
TEXT_CLR = ps.c("#e6edf3")
GRID_CLR = ps.c("#30363d")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, direction="in", which="both", top=True, right=True)
    ax.grid(True, color=GRID_CLR, alpha=0.22, linestyle=":")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ps.save(fig, path)


def parse_case_arg(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        name, path = raw.split("=", 1)
        return name.strip(), Path(path).expanduser()
    path = Path(raw).expanduser()
    return path.name, path


def load_case(name: str, path: Path) -> dict:
    manifests = list(path.glob("*_analysis_manifest.json")) or list(path.parent.glob("*_analysis_manifest.json"))
    if len(manifests) > 1:
        raise ValueError(f"Ambiguous analysis manifests for {name}: {manifests}")
    manifest = json.loads(manifests[0].read_text()) if manifests else {}
    rows = {}
    for table_name, table_path in [
        ("anisotropy", path / "anisotropy_table.csv"),
        ("fit", path / "fit_metrics.csv"),
        ("field", path / "field_fluctuation_table.csv"),
        ("energy", path / "energy_table.csv"),
    ]:
        for row in _read_csv(table_path):
            step = int(_to_float(row.get("step"), -1))
            if step < 0:
                continue
            rows.setdefault(step, {"case": name, "step": step})
            rows[step].update({f"{table_name}_{k}": v for k, v in row.items()})

    gamma_rows = _read_csv(path / "growth_rate_summary.csv")
    total = reference_growth_row(gamma_rows)
    gamma_raw = _to_float(total.get("gamma")) if total else np.nan
    gamma_err = _to_float(total.get("gamma_err")) if total else np.nan
    fit_ok = bool(total) and total.get("fit_ok", "").strip().lower() in ("1", "true")
    gamma_series = (total.get("series") or "total") if total else "none"
    gamma = gamma_raw if fit_ok else np.nan
    # Growth rate of each branch (mirror / ion-cyclotron for T_perp > T_par
    # ions): the reference gamma is that of whichever branch dominated, so a
    # kappa-vs-Maxwellian comparison of the mirror growth rate needs these.
    branch_gamma = {}
    for branch, row in branch_growth_rows(gamma_rows).items():
        ok = row.get("fit_ok", "").strip().lower() in ("1", "true")
        branch_gamma[branch] = (_to_float(row.get("gamma")) if ok else np.nan,
                                _to_float(row.get("gamma_err")) if ok else np.nan)
    driven = manifest.get("driven_species", "ion")
    s = "e" if driven == "electron" else "i"

    # Global energy budget (DiagEnergies), when the run has it.
    energy_rows = _read_csv(path / "global_energy_table.csv")
    energy_t = np.array([_to_float(r.get("omega_ci_t")) for r in energy_rows])
    energy_rel = np.array([_to_float(r.get("relative_change")) for r in energy_rows])
    # Third-moment heat flux of the driven species (06_heat_flux).
    heat_rows = [r for r in _read_csv(path.parent / "06_heat_flux" / "heat_flux_table.csv")
                 if r.get("species") == driven]
    heat_by_step = {int(_to_float(r.get("step"), -1)): r for r in heat_rows}
    merged = []
    for step in sorted(rows):
        row = rows[step]
        time = (
            row.get("anisotropy_omega_ci_t")
            or row.get("fit_omega_ci_t")
            or row.get("field_omega_ci_t")
            or row.get("energy_omega_ci_t")
        )
        merged.append({
            "case": name,
            "step": step,
            "omega_ci_t": _to_float(time),
            "driven_species": driven,
            "A_i": _to_float(row.get("anisotropy_A_i")),
            "A_e": _to_float(row.get("anisotropy_A_e")),
            "A_driven": _to_float(row.get(f"anisotropy_A_{s}")),
            "R_driven": _to_float(row.get(f"anisotropy_R_{s}")),
            "beta_parallel_driven": _to_float(row.get(f"anisotropy_beta_parallel_{s}")),
            "delta_B_vec_rms_over_B0": _to_float(
                row.get("field_delta_B_vec_rms_over_B0", row.get("field_delta_B_rms_over_B0"))),
            "delta_B_parallel_rms_over_B0": _to_float(row.get("field_delta_B_parallel_rms_over_B0")),
            "delta_B_perp_rms_over_B0": _to_float(row.get("field_delta_B_perp_rms_over_B0")),
            "gamma": gamma,
            "gamma_err": gamma_err,
            "gamma_raw": gamma_raw,
            "gamma_fit_ok": fit_ok,
            "kappa_fit": _to_float(row.get("fit_kappa_fit")),
            "F_supra": _to_float(row.get("fit_suprathermal_fraction")),
            "E_B": _to_float(row.get("energy_E_B")),
            "energy_relative_change": (
                float(np.interp(_to_float(time), energy_t, energy_rel, left=np.nan, right=np.nan))
                if energy_rows else np.nan),
            "abs_q_par_over_q0": _to_float(heat_by_step.get(step, {}).get("abs_q_par_over_q0_smax6",
                                           heat_by_step.get(step, {}).get("abs_q_par_over_q0"))),
        })
    return {"name": name, "rows": merged, "gamma": gamma, "gamma_err": gamma_err,
            "gamma_series": gamma_series, "manifest": manifest, "driven_species": driven,
            "branch_gamma": branch_gamma}


def validate_comparison(cases: list[dict], mode: str = "distribution") -> list[str]:
    """Check declared parameters; this does not certify the initial measured VDF."""
    fixed = ["mass_ratio", "n0", "B0", "beta_i_parallel", "A_i", "beta_e_parallel", "A_e"]
    if mode == "distribution":
        fixed += ["domain_di", "grid", "dt_code_from_profile", "nicell_from_profile"]
    elif mode == "convergence":
        fixed += ["kappa"]
    else:
        raise ValueError("comparison mode must be distribution or convergence")
    errors = []
    for case in cases:
        physics = case["manifest"].get("physics", {})
        for key in fixed:
            if key not in physics:
                errors.append(f"{case['name']}: missing {key}; regenerate its manifest")
    if errors or not cases:
        return errors
    reference = cases[0]["manifest"]["physics"]
    for case in cases[1:]:
        physics = case["manifest"]["physics"]
        for key in fixed:
            left, right = reference[key], physics[key]
            equal = left == right if left is None or right is None else np.allclose(left, right, rtol=1e-10, atol=1e-12)
            if not equal:
                errors.append(f"{case['name']}: {key}={right} differs from {cases[0]['name']} ({left})")
    return errors


def validate_estimators(cases):
    signatures = {(c.get("gamma_series", "none"),
                   c["manifest"].get("physics", {}).get("analysis_conventions_version"),
                   strict_dumps(c["manifest"].get("provenance", {}).get("algorithms", {}), sort_keys=True))
                  for c in cases}
    if len(signatures) > 1:
        raise ValueError("Mixed estimator or algorithm versions; regenerate all cases together")


CASE_COLORS = ["#58a6ff", "#ff7b72", "#56d364", "#d2a8ff", "#f2cc60", "#a8d4ff"]
LINESTYLES = ["-", "--", ":", "-."]


def plot_timeseries(cases: list[dict], ykeys: list[str], labels: list[str], path: Path, title: str, yscale=None):
    if not any(np.any(np.isfinite([r[k] for r in c["rows"]])) for c in cases for k in ykeys if c["rows"]):
        return
    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.patch.set_facecolor(DARK_BG)
    _style(ax)
    for i, case in enumerate(cases):
        rows = case["rows"]
        if not rows:
            continue
        t_all = np.array([r["omega_ci_t"] for r in rows], dtype=float)
        color = ps.c(CASE_COLORS[i % len(CASE_COLORS)])
        for j, (ykey, label) in enumerate(zip(ykeys, labels)):
            y = np.array([r[ykey] for r in rows], dtype=float)
            if yscale == "log":   # t = 0 is the uniform initial field (ps.measured_fluctuation)
                y = np.where(ps.measured_fluctuation(t_all), y, np.nan)
            # Columns sampled at the particle cadence (anisotropy, heat flux)
            # are NaN on every other field snapshot of the merged table; a
            # line through the NaNs draws nothing, so plot the finite samples.
            ok = np.isfinite(t_all) & np.isfinite(y)
            if np.any(ok):
                for segment, (ts, ys) in enumerate(gap_segments(t_all, y)):
                    ax.plot(ts, ys, LINESTYLES[j % len(LINESTYLES)], color=color, lw=1.7,
                            marker="o" if len(ts) < 60 else None, ms=3.5,
                            label=f"{case['name']} {label}" if segment == 0 else None)
    if yscale:
        ax.set_yscale(yscale)
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel(", ".join(labels), color=TEXT_CLR)
    ax.set_title(title, color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _save(fig, path)


def plot_growth_bars(cases: list[dict], path: Path):
    names = [case["name"] for case in cases]
    gamma = np.array([case["gamma"] for case in cases], dtype=float)
    err = np.array([case.get("gamma_err", np.nan) for case in cases], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style(ax)
    colors = [ps.c(CASE_COLORS[i % len(CASE_COLORS)]) for i in range(len(cases))]
    valid = np.isfinite(gamma)
    indices = np.arange(len(names))
    ax.bar(indices[valid], gamma[valid], yerr=np.where(np.isfinite(err[valid]), err[valid], 0.),
           color=np.asarray(colors)[valid], capsize=5, ecolor=TEXT_CLR)
    ax.set_xticks(indices, names)
    for i, g in enumerate(gamma):
        if not np.isfinite(g):
            ax.text(i, 0.0, "no valid fit", ha="center", va="bottom", color=TEXT_CLR, fontsize=10)
    series = {case.get("gamma_series", "total") for case in cases}
    amplitude = (r"dominant Fourier mode of $\delta\mathbf{B}$" if series == {"mode"}
                 else r"$|\delta\mathbf{B}|_{\rm rms}$" if series == {"total"}
                 else "mixed estimators: regenerate the v5 summaries")
    ax.set_ylabel(r"$\gamma/\Omega_{ci}$", color=TEXT_CLR)
    ax.set_xlabel(f"fit of the {amplitude}", color=TEXT_CLR)
    ax.set_title("Growth-rate comparison", color=TEXT_CLR, fontweight="bold")
    _save(fig, path)


#: Legend names of the geometric branches, and their instability for
#: T_perp > T_par ions (physical_diagnostics.physical_branch).
BRANCH_LABELS = {"compressive_oblique": ("compressive, oblique", "mirror"),
                 "transverse_parallel": ("transverse, parallel", "ion-cyclotron")}


def plot_branch_growth_bars(cases: list[dict], path: Path):
    """Grouped bars: growth rate of each branch in each case."""
    branches = [b for b in BRANCH_SERIES if any(b in c.get("branch_gamma", {}) for c in cases)]
    if not branches:
        return
    ion_driven = all(c["driven_species"] == "ion" and
                     (c["manifest"].get("physics", {}).get("A_i") or 0) > 1 for c in cases)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.patch.set_facecolor(DARK_BG)
    _style(ax)
    indices = np.arange(len(cases))
    width = 0.8 / len(branches)
    colors = ("#58a6ff", "#ffa657")
    for j, branch in enumerate(branches):
        values = np.array([c.get("branch_gamma", {}).get(branch, (np.nan, np.nan)) for c in cases], dtype=float)
        x = indices + (j - (len(branches) - 1) / 2) * width
        geometric, physical = BRANCH_LABELS[branch]
        label = physical if ion_driven else geometric
        ok = np.isfinite(values[:, 0])
        ax.bar(x[ok], values[ok, 0], width * 0.92,
               yerr=np.where(np.isfinite(values[ok, 1]), values[ok, 1], 0.),
               color=ps.c(colors[j % len(colors)]), capsize=4, ecolor=TEXT_CLR, label=label)
        for xi in x[~ok]:
            ax.text(xi, 0.0, "no fit", ha="center", va="bottom", color=TEXT_CLR, fontsize=9, rotation=90)
    ax.set_xticks(indices, [c["name"] for c in cases])
    ax.set_ylabel(r"$\gamma/\Omega_{ci}$", color=TEXT_CLR)
    ax.set_xlabel("strongest Fourier mode of each branch", color=TEXT_CLR)
    ax.set_title("Growth rate per branch", color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(branches))
    _save(fig, path)


def main():
    parser = argparse.ArgumentParser(description="Compare physical diagnostics between PSC cases.")
    parser.add_argument("cases", nargs="+", help="Case directories, optionally NAME=/path/to/09_physical_diagnostics")
    parser.add_argument("--outdir", default="comparison_physical", help="Output directory.")
    parser.add_argument("--comparison-mode", choices=["distribution", "convergence"], default="distribution")
    parser.add_argument("--allow-parameter-mismatch", action="store_true",
                        help="Produce an explicitly uncontrolled comparison despite missing/inconsistent metadata.")
    args = parser.parse_args()

    cases = [load_case(*parse_case_arg(raw)) for raw in args.cases]
    validate_estimators(cases)
    issues = validate_comparison(cases, args.comparison_mode)
    if issues and not args.allow_parameter_mismatch:
        raise SystemExit("Comparison is not controlled:\n" + "\n".join(issues))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "comparison_validation.json").write_text(strict_dumps({
        "mode": args.comparison_mode, "declared_parameters_match": not issues,
        "issues": issues, "requires_initial_moment_validation": True,
        "scientific_status": "UNVERIFIED",
        "reason": "Parameter parity alone does not establish energy conservation, mode classification or convergence",
        "common_time_coverage": [max((c["rows"][0]["omega_ci_t"] for c in cases if c["rows"]), default=None),
                                 min((c["rows"][-1]["omega_ci_t"] for c in cases if c["rows"]), default=None)],
    }, indent=2), encoding="utf-8")
    if issues:
        print("[WARN] Uncontrolled comparison:\n" + "\n".join(issues))
    rows = [row for case in cases for row in case["rows"]]
    _write_csv(outdir / "comparison_kappa_vs_maxwellian.csv", rows)

    drivers = {case["driven_species"] for case in cases}
    if len(drivers) > 1:
        raise SystemExit(f"Cases are driven by different species {sorted(drivers)}; "
                         "compare within one instability family.")
    s = "e" if drivers == {"electron"} else "i"
    plot_timeseries(cases, ["A_driven"], [rf"$A_{s}$"],
                    outdir / "comparison_anisotropy.png",
                    f"Anisotropy of the driven species ({next(iter(drivers))}s)")
    plot_timeseries(cases, ["delta_B_vec_rms_over_B0"],
                    [r"$\langle|\delta\mathbf{B}|^2\rangle^{1/2}/B_0$"],
                    outdir / "comparison_deltaB.png", "Magnetic-fluctuation comparison", yscale="log")
    plot_timeseries(cases, ["delta_B_parallel_rms_over_B0", "delta_B_perp_rms_over_B0"],
                    [r"$\delta B_\parallel$", r"$\delta B_\perp$"],
                    outdir / "comparison_deltaB_components.png",
                    "Compressive vs transverse fluctuation (/B0)", yscale="log")
    plot_growth_bars(cases, outdir / "comparison_growth_rate.png")
    plot_branch_growth_bars(cases, outdir / "comparison_growth_rate_branches.png")
    _write_csv(outdir / "comparison_growth_rate_branches.csv", [
        {"case": c["name"], "branch": b, "gamma": g, "gamma_err": e}
        for c in cases for b, (g, e) in sorted(c.get("branch_gamma", {}).items())])
    plot_timeseries(cases, ["energy_relative_change"], [r"$\Delta E_{\rm tot}/E_{\rm tot}$"],
                    outdir / "comparison_energy.png", "Global energy conservation (DiagEnergies)")
    plot_timeseries(cases, ["abs_q_par_over_q0"], [rf"$\langle|q_{{\parallel {s}}}|\rangle/q_0$"],
                    outdir / "comparison_heat_flux.png",
                    r"Heat flux of the driven species (third moment, $s_{\max}=6$)")
    print(f"Comparison written to {outdir}")


if __name__ == "__main__":
    main()
