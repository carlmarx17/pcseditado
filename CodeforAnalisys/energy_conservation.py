#!/usr/bin/env python3
"""Global PSC energy from DiagEnergies output, including restart segments.

EX2..BZ2 are volume integrals of squared fields, not energies. The factor
1/2 is applied here. E_electron and E_ion already contain m*(gamma-1),
particle weights, cell volume and the MPI sum. See DiagEnergiesField.h and
DiagEnergiesParticle.h. Valid for the periodic anisotropy runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps
from psc_units import OMEGA_CI

ps.apply()

FIELD_COLUMNS = ("EX2", "EY2", "EZ2", "BX2", "BY2", "BZ2")
REQUIRED_COLUMNS = ("time", *FIELD_COLUMNS, "E_electron", "E_ion")


def read_energy_segments(paths: list[Path]) -> tuple[list[dict], dict]:
    """Merge consistent overlaps; reject conflicting restarts and invalid data."""
    records = {}
    used = []
    for path in paths:
        if path.stat().st_size == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            data = np.genfromtxt(path, names=True, dtype=float, ndmin=1)
        if data.size == 0:
            continue
        if not set(REQUIRED_COLUMNS) <= set(data.dtype.names or ()):
            raise ValueError(f"{path}: expected DiagEnergies columns {REQUIRED_COLUMNS}")
        matrix = np.column_stack([data[key] for key in REQUIRED_COLUMNS])
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError(f"{path}: non-finite or negative time/energy")
        if np.any(np.diff(matrix[:, 0]) <= 0):
            raise ValueError(f"{path}: energy times are not strictly increasing")
        for row in matrix:
            time = float(row[0])
            if time in records and not np.allclose(records[time], row, rtol=5e-6, atol=1e-12):
                raise ValueError(f"Conflicting restart energy at t={time}; select one simulation history")
            records[time] = row
        used.append(str(path.resolve()))
    if not records:
        raise ValueError("No global energy samples; enable PSC_ENERGIES_EVERY for future runs")

    rows = []
    for time in sorted(records):
        raw = records[time]
        e_electric, e_magnetic = 0.5 * np.sum(raw[1:4]), 0.5 * np.sum(raw[4:7])
        electron, ion = raw[7], raw[8]
        rows.append({
            "time_code": time, "omega_ci_t": time * OMEGA_CI,
            "E_E": float(e_electric), "E_B": float(e_magnetic),
            "E_e": float(electron), "E_i": float(ion),
            "E_total": float(e_electric + e_magnetic + electron + ion),
        })
    baseline = rows[0]["E_total"]
    kinetic_baseline = rows[0]["E_e"] + rows[0]["E_i"]
    if baseline <= 0:
        raise ValueError("Initial total energy must be positive")
    for row in rows:
        change = row["E_total"] - baseline
        row["relative_change"] = change / baseline
        row["change_over_initial_kinetic"] = change / kinetic_baseline if kinetic_baseline > 0 else float("nan")
    # The relative change of E_total is diluted by the (large, constant) B0
    # and thermal energies. The criterion that matters for the physics is
    # whether the error is small compared with the energy the instability
    # actually moves between reservoirs.
    exchanged = max(max(abs(r[k] - rows[0][k]) for r in rows) for k in ("E_B", "E_i", "E_e"))
    max_error = max(abs(r["E_total"] - baseline) for r in rows)
    for row in rows:
        for key in ("E_E", "E_B", "E_e", "E_i", "E_total"):
            row[f"d{key}"] = row[key] - rows[0][key]
    summary = {
        "source_files": used, "n_samples": len(rows),
        "baseline_time_code": rows[0]["time_code"],
        "includes_simulation_t0": rows[0]["time_code"] == 0.0,
        "last_time_code": rows[-1]["time_code"],
        "max_abs_relative_change": max(abs(r["relative_change"]) for r in rows),
        "final_relative_change": rows[-1]["relative_change"],
        "max_exchanged_energy": exchanged,
        "max_error_over_exchanged": max_error / exchanged if exchanged > 0 else float("nan"),
        "energy_scope": "global domain, both species, electric and full magnetic field",
        "detrended": False,
    }
    return rows, summary


def write_energy_analysis(paths: list[Path], outdir: Path) -> dict:
    rows, summary = read_energy_segments(paths)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "global_energy_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (outdir / "global_energy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    time = [r["omega_ci_t"] for r in rows]
    e0 = rows[0]["E_total"]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.2), sharex=True)
    # Changes, not absolute values: the absolute energies are dominated by
    # the constant B0^2/2 and the initial thermal energy, which hides the
    # exchange the instability produces.
    for key, color, label in (("dE_B", "#56d364", r"$\Delta E_B$"),
                              ("dE_E", "#f2cc60", r"$\Delta E_E$"),
                              ("dE_i", "#ff7b72", r"$\Delta K_i$"),
                              ("dE_e", "#58a6ff", r"$\Delta K_e$"),
                              ("dE_total", "#111111", r"$\Delta E_{\rm tot}$")):
        axes[0].plot(time, [r[key] / e0 for r in rows], color=ps.c(color), label=label,
                     lw=2.2 if key == "dE_total" else 1.6)
    axes[0].axhline(0.0, color=ps.MUTED_CLR, linewidth=0.7)
    axes[0].set_ylabel(r"$\Delta E(t)/E_{\rm tot}(t_0)$")
    axes[0].set_title("Global energy budget (DiagEnergies, no detrending)", fontsize=13)
    ps.legend(axes[0], fontsize=10, ncol=3)
    axes[1].plot(time, [r["relative_change"] for r in rows], color=ps.c("#111111"))
    axes[1].axhline(0.0, color=ps.MUTED_CLR, linewidth=0.7)
    axes[1].set_ylabel(r"$(E_{\rm tot}(t)-E_{\rm tot}(t_0))/E_{\rm tot}(t_0)$")
    axes[1].set_xlabel(r"$t\,\Omega_{ci}$")
    ratio = summary["max_error_over_exchanged"]
    if np.isfinite(ratio):
        axes[1].text(0.02, 0.9, rf"max $|\Delta E_{{\rm tot}}|$ / max exchanged = {ratio:.2g}",
                     transform=axes[1].transAxes, fontsize=10, color=ps.TEXT_CLR, zorder=5,
                     bbox={"facecolor": ps.LEGEND_BG, "edgecolor": ps.GRID_CLR, "alpha": 0.9})
    for ax in axes:
        ps.style_axes(ax)
    fig.tight_layout()
    ps.save(fig, outdir / "global_energy_conservation.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="diag.asc and selected restart segments from the same run")
    parser.add_argument("--outdir", type=Path, default=Path("physical_diagnostics"))
    args = parser.parse_args()
    summary = write_energy_analysis(args.files, args.outdir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
