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
    summary = {
        "source_files": used, "n_samples": len(rows),
        "baseline_time_code": rows[0]["time_code"],
        "includes_simulation_t0": rows[0]["time_code"] == 0.0,
        "last_time_code": rows[-1]["time_code"],
        "max_abs_relative_change": max(abs(r["relative_change"]) for r in rows),
        "final_relative_change": rows[-1]["relative_change"],
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
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for key in ("E_E", "E_B", "E_e", "E_i", "E_total"):
        axes[0].plot(time, [r[key] for r in rows], label=key)
    axes[0].set_ylabel("Global energy [code]")
    axes[0].legend()
    axes[1].plot(time, [r["relative_change"] for r in rows])
    axes[1].axhline(0.0, color="gray", linewidth=0.7)
    axes[1].set_ylabel(r"$(E(t)-E(t_0))/E(t_0)$")
    axes[1].set_xlabel(r"$t\Omega_{ci}$")
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
