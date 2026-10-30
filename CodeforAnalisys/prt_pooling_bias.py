#!/usr/bin/env python3
"""How much energy the prt-window pooling moves from bulk to thermal.

``plot_prt._compute_particle_energies`` splits the particle energy of the prt
window using ONE global mean over the whole window:

    E_bulk = 0.5 m |<u>|^2        E_th = 0.5 m <|u - <u>|^2>

Averaging before centring is the same mistake the moment path avoids by
subtracting p_i p_j / (n m) cell by cell. Writing the window as cells c that
hold N_c macroparticles each, the law of total variance is exact:

    E_th(pooled)   = E_th(per cell)   + D
    E_bulk(pooled) = E_bulk(per cell) - D
    D = 0.5 m sum_c w_c |u_c - <u>|^2,        w_c = N_c / N

so pooling moves exactly D out of the bulk channel and into the thermal one.
D is the density-weighted variance of the cell-mean velocity across the
window, and the per-cell moments give it directly: u_c = p_c / (n_c m).

D is not all physics. The cell mean u_c is itself an average over N_c
particles, so it carries shot noise of variance (2 T_perp + T_par) / (m N_c)
per cell even in a perfectly uniform plasma. Since E_th = (2 T_perp + T_par)/2,
that floor is

    D_noise / E_th = 1 / N_c        (one over particles per cell)

which is why the quiet start, where no flow exists yet, is the calibration
point rather than a throwaway sample. The first-order deposit spreads each
particle over neighbouring cells, so this floor is an UPPER bound on the noise
actually present in p_c / n_c. What survives it,

    D_phys = D - D_noise,

is the unresolved flow energy the pooled split really misplaces. Two runs that
differ only in particles per cell separate the two terms: the floor scales as
1 / N_c and the physical part does not.

Two limits of reading it from the moments. The deposit is first order, so a
cell's p_c / n_c already mixes in its neighbours: the cell means are smoothed,
which lowers BOTH the measured D and its noise floor. D measured this way is
therefore a lower bound on the flow structure and the 1 / N_c floor is an
upper bound on the noise, and flow structure below about two cells is invisible
to it -- as it is to any per-cell correction.

Reads only moment files: the window is a fixed block of cells, so no particle
output is needed.

Usage:
    python prt_pooling_bias.py --data-dir RUN --nicell 1000 [--outdir OUT]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps
import psc_units as units
from data_reader import PICDataReader
from plasma_physics import central_pressure_tensor

ps.apply()

MOMENT_NAMES = ("rho", "px", "py", "pz", "txx", "tyy", "tzz", "txy", "tyz", "tzx")
#: ``OutputParticlesParams`` in psc_anisotropy_case.hxx: the central 20 % of
#: each resolved axis, as int(0.4 ngrid) .. int(0.6 ngrid).
WINDOW_LO_FRACTION, WINDOW_HI_FRACTION = 0.4, 0.6
SPECIES = {"ion": ("i", units.M_ION), "electron": ("e", units.M_ELEC)}


def read_moments(path: str, suffix: str) -> dict[str, np.ndarray]:
    raw = PICDataReader.read_multiple_fields_3d(
        path, "all_1st", [f"{name}_{suffix}/p0/3d" for name in MOMENT_NAMES])
    return {key.split("/")[0]: PICDataReader.flatten_2d_slice(value).astype(float)
            for key, value in raw.items()}


def window_slice(shape) -> tuple:
    """The block of cells PSC writes particles for, matching the C++ truncation."""
    return tuple(slice(int(WINDOW_LO_FRACTION * n), int(WINDOW_HI_FRACTION * n))
                 for n in shape[:2])


def dt_omegaci(ngrid: int) -> float:
    """Omega_ci t per step for a run on this grid, from the CFL of the case."""
    dt_code = 0.95 * (units.DOMAIN_DE / ngrid) / np.sqrt(2.0)
    return dt_code * units.OMEGA_CI


def pooling_bias(mom: dict, suffix: str, mass: float, nicell: int) -> dict:
    """Per-particle energies of the window, pooled and per cell.

    Everything is per macroparticle, the normalisation
    ``_compute_particle_energies`` uses, so the numbers are directly the ones
    that script would report.
    """
    tensor = central_pressure_tensor(mom, suffix, mass)
    win = window_slice(tensor["n"].shape)
    n = tensor["n"][win]
    u = np.stack([tensor[f"V{axis}"][win] for axis in "xyz"])
    trace_p = sum(tensor[f"P{axis}{axis}"][win] for axis in "xyz")

    good = (np.isfinite(n) & (n > 0) & np.isfinite(trace_p)
            & np.all(np.isfinite(u), axis=0))
    if not np.any(good):
        raise RuntimeError("no usable cell in the prt window")
    n, trace_p, u = n[good], trace_p[good], u[:, good]
    weight = n / n.sum()

    # Thermal energy with the flow removed per cell: 0.5 m <|u - u_c|^2> is
    # 0.5 tr(P_c) / n_c, the trace being P_par + 2 P_perp.
    e_th_cell = 0.5 * float((weight * trace_p / n).sum())
    u_mean = (weight * u).sum(axis=1)
    d_total = 0.5 * mass * float((weight * ((u - u_mean[:, None]) ** 2).sum(axis=0)).sum())
    e_bulk_pooled = 0.5 * mass * float((u_mean ** 2).sum())
    # Shot-noise floor: var(u_c) = sum_i var(u_i) / N_c with N_c = nicell n_c.
    d_noise = 0.5 * float((weight * trace_p / (n * nicell * n)).sum())
    return {
        "cells": int(n.size),
        "E_th_per_cell": e_th_cell,
        "E_th_pooled": e_th_cell + d_total,
        "E_bulk_pooled": e_bulk_pooled,
        "E_bulk_per_cell": e_bulk_pooled + d_total,
        "D_total": d_total,
        "D_noise_bound": d_noise,
        "D_phys_bound": d_total - d_noise,
        "D_over_E_th": d_total / e_th_cell,
        "D_noise_over_E_th": d_noise / e_th_cell,
        "thermal_inflation": (e_th_cell + d_total) / e_th_cell,
        "bulk_shortfall": d_total / e_bulk_pooled if e_bulk_pooled > 0 else np.nan,
    }


def collect(data_dir: Path, nicell: int, species: list[str]) -> list[dict]:
    moments = PICDataReader.find_files(str(data_dir / units.MOMENT_FILE_PATTERN))
    if not moments:
        raise RuntimeError(f"no moment file matching {units.MOMENT_FILE_PATTERN} in {data_dir}")
    rows = []
    for step in sorted(moments):
        for name in species:
            suffix, mass = SPECIES[name]
            mom = read_moments(moments[step], suffix)
            ngrid = mom[f"rho_{suffix}"].shape[1]
            row = {"step": step, "omega_ci_t": step * dt_omegaci(ngrid),
                   "species": name, "ngrid": ngrid, "nicell": nicell}
            row.update(pooling_bias(mom, suffix, mass, nicell))
            rows.append(row)
    return rows


def write_table(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], path: Path, nicell: int) -> None:
    """Two panels: what the thermal channel gains, what the bulk channel loses."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.8, 6.6), sharex=True)
    styles = (("ion", ps.c("#E69F00"), "s"), ("electron", ps.c("#0072B2"), "^"))

    for name, colour, marker in styles:
        series = [row for row in rows if row["species"] == name]
        if not series:
            continue
        t = np.array([row["omega_ci_t"] for row in series])
        top.plot(t, [100 * row["D_over_E_th"] for row in series], marker + "-",
                 color=colour, lw=1.6, ms=4.5, label=name)
        bottom.plot(t, [row["E_bulk_per_cell"] / row["E_th_per_cell"] for row in series],
                    ":", color=colour, lw=1.8, label=f"{name}: per cell")
        bottom.plot(t, [row["E_bulk_pooled"] / row["E_th_per_cell"] for row in series],
                    marker + "-", color=colour, lw=1.6, ms=4.5, label=f"{name}: one mean")

    # Both species sit on the same floor: n_c is 1 for both, so one line says it.
    top.axhline(100.0 / nicell, color=ps.GRID_CLR, lw=1.2, ls="--",
                label=rf"shot noise, $1/N_c$ at {nicell} ppc")
    top.set_ylabel(r"$D / E_{\rm th}$  [%]")
    top.set_yscale("log")
    ps.plain_log_axis(top, "y")
    ps.style_axes(top, "Energy the prt-window pooling books as thermal")
    top.text(0.30, 0.42, "no flow at $t=0$: the first point is the floor",
             transform=top.transAxes, fontsize=9, color=ps.TEXT_CLR)
    ps.legend(top, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)

    bottom.set_ylabel(r"$E_{\rm bulk} / E_{\rm th}$")
    bottom.set_xlabel(r"$t\,\Omega_{ci}$")
    bottom.set_yscale("log")
    ps.plain_log_axis(bottom, "y")
    ps.style_axes(bottom, "Resolved flow energy: per-cell means against one window mean")
    ps.legend(bottom, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)

    fig.subplots_adjust(right=0.74, hspace=0.32)
    ps.save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--outdir", default="prt_pooling_bias")
    parser.add_argument("--nicell", type=int, default=units.NICELL,
                        help="macroparticles per cell per species, for the noise floor")
    parser.add_argument("--species", nargs="+", default=list(SPECIES), choices=list(SPECIES))
    args = parser.parse_args()

    rows = collect(Path(args.data_dir), args.nicell, args.species)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_table(rows, outdir / "prt_pooling_bias.csv")
    plot(rows, outdir / "prt_pooling_bias.png", args.nicell)

    for name in args.species:
        series = [row for row in rows if row["species"] == name]
        if not series:
            continue
        last = series[-1]
        peak = max(series, key=lambda row: row["D_over_E_th"])
        print(f"[{name}] {len(series)} snapshots, {last['cells']} cells in the window, "
              f"{args.nicell} ppc")
        print(f"    shot-noise bound   D/E_th = {100 * last['D_noise_over_E_th']:.3f} %")
        print(f"    first snapshot     D/E_th = {100 * series[0]['D_over_E_th']:.3f} %")
        print(f"    peak               D/E_th = {100 * peak['D_over_E_th']:.3f} % "
              f"at t Omega_ci = {peak['omega_ci_t']:.1f}")
        print(f"    last               D/E_th = {100 * last['D_over_E_th']:.3f} %, "
              f"bulk channel short by {100 * last['bulk_shortfall']:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
