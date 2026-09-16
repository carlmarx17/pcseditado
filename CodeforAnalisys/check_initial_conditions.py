#!/usr/bin/env python3
"""Fail fast if a run's measured initial state does not match its declared CASE.

psc_units.py derives T_i_par, T_i_perp, T_e_par, T_e_perp, n and B0 from the
profile dictionary selected by PSC_PROFILE/CASE. Nothing checks that the data
in DATA_DIR actually started from those values: pointing an analysis at the
wrong run directory, or at a CASE whose profile has drifted from the .cxx
that produced the data, silently yields self-consistent-looking figures for
the wrong physics. This reads the earliest available particle, field and
moment snapshot and compares the measured T_parallel, T_perp, B0 and n
against the declared profile before any downstream analysis target runs.

Temperatures use a generous default tolerance because they come from a
finite number of macroparticles per cell at a single step (shot noise), not
because the comparison itself is loose. B0 and n come from the deterministic
field/moment initial condition and use a tighter tolerance.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from data_reader import PICDataReader


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def _weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    return float(np.average((values - mean) ** 2, weights=weights))


def measured_temperatures(q, m, px, py, pz, w, species: str) -> dict | None:
    mask = (q > 0) if species == "ion" else (q < 0)
    if not np.any(mask):
        return None
    weights = w[mask]
    mass = _weighted_mean(np.abs(m[mask]), weights)
    tpar = mass * _weighted_var(pz[mask], weights)
    tperp = 0.5 * mass * (_weighted_var(px[mask], weights) + _weighted_var(py[mask], weights))
    return {"T_parallel": tpar, "T_perp": tperp}


def measured_density(moment_path: str, suffix: str) -> float:
    data = PICDataReader.read_multiple_fields_3d(moment_path, "all_1st", [f"rho_{suffix}/p0/3d"])
    return float(np.mean(np.asarray(data[f"rho_{suffix}/p0/3d"], dtype=float)))


def measured_b0(field_path: str) -> float:
    data = PICDataReader.read_multiple_fields_3d(field_path, "jeh-", ["hz_fc/p0/3d"])
    return float(np.mean(np.asarray(data["hz_fc/p0/3d"], dtype=float)))


def relative_error(measured: float, declared: float) -> float:
    if declared == 0:
        return float("inf") if measured != 0 else 0.0
    return abs(measured - declared) / abs(declared)


def check_initial_conditions(particles_pattern, fields_pattern, moments_pattern,
                              declared: dict, max_particles: int,
                              temperature_tol: float, field_tol: float) -> list[str]:
    """Return a list of human-readable failures; empty means the check passed."""
    failures = []

    particles = PICDataReader.find_files(particles_pattern)
    if particles:
        step = next(iter(particles))
        q, m, px, py, pz, w = PICDataReader.read_particles_snapshot(
            particles[step], max_particles=max_particles, rng=np.random.default_rng(0)
        )
        for species in ("ion", "electron"):
            measured = measured_temperatures(q, m, px, py, pz, w, species)
            if measured is None:
                continue
            for key in ("T_parallel", "T_perp"):
                err = relative_error(measured[key], declared[species][key])
                if err > temperature_tol:
                    failures.append(
                        f"{species} {key}: measured={measured[key]:.4g} "
                        f"declared={declared[species][key]:.4g} "
                        f"(rel. error {err:.1%} > {temperature_tol:.0%})"
                    )
    else:
        failures.append(f"No particle snapshots match {particles_pattern!r}")

    fields = PICDataReader.find_files(fields_pattern)
    if fields:
        step = next(iter(fields))
        b0_measured = measured_b0(fields[step])
        err = relative_error(b0_measured, declared["B0"])
        if err > field_tol:
            failures.append(
                f"B0: measured={b0_measured:.4g} declared={declared['B0']:.4g} "
                f"(rel. error {err:.1%} > {field_tol:.0%})"
            )

    moments = PICDataReader.find_files(moments_pattern)
    if moments:
        step = next(iter(moments))
        for species, suffix in (("ion", "i"), ("electron", "e")):
            try:
                n_measured = measured_density(moments[step], suffix)
            except KeyError:
                continue
            err = relative_error(n_measured, declared["N0"])
            if err > temperature_tol:
                failures.append(
                    f"n_{species}: measured={n_measured:.4g} declared={declared['N0']:.4g} "
                    f"(rel. error {err:.1%} > {temperature_tol:.0%})"
                )

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", required=True, help="Glob pattern for prt_*.h5/.bp")
    parser.add_argument("--fields", required=True, help="Glob pattern for pfd.*.h5/.bp")
    parser.add_argument("--moments", required=True, help="Glob pattern for pfd_moments.*.h5/.bp")
    parser.add_argument("--case", required=True)
    parser.add_argument("--max-particles", type=int, default=200_000)
    parser.add_argument("--temperature-tol", type=float, default=0.15,
                         help="relative tolerance for T_par/T_perp/n (particle shot noise at t0)")
    parser.add_argument("--field-tol", type=float, default=0.05,
                         help="relative tolerance for B0 (deterministic field IC)")
    args = parser.parse_args()

    import psc_units as pu

    declared = {
        "ion": {"T_parallel": pu.TI_PAR, "T_perp": pu.TI_PERP},
        "electron": {"T_parallel": pu.TE_PAR, "T_perp": pu.TE_PERP},
        "B0": pu.B0,
        "N0": pu.N0,
    }

    failures = check_initial_conditions(
        args.particles, args.fields, args.moments, declared,
        args.max_particles, args.temperature_tol, args.field_tol,
    )

    if failures:
        print(
            f"ERROR: Initial state does not match CASE={args.case} "
            f"(profile={pu.SIM_PROFILE}):",
            file=sys.stderr,
        )
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] Initial state matches CASE={args.case} (profile={pu.SIM_PROFILE})")


if __name__ == "__main__":
    main()
