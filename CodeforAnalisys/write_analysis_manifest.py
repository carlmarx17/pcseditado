#!/usr/bin/env python3
"""Write a reproducibility manifest for one PSC analysis directory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from data_reader import PICDataReader
from psc_units import (
    B0, DOMAIN_DI, DRIVEN_SPECIES, INSTABILITY, MASS_RATIO, N_GRID_Y,
    N_GRID_Z, PARTICLE_BASENAME, PROFILE_LABEL, SIM_PROFILE,
    VA, VA_OVER_C, DT_CODE, OMEGA_CI, DI,
    N0, NICELL, KAPPA, BETA_I_PAR, BETA_E_PAR,
    BETA_I_PERP_OVER_PAR, BETA_E_PERP_OVER_PAR,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()

    discovered = PICDataReader.discover_outputs(str(args.data_dir))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    particle_series = discovered["particles"]
    expected_series = PARTICLE_BASENAME
    if particle_series and expected_series not in particle_series:
        names = ", ".join(sorted(particle_series))
        raise SystemExit(
            f"Particle series does not match CASE={args.case}: "
            f"expected {expected_series}, found {names}"
        )

    fields = discovered["fields"]
    moments = discovered["moments"]
    particles = particle_series.get(expected_series, {})
    common_steps = sorted(set(fields) & set(moments))
    detected_grid = None
    if moments:
        first_moment = moments[min(moments)]
        fields_at_start = PICDataReader.read_multiple_fields_3d(
            first_moment, "all_1st", ["rho_i/p0/3d"]
        )
        detected_grid = [int(size) for size in fields_at_start["rho_i/p0/3d"].shape if size > 1]
        if detected_grid != [N_GRID_Y, N_GRID_Z]:
            raise SystemExit(
                f"Grid/profile mismatch for CASE={args.case}: profile expects "
                f"{N_GRID_Y}x{N_GRID_Z}, HDF5 contains "
                f"{'x'.join(map(str, detected_grid))}"
            )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "case": args.case,
        "profile": SIM_PROFILE,
        "label": PROFILE_LABEL,
        "instability": INSTABILITY,
        "driven_species": DRIVEN_SPECIES,
        "input_directory": str(discovered["data_dir"]),
        "output_directory": str(output_dir),
        "expected_patterns": {
            "fields": "pfd.<step>_p<rank>.h5",
            "moments": "pfd_moments.<step>_p<rank>.h5",
            "particles": f"{PARTICLE_BASENAME}.<step>.h5",
        },
        "detected": {
            "field_files": len(fields),
            "moment_files": len(moments),
            "particle_files": len(particles),
            "paired_field_moment_snapshots": len(common_steps),
            "grid_from_hdf5": detected_grid,
            "first_paired_step": common_steps[0] if common_steps else None,
            "last_paired_step": common_steps[-1] if common_steps else None,
            "first_particle_step": min(particles) if particles else None,
            "last_particle_step": max(particles) if particles else None,
        },
        "physics": {
            "analysis_conventions_version": 3,
            "parameter_source": "analysis profile; verify against runtime log and initial moments",
            "mass_ratio": MASS_RATIO,
            "n0": N0,
            "beta_i_parallel": BETA_I_PAR,
            "A_i": BETA_I_PERP_OVER_PAR,
            "beta_e_parallel": BETA_E_PAR,
            "A_e": BETA_E_PERP_OVER_PAR,
            "kappa": KAPPA,
            "kappa_scope": "both species for maintained anisotropy cases",
            "nicell_from_profile": NICELL,
            "B0": B0,
            "vA_code": VA,
            "vA_over_c": VA_OVER_C,
            "dt_code_from_profile": DT_CODE,
            "omega_ci_code": OMEGA_CI,
            "di_code": DI,
            "mirror_reference": "cold-electron bi-Maxwellian: beta_parallel*A*(A-1)=1",
            "domain_di": DOMAIN_DI,
            "grid": [N_GRID_Y, N_GRID_Z],
        },
    }

    path = output_dir / f"{args.case}_analysis_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {path}")


if __name__ == "__main__":
    main()
