#!/usr/bin/env python3
"""Fail fast if the on-disk grid resolution does not match PSC_PROFILE.

psc_units.py hard-codes N_GRID_Y/N_GRID_Z per profile. If a run was actually
written at a different resolution (e.g. a restart with a different job
config), every downstream script silently reshapes/plots the wrong grid.
This check reads one real snapshot and compares its shape to the profile
before any analysis target runs.
"""

from __future__ import annotations

import argparse
import sys

from data_reader import PICDataReader
from psc_units import N_GRID_Y, N_GRID_Z, SIM_PROFILE


def _detect_grid(path: str, group_prefix: str, dataset: str) -> list[int]:
    fields = PICDataReader.read_multiple_fields_3d(path, group_prefix, [dataset])
    shape = fields[dataset].shape
    return [int(size) for size in shape if size > 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()

    discovered = PICDataReader.discover_outputs(args.data_dir)
    moments = discovered["moments"]
    fields = discovered["fields"]

    if moments:
        path = moments[min(moments)]
        detected = _detect_grid(path, "all_1st", "rho_i/p0/3d")
    elif fields:
        path = fields[min(fields)]
        detected = _detect_grid(path, "jeh-", "hx_fc/p0/3d")
    else:
        return

    expected = [N_GRID_Y, N_GRID_Z]
    if detected != expected:
        print(
            f"ERROR: Grid/profile mismatch for CASE={args.case} "
            f"(profile={SIM_PROFILE}): profile expects "
            f"{expected[0]}x{expected[1]}, HDF5/BP contains "
            f"{'x'.join(map(str, detected))} (from {path})",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
