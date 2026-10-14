#!/usr/bin/env python3
"""Resolve the actual run settings and report the last expected field output.

Output is four shell-readable values: last step, expected step, nmax, cadence.
A missing/ambiguous series or invalid runtime configuration exits nonzero.
The caller decides whether an incomplete run is allowed.
"""
import argparse
import os
from pathlib import Path


def completion(fields, nmax, fields_every):
    if not fields:
        raise ValueError('No field snapshots found')
    if nmax <= 0 or fields_every <= 0:
        raise ValueError('nmax and field cadence must be positive')
    return max(fields), (int(nmax) // int(fields_every)) * int(fields_every)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    os.environ['PSC_PROFILE'] = args.case
    os.environ['PSC_ANALYSIS_DATA_DIR'] = str(args.data_dir.resolve())
    runtime = args.data_dir / 'analysis_config.json'
    if runtime.is_file() and not os.environ.get('PSC_ANALYSIS_CONFIG'):
        os.environ['PSC_ANALYSIS_CONFIG'] = str(runtime.resolve())
    try:
        from data_reader import PICDataReader
        import psc_units as units
        fields = PICDataReader.discover_outputs(str(args.data_dir))['fields']
        last, expected = completion(fields, units.NMAX, units.FIELDS_EVERY)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(2, f'Cannot validate run {args.case}: {exc}\n')
    print(last, expected, units.NMAX, units.FIELDS_EVERY)


if __name__ == '__main__':
    main()
