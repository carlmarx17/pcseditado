"""Resolve run geometry without loading a complete field time series.

Coordinates are in code lengths, array storage is (z,y,x). Existing analysis
modules assume a square yz domain; reject unsupported geometry explicitly.
"""
import json
import os
from pathlib import Path
import numpy as np
from data_reader import PICDataReader


def resolve_run(profile):
    values = dict(profile)
    provenance = {"grid": "profile", "domain_di": "profile", "dt_code": "CFL estimate; verify runtime log", "nicell": "profile; verify runtime log"}
    config_path = os.environ.get("PSC_ANALYSIS_CONFIG")
    config = json.loads(Path(config_path).read_text()) if config_path else {}
    allowed = {"ngrid", "domain_di", "dt_code", "nicell", "nmax"}
    if set(config) - allowed:
        raise ValueError(f"Unknown run settings: {sorted(set(config)-allowed)}")
    for key, val in config.items():
        if isinstance(val, bool) or not np.isfinite(val) or val <= 0:
            raise ValueError(f"Invalid run setting {key}={val}")
        if key in {"ngrid", "nicell", "nmax"} and int(val) != val:
            raise ValueError(f"{key} must be an integer")
        values[key] = val
        provenance["grid" if key == "ngrid" else key] = str(Path(config_path).resolve())
    data_dir = os.environ.get("PSC_ANALYSIS_DATA_DIR")
    if data_dir:
        discovered = PICDataReader.discover_outputs(data_dir)
        files = discovered['fields'] or discovered['moments']
        if files:
            path = files[min(files)]
            with PICDataReader.open_data_file(path) as f:
                keys = f.keys()
                field = PICDataReader.resolve_dataset_path(keys, 'jeh-', 'hx_fc/p0/3d')
                if field is None:
                    field = PICDataReader.resolve_dataset_path(keys, 'all_1st', 'rho_i/p0/3d')
                if field is None:
                    raise ValueError(f"Cannot determine grid from {path}")
                shape = f.handle[field].shape if f.backend == 'hdf5' else f.read(field).shape
                if len(shape) != 3 or shape[2] != 1 or shape[0] != shape[1]:
                    raise ValueError(f"Expected square yz grid (Nz, Ny, 1), got {shape}; unsupported geometry")
                n = int(shape[0])
                if 'ngrid' in config and n != int(config['ngrid']):
                    raise ValueError(f"Configured ngrid={config['ngrid']} conflicts with snapshot {shape}")
                values['ngrid'] = n
                provenance['grid'] = str(path)
                lengths = []
                for axis in [1, 2]:
                    coord = PICDataReader.resolve_variable_path(keys, f'crd[{axis}]/p0/1d')
                    if coord is None:
                        coord = PICDataReader.resolve_dataset_path(keys, f'crd[{axis}]', f'crd[{axis}]/p0/1d')
                    if coord is None:
                        continue
                    x = np.asarray(f.read(coord), dtype=float).ravel()
                    if len(x) not in (n, n+1) or not np.all(np.isfinite(x)):
                        raise ValueError(f"Coordinate length does not match grid in {path}")
                    dx = np.diff(x)
                    if np.any(dx <= 0) or not np.allclose(dx, dx.mean(), rtol=2e-4, atol=1e-9):
                        raise ValueError(f"Nonuniform coordinates in {path}")
                    lengths.append(float(dx.mean()*n/np.sqrt(values['mass_ratio'])))
                if len(lengths) == 1:
                    raise ValueError(f"Only one coordinate axis available in {path}")
                if lengths:
                    if not np.isclose(*lengths, rtol=2e-4):
                        raise ValueError("Non-square physical domain is unsupported")
                    length = float(np.mean(lengths))
                    if 'domain_di' in config and not np.isclose(length, config['domain_di'], rtol=2e-4):
                        raise ValueError("Configured domain_di conflicts with snapshot coordinates")
                    values['domain_di'] = length
                    provenance['domain_di'] = str(path)
    values['ngrid'] = int(values['ngrid'])
    values['nicell'] = int(values['nicell'])
    return values, provenance
