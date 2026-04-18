#!/usr/bin/env python3
"""Helpers to read PSC particle ADIOS2 outputs in old and packed formats."""

from __future__ import annotations

import glob
import os
import re

import adios2
import numpy as np

from psc_units import MASS_RATIO

STEP_RE = re.compile(r"\.(\d+)(?:_p\d+)?\.bp$")

_DEFAULT_KIND_CHARGE = {
    0: -1.0,
    1: 1.0,
}

_DEFAULT_KIND_MASS = {
    0: 1.0,
    1: float(MASS_RATIO),
}


def _read_var_by_suffix(reader: adios2.FileReader, variables: dict, suffix: str):
    for key in variables:
        if key == suffix or key.endswith(f"::{suffix}") or key.endswith(f"/{suffix}"):
            variable = reader.inquire_variable(key)
            if variable is None:
                continue
            return reader.read(variable)
    raise KeyError(suffix)


def _read_var_by_suffix_optional(
    reader: adios2.FileReader, variables: dict, suffix: str, default=None
):
    try:
        return _read_var_by_suffix(reader, variables, suffix)
    except KeyError:
        return default


def _load_flat_particles(reader: adios2.FileReader, variables: dict) -> dict[str, np.ndarray]:
    q = _read_var_by_suffix(reader, variables, "q")
    m = _read_var_by_suffix(reader, variables, "m")
    px = _read_var_by_suffix(reader, variables, "px")
    py = _read_var_by_suffix(reader, variables, "py")
    pz = _read_var_by_suffix(reader, variables, "pz")
    x = _read_var_by_suffix_optional(reader, variables, "x")
    y = _read_var_by_suffix_optional(reader, variables, "y")
    z = _read_var_by_suffix_optional(reader, variables, "z")
    w = _read_var_by_suffix_optional(reader, variables, "w")

    if w is None:
        w = np.ones_like(q, dtype=float)

    return {
        "format": "flat",
        "q": np.asarray(q, dtype=float),
        "m": np.asarray(m, dtype=float),
        "w": np.asarray(w, dtype=float),
        "x": None if x is None else np.asarray(x, dtype=float),
        "y": None if y is None else np.asarray(y, dtype=float),
        "z": None if z is None else np.asarray(z, dtype=float),
        "px": np.asarray(px, dtype=float),
        "py": np.asarray(py, dtype=float),
        "pz": np.asarray(pz, dtype=float),
        "kind": None,
    }


def _load_mprts_particles(reader: adios2.FileReader, variables: dict) -> dict[str, np.ndarray]:
    x = _read_var_by_suffix(reader, variables, "x")
    y = _read_var_by_suffix(reader, variables, "y")
    z = _read_var_by_suffix(reader, variables, "z")
    ux = _read_var_by_suffix(reader, variables, "ux")
    uy = _read_var_by_suffix(reader, variables, "uy")
    uz = _read_var_by_suffix(reader, variables, "uz")
    kind = np.asarray(_read_var_by_suffix(reader, variables, "kind"), dtype=int)
    qni_wni = np.asarray(_read_var_by_suffix(reader, variables, "qni_wni"), dtype=float)

    unique_kinds = sorted(np.unique(kind).tolist())
    unknown_kinds = [k for k in unique_kinds if k not in _DEFAULT_KIND_CHARGE]
    if unknown_kinds:
        raise KeyError(
            f"Unsupported particle kind ids {unknown_kinds}; known mapping only covers {sorted(_DEFAULT_KIND_CHARGE)}"
        )

    q = np.asarray([_DEFAULT_KIND_CHARGE[int(k)] for k in kind], dtype=float)
    m = np.asarray([_DEFAULT_KIND_MASS[int(k)] for k in kind], dtype=float)
    w = qni_wni / q

    return {
        "format": "mprts",
        "q": q,
        "m": m,
        "w": w,
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
        "z": np.asarray(z, dtype=float),
        "px": np.asarray(ux, dtype=float),
        "py": np.asarray(uy, dtype=float),
        "pz": np.asarray(uz, dtype=float),
        "kind": kind,
    }


def load_particle_arrays(filepath: str, verbose: bool = False) -> dict[str, np.ndarray]:
    """Load PSC particle data from either flat variables or packed `mprts/*` output."""
    if verbose:
        print(f"Loading particles from: {filepath}")

    reader = adios2.FileReader(filepath)
    try:
        variables = reader.available_variables()
        try:
            data = _load_flat_particles(reader, variables)
        except KeyError:
            data = _load_mprts_particles(reader, variables)
    finally:
        reader.close()

    if verbose:
        fmt = data["format"]
        print(f"  Particle format: {fmt}")
        print(f"  Total particles: {len(data['q']):,}")
    return data


def build_structured_particles(filepath: str, include_position: bool = True, include_weight: bool = True,
                               include_mass: bool = True, verbose: bool = False) -> np.ndarray:
    """Return a structured particle array compatible with older plotting scripts."""
    arrays = load_particle_arrays(filepath, verbose=verbose)

    fields: list[tuple[str, str]] = [("q", "f8")]
    if include_mass:
        fields.append(("m", "f8"))
    if include_weight:
        fields.append(("w", "f8"))
    if include_position:
        fields.extend([("x", "f8"), ("y", "f8"), ("z", "f8")])
    fields.extend([("px", "f8"), ("py", "f8"), ("pz", "f8")])

    data = np.empty(len(arrays["q"]), dtype=np.dtype(fields))
    data["q"] = arrays["q"]
    if include_mass:
        data["m"] = arrays["m"]
    if include_weight:
        data["w"] = arrays["w"]
    if include_position:
        for axis in ("x", "y", "z"):
            values = arrays[axis]
            data[axis] = 0.0 if values is None else values
    data["px"] = arrays["px"]
    data["py"] = arrays["py"]
    data["pz"] = arrays["pz"]
    return data


def extract_step(filepath: str) -> int:
    match = STEP_RE.search(os.path.basename(filepath))
    if not match:
        raise ValueError(f"Could not extract step from filename: {filepath}")
    return int(match.group(1))


def is_particle_path(path: str) -> bool:
    return os.path.isfile(path) or os.path.isdir(path)


def resolve_particle_files(input_path: str) -> list[str]:
    """Resolve a file, directory, or glob pattern into an ordered particle-path list."""
    if os.path.isdir(input_path):
        candidates = sorted(glob.glob(os.path.join(input_path, "prt.*.bp")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(input_path, "prt_maxwellian.*.bp")))
    elif any(ch in input_path for ch in "*?[]"):
        candidates = sorted(glob.glob(input_path))
    else:
        candidates = [input_path]

    files = [path for path in candidates if is_particle_path(path)]
    if not files:
        raise FileNotFoundError(f"No particle files matched: {input_path}")

    return sorted(files, key=extract_step)
