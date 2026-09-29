"""Versioned, deterministic analysis metadata and portable numeric utilities."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

CONVENTIONS_VERSION = 6
ALGORITHMS = {"growth": "modal-transient-aware-hac-2", "vdf": "u-fit-v-density-2",
              "structures": "mean-centered-colocated-2", "heat_flux": "local-third-moment-2"}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def strict_dumps(value, **kwargs):
    kwargs['allow_nan'] = False
    return json.dumps(json_safe(value), **kwargs)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(strict_dumps(value, indent=2) + '\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def stable_rng(run, step, species, purpose):
    """Stable across process scheduling and subset selection, not particle reordering."""
    key = json.dumps([str(run), int(step), str(species), str(purpose)], separators=(',', ':'))
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], 'little'))


def sample_rng(path, species, purpose):
    from data_reader import PICDataReader
    p = Path(path)
    # Stable under relocating a run; PSC series basename and step carry identity.
    run = os.environ.get('PSC_ANALYSIS_RUN_ID', p.name.split('.')[0])
    return stable_rng(run, PICDataReader.get_step_from_filename(str(p)) or 0, species, purpose)


def effective_sample_size(weights):
    w = np.asarray(weights, float)
    w = w[np.isfinite(w) & (w > 0)]
    return float(w.sum() ** 2 / np.dot(w, w)) if w.size else 0.0


def provenance():
    root = Path(__file__).resolve().parent
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    h = hashlib.sha256()
    for p in sorted([*root.glob('*.py'), root / 'Makefile', root / 'requirements.txt']):
        h.update(p.name.encode() + b'\0' + p.read_bytes())
    versions = {}
    for pkg in ('numpy', 'scipy', 'matplotlib', 'h5py', 'adios2'):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    config = os.environ.get('PSC_ANALYSIS_CONFIG')
    return {"schema_version": 2, "algorithms": ALGORITHMS, "git_commit": git('rev-parse', 'HEAD'),
            "git_dirty": bool(git('status', '--porcelain')), "analysis_source_sha256": h.hexdigest(),
            "python": sys.version, "packages": versions, "argv": sys.argv,
            "run_config": json.loads(Path(config).read_text()) if config else None,
            "environment": {k: v for k, v in os.environ.items() if k.startswith('PSC_ANALYSIS_')},
            "missing_numeric_policy": "non-finite values serialized as null; consult diagnostic status/reason"}


def inventory_identity(paths):
    """Metadata fingerprint, explicitly not a content checksum of huge snapshots."""
    h = hashlib.sha256()
    count = 0
    for p in sorted(set(map(Path, paths))):
        members = sorted(x for x in p.rglob('*') if x.is_file()) if p.is_dir() else [p]
        for member in members:
            s = member.stat()
            name = str(member.relative_to(p.parent))
            h.update(f'{name}\0{s.st_size}\0{s.st_mtime_ns}\n'.encode())
            count += 1
    return {"kind": "name-size-mtime metadata fingerprint", "files": count, "sha256": h.hexdigest()}


def gap_segments(t, y, gap_factor=1.5):
    """Drop unobserved interleaved rows, retain gaps in the actual series cadence."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if t.size < 2:
        return [(t, y)] if t.size else []
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError('Series times must be strictly increasing')
    cuts = np.flatnonzero(dt > gap_factor * np.median(dt)) + 1
    return list(zip(np.split(t, cuts), np.split(y, cuts)))


def align_time(source_t, source_y, target_t, tolerance=0.0):
    """Interpolate only supported times; snap endpoints within explicit rounding tolerance."""
    x, y, t = map(lambda v: np.asarray(v, float), (source_t, source_y, target_t))
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError('Time tolerance must be finite and non-negative')
    if x.size < 2 or x.shape != y.shape or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError('Need increasing finite source times and matching values')
    q = t.copy()
    snap = ((q < x[0]) & (q >= x[0] - tolerance)) | ((q > x[-1]) & (q <= x[-1] + tolerance))
    q[snap] = np.clip(q[snap], x[0], x[-1])
    out = np.interp(q, x, y, left=np.nan, right=np.nan)
    ok = np.isfinite(out)
    return out, {"tolerance_code": tolerance, "endpoint_snaps": int(snap.sum()),
                 "coverage_fraction": float(ok.mean()) if ok.size else 0.,
                 "last_supported_time_code": float(t[np.flatnonzero(ok)[-1]]) if ok.any() else None}


def magnetic_cell_centres_yz(bx, by, bz):
    """Raw face-centred B -> cell centres on periodic (Nz, Ny) arrays."""
    return bx, .5 * (by + np.roll(by, -1, axis=1)), .5 * (bz + np.roll(bz, -1, axis=0))


def cylindrical_density(vpar, vperp, weights, par_edges, perp_edges, representation='gyrotropic'):
    counts, _, _ = np.histogram2d(vpar, vperp, bins=(par_edges, perp_edges), weights=weights)
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError('Positive total particle weight required')
    measure = np.diff(par_edges)[:, None] * np.diff(perp_edges)[None, :]
    if representation == 'gyrotropic':
        measure = np.diff(par_edges)[:, None] * (np.pi * np.diff(np.asarray(perp_edges) ** 2))[None, :]
    elif representation != 'probability':
        raise ValueError('Unknown VDF representation')
    return counts / (total * measure), {"representation": representation,
        "retained_probability": float(counts.sum() / total), "normalization": "full input weight",
        "coordinate": "v", "measure": "d3v (exact annular bins)" if representation == 'gyrotropic' else 'dv_parallel dv_perp'}
