"""
streaming_analysis.py
─────────────────────
Reads PSC ADIOS2/BP5 output files, computes key mirror-instability diagnostics,
and saves compact HDF5 results. Called by purge_and_analyze.slurm after each
simulation segment.

Modes:
  --mode fields     : process pfd.bp electromagnetic fields  (default)
  --mode particles  : subsample a particle BP5 file

Key design decisions:
  - NEVER loads the full 4096×4608 mesh into memory at once.
    Uses ADIOS2 SetSelection to read Bz as a single 2-D array (no ghost cells).
  - Fields Bx, By, Bz (PSC names: hx, hy, hz) are read individually and kept
    as float32 to minimise RAM (~90 MB per component for the full grid).
  - Spectra are computed every SPECTRA_STRIDE snapshots to save time.
  - Particle files are subsampled without loading the full array when possible.

Usage (from purge_and_analyze.slurm):
    python3 streaming_analysis.py \\
        --input-dir  ./Out1 \\
        --output-dir ./diagnostics_compact \\
        --n-workers  8

    python3 streaming_analysis.py \\
        --mode particles \\
        --input-dir  ./Out1/prt_000040000.bp \\
        --output-dir ./diagnostics_compact \\
        --sample-fraction 0.005
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

# ── Optional ADIOS2 import ────────────────────────────────────────────────────
try:
    import adios2  # type: ignore
    HAS_ADIOS2 = True
except ImportError:
    HAS_ADIOS2 = False
    print(
        "WARNING: adios2 Python bindings not found. "
        "Install with: pip install adios2",
        file=sys.stderr,
    )

# ── Physical constants (match psc_temp_aniso.cxx) ────────────────────────────
B0          = 0.1        # background field [code units]
GRID_NY     = 4096
GRID_NZ     = 4608
DCELL       = 0.023003   # cell size [c/wpe]
LY          = GRID_NY * DCELL   # 94.208 c/wpe
LZ          = GRID_NZ * DCELL   # 105.997 c/wpe
MASS_RATIO  = 200
LAMBDA_DE   = 0.07071    # sqrt(Te_perp) = sqrt(0.005)

# How often to compute spatial spectra (every N field snapshots saved)
SPECTRA_STRIDE = 10


# =============================================================================
# Field analysis helpers
# =============================================================================

def compute_magnetic_energy(
    Bx: np.ndarray, By: np.ndarray, Bz: np.ndarray,
    b0: float = B0,
) -> dict[str, float]:
    """Compute magnetic energy and fluctuation metrics. Inputs are 2-D arrays."""
    B_total = np.sqrt(Bx**2 + By**2 + Bz**2)
    dB = B_total - b0
    return {
        "B_rms":      float(np.sqrt(np.mean(Bz**2))),
        "dB_rms":     float(np.sqrt(np.mean(dB**2))),
        "dB_max":     float(np.max(np.abs(dB))),
        "dB_over_B0": float(np.sqrt(np.mean(dB**2))) / b0,
        "Bx_rms":     float(np.sqrt(np.mean(Bx**2))),
        "By_rms":     float(np.sqrt(np.mean(By**2))),
    }


def compute_spatial_spectrum(
    field2d: np.ndarray,
    Ly: float = LY,
    Lz: float = LZ,
) -> dict[str, np.ndarray]:
    """
    2D FFT power spectrum of a field slice.
    Returns wavenumber arrays and power spectrum (averaged over each axis).
    """
    ny, nz = field2d.shape
    fft2  = np.fft.rfft2(field2d)
    power = (np.abs(fft2) ** 2) / (ny * nz) ** 2

    ky = np.fft.fftfreq(ny,  d=Ly / ny) * 2 * np.pi
    kz = np.fft.rfftfreq(nz, d=Lz / nz) * 2 * np.pi

    # Averaged power vs ky and kz (only real-half in kz)
    power_ky = power.mean(axis=1)   # mean over kz → shape (ny,)
    power_kz = power.mean(axis=0)   # mean over ky → shape (nz//2+1,)

    return {"ky": ky, "kz": kz, "power_ky": power_ky, "power_kz": power_kz}


def mirror_instability_metric(
    Bz: np.ndarray, b0: float = B0
) -> dict[str, float]:
    """Detect mirror-mode structures: regions of |B| < B0 (magnetic holes)."""
    frac_depleted     = float(np.mean(np.abs(Bz) < 0.95 * b0))
    compression_ratio = float(np.max(np.abs(Bz)) / b0)
    return {
        "frac_magnetic_holes": frac_depleted,
        "B_compression_ratio": compression_ratio,
    }


# =============================================================================
# ADIOS2 helpers: safe variable reading without loading the full ghost-padded mesh
# =============================================================================

def _read_2d_field(
    engine: Any,
    io: Any,
    var_name: str,
) -> Optional[np.ndarray]:
    """
    Read a single 2D (Y-Z) field variable from the current ADIOS2 step.

    PSC writes fields with ghost-cell padding, so the raw shape is
    (1+ghosts, NY+2*ghosts, NZ+2*ghosts). We select only the interior
    NY × NZ slice to avoid loading ghost cells, reducing memory by ~10%.

    Returns a float32 array of shape (NY, NZ), or None if variable not found.
    """
    var = io.InquireVariable(var_name)
    if var is None:
        return None

    shape = list(var.Shape())   # e.g. [1, NY+4, NZ+4] or [NY+4, NZ+4]

    # Determine ghost thickness from the shape excess over expected grid
    # We only care about the last two spatial dims (Y, Z).
    if len(shape) < 2:
        return None

    ny_total = shape[-2]
    nz_total = shape[-1]
    ghost_y  = (ny_total - GRID_NY) // 2
    ghost_z  = (nz_total - GRID_NZ) // 2

    # Build start / count for SetSelection (no ghost cells)
    if len(shape) == 3:
        # [1 or Npatch, NY+ghost, NZ+ghost] — collapse first dim
        start = [0,       ghost_y, ghost_z]
        count = [shape[0], GRID_NY, GRID_NZ]
    else:
        start = [ghost_y, ghost_z]
        count = [GRID_NY, GRID_NZ]

    try:
        var.SetSelection([start, count])
        buf = np.zeros(count, dtype=np.float32)
        engine.Get(var, buf)
        engine.PerformGets()
        # If 3-D, squeeze the patch dimension (single patch in dim_yz)
        if buf.ndim == 3:
            buf = buf.squeeze(axis=0)
        return buf.reshape(GRID_NY, GRID_NZ)
    except Exception as exc:
        print(f"  WARNING: could not read variable '{var_name}': {exc}")
        return None


# =============================================================================
# ADIOS2 field reader (streaming-safe, no full-mesh allocation)
# =============================================================================

def process_fields(input_dir: Path, output_dir: Path, n_workers: int) -> None:
    """
    Read all steps from pfd.bp, compute mirror-instability diagnostics,
    and append to a compact HDF5 file. The full YZ mesh is NOT kept in RAM
    after each step — only scalar diagnostics and occasional spectra survive.
    """
    bp_path = input_dir / "pfd.bp"
    if not bp_path.exists():
        print(f"  pfd.bp not found at {bp_path}, skipping field analysis.")
        return

    if not HAS_ADIOS2:
        print("  ADIOS2 not available, cannot read BP5 files.")
        return

    out_file = output_dir / "field_diagnostics.h5"
    print(f"  Reading  : {bp_path}")
    print(f"  Writing  : {out_file}")
    print(f"  Grid     : {GRID_NY} × {GRID_NZ}  Δcell={DCELL}  λ_De={LAMBDA_DE:.4f}")

    adios  = adios2.ADIOS()
    io     = adios.DeclareIO("reader")
    io.SetEngine("BP5")
    engine = io.Open(str(bp_path), adios2.Mode.Read)

    # Accumulate time-series scalars
    steps: list[int]            = []
    B_rms_list: list[float]     = []
    dB_rms_list: list[float]    = []
    dB_max_list: list[float]    = []
    dB_over_B0_list: list[float] = []
    hole_frac_list: list[float] = []
    compress_list: list[float]  = []
    Bx_rms_list: list[float]    = []
    By_rms_list: list[float]    = []

    # Spectra (only every SPECTRA_STRIDE snapshots)
    spectra_steps: list[int]    = []
    power_ky_list: list[Any]    = []
    power_kz_list: list[Any]    = []
    ky_arr: Optional[np.ndarray] = None
    kz_arr: Optional[np.ndarray] = None

    snap = 0   # snapshot counter (≠ simulation step number)

    while True:
        status = engine.BeginStep()
        if status != adios2.StepStatus.OK:
            break

        current_step = int(engine.CurrentStep())

        # ── Read B-field components individually (PSC names: hx, hy, hz) ─────
        # hz = Bz (parallel to background B, along Z — the mirror direction)
        # hx = Bx, hy = By  (transverse perturbations)
        Bz = _read_2d_field(engine, io, "hz")
        Bx = _read_2d_field(engine, io, "hx")
        By = _read_2d_field(engine, io, "hy")

        engine.EndStep()   # release ADIOS2 buffers for this step

        if Bz is None:
            print(f"  WARNING: hz not found at step {current_step}, skipping.")
            snap += 1
            continue

        # Fallback: if transverse components missing use zeros (isotropic noise)
        if Bx is None:
            Bx = np.zeros_like(Bz)
        if By is None:
            By = np.zeros_like(Bz)

        # ── Scalar diagnostics ────────────────────────────────────────────────
        metrics = compute_magnetic_energy(Bx, By, Bz)
        mirror  = mirror_instability_metric(Bz)

        steps.append(current_step)
        B_rms_list.append(metrics["B_rms"])
        dB_rms_list.append(metrics["dB_rms"])
        dB_max_list.append(metrics["dB_max"])
        dB_over_B0_list.append(metrics["dB_over_B0"])
        Bx_rms_list.append(metrics["Bx_rms"])
        By_rms_list.append(metrics["By_rms"])
        hole_frac_list.append(mirror["frac_magnetic_holes"])
        compress_list.append(mirror["B_compression_ratio"])

        # ── Spatial spectra (every SPECTRA_STRIDE snapshots) ─────────────────
        if snap % SPECTRA_STRIDE == 0:
            spec = compute_spatial_spectrum(Bz)
            spectra_steps.append(current_step)
            power_ky_list.append(spec["power_ky"].astype(np.float32))
            power_kz_list.append(spec["power_kz"].astype(np.float32))
            if ky_arr is None:
                ky_arr = spec["ky"].astype(np.float32)
                kz_arr = spec["kz"].astype(np.float32)

        # ── Free large arrays immediately — do NOT accumulate the full mesh ───
        del Bz, Bx, By

        snap += 1
        if snap % 50 == 0:
            print(f"  Processed {snap} snapshots (sim step {current_step})...")

    engine.Close()
    print(f"  Closed BP5. Total snapshots processed: {snap}")

    # ── Write / append compact HDF5 ──────────────────────────────────────────
    mode = "a" if out_file.exists() else "w"
    with h5py.File(out_file, mode) as hf:
        # Time-series scalars
        grp = hf.require_group("timeseries")
        for key, arr_list, dtype in [
            ("step",       steps,           np.int32),
            ("B_rms",      B_rms_list,      np.float32),
            ("dB_rms",     dB_rms_list,     np.float32),
            ("dB_max",     dB_max_list,     np.float32),
            ("dB_over_B0", dB_over_B0_list, np.float32),
            ("Bx_rms",     Bx_rms_list,     np.float32),
            ("By_rms",     By_rms_list,     np.float32),
            ("frac_holes", hole_frac_list,  np.float32),
            ("B_compression", compress_list, np.float32),
        ]:
            arr = np.array(arr_list, dtype=dtype)
            if key in grp:
                del grp[key]
            grp.create_dataset(key, data=arr,
                               compression="gzip", compression_opts=6)

        # Spectra
        if spectra_steps:
            sp_grp = hf.require_group("spectra")
            for key, data in [
                ("step",     np.array(spectra_steps, dtype=np.int32)),
                ("ky",       ky_arr),
                ("kz",       kz_arr),
                ("power_ky", np.stack(power_ky_list, axis=0)),
                ("power_kz", np.stack(power_kz_list, axis=0)),
            ]:
                if key in sp_grp:
                    del sp_grp[key]
                sp_grp.create_dataset(key, data=data,
                                      compression="gzip", compression_opts=4)

        # Metadata
        hf.attrs["mass_ratio"]          = MASS_RATIO
        hf.attrs["grid_NY"]             = GRID_NY
        hf.attrs["grid_NZ"]             = GRID_NZ
        hf.attrs["dcell"]               = DCELL
        hf.attrs["Ly"]                  = LY
        hf.attrs["Lz"]                  = LZ
        hf.attrs["lambda_De"]           = LAMBDA_DE
        hf.attrs["pts_per_lambda_De"]   = LAMBDA_DE / DCELL
        hf.attrs["B0"]                  = B0

    size_mb = out_file.stat().st_size / 1e6
    print(f"  HDF5 written: {out_file}  ({size_mb:.1f} MB, {snap} snapshots)")


# =============================================================================
# Particle subsampler
# =============================================================================

def process_particles(
    input_dir: Path, output_dir: Path, sample_fraction: float
) -> None:
    """
    Read a PSC particle BP5 file, save a random subsample to HDF5.

    Strategy: read all particles for a species, then subsample — avoids
    the complexity of non-contiguous ADIOS2 selections over particles.
    Full particle arrays are freed immediately after subsampling.
    """
    if not HAS_ADIOS2:
        print("  ADIOS2 not available, cannot read particle BP5 files.")
        return

    bp_name  = input_dir.name.replace(".bp", "")
    out_file = output_dir / f"{bp_name}_subsample.h5"
    print(f"  Subsampling particles: {input_dir}  (fraction={sample_fraction:.1%})")

    adios  = adios2.ADIOS()
    io     = adios.DeclareIO("prt_reader")
    io.SetEngine("BP5")
    engine = io.Open(str(input_dir), adios2.Mode.Read)

    with h5py.File(out_file, "w") as hf:
        hf.attrs["sample_fraction"] = sample_fraction
        hf.attrs["source"]          = str(input_dir)
        hf.attrs["mass_ratio"]      = MASS_RATIO

        snap = 0
        while True:
            status = engine.BeginStep()
            if status != adios2.StepStatus.OK:
                break

            current_step = int(engine.CurrentStep())
            grp = hf.create_group(f"step_{current_step:08d}")

            # PSC particle variable naming convention: xi, yi, zi, ux, uy, uz
            # for each species index (0=electron, 1=ion).
            # Try both naming schemes used across PSC versions.
            for sp_idx, sp_name in enumerate(["electron", "ion"]):
                # Detect variable names dynamically
                pos_candidates = [
                    f"position_{sp_idx}",
                    f"x_{sp_idx}", f"y_{sp_idx}", f"z_{sp_idx}",
                ]
                vel_candidates = [
                    f"velocity_{sp_idx}",
                    f"px_{sp_idx}", f"py_{sp_idx}", f"pz_{sp_idx}",
                ]

                # Find position variable
                pos_var = None
                for name in pos_candidates:
                    v = io.InquireVariable(name)
                    if v is not None:
                        pos_var = v
                        break

                if pos_var is None:
                    continue   # species not present in this file

                try:
                    n_total  = pos_var.Shape()[0]
                    n_sample = max(1, int(n_total * sample_fraction))
                    idx      = np.sort(
                        np.random.choice(n_total, n_sample, replace=False)
                    )

                    # Read full position array, subsample — then free
                    pos = np.zeros((n_total, 3), dtype=np.float32)
                    engine.Get(pos_var, pos)
                    engine.PerformGets()
                    pos_sub = pos[idx].copy()
                    del pos  # free full array

                    sp_grp = grp.create_group(sp_name)
                    sp_grp.create_dataset("position", data=pos_sub,
                                          compression="gzip", compression_opts=6)
                    del pos_sub

                    # Velocity (optional)
                    vel_var = None
                    for name in vel_candidates:
                        v = io.InquireVariable(name)
                        if v is not None:
                            vel_var = v
                            break

                    if vel_var is not None:
                        vel = np.zeros((n_total, 3), dtype=np.float32)
                        engine.Get(vel_var, vel)
                        engine.PerformGets()
                        vel_sub = vel[idx].copy()
                        del vel
                        sp_grp.create_dataset("velocity", data=vel_sub,
                                              compression="gzip", compression_opts=6)
                        del vel_sub

                    sp_grp.attrs["n_total"]  = n_total
                    sp_grp.attrs["n_sample"] = n_sample
                    print(
                        f"    Step {current_step}, {sp_name}: "
                        f"saved {n_sample}/{n_total} particles"
                    )

                except Exception as exc:
                    print(
                        f"  WARNING: particle read error "
                        f"step={current_step} species={sp_name}: {exc}"
                    )

            engine.EndStep()
            snap += 1

    engine.Close()
    size_mb = out_file.stat().st_size / 1e6
    print(f"  Particle HDF5 written: {out_file}  ({size_mb:.1f} MB, {snap} steps)")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PSC streaming analysis & purge helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode",    choices=["fields", "particles"], default="fields")
    p.add_argument("--input-dir",  type=Path, required=True,
                   help="BP5 output directory (Out1/) or specific .bp dir")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory to write compact HDF5 diagnostics")
    p.add_argument("--n-workers",  type=int, default=4,
                   help="Parallel workers (currently informational)")
    p.add_argument("--sample-fraction", type=float, default=0.005,
                   help="Fraction of particles to keep (default 0.5%%)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.mode == "fields":
        process_fields(args.input_dir, args.output_dir, args.n_workers)
    else:
        process_particles(
            args.input_dir, args.output_dir, args.sample_fraction
        )

    elapsed = time.time() - t0
    print(f"\nAnalysis finished in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
