#!/usr/bin/env python3
"""
plot_prt.py — Particle diagnostic suite for PSC PIC simulations.

Reads HDF5 particle files and generates publication-quality diagnostics:
  1. Kappa vs. Maxwellian distribution comparison
  2. Goodness-of-fit tests (Anderson-Darling & Kolmogorov-Smirnov)
  3. 1D distribution temporal evolution (heatmap + suprathermal-tail overlay)
  4. Magnetic-fluctuation time series
  5. Energy partition (the heat flux is heat_flux_analysis.py)

2D/3D VDF snapshots and the T_perp/T_par ("Brazil") evolution are no longer
generated here -- physical_diagnostics.py (09_physical_diagnostics/) and
anisotropy_analysis.py (01_anisotropy/) already cover both, denser and
without this module's corner-noise-in-low-count-bins artifact.

Usage:
    python plot_prt.py [path_to_prt_file | directory | glob_pattern]

Defaults to  ../build/src/prt.000000000.h5  if no argument is given.
"""

import sys
import os
import glob
import re
import gc
import csv
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
try:
    from scipy.special import gamma as gamma_func
    from scipy import stats as scipy_stats
except ImportError:
    gamma_func = None
    scipy_stats = None

try:
    from data_reader import PICDataReader
except ImportError:
    PICDataReader = None
from plasma_physics import velocity_from_u
from psc_units import (
    B0, KAPPA, TI_PAR, TI_PERP, VA_OVER_C, ZI, BETA_I_PAR as _BETA_I_PAR_SIM,
    BETA_I_PERP_OVER_PAR as _TI_RATIO_SIM, M_ION, PROFILE_LABEL, step_to_omegaci,
)

import plot_style as ps

# Shared theme (paper: 300 dpi, PDF copy, content check at every save).
ps.apply()

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = "prt_plots"
MAX_EVOLUTION_FILES = 12
MAX_PARTICLES = 500_000  # submuestreo global si hay más partículas
RNG_SEED = 20260612
OUTPUT_PREFIX = ""

STEP_RE = re.compile(r"\.(\d+)(?:_p\d+)?\.h5$")


def _style_paper_axes(ax):
    ax.set_facecolor("white")
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=14)
    ax.minorticks_on()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _save_paper_figure(fig, path: str):
    ps.save(fig, path)
    print(f"  Saved: {path}")


def output_file(outdir: str, filename: str) -> str:
    return os.path.join(outdir, f"{OUTPUT_PREFIX}{filename}")


def _sample_indices(n_total: int, max_particles: int):
    """Return a deterministic uniform subsample for reproducible figures."""
    if n_total <= max_particles:
        return slice(None)
    rng = np.random.default_rng(RNG_SEED)
    return np.sort(rng.choice(n_total, max_particles, replace=False))

# ── Simulation parameters (from psc_units — single source of truth) ──────────
Zi: float = ZI
VA_OVER_C_: float = VA_OVER_C
BETA_I_PAR: float = _BETA_I_PAR_SIM
TI_PERP_OVER_TI_PAR: float = _TI_RATIO_SIM
BETA_NORM: float = 1.0




# ══════════════════════════════════════════════════════════════════════════════
#  I/O Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_particles(filepath: str, verbose: bool = True,
                   max_particles: int = MAX_PARTICLES) -> np.ndarray:
    """Load particle data from a PSC prt.*.h5 file.

    Only reads columns needed for analysis (q, m, px, py, pz) to save RAM.
    Subsamples uniformly if total particles exceeds max_particles.
    """
    if verbose:
        print(f"Loading particles from: {filepath}")
    with h5py.File(filepath, "r") as f:
        dset = f["particles"]["p0"]["1d"]
        n_total = dset["q"].shape[0]
        if n_total > max_particles:
            idx = _sample_indices(n_total, max_particles)
            if verbose:
                print(f"  Subsampling {max_particles:,} / {n_total:,} particles")
        else:
            idx = slice(None)
        # Read only needed fields — avoids loading x,y,z positions
        q  = dset["q"][idx].astype(np.float32)
        m  = dset["m"][idx].astype(np.float32)
        px = dset["px"][idx].astype(np.float32)
        py = dset["py"][idx].astype(np.float32)
        pz = dset["pz"][idx].astype(np.float32)

    # Pack into a structured array (same interface as before)
    dtype = np.dtype([("q", np.float32), ("m", np.float32),
                      ("px", np.float32), ("py", np.float32), ("pz", np.float32)])
    data = np.empty(len(q), dtype=dtype)
    data["q"]  = q;  data["m"]  = m
    data["px"] = px; data["py"] = py; data["pz"] = pz
    if verbose:
        print(f"  Loaded {len(data):,} particles ({data.nbytes / 1e6:.1f} MB)")
    return data


def load_particle_phase_space(filepath: str,
                              max_particles: int = MAX_PARTICLES) -> dict:
    """Load only momentum fields needed for distribution analysis (memory-light)."""
    with h5py.File(filepath, "r") as f:
        dset = f["particles"]["p0"]["1d"]
        n_total = dset["q"].shape[0]
        idx = _sample_indices(n_total, max_particles)
        q  = dset["q"][idx].astype(np.float32)
        px = dset["px"][idx].astype(np.float32)
        py = dset["py"][idx].astype(np.float32)
        pz = dset["pz"][idx].astype(np.float32)
        w = (dset["w"][idx].astype(np.float64) if "w" in (dset.dtype.names or ())
             else np.ones(q.shape[0]))

    ion_mask  = q > 0
    elec_mask = q < 0
    result = {
        "ions_px":        px[ion_mask],
        "ions_py":        py[ion_mask],
        "ions_pz":        pz[ion_mask],
        "ions_perp":      np.sqrt(px[ion_mask]**2 + py[ion_mask]**2),
        "ions_w":         w[ion_mask],
        "electrons_px":   px[elec_mask],
        "electrons_py":   py[elec_mask],
        "electrons_pz":   pz[elec_mask],
        "electrons_perp": np.sqrt(px[elec_mask]**2 + py[elec_mask]**2),
        "electrons_w":    w[elec_mask],
    }
    del q, px, py, pz, w, ion_mask, elec_mask
    gc.collect()
    return result


def load_field_fluctuation_metrics(filepath: str, b0: float = B0) -> dict:
    """Return RMS magnetic fluctuation metrics normalised to B0."""
    if PICDataReader is None:
        raise RuntimeError("data_reader.py is required to read magnetic field outputs.")

    fields = PICDataReader.read_multiple_fields_3d(
        filepath,
        "jeh-",
        ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"],
    )

    # PSC pfd files already store B in code units. Multiplying by B0 here
    # would suppress delta-B/B0 by an extra factor B0.
    bx = np.asarray(fields["hx_fc/p0/3d"], dtype=float).ravel()
    by = np.asarray(fields["hy_fc/p0/3d"], dtype=float).ravel()
    bz = np.asarray(fields["hz_fc/p0/3d"], dtype=float).ravel()

    dbx = bx - np.mean(bx)
    dby = by - np.mean(by)
    dbz = bz - np.mean(bz)

    b0_abs = max(abs(b0), 1e-30)
    return {
        "delta_b_total": np.sqrt(np.mean(dbx**2 + dby**2 + dbz**2)) / b0_abs,
        "delta_b_parallel": np.sqrt(np.mean(dbz**2)) / b0_abs,
        "delta_b_perp": np.sqrt(np.mean(dbx**2 + dby**2)) / b0_abs,
    }


def extract_step(filepath: str) -> int:
    """Extract the integer step from a PSC particle filename."""
    match = STEP_RE.search(os.path.basename(filepath))
    if not match:
        raise ValueError(f"Could not extract step from filename: {filepath}")
    return int(match.group(1))


def resolve_particle_files(input_path: str) -> list[str]:
    """Resolve a file, directory, or glob pattern into an ordered file list."""
    if os.path.isdir(input_path):
        candidates = sorted(glob.glob(os.path.join(input_path, "prt*.h5")))
    elif any(ch in input_path for ch in "*?[]"):
        candidates = sorted(glob.glob(input_path))
    else:
        candidates = [input_path]

    files = [p for p in candidates if os.path.isfile(p)]
    if not files:
        raise FileNotFoundError(f"No particle files matched: {input_path}")

    return sorted(files, key=extract_step)


def resolve_field_files(
    reference_dir: str, pattern: str = "pfd.*_p*.h5"
) -> dict[int, str]:
    """Map magnetic field files by simulation step."""
    candidates = sorted(glob.glob(os.path.join(reference_dir, pattern)))
    result = {}
    for path in candidates:
        if os.path.isfile(path):
            result[extract_step(path)] = path
    return result


def sample_filepaths(
    filepaths: list[str], max_files: int = MAX_EVOLUTION_FILES
) -> list[str]:
    """Uniformly sub-sample filepaths for temporal scans."""
    if len(filepaths) <= max_files:
        return filepaths
    indices = np.unique(np.linspace(0, len(filepaths) - 1, max_files, dtype=int))
    return [filepaths[i] for i in indices]


# ══════════════════════════════════════════════════════════════════════════════
#  Species helpers
# ══════════════════════════════════════════════════════════════════════════════

def separate_species(data: np.ndarray, verbose: bool = True):
    """Separate particles by charge sign into ions and electrons."""
    ions = data[data["q"] > 0]
    electrons = data[data["q"] < 0]
    if verbose:
        print(f"  Ions: {len(ions):,},  Electrons: {len(electrons):,}")
    return ions, electrons


# ══════════════════════════════════════════════════════════════════════════════
#  Theoretical distributions
# ══════════════════════════════════════════════════════════════════════════════

def kappa_1d(v: np.ndarray, kappa: float, v_th: float) -> np.ndarray:
    """1-D Kappa (suprathermal) velocity distribution function."""
    A = (
        1.0
        / (np.sqrt(np.pi * (2 * kappa - 3)) * v_th)
        * (gamma_func(kappa) / gamma_func(kappa - 0.5))
    )
    return A * (1 + v**2 / ((2 * kappa - 3) * v_th**2)) ** (-kappa)


def maxwellian_1d(v: np.ndarray, v_th: float) -> np.ndarray:
    """1-D Maxwellian velocity distribution function."""
    return (1.0 / (np.sqrt(2 * np.pi) * v_th)) * np.exp(-v**2 / (2 * v_th**2))


def _kappa_cdf(v_sorted: np.ndarray, kappa: float, v_th: float) -> np.ndarray:
    """Numerically integrate the 1-D Kappa PDF to build a CDF."""
    dv = np.diff(v_sorted, prepend=v_sorted[0] - (v_sorted[1] - v_sorted[0]))
    pdf = kappa_1d(v_sorted, kappa, v_th)
    cdf = np.cumsum(pdf * np.abs(dv))
    return np.clip(cdf / cdf[-1], 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 2: Kappa vs. Maxwellian comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_kappa_comparison(ions, outdir: str):
    """Compare measured ion distributions to theoretical Kappa and Maxwellian.

    Three rows:
      Row 0 — f(p) vs p          (semilog, tail visibility)
      Row 1 — f(p) vs p^2        (Maxwellian linearisation)
      Row 2 — f(p) vs |p|        (log-log, power-law check)
    """
    if gamma_func is None:
        raise RuntimeError(
            "SciPy is required for Kappa diagnostics. "
            "Install CodeforAnalisys/requirements.txt."
        )
    mom_fields = ["px", "py", "pz"]
    comp_labels = [
        r"$u_x/v_A$  [perp.]",
        r"$u_y/v_A$  [perp.]",
        r"$u_z/v_A$  [par.]",
    ]
    comp_short = [r"$u_x$", r"$u_y$", r"$u_z$"]
    comp_dir = ["perpendicular", "perpendicular", "parallel"]
    kappa = KAPPA
    norm_vth = {
        "px": BETA_NORM * np.sqrt(TI_PERP / M_ION),
        "py": BETA_NORM * np.sqrt(TI_PERP / M_ION),
        "pz": BETA_NORM * np.sqrt(TI_PAR / M_ION),
    }

    for field, xlabel, short, direction in zip(mom_fields, comp_labels, comp_short, comp_dir):
        ion_p = np.asarray(ions[field], dtype=float) / VA_OVER_C_
        ion_p = ion_p - np.nanmedian(ion_p)
        v_th = norm_vth[field] / VA_OVER_C_
        p_lo = np.percentile(ion_p, 0.05)
        p_hi = np.percentile(ion_p, 99.95)
        bins = np.linspace(p_lo, p_hi, 200)
        hist, bin_edges = np.histogram(ion_p, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        hist_plot = hist.astype(float)
        hist_plot[hist_plot == 0] = np.nan
        valid = hist[hist > 0]
        y_min = max(valid.min() * 0.1, 1e-8) if len(valid) else 1e-8
        v_range = np.linspace(p_lo, p_hi, 1000)
        slug = field.replace("p", "u")

        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        _style_paper_axes(ax)
        ax.step(bin_centers, hist_plot, where="mid", color="#D55E00", linewidth=1.5, label="Ion data")
        ax.semilogy(v_range, kappa_1d(v_range, kappa, v_th), "-", color="#009E73", linewidth=2.2,
                    label=rf"Kappa ($\kappa$={kappa})")
        ax.semilogy(v_range, maxwellian_1d(v_range, v_th), "--", color="#0072B2", linewidth=2.2,
                    label="Maxwellian")
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(r"$f(u)$ [PDF]", fontsize=15)
        ax.set_title(rf"Ion {short} distribution ({direction})", fontsize=15, fontweight="bold")
        ax.set_yscale("log")
        ax.set_ylim(bottom=y_min)
        # The first x label sits in the corner of the lowest decade label.
        ax.set_xlim(p_lo, p_hi)
        ps.drop_corner_tick(ax, "x")
        ax.legend(fontsize=12)
        _save_paper_figure(fig, output_file(outdir, f"kappa_comparison_{slug}_semilog.png"))

        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        _style_paper_axes(ax)
        p2_centers = bin_centers**2
        p2_range = v_range**2
        sort_idx = np.argsort(p2_range)
        ax.step(p2_centers, hist_plot, where="mid", color="#D55E00", linewidth=1.5, label="Ion data")
        ax.semilogy(p2_range[sort_idx], kappa_1d(v_range[sort_idx], kappa, v_th), "-",
                    color="#009E73", linewidth=2.2, label=rf"Kappa ($\kappa$={kappa})")
        ax.semilogy(p2_range[sort_idx], maxwellian_1d(v_range[sort_idx], v_th), "--",
                    color="#0072B2", linewidth=2.2, label="Maxwellian")
        ax.set_xlabel(rf"{short}$^2$ [$v_A^2$]", fontsize=15)
        ax.set_ylabel(r"$f(u)$ [PDF]", fontsize=15)
        ax.set_title(rf"Maxwellian linearisation: {short}$^2$", fontsize=15, fontweight="bold")
        ax.set_yscale("log")
        ax.set_ylim(bottom=y_min)
        ax.legend(fontsize=12)
        _save_paper_figure(fig, output_file(outdir, f"kappa_comparison_{slug}_linearized.png"))

        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        _style_paper_axes(ax)
        pos_mask = bin_centers > 0
        bin_centers_pos = bin_centers[pos_mask]
        hist_plot_pos = hist_plot[pos_mask]
        ax.step(bin_centers_pos, hist_plot_pos, where="mid", color="#D55E00", linewidth=1.5,
                label="Ion data")
        if len(bin_centers_pos) > 1:
            v_pos = np.linspace(bin_centers_pos[0], bin_centers_pos[-1], 1000)
            ax.loglog(v_pos, kappa_1d(v_pos, kappa, v_th), "-", color="#009E73", linewidth=2.2,
                      label=rf"Kappa")
            ax.loglog(v_pos, maxwellian_1d(v_pos, v_th), "--", color="#0072B2", linewidth=2.2,
                      label="Maxwellian")
        ax.set_xlabel(rf"$|${short}$|$ [$v_A$]", fontsize=15)
        ax.set_ylabel(r"$f(u)$ [PDF]", fontsize=15)
        ax.set_title(rf"Positive-tail check: {short}", fontsize=15, fontweight="bold")
        ax.set_ylim(bottom=y_min)
        ax.legend(fontsize=12)
        _save_paper_figure(fig, output_file(outdir, f"kappa_comparison_{slug}_tail.png"))
    return

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        r"Distribution Comparison: Data vs Kappa ($\kappa$=3) vs Maxwellian — Ions",
        fontsize=20, fontweight="bold",
    )

    mom_fields = ["px", "py", "pz"]
    comp_labels = [
        r"$u_x / v_A$  [perp.]",
        r"$u_y / v_A$  [perp.]",
        r"$u_z / v_A$  [par.]",
    ]
    comp_short = [r"$p_x$", r"$p_y$", r"$p_z$"]
    comp_dir = ["perpendicular", "perpendicular", "parallel"]
    kappa = KAPPA

    norm_vth = {
        "px": BETA_NORM * np.sqrt(TI_PERP / M_ION),
        "py": BETA_NORM * np.sqrt(TI_PERP / M_ION),
        "pz": BETA_NORM * np.sqrt(TI_PAR / M_ION),
    }

    for col, (field, xlabel, short, direction) in enumerate(
        zip(mom_fields, comp_labels, comp_short, comp_dir)
    ):
        ion_p = np.asarray(ions[field], dtype=float)
        v_th = norm_vth[field]

        p_lo = np.percentile(ion_p, 0.05)
        p_hi = np.percentile(ion_p, 99.95)
        bins = np.linspace(p_lo, p_hi, 200)
        hist, bin_edges = np.histogram(ion_p, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        hist_plot = hist.copy().astype(float)
        hist_plot[hist_plot == 0] = np.nan

        valid = hist[hist > 0]
        y_min = max(valid.min() * 0.1, 1e-8) if len(valid) else 1e-8
        v_range = np.linspace(p_lo, p_hi, 1000)

        # ── Row 0: f(p) vs p — Semilog ──────────────────────────────────
        ax0 = axes[0, col]
        ax0.step(bin_centers, hist_plot, where="mid",
                 color="#D55E00", linewidth=1.5, alpha=0.85,
                 label="Ion data", zorder=2)
        ax0.semilogy(v_range, kappa_1d(v_range, kappa, v_th), "-",
                     color="#009E73", linewidth=2.5, zorder=3,
                     label=rf"Kappa ($\kappa$={kappa})")
        ax0.semilogy(v_range, maxwellian_1d(v_range, v_th), "--",
                     color="#0072B2", linewidth=2.5, zorder=3,
                     label="Maxwellian")
        ax0.set_xlabel(xlabel, fontsize=15)
        ax0.set_ylabel(r"$f(p)$  [PDF]", fontsize=15)
        ax0.set_title(rf"Semilog: $f$ vs {short}  ({direction})",
                      fontsize=15, fontweight="bold")
        ax0.legend(fontsize=12)
        ax0.grid(True, alpha=0.3)
        ax0.set_yscale("log")
        ax0.set_ylim(bottom=y_min)

        # ── Row 1: f(p) vs p^2 — Maxwellian linearisation ───────────────
        ax1 = axes[1, col]
        p2_centers = bin_centers**2
        p2_range = v_range**2
        sort_idx = np.argsort(p2_range)

        ax1.step(p2_centers, hist_plot, where="mid",
                 color="#D55E00", linewidth=1.5, alpha=0.85,
                 label="Ion data", zorder=2)
        ax1.semilogy(p2_range[sort_idx], kappa_1d(v_range[sort_idx], kappa, v_th), "-",
                     color="#009E73", linewidth=2.5, zorder=3,
                     label=rf"Kappa ($\kappa$={kappa})")
        ax1.semilogy(p2_range[sort_idx], maxwellian_1d(v_range[sort_idx], v_th), "--",
                     color="#0072B2", linewidth=2.5, zorder=3,
                     label="Maxwellian (straight line)")
        ax1.set_xlabel(rf"{short}$^2$  $[(m_i v_A)^2]$", fontsize=15)
        ax1.set_ylabel(r"$f(p)$  [PDF]", fontsize=15)
        ax1.set_title(rf"Semilog: $f$ vs {short}$^2$  — Maxwellian linearisation",
                      fontsize=15, fontweight="bold")
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")
        ax1.set_ylim(bottom=y_min)

        # ── Row 2: f(p) vs |p| — Log-log (power-law tail) ───────────────
        ax2 = axes[2, col]
        pos_mask = bin_centers > 0
        bin_centers_pos = bin_centers[pos_mask]
        hist_plot_pos = hist_plot[pos_mask]

        ax2.step(bin_centers_pos, hist_plot_pos, where="mid",
                 color="#D55E00", linewidth=1.5, alpha=0.85,
                 label="Ion data (positive tail)", zorder=2)

        if len(bin_centers_pos) > 1:
            v_pos = np.linspace(bin_centers_pos[0], bin_centers_pos[-1], 1000)
            ax2.loglog(v_pos, kappa_1d(v_pos, kappa, v_th), "-",
                       color="#009E73", linewidth=2.5, zorder=3,
                       label=rf"Kappa (slope $\propto p^{{-2\kappa}}$)")
            ax2.loglog(v_pos, maxwellian_1d(v_pos, v_th), "--",
                       color="#0072B2", linewidth=2.5, zorder=3,
                       label="Maxwellian (exponential drop)")

        ax2.set_xlabel(rf"$|${short}$|$  $[m_i v_A]$  (log scale)", fontsize=15)
        ax2.set_ylabel(r"$f(p)$  [PDF]  (log scale)", fontsize=15)
        ax2.set_title(rf"Log–Log: power-law tail of {short}  ($p > 0$)",
                      fontsize=15, fontweight="bold")
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3, which="both", ls="--")
        ax2.set_ylim(bottom=y_min)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_file(outdir, "kappa_vs_maxwellian.png")
    ps.save(plt.gcf(), path)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 3: Goodness-of-Fit (A-D & K-S)
# ══════════════════════════════════════════════════════════════════════════════

def _ad_p_value(ad_stat_norm: float) -> float:
    """Approximate p-value for the Anderson-Darling statistic (Stephens 1974)."""
    a = ad_stat_norm
    if a >= 0.6:
        p = np.exp(1.2937 - 5.709 * a + 0.0186 * a**2)
    elif a >= 0.34:
        p = np.exp(0.9177 - 4.279 * a - 1.38 * a**2)
    elif a >= 0.2:
        p = 1 - np.exp(-8.318 + 42.796 * a - 59.938 * a**2)
    else:
        p = 1 - np.exp(-13.436 + 101.14 * a - 223.73 * a**2)
    return float(np.clip(p, 0.0, 1.0))


def plot_goodness_of_fit(ions, outdir: str):
    """
    Statistical goodness-of-fit tests for the Kappa distribution.

    For each momentum component (px, py, pz):
      - Kolmogorov-Smirnov (K-S): sensitive to bulk of distribution.
      - Anderson-Darling (A-D): higher weight on tails.

    Output: CDF comparison + results table.
    """
    if scipy_stats is None:
        raise RuntimeError(
            "SciPy is required for Kappa goodness-of-fit diagnostics. "
            "Install CodeforAnalisys/requirements.txt."
        )
    mom_fields = ["px", "py", "pz"]
    comp_labels = [
        r"$p_x$  (perpendicular)",
        r"$p_y$  (perpendicular)",
        r"$p_z$  (parallel)",
    ]
    norm_vth = {
        "px": BETA_NORM * np.sqrt(TI_PERP / M_ION),
        "py": BETA_NORM * np.sqrt(TI_PERP / M_ION),
        "pz": BETA_NORM * np.sqrt(TI_PAR / M_ION),
    }
    kappa = KAPPA

    results_rows = []
    NSAMPLE = 5_000
    rng = np.random.default_rng(42)

    for field, label in zip(mom_fields, comp_labels):
        ion_p = np.asarray(ions[field], dtype=float) / VA_OVER_C_
        ion_p = ion_p - np.nanmedian(ion_p)
        v_th = norm_vth[field] / VA_OVER_C_
        sample = (
            rng.choice(ion_p, size=NSAMPLE, replace=False)
            if len(ion_p) > NSAMPLE
            else ion_p.copy()
        )
        sample.sort()
        kappa_cdf_vals = _kappa_cdf(sample, kappa, v_th)
        maxw_cdf_vals = scipy_stats.norm.cdf(sample, loc=0.0, scale=v_th)
        n = len(sample)
        ecdf_y = np.arange(1, n + 1) / n

        ks_stat_kappa = np.max(np.abs(ecdf_y - kappa_cdf_vals))
        ks_p_kappa = scipy_stats.kstwobign.sf(np.sqrt(n) * ks_stat_kappa)
        # Frozen distribution: the ("norm", args=...) form breaks on recent SciPy.
        ks_res_maxw = scipy_stats.kstest(sample, scipy_stats.norm(loc=0.0, scale=v_th).cdf)
        ks_stat_maxw = ks_res_maxw.statistic
        ks_p_maxw = ks_res_maxw.pvalue

        i_idx = np.arange(1, n + 1)
        cdf_lo = np.clip(kappa_cdf_vals, 1e-12, 1 - 1e-12)
        cdf_hi = np.clip(kappa_cdf_vals[::-1], 1e-12, 1 - 1e-12)
        ad_kappa = -n - np.mean(
            (2 * i_idx - 1) * (np.log(cdf_lo) + np.log(1 - cdf_hi))
        )
        ad_p_kappa = _ad_p_value(ad_kappa * (1 + 4 / n - 25 / n**2))

        cdf_lo_m = np.clip(maxw_cdf_vals, 1e-12, 1 - 1e-12)
        cdf_hi_m = np.clip(maxw_cdf_vals[::-1], 1e-12, 1 - 1e-12)
        ad_maxw = -n - np.mean(
            (2 * i_idx - 1) * (np.log(cdf_lo_m) + np.log(1 - cdf_hi_m))
        )
        ad_p_maxw = _ad_p_value(ad_maxw * (1 + 4 / n - 25 / n**2))

        results_rows.append([
            field, ks_stat_kappa, ks_p_kappa, ks_stat_maxw, ks_p_maxw,
            ad_kappa, ad_p_kappa, ad_maxw, ad_p_maxw,
        ])

        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        _style_paper_axes(ax)
        ax.step(sample, ecdf_y, where="post", color="#D55E00", linewidth=1.8,
                label="Empirical CDF")
        ax.plot(sample, kappa_cdf_vals, "-", color="#009E73", linewidth=2.2,
                label=rf"Kappa CDF ($\kappa={kappa}$)")
        ax.plot(sample, maxw_cdf_vals, "--", color="#0072B2", linewidth=2.2,
                label="Maxwellian CDF")
        ks_idx = np.argmax(np.abs(ecdf_y - kappa_cdf_vals))
        ax.vlines(sample[ks_idx], ecdf_y[ks_idx], kappa_cdf_vals[ks_idx],
                  colors="#E69F00", linewidths=2.0, label=rf"$D_{{KS}}={ks_stat_kappa:.3g}$")
        ax.set_xlabel(label.replace("p_", "u_"), fontsize=15)
        ax.set_ylabel("Cumulative probability", fontsize=15)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"Empirical and model CDFs: {field}", fontsize=15, fontweight="bold")
        ax.legend(fontsize=12, loc="lower right")
        _save_paper_figure(fig, output_file(outdir, f"goodness_of_fit_{field}_cdf.png"))

    csv_path = output_file(outdir, "goodness_of_fit_metrics.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "component", "ks_stat_kappa", "ks_p_kappa", "ks_stat_maxwellian",
            "ks_p_maxwellian", "ad_stat_kappa", "ad_p_kappa",
            "ad_stat_maxwellian", "ad_p_maxwellian",
        ])
        writer.writerows(results_rows)
    print(f"  Saved: {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════

def print_summary(ions, electrons, step: int):
    """Print summary statistics of the particle data."""
    print("\n" + "=" * 60)
    print(f"PARTICLE DATA SUMMARY (step {step}, t*Omega_ci={step_to_omegaci(step):.3f})")
    print("=" * 60)

    for name, sp in [("IONS", ions), ("ELECTRONS", electrons)]:
        m = np.abs(sp["m"][0])
        T_perp = 0.5 * m * (np.var(sp["px"]) + np.var(sp["py"]))
        T_par = m * np.var(sp["pz"])
        print(f"\n  {name}:")
        print(f"    Count: {len(sp):,}")
        print(f"    Mass (m): {sp['m'][0]:.4f}")
        print(f"    Charge (q): {sp['q'][0]:.4f}")
        print(f"    T_perp = {T_perp:.5f}, T_par = {T_par:.5f}")
        print(f"    T_perp/T_par = {T_perp / T_par:.4f}")

    print("\n" + "=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 6: Distribution temporal evolution
# ══════════════════════════════════════════════════════════════════════════════

def plot_distribution_evolution(
    filepaths: list[str],
    outdir: str,
    bins_parallel: int = 180,
    bins_perp: int = 120,
):
    """f(v_z, t) and f(|v_perp|, t) of ions and electrons, in units of v_A.

    PSC stores u = gamma v; velocities are converted before binning. Each
    column is normalised by the total particle weight of that snapshot, so the
    probability outside the plotted range is not folded back into it. Colours
    below 1e-4 of the maximum are cut: single-particle bins are noise.
    """
    print("\nBuilding distribution evolution plot...")

    sampled_paths = sample_filepaths(filepaths)
    if len(sampled_paths) != len(filepaths):
        print(
            f"  Using {len(sampled_paths)} uniformly sampled snapshots "
            f"out of {len(filepaths)} for temporal evolution."
        )

    def velocities(phase, species):
        vx, vy, vz, _ = velocity_from_u(phase[f"{species}_px"], phase[f"{species}_py"],
                                        phase[f"{species}_pz"])
        return vz / VA_OVER_C, np.hypot(vx, vy) / VA_OVER_C, phase[f"{species}_w"]

    reference = load_particle_phase_space(sampled_paths[-1])
    edges = {}
    for species in ("ions", "electrons"):
        vz, vperp, _ = velocities(reference, species)
        par_max, perp_max = np.percentile(np.abs(vz), 99.5), np.percentile(vperp, 99.5)
        edges[species] = (np.linspace(-par_max, par_max, bins_parallel + 1),
                          np.linspace(0.0, perp_max, bins_perp + 1))
    del reference

    matrices = {(sp, c): np.zeros((len(edges[sp][i]) - 1, len(sampled_paths)))
                for sp in ("ions", "electrons") for i, c in enumerate(("par", "perp"))}
    times = []
    for idx, filepath in enumerate(sampled_paths):
        phase = load_particle_phase_space(filepath)
        times.append(step_to_omegaci(extract_step(filepath)))
        for species in ("ions", "electrons"):
            vz, vperp, w = velocities(phase, species)
            for i, (component, values) in enumerate((("par", vz), ("perp", vperp))):
                bins = edges[species][i]
                counts, _ = np.histogram(values, bins=bins, weights=w)
                matrices[(species, component)][:, idx] = counts / (w.sum() * np.diff(bins))
        del phase; gc.collect()

    times = np.asarray(times, dtype=float)
    if len(times) == 1:
        time_edges = np.array([times[0] - 0.5, times[0] + 0.5], dtype=float)
    else:
        deltas = np.diff(times)
        time_edges = np.concatenate([[times[0] - 0.5 * deltas[0]],
                                     0.5 * (times[:-1] + times[1:]),
                                     [times[-1] + 0.5 * deltas[-1]]])

    panels = [
        (("ions", "par"), r"Ions: $f(v_z, t)$", r"$v_z/v_A$  ($\parallel B_0$)",
         r"$f(v_z)$  [per $v_A$]", "distribution_evolution_ions_parallel.png"),
        (("ions", "perp"), r"Ions: $f(|v_\perp|, t)$", r"$|v_\perp|/v_A$",
         r"$f(|v_\perp|)$  [per $v_A$]", "distribution_evolution_ions_perp.png"),
        (("electrons", "par"), r"Electrons: $f(v_z, t)$", r"$v_z/v_A$  ($\parallel B_0$)",
         r"$f(v_z)$  [per $v_A$]", "distribution_evolution_electrons_parallel.png"),
        (("electrons", "perp"), r"Electrons: $f(|v_\perp|, t)$", r"$|v_\perp|/v_A$",
         r"$f(|v_\perp|)$  [per $v_A$]", "distribution_evolution_electrons_perp.png"),
    ]
    cmap = plt.get_cmap(ps.CMAP_SEQUENTIAL).copy()
    cmap.set_bad(ps.PANEL_BG)
    for key, title, ylabel, clabel, filename in panels:
        matrix = matrices[key]
        vmax = float(matrix.max()) if matrix.size and matrix.max() > 0 else 1.0
        vmin = vmax * 1e-4
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        _style_paper_axes(ax)
        im = ax.pcolormesh(time_edges, edges[key[0]][0 if key[1] == "par" else 1],
                           np.ma.masked_less(matrix, vmin), shading="auto",
                           norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, rasterized=True)
        ax.set_title(f"{title} — {PROFILE_LABEL}", fontsize=14, fontweight="bold")
        ax.set_xlabel(r"$t\Omega_{ci}$")
        ax.set_ylabel(ylabel)
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label(clabel)
        fig.tight_layout()
        _save_paper_figure(fig, output_file(outdir, filename))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 7: Magnetic fluctuation evolution
# ══════════════════════════════════════════════════════════════════════════════
#
# The particle-anisotropy-vs-time panels this function used to also produce
# (T_perp/T_par and its inverse for ions/electrons) are dropped: they
# duplicated 01_anisotropy's anisotropy_ratio_vs_time.png/inverse_anisotropy_
# vs_time.png (denser, computed from all field/moment snapshots) and
# 09_physical_diagnostics's anisotropy_vs_time.png/temperature_parallel_
# perp_vs_time.png -- three independent implementations of the same T_perp/
# T_par(t) curve. This module now only produces the one thing that was
# actually unique here: the field-derived delta_B fluctuation time series.

def plot_macro_evolution(
    outdir: str,
    field_files: dict[int, str] | None = None,
):
    """Track magnetic-field fluctuations vs. time."""
    print("\nBuilding magnetic-fluctuation evolution plot...")

    field_steps = []
    delta_b_total = []
    delta_b_parallel = []
    delta_b_perp = []

    if field_files:
        field_items = sorted(field_files.items())
        if len(field_items) > MAX_EVOLUTION_FILES:
            keep = np.unique(
                np.linspace(0, len(field_items) - 1, MAX_EVOLUTION_FILES, dtype=int)
            )
            field_items = [field_items[index] for index in keep]
        for step, filepath in field_items:
            metrics = load_field_fluctuation_metrics(filepath)
            field_steps.append(step_to_omegaci(step))
            delta_b_total.append(metrics["delta_b_total"])
            delta_b_parallel.append(metrics["delta_b_parallel"])
            delta_b_perp.append(metrics["delta_b_perp"])

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    _style_paper_axes(ax)
    if field_steps:
        field_steps = np.asarray(field_steps, dtype=float)
        ax.plot(field_steps, delta_b_total, marker="o", linewidth=2.0,
                color="#0072B2", label=r"$\delta B_{\rm rms} / B_0$")
        ax.plot(field_steps, delta_b_parallel, marker="^", linewidth=1.8,
                color="#56B4E9", label=r"$\delta B_{\parallel,\rm rms} / B_0$")
        ax.plot(field_steps, delta_b_perp, marker="d", linewidth=1.8,
                color="#E69F00", label=r"$\delta B_{\perp,\rm rms} / B_0$")
        ax.legend()
    else:
        ax.text(
            0.5, 0.5,
            "No matching pfd.*.h5 field files found for these steps.",
            ha="center", va="center", transform=ax.transAxes,
        )
    ax.set_ylabel("Normalised fluctuation amplitude")
    ax.set_xlabel(r"$t\Omega_{ci}$")
    ax.set_title("Magnetic-field fluctuations")
    fig.tight_layout()
    _save_paper_figure(fig, output_file(outdir, "magnetic_fluctuations_vs_time.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 9: 1-D VDF evolution with suprathermal-tail quantification
# ══════════════════════════════════════════════════════════════════════════════

def plot_1d_vdf_evolution(
    filepaths: list[str],
    outdir: str,
    n_times: int = 6,
    nbins: int = 200,
):
    """Ion f(v_z) and f(|v_perp|) at selected times, against t = 0 Maxwellians.

    PSC stores u = gamma v, so the velocities are converted before binning;
    z is the direction of the background field B0 (global frame, not the
    local field). Each curve is normalised by the total particle weight, so
    the probability outside the plotted range is not redistributed into it.
    The references carry the t = 0 variances: a Gaussian for v_z and, for the
    magnitude of the two perpendicular components, the Rayleigh distribution
    (v/s^2) exp(-v^2 / 2 s^2) with s^2 = <v_perp^2>/2 -- a Gaussian in |v_perp|
    would be the wrong reference. The printed tail fractions count
    |v_z| > 3 s_par and |v_perp| > 3 s_perp.
    """
    print("\nBuilding 1-D VDF temporal evolution with tail diagnostics...")

    paths = sample_filepaths(filepaths, max_files=n_times)
    times = [step_to_omegaci(extract_step(p)) for p in paths]
    cmap = plt.get_cmap(ps.CMAP_SEQUENTIAL)
    colors = [cmap(0.9 * i / max(len(paths) - 1, 1)) for i in range(len(paths))]

    def ion_velocities(path):
        phase = load_particle_phase_space(path)
        vx, vy, vz, _ = velocity_from_u(phase["ions_px"], phase["ions_py"], phase["ions_pz"])
        return vz / VA_OVER_C, np.hypot(vx, vy) / VA_OVER_C, phase["ions_w"]

    vz_ref, vperp_ref, _ = ion_velocities(paths[-1])
    vpar_max = float(np.percentile(np.abs(vz_ref), 99.5))
    vperp_max = float(np.percentile(vperp_ref, 99.5))
    del vz_ref, vperp_ref; gc.collect()

    vpar_edges = np.linspace(-vpar_max, vpar_max, nbins + 1)
    vperp_edges = np.linspace(0.0, vperp_max, nbins + 1)
    vc_par = 0.5 * (vpar_edges[:-1] + vpar_edges[1:])
    vc_perp = 0.5 * (vperp_edges[:-1] + vperp_edges[1:])

    def pdf(values, weights, edges):
        counts, _ = np.histogram(values, bins=edges, weights=weights)
        f = counts / (weights.sum() * np.diff(edges))
        return np.where(f > 0, f, np.nan)            # empty bins: gaps, not log(0)

    fig_par, ax_par = plt.subplots(figsize=(9.2, 5.2))
    fig_perp, ax_perp = plt.subplots(figsize=(9.2, 5.2))
    axes = [ax_par, ax_perp]
    for ax in axes:
        _style_paper_axes(ax)

    s_par = s_perp = None
    tails = []
    for idx, (path, toci, col) in enumerate(zip(paths, times, colors)):
        vz, vperp, w = ion_velocities(path)
        label = rf"$t\Omega_{{ci}} = {toci:.1f}$"
        axes[0].semilogy(vc_par, pdf(vz, w, vpar_edges), color=col, linewidth=1.6, label=label)
        axes[1].semilogy(vc_perp, pdf(vperp, w, vperp_edges), color=col, linewidth=1.6, label=label)
        if idx == 0:
            mean_z = np.average(vz, weights=w)
            s_par = float(np.sqrt(np.average((vz - mean_z) ** 2, weights=w)))
            s_perp = float(np.sqrt(0.5 * np.average(vperp ** 2, weights=w)))
        tails.append((toci, float(np.average(np.abs(vz) > 3 * s_par, weights=w)),
                      float(np.average(vperp > 3 * s_perp, weights=w))))
        del vz, vperp, w; gc.collect()

    axes[0].semilogy(vc_par, maxwellian_1d(vc_par, s_par), "k--", linewidth=1.8, alpha=0.7,
                     label=r"Gaussian, $t=0$ variance")
    rayleigh = vc_perp / s_perp ** 2 * np.exp(-vc_perp ** 2 / (2 * s_perp ** 2))
    axes[1].semilogy(vc_perp, np.where(rayleigh > 0, rayleigh, np.nan), "k--", linewidth=1.8,
                     alpha=0.7, label=r"2-D Maxwellian (Rayleigh), $t=0$ variance")
    for ax, s, vmax, lo in ((axes[0], s_par, vpar_max, -vpar_max), (axes[1], s_perp, vperp_max, 0.0)):
        ax.axvspan(3 * s, vmax, alpha=0.08, color="#D55E00", label=r"tail beyond $3\sigma$ at $t=0$")
        if lo < 0:
            ax.axvspan(lo, -3 * s, alpha=0.08, color="#D55E00")

    for ax, xlabel, ylabel, title in (
        (axes[0], r"$v_z/v_A$  ($\parallel B_0$)", r"$f(v_z)$  [per $v_A$]", "Ion parallel VDF"),
        (axes[1], r"$|v_\perp|/v_A$", r"$f(|v_\perp|)$  [per $v_A$]", "Ion perpendicular speed distribution"),
    ):
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_title(f"{title} — {PROFILE_LABEL}", fontsize=15, fontweight="bold")
        ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        ax.grid(True, alpha=0.3, which="both", ls="--")
        ax.tick_params(which="both", direction="in", top=True, right=True)

    fig_par.tight_layout()
    fig_perp.tight_layout()
    _save_paper_figure(fig_par, output_file(outdir, "vdf_1d_parallel_evolution.png"))
    _save_paper_figure(fig_perp, output_file(outdir, "vdf_1d_perp_evolution.png"))

    print("  Ion tail fractions (|v_z| > 3 s_par, |v_perp| > 3 s_perp; s at t = 0):")
    for toci, tp, tperp in tails:
        print(f"    t Omega_ci = {toci:8.2f}  par={tp:.4f}  perp={tperp:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 10: Energy partition (magnetic, kinetic, thermal)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_particle_energies(filepath: str) -> dict:
    """Return kinetic and thermal energies from a particle snapshot.

    PSC stores u = gamma*v. For this non-relativistic run:
      E_bulk = 0.5*m*|<u>|^2
      E_th   = 0.5*m*<|u-<u>|^2>
    Values are per macroparticle, so species with the same macro-weight can be
    compared and summed without introducing sample-size-dependent factors.
    """
    with h5py.File(filepath, "r") as f:
        dset  = f["particles"]["p0"]["1d"]
        n_tot = dset["q"].shape[0]
        idx = _sample_indices(n_tot, MAX_PARTICLES)
        q  = dset["q"][idx].astype(np.float64)
        m  = dset["m"][idx].astype(np.float64)
        px = dset["px"][idx].astype(np.float64)
        py = dset["py"][idx].astype(np.float64)
        pz = dset["pz"][idx].astype(np.float64)

    ion_mask  = q > 0
    elec_mask = q < 0

    result = {}
    for name, mask in [("ion", ion_mask), ("elec", elec_mask)]:
        mi     = float(np.abs(m[mask][0])) if mask.any() else 1.0
        vx     = px[mask]
        vy     = py[mask]
        vz     = pz[mask]
        bulk_v2 = np.mean(vx)**2 + np.mean(vy)**2 + np.mean(vz)**2
        rand_v2 = np.mean((vx - np.mean(vx))**2 +
                           (vy - np.mean(vy))**2 +
                           (vz - np.mean(vz))**2)
        t_perp = 0.5 * mi * (np.var(vx) + np.var(vy))
        t_par  = mi * np.var(vz)
        result[f"{name}_kinetic_bulk"]    = 0.5 * mi * bulk_v2
        result[f"{name}_thermal_energy"]  = 0.5 * mi * rand_v2
        result[f"{name}_t_perp"] = t_perp
        result[f"{name}_t_par"]  = t_par

    return result


def plot_energy_partition(
    filepaths: list[str],
    outdir: str,
    field_files: dict[int, str] | None = None,
):
    """Track magnetic, bulk-kinetic, and thermal energies over simulation time.

    Energy budget normalised to total initial energy E_0.
    Follows the methodology of PIC anisotropy-instability studies
    (e.g. Hellinger & Travnicek 2008; Kunz et al. 2014).
    """
    print("\nBuilding energy partition plot...")

    paths  = sample_filepaths(filepaths, max_files=MAX_EVOLUTION_FILES)
    steps  = np.array([extract_step(p) for p in paths], dtype=float)
    times  = np.array([step_to_omegaci(int(st)) for st in steps])

    ion_kin_bulk    = []
    ion_thermal     = []
    elec_kin_bulk   = []
    elec_thermal    = []
    mag_energy      = []

    for path, step in zip(paths, steps.astype(int)):
        e = _compute_particle_energies(path)
        ion_kin_bulk.append(e["ion_kinetic_bulk"])
        ion_thermal.append(e["ion_thermal_energy"])
        elec_kin_bulk.append(e["elec_kinetic_bulk"])
        elec_thermal.append(e["elec_thermal_energy"])

        # Magnetic energy from field file if available
        eb = np.nan
        if field_files and step in field_files:
            try:
                m = load_field_fluctuation_metrics(field_files[step])
                # delta_b_total = rms|dB|/B0, so E_dB/(B0^2/2) = (dB_rms/B0)^2
                eb = m["delta_b_total"]**2
            except Exception:
                pass
        mag_energy.append(eb)
        gc.collect()

    ion_kin_bulk    = np.array(ion_kin_bulk)
    ion_thermal     = np.array(ion_thermal)
    elec_kin_bulk   = np.array(elec_kin_bulk)
    elec_thermal    = np.array(elec_thermal)
    mag_energy      = np.array(mag_energy)

    # Normalise to initial total particle energy
    E0 = ion_kin_bulk[0] + elec_kin_bulk[0] + ion_thermal[0] + elec_thermal[0]
    if E0 < 1e-30:
        E0 = 1.0

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    _style_paper_axes(ax)
    ax.plot(times, ion_thermal / E0,     "s-", color="#E69F00", linewidth=2,
            label=r"Ion thermal energy  ($\frac{1}{2}m_i\langle\delta u^2\rangle$)")
    ax.plot(times, elec_thermal / E0,    "^-", color="#0072B2", linewidth=2,
            label=r"Electron thermal energy")
    ax.plot(times, ion_kin_bulk / E0,    "d-", color="#009E73", linewidth=1.5,
            label=r"Ion bulk kinetic")
    ax.plot(times, elec_kin_bulk / E0,   "x-", color="#56B4E9", linewidth=1.5,
            label=r"Electron bulk kinetic")
    ax.set_ylabel(r"Energy  [$E_0$]", fontsize=15)
    ax.set_xlabel(r"$t\Omega_{ci}$", fontsize=14)
    ax.set_title("Particle energy components (normalised to $E_0$)",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout()
    _save_paper_figure(fig, output_file(outdir, "particle_energy_partition.png"))

    fig, ax2 = plt.subplots(figsize=(7.8, 5.4))
    _style_paper_axes(ax2)
    if not np.all(np.isnan(mag_energy)):
        ax2.plot(times, mag_energy, "o-", color="#CC79A7", linewidth=2,
                 label=r"$(\delta B_{\rm rms}/B_0)^2$  (from field files)")
        ax2.set_ylabel(r"$E_{\delta B}$  [$B_0^2/2$]", fontsize=15)
        ax2.legend(fontsize=12)
    else:
        ax2.text(0.5, 0.5,
                 "Magnetic energy requires matching pfd.*.h5 field files.",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=14, color="gray")
    ax2.set_xlabel(r"$t\Omega_{ci}$", fontsize=14)
    ax2.set_title("Magnetic energy from field fluctuations",
                  fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save_paper_figure(fig, output_file(outdir, "magnetic_energy_fluctuation.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global OUTPUT_PREFIX
    parser = argparse.ArgumentParser(
        description=f"Particle diagnostics for {PROFILE_LABEL}."
    )
    parser.add_argument(
        "input", nargs="?",
        default=os.path.join(os.path.dirname(__file__), "..", "build", "src"),
        help="Particle file, data directory, or glob pattern.",
    )
    parser.add_argument(
        "--outdir",
        help="Output directory (default: <data-dir>/prt_plots).",
    )
    parser.add_argument(
        "--run-name", default="",
        help="Name prefixed to every generated file.",
    )
    args = parser.parse_args()
    input_path = args.input
    clean_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name).strip("_")
    OUTPUT_PREFIX = f"{clean_name}_" if clean_name else ""

    try:
        filepaths = resolve_particle_files(input_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Usage: python plot_prt.py [path_to_prt_file.h5 | directory | glob]")
        sys.exit(1)

    filepath = filepaths[-1]
    outdir = (
        os.path.abspath(args.outdir)
        if args.outdir
        else os.path.join(os.path.dirname(os.path.abspath(filepath)), OUTPUT_DIR)
    )
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}\n")

    if len(filepaths) > 1:
        print(
            f"Resolved {len(filepaths)} particle files. "
            f"Using latest snapshot for static plots: {os.path.basename(filepath)}"
        )
        field_files = resolve_field_files(os.path.dirname(os.path.abspath(filepath)))
        plot_distribution_evolution(filepaths, outdir);                  gc.collect()
        plot_macro_evolution(outdir, field_files=field_files);           gc.collect()
        # ── New diagnostics ────────────────────────────────────────────
        print("\nRunning distribution-tail and energy diagnostics...")
        plot_1d_vdf_evolution(filepaths, outdir);                         gc.collect()
        plot_energy_partition(filepaths, outdir, field_files=field_files); gc.collect()

    data = load_particles(filepath)
    ions, electrons = separate_species(data)
    del data; gc.collect()

    print_summary(ions, electrons, extract_step(filepath))

    print("\nGenerating plots...")
    if KAPPA is not None:
        plot_kappa_comparison(ions, outdir); gc.collect()
        print("\nRunning goodness-of-fit tests (Anderson-Darling & Kolmogorov-Smirnov)...")
        plot_goodness_of_fit(ions, outdir);  gc.collect()
    else:
        print("  Skipping Kappa-only comparison and goodness-of-fit plots for a Bi-Maxwellian profile.")

    print(f"\nAll plots saved to: {outdir}")


if __name__ == "__main__":
    main()
