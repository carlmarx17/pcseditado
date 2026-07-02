#!/usr/bin/env python3
"""
plot_prt.py — Particle diagnostic suite for PSC PIC simulations.

Reads HDF5 particle files and generates publication-quality diagnostics:
  1. 2D VDF heatmap: f(v_perp, v_parallel)
  2. Kappa vs. Maxwellian distribution comparison
  3. Goodness-of-fit tests (Anderson-Darling & Kolmogorov-Smirnov)
  4. Multi-time VDF snapshots
  5. 1D distribution temporal evolution
  6. Anisotropy + magnetic-fluctuation time series

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
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
try:
    from scipy.ndimage import gaussian_filter
    from scipy.special import gamma as gamma_func
    from scipy import stats as scipy_stats
except ImportError:
    gaussian_filter = lambda values, sigma=1.0: values
    gamma_func = None
    scipy_stats = None

try:
    from data_reader import PICDataReader
except ImportError:
    PICDataReader = None
from psc_units import (
    B0, KAPPA, MASS_RATIO, TI_PAR, TI_PERP, VA_OVER_C, ZI,
    BETA_I_PAR as _BETA_I_PAR_SIM,
    BETA_I_PERP_OVER_PAR as _TI_RATIO_SIM,
    DRIVEN_SPECIES, INSTABILITY, M_ION, M_ELEC, PROFILE_LABEL,
    step_to_omegaci,
)

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = "prt_plots"
DPI = 150          # reducido de 200 para menor uso de RAM en savefig
MAX_EVOLUTION_FILES = 12
MAX_VDF_SNAPSHOTS = 5
NBINS_VDF = 250    # reducido de 400; suficiente para resolución publicable
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
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def _centered_vdf_coordinates(species: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vpar = np.asarray(species["pz"], dtype=float) / VA_OVER_C_
    vx = np.asarray(species["px"], dtype=float) / VA_OVER_C_
    vy = np.asarray(species["py"], dtype=float) / VA_OVER_C_
    vpar = vpar - np.nanmedian(vpar)
    vx = vx - np.nanmedian(vx)
    vy = vy - np.nanmedian(vy)
    vperp = np.sqrt(vx * vx + vy * vy)
    return vpar, vperp


def _plot_vdf_single(
    vpar: np.ndarray,
    vperp: np.ndarray,
    title: str,
    path: str,
    *,
    par_abs: float | None = None,
    perp_hi: float | None = None,
    bins_parallel: int = NBINS_VDF,
    bins_perp: int | None = None,
    cmap: str = "magma",
):
    bins_perp = bins_perp or max(96, int(bins_parallel * 0.65))
    finite = np.isfinite(vpar) & np.isfinite(vperp)
    vpar = vpar[finite]
    vperp = vperp[finite]
    if vpar.size < 10:
        return
    if par_abs is None:
        par_abs = float(np.nanpercentile(np.abs(vpar), 99.7))
    if perp_hi is None:
        perp_hi = float(np.nanpercentile(vperp, 99.7))
    if par_abs <= 0 or perp_hi <= 0:
        return

    hist, xedges, yedges = np.histogram2d(
        vpar,
        vperp,
        bins=[bins_parallel, bins_perp],
        range=[[-par_abs, par_abs], [0.0, perp_hi]],
        density=True,
    )
    hist = gaussian_filter(hist.astype(float), sigma=0.8)
    positive = hist[np.isfinite(hist) & (hist > 0)]
    if positive.size == 0:
        return
    vmin = max(float(np.nanpercentile(positive, 1.0)), float(positive.max()) * 1e-5)
    vmax = float(positive.max())

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    _style_paper_axes(ax)
    im = ax.pcolormesh(
        xedges,
        yedges,
        hist.T,
        shading="auto",
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    levels = vmax * np.array([1e-4, 1e-3, 1e-2, 1e-1])
    levels = levels[(levels > vmin) & (levels < vmax)]
    if levels.size:
        ax.contour(
            0.5 * (xedges[:-1] + xedges[1:]),
            0.5 * (yedges[:-1] + yedges[1:]),
            hist.T,
            levels=levels,
            colors="white",
            linewidths=0.65,
            alpha=0.75,
        )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$f(v_\parallel,v_\perp)$ [PDF]", fontsize=14)
    cbar.ax.tick_params(which="both", direction="in", labelsize=13)
    ax.axvline(0.0, color="white", lw=0.8, ls=":", alpha=0.8)
    ax.set_xlim(-par_abs, par_abs)
    ax.set_ylim(0.0, perp_hi)
    ax.set_xlabel(r"$(v_\parallel-\langle v_\parallel\rangle)/v_A$", fontsize=15)
    ax.set_ylabel(r"$v_\perp/v_A$", fontsize=15)
    ax.set_title(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save_paper_figure(fig, path)


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

    ion_mask  = q > 0
    elec_mask = q < 0
    result = {
        "ions_px":        px[ion_mask],
        "ions_py":        py[ion_mask],
        "ions_pz":        pz[ion_mask],
        "ions_perp":      np.sqrt(px[ion_mask]**2 + py[ion_mask]**2),
        "electrons_px":   px[elec_mask],
        "electrons_py":   py[elec_mask],
        "electrons_pz":   pz[elec_mask],
        "electrons_perp": np.sqrt(px[elec_mask]**2 + py[elec_mask]**2),
    }
    del q, px, py, pz, ion_mask, elec_mask
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


def compute_species_temperatures(species: np.ndarray) -> tuple[float, float]:
    """Compute temperatures from PSC normalized momentum u = gamma*v."""
    mass = abs(float(species["m"][0]))
    px = np.asarray(species["px"], dtype=float)
    py = np.asarray(species["py"], dtype=float)
    pz = np.asarray(species["pz"], dtype=float)
    t_perp = 0.5 * mass * (np.var(px) + np.var(py))
    t_par = mass * np.var(pz)
    return t_perp, t_par


def summarize_particle_snapshot(filepath: str) -> dict:
    """Compute anisotropy summary from one snapshot."""
    data = load_particles(filepath, verbose=False)
    ions, electrons = separate_species(data, verbose=False)
    ion_tp, ion_tl = compute_species_temperatures(ions)
    elec_tp, elec_tl = compute_species_temperatures(electrons)
    return {
        "step": extract_step(filepath),
        "ion_anisotropy": ion_tp / max(ion_tl, 1e-30),
        "electron_anisotropy": elec_tp / max(elec_tl, 1e-30),
        "ion_tperp": ion_tp,
        "ion_tpar": ion_tl,
        "electron_tperp": elec_tp,
        "electron_tpar": elec_tl,
    }


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
#  Plot 1: 2D VDF — f(v_perp, v_parallel)
# ══════════════════════════════════════════════════════════════════════════════

def plot_vdf(ions, electrons, outdir: str):
    """
    2D reduced VDF: f(v_perp, v_parallel) for ions and electrons.

    Each species is saved as a separate paper-style PNG. The perpendicular
    velocity is not mirrored; it is a positive magnitude.
    """
    species_cfg = [
        ("Ions", ions, "vdf_2d_ions.png", "magma"),
        ("Electrons", electrons, "vdf_2d_electrons.png", "viridis"),
    ]

    for label, sp, filename, cmap in species_cfg:
        vpar, vperp = _centered_vdf_coordinates(sp)
        _plot_vdf_single(
            vpar,
            vperp,
            rf"{label} VDF, initial snapshot",
            output_file(outdir, filename),
            cmap=cmap,
        )


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
        ax.step(bin_centers, hist_plot, where="mid", color="#c0392b", linewidth=1.5, label="Ion data")
        ax.semilogy(v_range, kappa_1d(v_range, kappa, v_th), "-", color="#27ae60", linewidth=2.2,
                    label=rf"Kappa ($\kappa$={kappa})")
        ax.semilogy(v_range, maxwellian_1d(v_range, v_th), "--", color="#8e44ad", linewidth=2.2,
                    label="Maxwellian")
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(r"$f(u)$ [PDF]", fontsize=15)
        ax.set_title(rf"Ion {short} distribution ({direction})", fontsize=15, fontweight="bold")
        ax.set_yscale("log")
        ax.set_ylim(bottom=y_min)
        ax.legend(fontsize=12)
        _save_paper_figure(fig, output_file(outdir, f"kappa_comparison_{slug}_semilog.png"))

        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        _style_paper_axes(ax)
        p2_centers = bin_centers**2
        p2_range = v_range**2
        sort_idx = np.argsort(p2_range)
        ax.step(p2_centers, hist_plot, where="mid", color="#c0392b", linewidth=1.5, label="Ion data")
        ax.semilogy(p2_range[sort_idx], kappa_1d(v_range[sort_idx], kappa, v_th), "-",
                    color="#27ae60", linewidth=2.2, label=rf"Kappa ($\kappa$={kappa})")
        ax.semilogy(p2_range[sort_idx], maxwellian_1d(v_range[sort_idx], v_th), "--",
                    color="#8e44ad", linewidth=2.2, label="Maxwellian")
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
        ax.step(bin_centers_pos, hist_plot_pos, where="mid", color="#c0392b", linewidth=1.5,
                label="Ion data")
        if len(bin_centers_pos) > 1:
            v_pos = np.linspace(bin_centers_pos[0], bin_centers_pos[-1], 1000)
            ax.loglog(v_pos, kappa_1d(v_pos, kappa, v_th), "-", color="#27ae60", linewidth=2.2,
                      label=rf"Kappa")
            ax.loglog(v_pos, maxwellian_1d(v_pos, v_th), "--", color="#8e44ad", linewidth=2.2,
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
                 color="#c0392b", linewidth=1.5, alpha=0.85,
                 label="Ion data", zorder=2)
        ax0.semilogy(v_range, kappa_1d(v_range, kappa, v_th), "-",
                     color="#27ae60", linewidth=2.5, zorder=3,
                     label=rf"Kappa ($\kappa$={kappa})")
        ax0.semilogy(v_range, maxwellian_1d(v_range, v_th), "--",
                     color="#8e44ad", linewidth=2.5, zorder=3,
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
                 color="#c0392b", linewidth=1.5, alpha=0.85,
                 label="Ion data", zorder=2)
        ax1.semilogy(p2_range[sort_idx], kappa_1d(v_range[sort_idx], kappa, v_th), "-",
                     color="#27ae60", linewidth=2.5, zorder=3,
                     label=rf"Kappa ($\kappa$={kappa})")
        ax1.semilogy(p2_range[sort_idx], maxwellian_1d(v_range[sort_idx], v_th), "--",
                     color="#8e44ad", linewidth=2.5, zorder=3,
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
                 color="#c0392b", linewidth=1.5, alpha=0.85,
                 label="Ion data (positive tail)", zorder=2)

        if len(bin_centers_pos) > 1:
            v_pos = np.linspace(bin_centers_pos[0], bin_centers_pos[-1], 1000)
            ax2.loglog(v_pos, kappa_1d(v_pos, kappa, v_th), "-",
                       color="#27ae60", linewidth=2.5, zorder=3,
                       label=rf"Kappa (slope $\propto p^{{-2\kappa}}$)")
            ax2.loglog(v_pos, maxwellian_1d(v_pos, v_th), "--",
                       color="#8e44ad", linewidth=2.5, zorder=3,
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
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
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
        ks_res_maxw = scipy_stats.kstest(sample, "norm", args=(0.0, v_th))
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
        ax.step(sample, ecdf_y, where="post", color="#c0392b", linewidth=1.8,
                label="Empirical CDF")
        ax.plot(sample, kappa_cdf_vals, "-", color="#27ae60", linewidth=2.2,
                label=rf"Kappa CDF ($\kappa={kappa}$)")
        ax.plot(sample, maxw_cdf_vals, "--", color="#8e44ad", linewidth=2.2,
                label="Maxwellian CDF")
        ks_idx = np.argmax(np.abs(ecdf_y - kappa_cdf_vals))
        ax.vlines(sample[ks_idx], ecdf_y[ks_idx], kappa_cdf_vals[ks_idx],
                  colors="#e67e22", linewidths=2.0, label=rf"$D_{{KS}}={ks_stat_kappa:.3g}$")
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
    return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        rf"Goodness-of-Fit Tests — Kappa ($\kappa = {kappa}$) vs Maxwellian — Ions",
        fontsize=18, fontweight="bold", y=1.01,
    )

    NSAMPLE = 5_000
    rng = np.random.default_rng(42)

    for col, (field, label) in enumerate(zip(mom_fields, comp_labels)):
        ion_p = np.asarray(ions[field], dtype=float)
        v_th = norm_vth[field]

        # Sub-sample for A-D (O(n^2) memory)
        sample = (
            rng.choice(ion_p, size=NSAMPLE, replace=False)
            if len(ion_p) > NSAMPLE
            else ion_p.copy()
        )
        sample.sort()

        # Theoretical CDFs on the sample grid
        kappa_cdf_vals = _kappa_cdf(sample, kappa, v_th)
        norm_sigma = v_th
        maxw_cdf_vals = scipy_stats.norm.cdf(sample, loc=0.0, scale=norm_sigma)

        n = len(sample)
        ecdf_y = np.arange(1, n + 1) / n

        # ── K-S vs Kappa ─────────────────────────────────────────────────
        ks_stat_kappa = np.max(np.abs(ecdf_y - kappa_cdf_vals))
        ks_p_kappa = scipy_stats.kstwobign.sf(np.sqrt(n) * ks_stat_kappa)

        # ── K-S vs Maxwellian ────────────────────────────────────────────
        ks_res_maxw = scipy_stats.kstest(sample, "norm", args=(0.0, norm_sigma))
        ks_stat_maxw = ks_res_maxw.statistic
        ks_p_maxw = ks_res_maxw.pvalue

        # ── A-D vs Kappa ─────────────────────────────────────────────────
        i_idx = np.arange(1, n + 1)
        cdf_lo = np.clip(kappa_cdf_vals, 1e-12, 1 - 1e-12)
        cdf_hi = np.clip(kappa_cdf_vals[::-1], 1e-12, 1 - 1e-12)
        ad_kappa = -n - np.mean(
            (2 * i_idx - 1) * (np.log(cdf_lo) + np.log(1 - cdf_hi))
        )
        ad_kappa_norm = ad_kappa * (1 + 4 / n - 25 / n**2)
        ad_p_kappa = _ad_p_value(ad_kappa_norm)

        # ── A-D vs Maxwellian ────────────────────────────────────────────
        cdf_lo_m = np.clip(maxw_cdf_vals, 1e-12, 1 - 1e-12)
        cdf_hi_m = np.clip(maxw_cdf_vals[::-1], 1e-12, 1 - 1e-12)
        ad_maxw = -n - np.mean(
            (2 * i_idx - 1) * (np.log(cdf_lo_m) + np.log(1 - cdf_hi_m))
        )
        ad_maxw_norm = ad_maxw * (1 + 4 / n - 25 / n**2)
        ad_p_maxw = _ad_p_value(ad_maxw_norm)

        # ── Row 0: CDF comparison ────────────────────────────────────────
        ax_cdf = axes[0, col]
        ax_cdf.step(sample, ecdf_y, where="post",
                    color="#c0392b", linewidth=1.8, alpha=0.9,
                    label="Empirical CDF", zorder=3)
        ax_cdf.plot(sample, kappa_cdf_vals, "-",
                    color="#27ae60", linewidth=2.2, zorder=4,
                    label=rf"Kappa CDF ($\kappa={kappa}$)")
        ax_cdf.plot(sample, maxw_cdf_vals, "--",
                    color="#8e44ad", linewidth=2.2, zorder=4,
                    label="Maxwellian CDF")

        ks_idx = np.argmax(np.abs(ecdf_y - kappa_cdf_vals))
        ax_cdf.vlines(sample[ks_idx], ecdf_y[ks_idx], kappa_cdf_vals[ks_idx],
                      colors="#e67e22", linewidths=2.5, linestyles="solid",
                      label=rf"$D_{{KS}}$ = {ks_stat_kappa:.4f}")

        ax_cdf.set_xlabel(label, fontsize=14)
        ax_cdf.set_ylabel("Cumulative probability  $F(p)$", fontsize=14)
        ax_cdf.set_title(f"Empirical vs theoretical CDF\n{label}",
                         fontsize=14, fontweight="bold")
        ax_cdf.legend(fontsize=12, loc="upper left")
        ax_cdf.grid(True, alpha=0.3)
        ax_cdf.set_ylim(-0.02, 1.05)

        # ── Row 1: Results table ─────────────────────────────────────────
        ax_tbl = axes[1, col]
        ax_tbl.axis("off")

        def _fmt_p(p):
            return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"

        def _verdict(p, threshold=0.05):
            return ("✔  H₀ not rejected" if p > threshold else "✘  H₀ rejected")

        table_data = [
            ["Test", "Statistic", "p-value", "Conclusion (α=0.05)"],
            ["K-S  vs Kappa",      f"{ks_stat_kappa:.5f}", _fmt_p(ks_p_kappa), _verdict(ks_p_kappa)],
            ["K-S  vs Maxwellian", f"{ks_stat_maxw:.5f}",  _fmt_p(ks_p_maxw),  _verdict(ks_p_maxw)],
            ["A-D  vs Kappa",      f"{ad_kappa:.4f}",      _fmt_p(ad_p_kappa), _verdict(ad_p_kappa)],
            ["A-D  vs Maxwellian", f"{ad_maxw:.4f}",       _fmt_p(ad_p_maxw),  _verdict(ad_p_maxw)],
        ]

        col_widths = [0.30, 0.20, 0.18, 0.32]
        row_colors = ["#2c3e50", "#1a252f", "#22303c", "#1a252f", "#22303c"]
        text_colors = ["white", "#ecf0f1", "#ecf0f1", "#ecf0f1", "#ecf0f1"]

        y_start = 0.95
        row_h = 0.16
        for r_idx, row in enumerate(table_data):
            bg = row_colors[r_idx]
            tcol = text_colors[r_idx]
            x_cur = 0.0
            for c_idx, (cell, cw) in enumerate(zip(row, col_widths)):
                rect = FancyBboxPatch(
                    (x_cur, y_start - (r_idx + 1) * row_h),
                    cw - 0.005, row_h - 0.01,
                    boxstyle="round,pad=0.01",
                    facecolor=bg, edgecolor="none",
                    transform=ax_tbl.transAxes, clip_on=False,
                )
                ax_tbl.add_patch(rect)
                weight = "bold" if r_idx == 0 else "normal"
                ax_tbl.text(
                    x_cur + cw / 2,
                    y_start - (r_idx + 0.5) * row_h,
                    cell,
                    ha="center", va="center",
                    fontsize=12, color=tcol, fontweight=weight,
                    transform=ax_tbl.transAxes,
                )
                x_cur += cw

        ax_tbl.set_xlim(0, 1)
        ax_tbl.set_ylim(0, 1)
        ax_tbl.set_title(f"Test results — {label}",
                         fontsize=13, fontweight="bold", pad=8)

        print(
            f"  [{label}]  "
            f"KS-Kappa: D={ks_stat_kappa:.4f} p={_fmt_p(ks_p_kappa)}  "
            f"KS-Maxw: D={ks_stat_maxw:.4f} p={_fmt_p(ks_p_maxw)}  "
            f"AD-Kappa: A²={ad_kappa:.3f} p={_fmt_p(ad_p_kappa)}  "
            f"AD-Maxw: A²={ad_maxw:.3f} p={_fmt_p(ad_p_maxw)}"
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = output_file(outdir, "goodness_of_fit.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 4: Summary statistics
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
#  Plot 5: Multi-time VDF snapshots
# ══════════════════════════════════════════════════════════════════════════════

def plot_vdf_snapshots(
    filepaths: list[str],
    outdir: str,
    max_snapshots: int = MAX_VDF_SNAPSHOTS,
    bins_parallel: int = 180,
    bins_perp: int = 120,
):
    """Plot reduced VDFs for a few representative time steps."""
    print("\nBuilding separated VDF snapshots...")

    sampled_paths = sample_filepaths(filepaths, max_files=max_snapshots)
    sampled_data = [
        (extract_step(path), load_particle_phase_space(path))
        for path in sampled_paths
    ]

    ion_par_max = max(np.percentile(np.abs(d["ions_pz"] / VA_OVER_C_), 99.7) for _, d in sampled_data)
    ion_perp_max = max(np.percentile(d["ions_perp"] / VA_OVER_C_, 99.7) for _, d in sampled_data)
    elec_par_max = max(np.percentile(np.abs(d["electrons_pz"] / VA_OVER_C_), 99.7) for _, d in sampled_data)
    elec_perp_max = max(np.percentile(d["electrons_perp"] / VA_OVER_C_, 99.7) for _, d in sampled_data)

    species_config = [
        ("ions_pz", "ions_perp", ion_par_max, ion_perp_max, "ions", "Ions", "magma"),
        ("electrons_pz", "electrons_perp", elec_par_max, elec_perp_max, "electrons", "Electrons", "viridis"),
    ]

    for par_key, perp_key, par_max, perp_max, slug, label, cmap in species_config:
        for step, data in sampled_data:
            vpar = np.asarray(data[par_key], dtype=float) / VA_OVER_C_
            vperp = np.asarray(data[perp_key], dtype=float) / VA_OVER_C_
            vpar = vpar - np.nanmedian(vpar)
            _plot_vdf_single(
                vpar,
                vperp,
                rf"{label} VDF, step {step}, $t\Omega_{{ci}}={step_to_omegaci(step):.2f}$",
                output_file(outdir, f"vdf_2d_{slug}_step{step}.png"),
                par_abs=par_max,
                perp_hi=perp_max,
                bins_parallel=bins_parallel,
                bins_perp=bins_perp,
                cmap=cmap,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 6: Distribution temporal evolution
# ══════════════════════════════════════════════════════════════════════════════

def plot_distribution_evolution(
    filepaths: list[str],
    outdir: str,
    bins_parallel: int = 180,
    bins_perp: int = 120,
):
    """Plot the temporal evolution of 1D ion/electron distributions."""
    print("\nBuilding distribution evolution plot...")

    sampled_paths = sample_filepaths(filepaths)
    if len(sampled_paths) != len(filepaths):
        print(
            f"  Using {len(sampled_paths)} uniformly sampled snapshots "
            f"out of {len(filepaths)} for temporal evolution."
        )

    range_sample = load_particle_phase_space(sampled_paths[-1])
    ion_par_max = np.percentile(np.abs(range_sample["ions_pz"]), 99.5)
    ion_perp_max = np.percentile(range_sample["ions_perp"], 99.5)
    elec_par_max = np.percentile(np.abs(range_sample["electrons_pz"]), 99.5)
    elec_perp_max = np.percentile(range_sample["electrons_perp"], 99.5)

    ion_par_edges = np.linspace(-ion_par_max, ion_par_max, bins_parallel + 1)
    ion_perp_edges = np.linspace(0.0, ion_perp_max, bins_perp + 1)
    elec_par_edges = np.linspace(-elec_par_max, elec_par_max, bins_parallel + 1)
    elec_perp_edges = np.linspace(0.0, elec_perp_max, bins_perp + 1)

    ion_par_matrix = np.zeros((len(ion_par_edges) - 1, len(sampled_paths)))
    ion_perp_matrix = np.zeros((len(ion_perp_edges) - 1, len(sampled_paths)))
    elec_par_matrix = np.zeros((len(elec_par_edges) - 1, len(sampled_paths)))
    elec_perp_matrix = np.zeros((len(elec_perp_edges) - 1, len(sampled_paths)))

    steps = []
    for idx, filepath in enumerate(sampled_paths):
        phase_space = load_particle_phase_space(filepath)
        steps.append(extract_step(filepath))
        ion_par_matrix[:, idx], _ = np.histogram(
            phase_space["ions_pz"], bins=ion_par_edges, density=True
        )
        ion_perp_matrix[:, idx], _ = np.histogram(
            phase_space["ions_perp"], bins=ion_perp_edges, density=True
        )
        elec_par_matrix[:, idx], _ = np.histogram(
            phase_space["electrons_pz"], bins=elec_par_edges, density=True
        )
        elec_perp_matrix[:, idx], _ = np.histogram(
            phase_space["electrons_perp"], bins=elec_perp_edges, density=True
        )

    steps = np.asarray(steps, dtype=float)
    if len(steps) == 1:
        time_edges = np.array([steps[0] - 0.5, steps[0] + 0.5], dtype=float)
    else:
        deltas = np.diff(steps)
        left_edge = steps[0] - 0.5 * deltas[0]
        right_edge = steps[-1] + 0.5 * deltas[-1]
        interior = 0.5 * (steps[:-1] + steps[1:])
        time_edges = np.concatenate([[left_edge], interior, [right_edge]])

    panels = [
        (ion_par_matrix, ion_par_edges, r"Ions: $f(v_\parallel, t)$", "distribution_evolution_ions_parallel.png"),
        (ion_perp_matrix, ion_perp_edges, r"Ions: $f(v_\perp, t)$", "distribution_evolution_ions_perp.png"),
        (elec_par_matrix, elec_par_edges, r"Electrons: $f(v_\parallel, t)$", "distribution_evolution_electrons_parallel.png"),
        (elec_perp_matrix, elec_perp_edges, r"Electrons: $f(v_\perp, t)$", "distribution_evolution_electrons_perp.png"),
    ]

    for matrix, edges, title, filename in panels:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        _style_paper_axes(ax)
        positive = matrix[matrix > 0]
        vmin = positive.min() if positive.size else 1e-12
        vmax = matrix.max() if matrix.size else 1.0
        im = ax.pcolormesh(
            time_edges, edges, matrix,
            shading="auto",
            norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10.0)),
            cmap="viridis",
        )
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("Simulation step")
        ax.set_ylabel("Velocity")
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label("PDF")
        fig.tight_layout()
        _save_paper_figure(fig, output_file(outdir, filename))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 7: Anisotropy + magnetic fluctuation evolution
# ══════════════════════════════════════════════════════════════════════════════

def plot_macro_evolution(
    filepaths: list[str],
    outdir: str,
    field_files: dict[int, str] | None = None,
):
    """Track particle anisotropy and magnetic fluctuations vs. time."""
    print("\nBuilding anisotropy and magnetic-fluctuation evolution plot...")

    sampled_paths = sample_filepaths(filepaths)
    particle_rows = [summarize_particle_snapshot(p) for p in sampled_paths]

    steps = np.asarray([r["step"] for r in particle_rows], dtype=float)
    times = np.asarray([step_to_omegaci(int(step)) for step in steps])
    ion_anis = np.asarray([r["ion_anisotropy"] for r in particle_rows], dtype=float)
    elec_anis = np.asarray([r["electron_anisotropy"] for r in particle_rows], dtype=float)

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
    ax.plot(times, ion_anis, marker="o", linewidth=2.0, color="#c0392b", label="Ions")
    ax.plot(times, elec_anis, marker="s", linewidth=2.0, color="#2980b9", label="Electrons")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_ylabel(r"$T_\perp / T_\parallel$")
    ax.set_xlabel(r"$t\Omega_{ci}$")
    ax.set_title("Particle anisotropy")
    ax.legend()
    fig.tight_layout()
    _save_paper_figure(fig, output_file(outdir, "particle_anisotropy_vs_time.png"))

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    _style_paper_axes(ax)
    driven_anis = ion_anis if DRIVEN_SPECIES == "ion" else elec_anis
    ax.plot(
        times, 1.0 / np.maximum(driven_anis, 1e-30),
        marker="o", linewidth=2.0, color="#d4a017",
        label=rf"{DRIVEN_SPECIES}: $T_\parallel/T_\perp$",
    )
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_ylabel(r"$T_\parallel/T_\perp$")
    ax.set_xlabel(r"$t\Omega_{ci}$")
    ax.set_title("Inverse anisotropy of the driven species")
    ax.legend()
    fig.tight_layout()
    _save_paper_figure(fig, output_file(outdir, "driven_species_inverse_anisotropy_vs_time.png"))

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    _style_paper_axes(ax)
    if field_steps:
        field_steps = np.asarray(field_steps, dtype=float)
        ax.plot(field_steps, delta_b_total, marker="o", linewidth=2.0,
                color="#8e44ad", label=r"$\delta B_{\rm rms} / B_0$")
        ax.plot(field_steps, delta_b_parallel, marker="^", linewidth=1.8,
                color="#16a085", label=r"$\delta B_{\parallel,\rm rms} / B_0$")
        ax.plot(field_steps, delta_b_perp, marker="d", linewidth=1.8,
                color="#f39c12", label=r"$\delta B_{\perp,\rm rms} / B_0$")
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
#  Plot 8: Brazil plot — T_perp/T_par vs beta_par (instability thresholds)
# ══════════════════════════════════════════════════════════════════════════════

def plot_brazil(filepaths: list[str], outdir: str):
    """Canonical Brazil plot: T_perp/T_par vs beta_parallel for ions.

    Threshold curves (Gary 1993 / Hellinger 2006):
      Mirror:         A_i = 1 + 0.77 / beta_par
      Firehose:       A_i = 1 - 2   / beta_par
      Oblique FH:     A_i = 1 - 1.4 / (beta_par - 0.11)^0.55
    Each snapshot is one point; colour encodes simulation step.
    """
    print("\nBuilding Brazil plot (T_perp/T_par vs beta_par)...")

    sampled_paths = sample_filepaths(filepaths, max_files=MAX_EVOLUTION_FILES)
    steps_arr = np.array([extract_step(p) for p in sampled_paths], dtype=float)
    times_arr = np.array([step_to_omegaci(int(step)) for step in steps_arr])

    aniso_vals    = []
    beta_par_vals = []

    for path in sampled_paths:
        ps = load_particle_phase_space(path)
        # PSC stores normalized momentum u=gamma*v. In this non-relativistic
        # setup, T = m*Var(u); subtract drift before computing beta.
        prefix = "ions" if DRIVEN_SPECIES == "ion" else "electrons"
        mass = M_ION if DRIVEN_SPECIES == "ion" else M_ELEC
        tpar = mass * float(np.var(ps[f"{prefix}_pz"]))
        tperp = 0.5 * mass * float(
            np.var(ps[f"{prefix}_px"]) + np.var(ps[f"{prefix}_py"])
        )
        beta_par = 2.0 * tpar / (B0 ** 2)      # beta_i_par = 2 n T_par / B0^2 (n=1)
        aniso    = tperp / max(tpar, 1e-30)
        aniso_vals.append(aniso)
        beta_par_vals.append(beta_par)
        del ps
        gc.collect()

    aniso_vals    = np.asarray(aniso_vals,    dtype=float)
    beta_par_vals = np.asarray(beta_par_vals, dtype=float)

    # ── Threshold curves ──────────────────────────────────────────────────────
    beta_range = np.logspace(-2, 2, 500)
    mirror_thresh   = 1.0 + 0.77 / beta_range
    firehose_thresh = 1.0 - 2.0  / beta_range
    whistler_thresh = 1.0 + 0.21 / beta_range**0.6
    with np.errstate(invalid="ignore", divide="ignore"):
        valid_bf = beta_range > 0.11
        oblique_fh = np.where(valid_bf,
                               1.0 - 1.4 / (beta_range - 0.11) ** 0.55,
                               np.nan)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#111827")

    # Instability regions (shaded)
    if INSTABILITY == "whistler":
        ax.fill_between(beta_range, whistler_thresh, 10.0, alpha=0.12,
                        color="#9b59b6", label="_nolegend_")
        ax.plot(beta_range, whistler_thresh, "--", color="#9b59b6", linewidth=2.0,
                label=r"Whistler  $A_e = 1 + 0.21/\beta_{e\parallel}^{0.6}$")
    else:
        ax.fill_between(beta_range, mirror_thresh, 10.0, alpha=0.12,
                        color="#e74c3c", label="_nolegend_")
        ax.fill_between(beta_range, -10.0, np.clip(firehose_thresh, -10, 10),
                        alpha=0.12, color="#3498db", label="_nolegend_")
        ax.plot(beta_range, mirror_thresh, "--", color="#e74c3c", linewidth=2.0,
                label=r"Mirror  $A_i = 1 + 0.77/\beta_{\parallel}$")
        ax.plot(beta_range, firehose_thresh, "--", color="#3498db", linewidth=2.0,
                label=r"Firehose  $A_i = 1 - 2/\beta_{\parallel}$")
        ax.plot(beta_range[valid_bf], oblique_fh[valid_bf], ":",
                color="#9b59b6", linewidth=1.8,
                label=r"Oblique firehose (Hellinger 2006)")

    # Isotropic line
    ax.axhline(1.0, color="#aaaaaa", linestyle=":", linewidth=1.0, alpha=0.6)

    # Data scatter
    sc = ax.scatter(beta_par_vals, aniso_vals, c=times_arr,
                    cmap="plasma", s=90, zorder=5,
                    edgecolors="white", linewidths=0.6,
                    norm=plt.Normalize(times_arr.min(), times_arr.max()))
    ax.plot(beta_par_vals, aniso_vals, "-", color="white",
            linewidth=0.8, alpha=0.4, zorder=4)
    ax.scatter(beta_par_vals[0],  aniso_vals[0],  marker="D", s=130,
               color="#2ecc71", zorder=6, label=f"t=0  (step {int(steps_arr[0])})")
    ax.scatter(beta_par_vals[-1], aniso_vals[-1], marker="*", s=220,
               color="#f1c40f", zorder=6, label=f"Final (step {int(steps_arr[-1])})")

    # Region labels
    if INSTABILITY == "whistler":
        ax.text(0.03, 0.94, "WHISTLER\nUNSTABLE", transform=ax.transAxes,
                color="#9b59b6", fontsize=12, alpha=0.9, va="top")
    else:
        ax.text(0.03, 0.94, "MIRROR\nUNSTABLE", transform=ax.transAxes,
                color="#e74c3c", fontsize=12, alpha=0.9, va="top")
        ax.text(0.03, 0.08, "FIREHOSE\nUNSTABLE", transform=ax.transAxes,
                color="#3498db", fontsize=12, alpha=0.9, va="bottom")
    ax.text(0.72, 0.72, "STABLE", transform=ax.transAxes,
            color="#aaaaaa", fontsize=12, alpha=0.6, va="bottom")

    # Colourbar for time
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$t\Omega_{ci}$", color="white", fontsize=14)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color="white")

    ax.set_xscale("log")
    xmin = max(0.01, float(np.nanmin(beta_par_vals)) / 1.5)
    xmax = max(xmin * 2.0, float(np.nanmax(beta_par_vals)) * 1.5)
    ymin = max(0.03, float(np.nanmin(aniso_vals)) / 1.5)
    ymax = max(1.25, float(np.nanmax(aniso_vals)) * 1.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    species_symbol = "i" if DRIVEN_SPECIES == "ion" else "e"
    ax.set_xlabel(rf"$\beta_{{\parallel,{species_symbol}}}"
                  rf" = 2\,n\,T_{{\parallel,{species_symbol}}}\,/\,B_0^2$",
                  fontsize=16, color="white")
    ax.set_ylabel(rf"$A_{species_symbol} = T_{{\perp,{species_symbol}}}"
                  rf"\,/\,T_{{\parallel,{species_symbol}}}$",
                  fontsize=16, color="white")
    ax.set_title(
        rf"Brazil Plot — {PROFILE_LABEL}",
        fontsize=17, fontweight="bold", color="white", pad=12,
    )
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, alpha=0.15, color="white", linestyle="--", which="both")
    ax.legend(fontsize=12, loc="upper right",
              facecolor="#1a1a2e", edgecolor="#444",
              labelcolor="white", framealpha=0.85)

    plt.tight_layout()
    path = output_file(outdir, "brazil_plot.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved: {path}")

    csv_path = output_file(outdir, "particle_anisotropy_evolution.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case", "species", "step", "omega_ci_t", "anisotropy",
            "parallel_over_perpendicular", "beta_parallel",
        ])
        for step, time, aniso, beta in zip(
            steps_arr.astype(int), times_arr, aniso_vals, beta_par_vals
        ):
            writer.writerow([
                PROFILE_LABEL, DRIVEN_SPECIES, step, time, aniso,
                1.0 / max(aniso, 1e-30), beta,
            ])
    print(f"  Saved: {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 9: 1-D VDF evolution with suprathermal-tail quantification
# ══════════════════════════════════════════════════════════════════════════════

def plot_1d_vdf_evolution(
    filepaths: list[str],
    outdir: str,
    n_times: int = 6,
    nbins: int = 200,
):
    """Overlay 1-D f(v_par) and f(v_perp) at selected times.

    Suprathermal tail excess is quantified as the ratio of the measured
    high-velocity tail (|v| > 3 v_th) population to the Maxwellian prediction.
    Colour encodes time (early=blue, late=red), reference Maxwellian dashed.
    """
    print("\nBuilding 1-D VDF temporal evolution with tail diagnostics...")

    paths = sample_filepaths(filepaths, max_files=n_times)
    steps = [extract_step(p) for p in paths]
    cmap  = plt.colormaps["coolwarm"]
    colors = [cmap(i / max(len(paths) - 1, 1)) for i in range(len(paths))]

    # Determine shared axes from last snapshot
    ref = load_particle_phase_space(paths[-1])
    vpar_max  = float(np.percentile(np.abs(ref["ions_pz"]),  99.5))
    vperp_max = float(np.percentile(ref["ions_perp"], 99.5))
    del ref; gc.collect()

    vpar_edges  = np.linspace(-vpar_max,  vpar_max,  nbins + 1)
    vperp_edges = np.linspace(0.0,        vperp_max, nbins + 1)
    vc_par  = 0.5 * (vpar_edges[:-1]  + vpar_edges[1:])
    vc_perp = 0.5 * (vperp_edges[:-1] + vperp_edges[1:])

    fig_par, ax_par = plt.subplots(figsize=(7.4, 5.2))
    fig_perp, ax_perp = plt.subplots(figsize=(7.4, 5.2))
    axes = [ax_par, ax_perp]
    for ax in axes:
        _style_paper_axes(ax)

    tail_excess_par  = []
    tail_excess_perp = []
    vth_par_est  = None
    vth_perp_est = None

    for idx, (path, step, col) in enumerate(zip(paths, steps, colors)):
        ps = load_particle_phase_space(path)
        pz   = ps["ions_pz"]
        perp = ps["ions_perp"]

        f_par,  _ = np.histogram(pz,   bins=vpar_edges,  density=True)
        f_perp, _ = np.histogram(perp, bins=vperp_edges, density=True)

        axes[0].semilogy(vc_par,  f_par,  color=col, linewidth=1.6,
                         alpha=0.85, label=f"step {step}")
        axes[1].semilogy(vc_perp, f_perp, color=col, linewidth=1.6,
                         alpha=0.85, label=f"step {step}")

        # Tail excess at t=0 baseline
        if idx == 0:
            vth_par_est  = float(np.std(pz))
            vth_perp_est = float(np.std(perp))

        # Fraction of ions with |v| > 3 vth
        if vth_par_est and vth_par_est > 0:
            tail_par  = float(np.mean(np.abs(pz)   > 3 * vth_par_est))
            tail_perp = float(np.mean(perp          > 3 * vth_perp_est))
            tail_excess_par.append(tail_par)
            tail_excess_perp.append(tail_perp)

        del ps; gc.collect()

    # Reference Maxwellian at t=0 temperatures
    if vth_par_est:
        mw_par  = maxwellian_1d(vc_par,  vth_par_est)
        mw_perp = maxwellian_1d(vc_perp, vth_perp_est)
        axes[0].semilogy(vc_par,  mw_par,  "k--", linewidth=1.8,
                         alpha=0.6, label=r"Maxwellian ($t=0$ $v_{th}$)")
        axes[1].semilogy(vc_perp, mw_perp, "k--", linewidth=1.8,
                         alpha=0.6, label=r"Maxwellian ($t=0$ $v_{th}$)")
        # Shade 3 vth tail region
        for ax, vth, vmax in [(axes[0], vth_par_est, vpar_max),
                               (axes[1], vth_perp_est, vperp_max)]:
            ax.axvspan(3 * vth, vmax, alpha=0.08, color="#e74c3c",
                       label=r"Suprathermal tail ($|v|>3v_{th}$)")
            if ax is axes[0]:
                ax.axvspan(-vmax, -3 * vth, alpha=0.08, color="#e74c3c")

    for ax, xlabel, title in [
        (axes[0], r"$v_\parallel\ [\mathrm{code\ units}]$",
         r"$f(v_\parallel)$ — Parallel VDF"),
        (axes[1], r"$v_\perp\ [\mathrm{code\ units}]$",
         r"$f(v_\perp)$ — Perpendicular VDF"),
    ]:
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(r"$f(v)$  [PDF]", fontsize=15)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.legend(fontsize=11, loc="upper right")
        ax.grid(True, alpha=0.3, which="both", ls="--")
        ax.tick_params(which="both", direction="in", top=True, right=True)

    fig_par.tight_layout()
    fig_perp.tight_layout()
    _save_paper_figure(fig_par, output_file(outdir, "vdf_1d_parallel_evolution.png"))
    _save_paper_figure(fig_perp, output_file(outdir, "vdf_1d_perp_evolution.png"))

    # Print tail-excess summary
    if tail_excess_par:
        print("  Suprathermal tail fraction (|v| > 3 v_th):")
        for s, tp, tperp in zip(steps, tail_excess_par, tail_excess_perp):
            print(f"    step {s:>8d}  par={tp:.4f}  perp={tperp:.4f}")


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
    times  = steps  # steps already in code units; convert if needed

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
                # E_B ~ (delta_B)^2 / 2  (normalised)
                eb = 0.5 * m["delta_b_total"]**2
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
    ax.plot(times, ion_thermal / E0,     "s-", color="#e67e22", linewidth=2,
            label=r"Ion thermal energy  ($\frac{1}{2}m_i\langle\delta u^2\rangle$)")
    ax.plot(times, elec_thermal / E0,    "^-", color="#3498db", linewidth=2,
            label=r"Electron thermal energy")
    ax.plot(times, ion_kin_bulk / E0,    "d-", color="#2ecc71", linewidth=1.5,
            label=r"Ion bulk kinetic")
    ax.plot(times, elec_kin_bulk / E0,   "x-", color="#16a085", linewidth=1.5,
            label=r"Electron bulk kinetic")
    ax.set_ylabel(r"Energy  [$E_0$]", fontsize=15)
    ax.set_xlabel("Simulation step", fontsize=14)
    ax.set_title("Particle energy components (normalised to $E_0$)",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    fig.tight_layout()
    _save_paper_figure(fig, output_file(outdir, "particle_energy_partition.png"))

    fig, ax2 = plt.subplots(figsize=(7.8, 5.4))
    _style_paper_axes(ax2)
    if not np.all(np.isnan(mag_energy)):
        ax2.plot(times, mag_energy, "o-", color="#9b59b6", linewidth=2,
                 label=r"$E_B = (\delta B_{\rm rms})^2 / 2$  (from field files)")
        ax2.set_ylabel(r"Magnetic fluctuation energy  [$B_0^2/2$]", fontsize=15)
        ax2.legend(fontsize=12)
    else:
        ax2.text(0.5, 0.5,
                 "Magnetic energy requires matching pfd.*.h5 field files.",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=14, color="gray")
    ax2.set_xlabel("Simulation step", fontsize=14)
    ax2.set_title("Magnetic energy from field fluctuations",
                  fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save_paper_figure(fig, output_file(outdir, "magnetic_energy_fluctuation.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 11: Heat flux diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def _compute_heat_flux(filepath: str,
                       n_regions: int = 4) -> dict:
    """Experimental velocity-quantile heat-flux diagnostic.

    The heat flux vector is defined as:
        q_i = (m/2) * <(v - <v>)^2 * (v_i - <v_i>)>

    Parallel component:   q_par  = (m/2) * <delta_v^2 * delta_vz>
    Perpendicular:        q_perp = (m/2) * <delta_v^2 * delta_vperp>

    Particles are grouped by u_z quantile. These groups are velocity
    populations, not spatial regions, so this diagnostic is not part of the
    standard analysis pipeline.
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

    ion_mask = q > 0
    mi = float(np.abs(m[ion_mask][0])) if ion_mask.any() else M_ION
    pz_i  = pz[ion_mask]
    px_i  = px[ion_mask]
    py_i  = py[ion_mask]

    # Proxy spatial bins: divide by pz quantile (streaming direction)
    pz_bins = np.percentile(pz_i, np.linspace(0, 100, n_regions + 1))

    q_par_regions  = []
    q_perp_regions = []
    region_centers = []

    for k in range(n_regions):
        mask = (pz_i >= pz_bins[k]) & (pz_i < pz_bins[k + 1])
        if mask.sum() < 10:
            q_par_regions.append(np.nan)
            q_perp_regions.append(np.nan)
            region_centers.append(0.5 * (pz_bins[k] + pz_bins[k + 1]))
            continue

        vz   = pz_i[mask]
        vx   = px_i[mask]
        vy   = py_i[mask]
        dvz  = vz - np.mean(vz)
        dvx  = vx - np.mean(vx)
        dvy  = vy - np.mean(vy)
        dv2  = dvx**2 + dvy**2 + dvz**2
        dvperp = np.sqrt(dvx**2 + dvy**2)

        q_par  = 0.5 * mi * np.mean(dv2 * dvz)
        q_perp = 0.5 * mi * np.mean(dv2 * dvperp)
        q_par_regions.append(q_par)
        q_perp_regions.append(q_perp)
        region_centers.append(0.5 * (pz_bins[k] + pz_bins[k + 1]))

    return {
        "region_centers": np.array(region_centers),
        "q_par":  np.array(q_par_regions),
        "q_perp": np.array(q_perp_regions),
    }


def plot_heat_flux(
    filepaths: list[str],
    outdir: str,
    n_times: int = 8,
    n_regions: int = 4,
):
    """Plot experimental heat flux by parallel-velocity population.

    Characterises non-thermal energy transport associated with instability
    dynamics for both Maxwellian and kappa initial distributions.
    """
    print("\nBuilding heat flux diagnostics...")

    paths  = sample_filepaths(filepaths, max_files=n_times)
    steps  = np.array([extract_step(p) for p in paths], dtype=float)
    cmap   = plt.colormaps["plasma"]
    colors = [cmap(i / max(len(paths) - 1, 1)) for i in range(len(paths))]

    q_par_all  = []   # shape (n_times, n_regions)
    q_perp_all = []

    for path in paths:
        hf = _compute_heat_flux(path, n_regions=n_regions)
        q_par_all.append(hf["q_par"])
        q_perp_all.append(hf["q_perp"])
        gc.collect()

    q_par_all  = np.array(q_par_all,  dtype=float)   # (n_times, n_regions)
    q_perp_all = np.array(q_perp_all, dtype=float)

    region_labels = [f"R{k+1}" for k in range(n_regions)]
    x = np.arange(n_regions)
    width = 0.8 / len(paths)

    for data, ylabel, title, filename in [
        (q_par_all,
         r"$q_\parallel = \frac{m}{2}\langle\delta v^2\,\delta v_z\rangle$",
         r"Parallel heat flux $q_\parallel$",
         "heat_flux_regions_parallel.png"),
        (q_perp_all,
         r"$q_\perp = \frac{m}{2}\langle\delta v^2\,\delta v_\perp\rangle$",
         r"Perpendicular heat flux $q_\perp$",
         "heat_flux_regions_perp.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.8, 5.4))
        _style_paper_axes(ax)
        for i, (row, step, col) in enumerate(zip(data, steps, colors)):
            offset = (i - len(paths) / 2) * width
            ax.bar(x + offset, row, width=width * 0.9,
                   color=col, alpha=0.85, label=f"step {int(step)}")

        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels(region_labels)
        ax.set_xlabel(r"Population (binned by $u_z$ quantile)", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.legend(fontsize=11, ncol=2)
        fig.tight_layout()
        _save_paper_figure(fig, output_file(outdir, filename))

    # Time-series panel at bottom
    fig2, ax3 = plt.subplots(figsize=(12, 5))
    fig2.patch.set_facecolor("white")
    for k in range(n_regions):
        ax3.plot(steps, q_par_all[:, k],  "o-", linewidth=1.8,
                 label=rf"$q_\parallel$ R{k+1}")
        ax3.plot(steps, q_perp_all[:, k], "s--", linewidth=1.4, alpha=0.7,
                 label=rf"$q_\perp$ R{k+1}")
    ax3.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax3.set_xlabel("Simulation step", fontsize=15)
    ax3.set_ylabel("Heat flux  [code units]", fontsize=15)
    ax3.set_title(
        r"Temporal Evolution of $q_\parallel$ and $q_\perp$ per Region",
        fontsize=16, fontweight="bold",
    )
    ax3.legend(fontsize=11, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(which="both", direction="in", top=True, right=True)
    fig2.tight_layout()

    path2 = output_file(outdir, "heat_flux_timeseries.png")
    fig2.savefig(path2, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  Saved: {path2}")


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
        plot_vdf_snapshots(filepaths, outdir);                          gc.collect()
        plot_distribution_evolution(filepaths, outdir);                  gc.collect()
        plot_macro_evolution(filepaths, outdir, field_files=field_files); gc.collect()
        plot_brazil(filepaths, outdir);                                   gc.collect()
        # ── New diagnostics ────────────────────────────────────────────
        print("\nRunning distribution-tail and energy diagnostics...")
        plot_1d_vdf_evolution(filepaths, outdir);                         gc.collect()
        plot_energy_partition(filepaths, outdir, field_files=field_files); gc.collect()

    data = load_particles(filepath)
    ions, electrons = separate_species(data)
    del data; gc.collect()

    print_summary(ions, electrons, extract_step(filepath))

    print("\nGenerating plots...")
    plot_vdf(ions, electrons, outdir);      gc.collect()
    if KAPPA is not None:
        plot_kappa_comparison(ions, outdir); gc.collect()
        print("\nRunning goodness-of-fit tests (Anderson-Darling & Kolmogorov-Smirnov)...")
        plot_goodness_of_fit(ions, outdir);  gc.collect()
    else:
        print("  Skipping Kappa-only comparison and goodness-of-fit plots for a Bi-Maxwellian profile.")

    print(f"\nAll plots saved to: {outdir}")


if __name__ == "__main__":
    main()
