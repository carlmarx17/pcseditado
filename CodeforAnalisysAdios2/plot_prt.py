#!/usr/bin/env python3
"""
plot_prt.py — Particle diagnostic suite for PSC PIC simulations.

Reads HDF5 particle files and generates publication-quality diagnostics:
  1. 2D VDF heatmap: f(v_perp, v_parallel)
  2. Kappa vs. Maxwellian distribution comparison (3 views × 3 components)
  3. Goodness-of-fit tests (Anderson-Darling & Kolmogorov-Smirnov)
  4. Multi-time VDF snapshot panel
  5. 1D distribution temporal evolution
  6. Anisotropy + magnetic-fluctuation time series

Usage:
    python plot_prt.py [path_to_prt_file | directory | glob_pattern]

Defaults to  ../build/src/prt.000000000.bp  if no argument is given.
"""

import sys
import os
import glob
import re
import adios2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter
from scipy.special import gamma as gamma_func
from scipy import stats as scipy_stats

try:
    from data_reader import PICDataReader
except ImportError:
    PICDataReader = None
from psc_units import B0, KAPPA, MASS_RATIO

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = "prt_plots"
DPI = 200
MAX_EVOLUTION_FILES = 12
MAX_VDF_SNAPSHOTS = 5

STEP_RE = re.compile(r"\.(\d+)(?:_p\d+)?\.bp$")

# ── Simulation parameters (from psc_temp_aniso.cxx) ─────────────────────────
Zi: float = 1.0
VA_OVER_C: float = B0
BETA_I_PAR: float = 10.0
TI_PERP_OVER_TI_PAR: float = 3.5
BETA_NORM: float = 1.0

TI_PAR: float = BETA_I_PAR * B0**2 / 2.0             # 0.05
TI_PERP: float = TI_PERP_OVER_TI_PAR * TI_PAR        # 0.175
M_ION: float = MASS_RATIO * Zi                        # 64.0


# ══════════════════════════════════════════════════════════════════════════════
#  I/O Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_particles(filepath: str, verbose: bool = True) -> np.ndarray:
    """Load particle data from a PSC prt.*.bp file/dir."""
    if verbose:
        print(f"Loading particles from: {filepath}")
    
    f = adios2.FileReader(filepath)
    try:
        vars = f.available_variables()
        
        # Determine variable names in ADIOS2 (they might be flat or prefixed)
        def get_var(name):
            for k in vars.keys():
                if k.endswith(name):
                    variable = f.inquire_variable(k)
                    if variable is None:
                        break
                    return f.read(variable)
            raise KeyError(f"Variable ending in '{name}' not found in {filepath}")

        q = get_var("q")
        m = get_var("m")
        x = get_var("x")
        y = get_var("y")
        z = get_var("z")
        px = get_var("px")
        py = get_var("py")
        pz = get_var("pz")
    finally:
        f.close()

    # Reconstruct a structured array to maintain compatibility with the rest of the script
    dt = np.dtype([('q', 'f8'), ('m', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('px', 'f8'), ('py', 'f8'), ('pz', 'f8')])
    data = np.zeros(len(q), dtype=dt)
    data['q'] = q
    data['m'] = m
    data['x'] = x
    data['y'] = y
    data['z'] = z
    data['px'] = px
    data['py'] = py
    data['pz'] = pz

    if verbose:
        print(f"  Total particles: {len(data):,}")
    return data


def load_particle_phase_space(filepath: str) -> dict:
    """Load only the momentum fields needed for distribution analysis."""
    f = adios2.FileReader(filepath)
    try:
        vars = f.available_variables()
        
        def get_var(name):
            for k in vars.keys():
                if k.endswith(name):
                    variable = f.inquire_variable(k)
                    if variable is None:
                        break
                    return f.read(variable)
            raise KeyError(f"Variable ending in '{name}' not found in {filepath}")
            
        q = get_var("q")
        px = get_var("px")
        py = get_var("py")
        pz = get_var("pz")
    finally:
        f.close()

    ion_mask = q > 0
    elec_mask = q < 0
    return {
        "ions_pz": pz[ion_mask],
        "ions_perp": np.sqrt(px[ion_mask] ** 2 + py[ion_mask] ** 2),
        "electrons_pz": pz[elec_mask],
        "electrons_perp": np.sqrt(px[elec_mask] ** 2 + py[elec_mask] ** 2),
    }


def load_field_fluctuation_metrics(filepath: str, b0: float = B0) -> dict:
    """Return RMS magnetic fluctuation metrics normalised to B0."""
    if PICDataReader is None:
        raise RuntimeError("data_reader.py is required to read magnetic field outputs.")

    fields = PICDataReader.read_multiple_fields_3d(
        filepath,
        "jeh-",
        ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"],
    )

    bx = np.asarray(fields["hx_fc/p0/3d"], dtype=float).ravel() * b0
    by = np.asarray(fields["hy_fc/p0/3d"], dtype=float).ravel() * b0
    bz = np.asarray(fields["hz_fc/p0/3d"], dtype=float).ravel() * b0

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
        candidates = sorted(glob.glob(os.path.join(input_path, "prt.*.bp")))
    elif any(ch in input_path for ch in "*?[]"):
        candidates = sorted(glob.glob(input_path))
    else:
        candidates = [input_path]

    files = [p for p in candidates if os.path.isfile(p)]
    if not files:
        raise FileNotFoundError(f"No particle files matched: {input_path}")

    return sorted(files, key=extract_step)


def resolve_field_files(
    reference_dir: str, pattern: str = "pfd.*.bp"
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
    """Compute T_perp and T_par from velocity variance (removing bulk drift)."""
    mass = abs(float(species["m"][0]))
    px = np.asarray(species["px"], dtype=float)
    py = np.asarray(species["py"], dtype=float)
    pz = np.asarray(species["pz"], dtype=float)
    t_perp = 0.5 * (np.var(px) + np.var(py)) / mass
    t_par = np.var(pz) / mass
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

    - Axis range set to 6 × thermal speed to capture the Kappa power-law tail.
    - Independent LogNorm per species.
    - Filled contour plot with viridis colourmap.
    """
    VA_ = VA_OVER_C
    M_ELE = 1.0
    B0_ = VA_OVER_C
    KAPPA_ = KAPPA

    TI_PAR_ = BETA_I_PAR * B0_**2 / 2.0
    TI_PERP_ = TI_PERP_OVER_TI_PAR * TI_PAR_
    TE_PAR_ = 1.0 * B0_**2 / 2.0
    TE_PERP_ = TE_PAR_

    # Thermal speeds in v_A units
    vth_i_par = np.sqrt(TI_PAR_ / M_ION) / VA_
    vth_i_perp = np.sqrt(TI_PERP_ / M_ION) / VA_
    vth_e_par = np.sqrt(TE_PAR_ / M_ELE) / VA_
    vth_e_perp = np.sqrt(TE_PERP_ / M_ELE) / VA_

    NBINS = 400

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        r"Ion and Electron VDF: $f(v_\perp,\, v_\parallel)$  (t = 0)",
        fontsize=16, fontweight="bold", color="black", y=1.01,
    )

    species_cfg = [
        ("Ions",      ions,      M_ION, TI_PERP_, TI_PAR_, vth_i_perp, vth_i_par, "plasma"),
        ("Electrons", electrons, M_ELE, TE_PERP_, TE_PAR_, vth_e_perp, vth_e_par, "viridis"),
    ]

    for ax, (label, sp, mass, T_perp, T_par, vth_p, vth_z, cmap_name) in zip(
        axes, species_cfg
    ):
        px_, py_, pz_ = sp["px"], sp["py"], sp["pz"]
        v_perp = np.sqrt(px_**2 + py_**2) / VA_
        v_par = pz_ / VA_

        perp_max = 6.0 * vth_p
        par_abs = 6.0 * vth_z

        h, xedges, yedges = np.histogram2d(
            v_par, v_perp,
            bins=[NBINS, int(NBINS * 0.6)],
            range=[[-par_abs, par_abs], [0, perp_max]],
            density=True,
        )

        # Mirror negative v_perp for visual symmetry
        h_smooth = gaussian_filter(h, sigma=1.5)
        h_full = np.concatenate([h_smooth[:, ::-1], h_smooth], axis=1)
        yedges_full = np.concatenate([-yedges[::-1], yedges[1:]])

        f_max = h_full.max()
        with np.errstate(divide="ignore"):
            log_f = np.log10(h_full / f_max)
        log_f = np.clip(log_f, -4.5, 0.0)

        cmap = plt.colormaps["viridis"].copy()
        x_c = 0.5 * (xedges[:-1] + xedges[1:])
        y_c = 0.5 * (yedges_full[:-1] + yedges_full[1:])

        ax.set_facecolor("white")

        levels = np.linspace(-4.5, 0.0, 50)
        im = ax.contourf(x_c, y_c, log_f.T, levels=levels, cmap=cmap, extend="min")

        line_levels = [-4.0, -3.0, -2.0, -1.0]
        contours = ax.contour(
            x_c, y_c, log_f.T, levels=line_levels,
            colors="black", linestyles="dashed", linewidths=0.8, alpha=0.7,
        )
        ax.clabel(contours, inline=True, fontsize=10, fmt="%1.1f")

        cbar = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.03, ticks=[0, -1, -2, -3, -4])
        cbar.set_label(r"$\log_{10}(f / f_{max})$", fontsize=12)
        cbar.ax.yaxis.set_tick_params(labelsize=10)

        ax.set_xlabel(r"$v_\parallel\ [v_A]$", fontsize=13)
        ax.set_ylabel(r"$v_\perp\ [v_A]$", fontsize=13)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.set_title(
            rf"{label}: $\log_{{10}}(f / f_{{max}})$",
            fontsize=14, fontweight="bold", pad=10,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(outdir, "vdf_2d.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


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
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        r"Distribution Comparison: Data vs Kappa ($\kappa$=3) vs Maxwellian — Ions",
        fontsize=18, fontweight="bold",
    )

    mom_fields = ["px", "py", "pz"]
    comp_labels = [
        r"$p_x\ /\ (m_i v_A)$  [perp.]",
        r"$p_y\ /\ (m_i v_A)$  [perp.]",
        r"$p_z\ /\ (m_i v_A)$  [par.]",
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
        ax0.set_xlabel(xlabel, fontsize=12)
        ax0.set_ylabel(r"$f(p)$  [PDF]", fontsize=12)
        ax0.set_title(rf"Semilog: $f$ vs {short}  ({direction})",
                      fontsize=12, fontweight="bold")
        ax0.legend(fontsize=9)
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
        ax1.set_xlabel(rf"{short}$^2$  $[(m_i v_A)^2]$", fontsize=12)
        ax1.set_ylabel(r"$f(p)$  [PDF]", fontsize=12)
        ax1.set_title(rf"Semilog: $f$ vs {short}$^2$  — Maxwellian linearisation",
                      fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9)
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

        ax2.set_xlabel(rf"$|${short}$|$  $[m_i v_A]$  (log scale)", fontsize=12)
        ax2.set_ylabel(r"$f(p)$  [PDF]  (log scale)", fontsize=12)
        ax2.set_title(rf"Log–Log: power-law tail of {short}  ($p > 0$)",
                      fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, which="both", ls="--")
        ax2.set_ylim(bottom=y_min)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, "kappa_vs_maxwellian.png")
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

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        rf"Goodness-of-Fit Tests — Kappa ($\kappa = {kappa}$) vs Maxwellian — Ions",
        fontsize=15, fontweight="bold", y=1.01,
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

        ax_cdf.set_xlabel(label, fontsize=11)
        ax_cdf.set_ylabel("Cumulative probability  $F(p)$", fontsize=11)
        ax_cdf.set_title(f"Empirical vs theoretical CDF\n{label}",
                         fontsize=11, fontweight="bold")
        ax_cdf.legend(fontsize=8.5, loc="upper left")
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
                    fontsize=8.5, color=tcol, fontweight=weight,
                    transform=ax_tbl.transAxes,
                )
                x_cur += cw

        ax_tbl.set_xlim(0, 1)
        ax_tbl.set_ylim(0, 1)
        ax_tbl.set_title(f"Test results — {label}",
                         fontsize=10, fontweight="bold", pad=8)

        print(
            f"  [{label}]  "
            f"KS-Kappa: D={ks_stat_kappa:.4f} p={_fmt_p(ks_p_kappa)}  "
            f"KS-Maxw: D={ks_stat_maxw:.4f} p={_fmt_p(ks_p_maxw)}  "
            f"AD-Kappa: A²={ad_kappa:.3f} p={_fmt_p(ad_p_kappa)}  "
            f"AD-Maxw: A²={ad_maxw:.3f} p={_fmt_p(ad_p_maxw)}"
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(outdir, "goodness_of_fit.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 4: Summary statistics
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(ions, electrons):
    """Print summary statistics of the particle data."""
    print("\n" + "=" * 60)
    print("PARTICLE DATA SUMMARY (t=0)")
    print("=" * 60)

    for name, sp in [("IONS", ions), ("ELECTRONS", electrons)]:
        m = np.abs(sp["m"][0])
        T_perp = 0.5 * (np.mean(sp["px"] ** 2) + np.mean(sp["py"] ** 2)) / m
        T_par = np.mean(sp["pz"] ** 2) / m
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
    print("\nBuilding multi-time VDF snapshot panel...")

    sampled_paths = sample_filepaths(filepaths, max_files=max_snapshots)
    sampled_data = [
        (extract_step(path), load_particle_phase_space(path))
        for path in sampled_paths
    ]

    ion_par_max = max(np.percentile(np.abs(d["ions_pz"]), 99.5) for _, d in sampled_data)
    ion_perp_max = max(np.percentile(d["ions_perp"], 99.5) for _, d in sampled_data)
    elec_par_max = max(np.percentile(np.abs(d["electrons_pz"]), 99.5) for _, d in sampled_data)
    elec_perp_max = max(np.percentile(d["electrons_perp"], 99.5) for _, d in sampled_data)

    species_config = [
        ("ions_pz", "ions_perp", ion_par_max, ion_perp_max, "Ions"),
        ("electrons_pz", "electrons_perp", elec_par_max, elec_perp_max, "Electrons"),
    ]

    fig, axes = plt.subplots(
        2, len(sampled_data),
        figsize=(4.4 * len(sampled_data), 8.0),
        constrained_layout=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle("Velocity Distribution Function at Different Times",
                 fontsize=17, fontweight="bold")

    for row, (par_key, perp_key, par_max, perp_max, label) in enumerate(species_config):
        for col, (step, data) in enumerate(sampled_data):
            ax = axes[row, col]
            hist, xedges, yedges = np.histogram2d(
                data[par_key], data[perp_key],
                bins=[bins_parallel, bins_perp],
                range=[[-par_max, par_max], [0.0, perp_max]],
                density=True,
            )
            hist = gaussian_filter(hist, sigma=1.0)

            positive = hist[hist > 0]
            vmax = positive.max() if positive.size else 1.0
            log_hist = np.full_like(hist, -5.0)
            if positive.size:
                with np.errstate(divide="ignore"):
                    log_hist = np.log10(np.maximum(hist / vmax, 1e-5))

            im = ax.pcolormesh(
                xedges, yedges, log_hist.T,
                shading="auto", cmap="viridis", vmin=-5.0, vmax=0.0,
            )
            ax.set_title(f"{label} | step {step}", fontsize=12, fontweight="bold")
            ax.set_xlabel(r"$v_\parallel$")
            ax.set_ylabel(r"$v_\perp$")
            ax.tick_params(which="both", direction="in", top=True, right=True)

            if col == len(sampled_data) - 1:
                cbar = fig.colorbar(im, ax=ax, pad=0.01)
                cbar.set_label(r"$\log_{10}(f / f_{max})$")

    path = os.path.join(outdir, "vdf_time_snapshots.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


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
        (ion_par_matrix, ion_par_edges, r"Ions: $f(v_\parallel, t)$"),
        (ion_perp_matrix, ion_perp_edges, r"Ions: $f(v_\perp, t)$"),
        (elec_par_matrix, elec_par_edges, r"Electrons: $f(v_\parallel, t)$"),
        (elec_perp_matrix, elec_perp_edges, r"Electrons: $f(v_\perp, t)$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Temporal Evolution of Velocity Distributions",
        fontsize=17, fontweight="bold",
    )

    for ax, (matrix, edges, title) in zip(axes.flat, panels):
        positive = matrix[matrix > 0]
        vmin = positive.min() if positive.size else 1e-12
        vmax = matrix.max() if matrix.size else 1.0
        im = ax.pcolormesh(
            time_edges, edges, matrix,
            shading="auto",
            norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10.0)),
            cmap="viridis",
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Simulation step")
        ax.set_ylabel("Velocity")
        ax.tick_params(which="both", direction="in", top=True, right=True)
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label("PDF")

    path = os.path.join(outdir, "distribution_evolution.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


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
    ion_anis = np.asarray([r["ion_anisotropy"] for r in particle_rows], dtype=float)
    elec_anis = np.asarray([r["electron_anisotropy"] for r in particle_rows], dtype=float)

    field_steps = []
    delta_b_total = []
    delta_b_parallel = []
    delta_b_perp = []

    if field_files:
        for step in steps.astype(int):
            filepath = field_files.get(step)
            if filepath is None:
                continue
            metrics = load_field_fluctuation_metrics(filepath)
            field_steps.append(step)
            delta_b_total.append(metrics["delta_b_total"])
            delta_b_parallel.append(metrics["delta_b_parallel"])
            delta_b_perp.append(metrics["delta_b_perp"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Temporal Evolution of Anisotropy and Magnetic Fluctuations",
        fontsize=17, fontweight="bold",
    )

    ax = axes[0]
    ax.plot(steps, ion_anis, marker="o", linewidth=2.0, color="#c0392b", label="Ions")
    ax.plot(steps, elec_anis, marker="s", linewidth=2.0, color="#2980b9", label="Electrons")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_ylabel(r"$T_\perp / T_\parallel$")
    ax.set_xlabel("Simulation step")
    ax.set_title("Particle anisotropy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
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
            "No matching pfd.*.bp field files found for these steps.",
            ha="center", va="center", transform=ax.transAxes,
        )
    ax.set_ylabel("Normalised fluctuation amplitude")
    ax.set_xlabel("Simulation step")
    ax.set_title("Magnetic-field fluctuations")
    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, "anisotropy_and_field_evolution.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = os.path.join(
            os.path.dirname(__file__), "..", "build", "src", "prt.000000000.bp"
        )

    try:
        filepaths = resolve_particle_files(input_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Usage: python plot_prt.py [path_to_prt_file.bp | directory | glob]")
        sys.exit(1)

    filepath = filepaths[-1]
    outdir = os.path.join(os.path.dirname(os.path.abspath(filepath)), OUTPUT_DIR)
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}\n")

    if len(filepaths) > 1:
        print(
            f"Resolved {len(filepaths)} particle files. "
            f"Using latest snapshot for static plots: {os.path.basename(filepath)}"
        )
        field_files = resolve_field_files(os.path.dirname(os.path.abspath(filepath)))
        plot_vdf_snapshots(filepaths, outdir)
        plot_distribution_evolution(filepaths, outdir)
        plot_macro_evolution(filepaths, outdir, field_files=field_files)

    data = load_particles(filepath)
    ions, electrons = separate_species(data)

    print_summary(ions, electrons)

    print("\nGenerating plots...")
    plot_vdf(ions, electrons, outdir)
    plot_kappa_comparison(ions, outdir)

    print("\nRunning goodness-of-fit tests (Anderson-Darling & Kolmogorov-Smirnov)...")
    plot_goodness_of_fit(ions, outdir)

    print(f"\nAll plots saved to: {outdir}")


if __name__ == "__main__":
    main()
