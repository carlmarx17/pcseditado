#!/usr/bin/env python3
"""
plot_prt.py — Visualize the first particle output from PSC simulation.
Reads the HDF5 particle file and creates diagnostic plots:
  1. Velocity Distribution Function f(v_perp, v_parallel) — 2D heatmap
  2. Kappa vs Maxwellian distribution comparison for ions

Usage:
    python plot_prt.py [path_to_prt_file]

If no path is given, it defaults to ../build/src/prt.000000000.h5
"""

import sys
import os
import glob
import re
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from scipy.special import gamma as gamma_func

try:
    from data_reader import PICDataReader
except ImportError:
    PICDataReader = None

# ── Configuration ───────────────────────────────────────────────────
OUTPUT_DIR = "prt_plots"
DPI = 200
MAX_EVOLUTION_FILES = 12
MAX_VDF_SNAPSHOTS = 5

STEP_RE = re.compile(r"\.(\d+)(?:_p\d+)?\.h5$")

# ── Simulation parameters (from psc_temp_aniso.cxx) ────────────────
MASS_RATIO = 64.0
Zi = 1.0
VA_OVER_C = 0.1
BETA_I_PAR = 10.0
TI_PERP_OVER_TI_PAR = 3.5
BETA_NORM = 1.0

B0 = VA_OVER_C                          # 0.1
TI_PAR = BETA_I_PAR * B0**2 / 2.0      # 0.05
TI_PERP = TI_PERP_OVER_TI_PAR * TI_PAR # 0.175
M_ION = MASS_RATIO * Zi                # 64.0
KAPPA = 3.0                             # kappa parameter


# ── Helper: Load particles from PSC HDF5 file ──────────────────────
def load_particles(filepath, verbose=True):
    """Load particle data from a PSC prt.*.h5 file."""
    if verbose:
        print(f"Loading particles from: {filepath}")
    with h5py.File(filepath, 'r') as f:
        data = f['particles']['p0']['1d'][:]
    if verbose:
        print(f"  Total particles: {len(data):,}")
    return data


def load_particle_phase_space(filepath):
    """Load only the fields needed for distribution evolution."""
    with h5py.File(filepath, 'r') as f:
        dset = f['particles']['p0']['1d']
        q = dset['q'][:]
        px = dset['px'][:]
        py = dset['py'][:]
        pz = dset['pz'][:]

    ion_mask = q > 0
    elec_mask = q < 0
    return {
        'ions_pz': pz[ion_mask],
        'ions_perp': np.sqrt(px[ion_mask]**2 + py[ion_mask]**2),
        'electrons_pz': pz[elec_mask],
        'electrons_perp': np.sqrt(px[elec_mask]**2 + py[elec_mask]**2),
    }


def load_field_fluctuation_metrics(filepath, b0=B0):
    """Return RMS magnetic fluctuation metrics normalized to B0."""
    if PICDataReader is None:
        raise RuntimeError("data_reader.py is required to read magnetic field outputs.")

    fields = PICDataReader.read_multiple_fields_3d(
        filepath,
        'jeh-',
        ['hx_fc/p0/3d', 'hy_fc/p0/3d', 'hz_fc/p0/3d'],
    )

    bx = np.asarray(fields['hx_fc/p0/3d'], dtype=float).ravel() * b0
    by = np.asarray(fields['hy_fc/p0/3d'], dtype=float).ravel() * b0
    bz = np.asarray(fields['hz_fc/p0/3d'], dtype=float).ravel() * b0

    dbx = bx - np.mean(bx)
    dby = by - np.mean(by)
    dbz = bz - np.mean(bz)

    delta_b_total = np.sqrt(np.mean(dbx**2 + dby**2 + dbz**2)) / max(abs(b0), 1e-30)
    delta_b_parallel = np.sqrt(np.mean(dbz**2)) / max(abs(b0), 1e-30)
    delta_b_perp = np.sqrt(np.mean(dbx**2 + dby**2)) / max(abs(b0), 1e-30)

    return {
        'delta_b_total': delta_b_total,
        'delta_b_parallel': delta_b_parallel,
        'delta_b_perp': delta_b_perp,
    }


def extract_step(filepath):
    """Extract the integer step from a PSC particle filename."""
    match = STEP_RE.search(os.path.basename(filepath))
    if not match:
        raise ValueError(f"Could not extract step from filename: {filepath}")
    return int(match.group(1))


def resolve_particle_files(input_path):
    """Resolve a file, directory, or glob pattern into an ordered file list."""
    if os.path.isdir(input_path):
        candidates = sorted(glob.glob(os.path.join(input_path, "prt.*.h5")))
    elif any(ch in input_path for ch in "*?[]"):
        candidates = sorted(glob.glob(input_path))
    else:
        candidates = [input_path]

    files = [path for path in candidates if os.path.isfile(path)]
    if not files:
        raise FileNotFoundError(f"No particle files matched: {input_path}")

    return sorted(files, key=extract_step)


def resolve_field_files(reference_dir, pattern="pfd.*_p*.h5"):
    """Resolve magnetic field files and map them by simulation step."""
    candidates = sorted(glob.glob(os.path.join(reference_dir, pattern)))
    files = {}
    for path in candidates:
        if os.path.isfile(path):
            files[extract_step(path)] = path
    return files


def sample_filepaths(filepaths, max_files=MAX_EVOLUTION_FILES):
    """Uniformly sample filepaths to keep temporal scans tractable."""
    if len(filepaths) <= max_files:
        return filepaths

    indices = np.linspace(0, len(filepaths) - 1, max_files, dtype=int)
    indices = np.unique(indices)
    return [filepaths[idx] for idx in indices]


# ── Helper: Separate species ────────────────────────────────────────
def separate_species(data, verbose=True):
    """Separate particles by species using charge (q field)."""
    ions = data[data['q'] > 0]
    electrons = data[data['q'] < 0]
    if verbose:
        print(f"  Ions: {len(ions):,},  Electrons: {len(electrons):,}")
    return ions, electrons


def compute_species_temperatures(species):
    """Compute perpendicular/parallel temperatures removing bulk drift."""
    mass = abs(float(species['m'][0]))
    px = np.asarray(species['px'], dtype=float)
    py = np.asarray(species['py'], dtype=float)
    pz = np.asarray(species['pz'], dtype=float)

    t_perp = 0.5 * (np.var(px) + np.var(py)) / mass
    t_par = np.var(pz) / mass
    return t_perp, t_par


def summarize_particle_snapshot(filepath):
    """Compute anisotropy summary from a single particle snapshot."""
    data = load_particles(filepath, verbose=False)
    ions, electrons = separate_species(data, verbose=False)

    ion_tperp, ion_tpar = compute_species_temperatures(ions)
    elec_tperp, elec_tpar = compute_species_temperatures(electrons)

    return {
        'step': extract_step(filepath),
        'ion_anisotropy': ion_tperp / max(ion_tpar, 1e-30),
        'electron_anisotropy': elec_tperp / max(elec_tpar, 1e-30),
        'ion_tperp': ion_tperp,
        'ion_tpar': ion_tpar,
        'electron_tperp': elec_tperp,
        'electron_tpar': elec_tpar,
    }


# ── Plot 1: VDF 2D — f(v_perp, v_parallel) heatmap ─────────────────
def plot_vdf(ions, electrons, outdir):
    """
    2D reduced VDF: f(v_perp, v_parallel) for ions and electrons.
      v_perp = sqrt(px^2 + py^2) / (m * vA)   [in units of vA, >= 0]
      v_par  = pz / (m * vA)                   [in units of vA, symmetric]

    Publication-quality design (white background):
      - Axis range set to 5 * vth_perp/par so the Kappa power-law tail
        is visible falling to the noise floor (clear contrast against white)
      - Independent LogNorm per species (ions and electrons have very
        different temperatures; shared colorscale would crush one of them)
      - Analytical 2D Kappa contours (dashed blue) and isotropic Maxwellian
        contours (dotted orange) overlaid at 50%, 20%, 5%, 1% of the peak
      - All axes in Alfven speed units [vA]
    """
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    # ── Physical parameters (from psc_temp_aniso.cxx) ────────────────
    VA_    = VA_OVER_C                           # Alfven speed in code units (c=1)
    M_ELE  = 1.0                                 # Artificial electron mass
    B0_    = VA_OVER_C
    KAPPA_ = KAPPA

    TI_PAR_  = BETA_I_PAR * B0_**2 / 2.0        # Ion parallel temperature
    TI_PERP_ = TI_PERP_OVER_TI_PAR * TI_PAR_   # Ion perpendicular temperature
    TE_PAR_  = 1.0 * B0_**2 / 2.0               # Electron parallel temperature
    TE_PERP_ = TE_PAR_                           # Electron perp temp (isotropic)

    # Thermal speeds in units of vA: vth = sqrt(T/m) / vA
    vth_i_par  = np.sqrt(TI_PAR_  / M_ION)  / VA_
    vth_i_perp = np.sqrt(TI_PERP_ / M_ION)  / VA_
    vth_e_par  = np.sqrt(TE_PAR_  / M_ELE)  / VA_
    vth_e_perp = np.sqrt(TE_PERP_ / M_ELE)  / VA_

    NBINS = 400   # histogram resolution

    # ── Figure layout — white background for publication ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        r"Ion and Electron VDF: $f(v_\perp,\, v_\parallel)$  (t = 0)",
        fontsize=16, fontweight='bold', color='black', y=1.01
    )

    species_cfg = [
        # (label, data, mass, T_perp, T_par, vth_perp, vth_par, colormap)
        ("Ions",      ions,      M_ION, TI_PERP_, TI_PAR_,
         vth_i_perp, vth_i_par,  'plasma'),
        ("Electrons", electrons, M_ELE, TE_PERP_, TE_PAR_,
         vth_e_perp, vth_e_par,  'viridis'),
    ]

    for ax, (label, sp, mass, T_perp, T_par, vth_p, vth_z, cmap_name) in \
            zip(axes, species_cfg):

        # Compute velocities in vA units
        # PSC 'px', 'py', 'pz' arrays store velocity u = p/m (for non-relativistic).
        px_, py_, pz_ = sp['px'], sp['py'], sp['pz']
        v_perp = np.sqrt(px_**2 + py_**2) / VA_
        v_par  = pz_ / VA_

        # Axis range: 6 * thermal speed
        perp_max = 6.0 * vth_p
        par_abs  = 6.0 * vth_z

        h, xedges, yedges = np.histogram2d(
            v_par, v_perp,
            bins=[NBINS, int(NBINS * 0.6)],
            range=[[-par_abs, par_abs], [0, perp_max]],
            density=True
        )
        
        # Mirror the data to show negative v_perp for illustrative purposes
        h_smooth = gaussian_filter(h, sigma=1.5)
        h_full = np.concatenate([h_smooth[:, ::-1], h_smooth], axis=1)
        yedges_full = np.concatenate([-yedges[::-1], yedges[1:]])

        # Normalize to maximum and take log10
        f_max = h_full.max()
        with np.errstate(divide='ignore'):
            log_f = np.log10(h_full / f_max)
        
        # Clamp minimum value for plotting (e.g. -4 decades)
        log_f = np.clip(log_f, -4.5, 0.0)

        # Use viridis
        cmap = plt.colormaps['viridis'].copy()

        x_c = 0.5 * (xedges[:-1] + xedges[1:])
        y_c = 0.5 * (yedges_full[:-1] + yedges_full[1:])

        ax.set_facecolor('white')

        # Filled contours (must use bin centers x_c, y_c for contourf)
        levels = np.linspace(-4.5, 0.0, 50)
        im = ax.contourf(x_c, y_c, log_f.T, levels=levels, cmap=cmap, extend='min')

        # Black contour lines for readability
        line_levels = [-4.0, -3.0, -2.0, -1.0]
        contours = ax.contour(x_c, y_c, log_f.T, levels=line_levels, colors='black',
                              linestyles='dashed', linewidths=0.8, alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=10, fmt='%1.1f')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.03, ticks=[0, -1, -2, -3, -4])
        cbar.set_label(r'$\log_{10}(f / f_{max})$', fontsize=12)
        cbar.ax.yaxis.set_tick_params(labelsize=10)

        # Axis labels and ticks
        ax.set_xlabel(r'$v_\parallel\ [v_A]$', fontsize=13)
        ax.set_ylabel(r'$v_\perp\ [v_A]$',     fontsize=13)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        
        # Grid/origin marking
        ax.axvline(0, color='black', linewidth=0.6, linestyle=':')
        ax.axhline(0, color='black', linewidth=0.6, linestyle=':')

        title_label = rf"{label}: $\log_{{10}}(f / f_{{max}})$"
        ax.set_title(title_label, fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(outdir, "vdf_2d.png")
    plt.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")



# ── Plot 2: Kappa vs Maxwellian fit comparison ──────────────────────
def kappa_1d(v, kappa, v_th):
    """1D Kappa velocity distribution function (marginal PDF)."""
    A = (1.0 / (np.sqrt(np.pi * (2*kappa - 3)) * v_th)) * \
        (gamma_func(kappa) / (gamma_func(kappa - 0.5)))
    return A * (1 + v**2 / ((2*kappa - 3) * v_th**2))**(-kappa)

def maxwellian_1d(v, v_th):
    """1D Maxwellian velocity distribution function."""
    return (1.0 / (np.sqrt(2 * np.pi) * v_th)) * np.exp(-v**2 / (2 * v_th**2))

def plot_kappa_comparison(ions, outdir):
    """Compare measured ion distributions to theoretical Kappa and Maxwellian."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.patch.set_facecolor('white')
    fig.suptitle(r"Distribution Comparison: Data vs Kappa ($\kappa$=3) vs Maxwellian — Ions",
                 fontsize=18, fontweight='bold')

    mom_fields = ['px', 'py', 'pz']
    titles = [r'$p_x$ (perp)', r'$p_y$ (perp)', r'$p_z$ (parallel)']
    kappa = KAPPA

    # Theoretical v_th per component
    theoretical_vth = {
        'px': BETA_NORM * np.sqrt(TI_PERP / M_ION),
        'py': BETA_NORM * np.sqrt(TI_PERP / M_ION),
        'pz': BETA_NORM * np.sqrt(TI_PAR / M_ION),
    }

    for col, (field, title) in enumerate(zip(mom_fields, titles)):
        ion_p = ions[field]
        v_th = theoretical_vth[field]

        # Histogram (use step style so theoretical curves are clearly visible)
        bins = np.linspace(np.percentile(ion_p, 0.05),
                           np.percentile(ion_p, 99.95), 200)
        hist, bin_edges = np.histogram(ion_p, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        hist_plot = hist.copy().astype(float)
        hist_plot[hist_plot == 0] = np.nan

        valid = hist[hist > 0]
        y_min = max(valid.min() * 0.1, 1e-8) if len(valid) else 1e-8

        # --- ROW 0: f(p) vs p (Linear P, Log F) ---
        ax0 = axes[0, col]
        ax0.step(bin_centers, hist_plot, where='mid', color='#c0392b', linewidth=1.5, alpha=0.85, label='Ion data', zorder=2)
        v_range = np.linspace(bin_centers[0], bin_centers[-1], 1000)
        
        ax0.semilogy(v_range, kappa_1d(v_range, kappa, v_th), '-', color='#27ae60', linewidth=2.5, zorder=3, label=rf'Kappa ($\kappa$={kappa})')
        ax0.semilogy(v_range, maxwellian_1d(v_range, v_th), '--', color='#8e44ad', linewidth=2.5, zorder=3, label='Maxwellian')
        
        ax0.set_xlabel(title, fontsize=13)
        ax0.set_ylabel('f(p)', fontsize=13)
        ax0.set_title(f"Log F vs P: {title}", fontsize=13)
        ax0.legend(fontsize=10)
        ax0.grid(True, alpha=0.3)
        ax0.set_yscale('log')
        ax0.set_ylim(bottom=y_min)

        # --- ROW 1: f(p) vs p^2 (P^2, Log F) ---
        ax1 = axes[1, col]
        ax1.step(bin_centers**2, hist_plot, where='mid', color='#c0392b', linewidth=1.5, alpha=0.85, label='Ion data', zorder=2)
        
        # Sort by p^2 for plotting theoretical lines properly
        p2_sorted_idx = np.argsort(v_range**2)
        v_range_sorted = v_range[p2_sorted_idx]
        
        ax1.semilogy(v_range_sorted**2, kappa_1d(v_range_sorted, kappa, v_th), '-', color='#27ae60', linewidth=2.5, zorder=3, label=rf'Kappa ($\kappa$={kappa})')
        ax1.semilogy(v_range_sorted**2, maxwellian_1d(v_range_sorted, v_th), '--', color='#8e44ad', linewidth=2.5, zorder=3, label='Maxwellian (Straight)')
        
        ax1.set_xlabel(title + r'$^2$', fontsize=13)
        ax1.set_ylabel('f(p)', fontsize=13)
        ax1.set_title(f"Log F vs P^2: {title}", fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=y_min)

        # --- ROW 2: f(p) vs p (Log P, Log F) ---
        ax2 = axes[2, col]
        
        # Select only positive tail for log-log plot to see power law
        pos_mask = bin_centers > 0
        bin_centers_pos = bin_centers[pos_mask]
        hist_plot_pos = hist_plot[pos_mask]
        
        ax2.step(bin_centers_pos, hist_plot_pos, where='mid', color='#c0392b', linewidth=1.5, alpha=0.85, label='Ion data', zorder=2)
        
        if len(bin_centers_pos) > 0:
            v_range_pos = np.linspace(bin_centers_pos[0], bin_centers[-1], 1000)
            ax2.loglog(v_range_pos, kappa_1d(v_range_pos, kappa, v_th), '-', color='#27ae60', linewidth=2.5, zorder=3, label=rf'Kappa (Linear Tail)')
            ax2.loglog(v_range_pos, maxwellian_1d(v_range_pos, v_th), '--', color='#8e44ad', linewidth=2.5, zorder=3, label='Maxwellian (Plunge)')
        
        ax2.set_xlabel(title + ' (Log scale)', fontsize=13)
        ax2.set_ylabel('f(p)', fontsize=13)
        ax2.set_title(f"Log F vs Log P: {title} > 0", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which="both", ls="--")
        ax2.set_ylim(bottom=y_min)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, "kappa_vs_maxwellian.png")
    plt.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 3: Summary statistics ──────────────────────────────────────
def print_summary(ions, electrons):
    """Print summary statistics of the particle data."""
    print("\n" + "="*60)
    print("PARTICLE DATA SUMMARY (t=0)")
    print("="*60)

    for name, sp in [("IONS", ions), ("ELECTRONS", electrons)]:
        print(f"\n  {name}:")
        print(f"    Count: {len(sp):,}")
        print(f"    Mass (m): {sp['m'][0]:.4f}")
        print(f"    Charge (q): {sp['q'][0]:.4f}")

        m = np.abs(sp['m'][0])
        # px, py, pz are momentum (p=mv). Thus T = var(p) / m
        T_perp = 0.5 * (np.mean(sp['px']**2) + np.mean(sp['py']**2)) / m
        T_par  = np.mean(sp['pz']**2) / m
        print(f"    T_perp = {T_perp:.5f}, T_par = {T_par:.5f}")
        print(f"    T_perp/T_par = {T_perp/T_par:.4f}")

    print("\n" + "="*60)


def plot_vdf_snapshots(filepaths, outdir, max_snapshots=MAX_VDF_SNAPSHOTS, bins_parallel=180, bins_perp=120):
    """Plot reduced VDFs for a few representative times."""
    print("\nBuilding multi-time VDF snapshot panel...")

    sampled_paths = sample_filepaths(filepaths, max_files=max_snapshots)
    sampled_data = [(extract_step(path), load_particle_phase_space(path)) for path in sampled_paths]

    ion_par_max = max(np.percentile(np.abs(data['ions_pz']), 99.5) for _, data in sampled_data)
    ion_perp_max = max(np.percentile(data['ions_perp'], 99.5) for _, data in sampled_data)
    elec_par_max = max(np.percentile(np.abs(data['electrons_pz']), 99.5) for _, data in sampled_data)
    elec_perp_max = max(np.percentile(data['electrons_perp'], 99.5) for _, data in sampled_data)

    species_config = [
        ('ions_pz', 'ions_perp', ion_par_max, ion_perp_max, 'Ions'),
        ('electrons_pz', 'electrons_perp', elec_par_max, elec_perp_max, 'Electrons'),
    ]

    fig, axes = plt.subplots(
        2,
        len(sampled_data),
        figsize=(4.4 * len(sampled_data), 8.0),
        constrained_layout=True,
        squeeze=False,
    )
    fig.patch.set_facecolor('white')
    fig.suptitle('Velocity Distribution Function at Different Times', fontsize=17, fontweight='bold')

    for row, (par_key, perp_key, par_max, perp_max, label) in enumerate(species_config):
        for col, (step, data) in enumerate(sampled_data):
            ax = axes[row, col]
            hist, xedges, yedges = np.histogram2d(
                data[par_key],
                data[perp_key],
                bins=[bins_parallel, bins_perp],
                range=[[-par_max, par_max], [0.0, perp_max]],
                density=True,
            )
            hist = gaussian_filter(hist, sigma=1.0)

            positive = hist[hist > 0]
            vmax = positive.max() if positive.size else 1.0
            log_hist = np.full_like(hist, -5.0)
            if positive.size:
                with np.errstate(divide='ignore'):
                    log_hist = np.log10(np.maximum(hist / vmax, 1e-5))

            im = ax.pcolormesh(
                xedges,
                yedges,
                log_hist.T,
                shading='auto',
                cmap='viridis',
                vmin=-5.0,
                vmax=0.0,
            )
            ax.set_title(f'{label} | step {step}', fontsize=12, fontweight='bold')
            ax.set_xlabel(r'$v_\parallel$')
            ax.set_ylabel(r'$v_\perp$')
            ax.tick_params(which='both', direction='in', top=True, right=True)

            if col == len(sampled_data) - 1:
                cbar = fig.colorbar(im, ax=ax, pad=0.01)
                cbar.set_label(r'$\log_{10}(f / f_{max})$')

    path = os.path.join(outdir, "vdf_time_snapshots.png")
    plt.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def plot_distribution_evolution(filepaths, outdir, bins_parallel=180, bins_perp=120):
    """Plot the temporal evolution of 1D ion/electron distributions."""
    print("\nBuilding distribution evolution plot...")

    sampled_paths = sample_filepaths(filepaths)
    if len(sampled_paths) != len(filepaths):
        print(f"  Using {len(sampled_paths)} uniformly sampled snapshots out of {len(filepaths)} for temporal evolution.")

    steps = []
    range_sample = load_particle_phase_space(sampled_paths[-1])
    ion_par_max = np.percentile(np.abs(range_sample['ions_pz']), 99.5)
    ion_perp_max = np.percentile(range_sample['ions_perp'], 99.5)
    elec_par_max = np.percentile(np.abs(range_sample['electrons_pz']), 99.5)
    elec_perp_max = np.percentile(range_sample['electrons_perp'], 99.5)

    ion_par_edges = np.linspace(-ion_par_max, ion_par_max, bins_parallel + 1)
    ion_perp_edges = np.linspace(0.0, ion_perp_max, bins_perp + 1)
    elec_par_edges = np.linspace(-elec_par_max, elec_par_max, bins_parallel + 1)
    elec_perp_edges = np.linspace(0.0, elec_perp_max, bins_perp + 1)

    ion_par_matrix = np.zeros((len(ion_par_edges) - 1, len(sampled_paths)))
    ion_perp_matrix = np.zeros((len(ion_perp_edges) - 1, len(sampled_paths)))
    elec_par_matrix = np.zeros((len(elec_par_edges) - 1, len(sampled_paths)))
    elec_perp_matrix = np.zeros((len(elec_perp_edges) - 1, len(sampled_paths)))

    for idx, filepath in enumerate(sampled_paths):
        phase_space = load_particle_phase_space(filepath)
        steps.append(extract_step(filepath))

        ion_par_matrix[:, idx], _ = np.histogram(phase_space['ions_pz'], bins=ion_par_edges, density=True)
        ion_perp_matrix[:, idx], _ = np.histogram(phase_space['ions_perp'], bins=ion_perp_edges, density=True)
        elec_par_matrix[:, idx], _ = np.histogram(phase_space['electrons_pz'], bins=elec_par_edges, density=True)
        elec_perp_matrix[:, idx], _ = np.histogram(phase_space['electrons_perp'], bins=elec_perp_edges, density=True)

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
        (ion_par_matrix, ion_par_edges, r'Ions: $f(v_\parallel, t)$'),
        (ion_perp_matrix, ion_perp_edges, r'Ions: $f(v_\perp, t)$'),
        (elec_par_matrix, elec_par_edges, r'Electrons: $f(v_\parallel, t)$'),
        (elec_perp_matrix, elec_perp_edges, r'Electrons: $f(v_\perp, t)$'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.patch.set_facecolor('white')
    fig.suptitle("Temporal Evolution of Velocity Distributions", fontsize=17, fontweight='bold')

    for ax, (matrix, edges, title) in zip(axes.flat, panels):
        positive = matrix[matrix > 0]
        vmin = positive.min() if positive.size else 1e-12
        vmax = matrix.max() if matrix.size else 1.0
        im = ax.pcolormesh(
            time_edges,
            edges,
            matrix,
            shading='auto',
            norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10.0)),
            cmap='viridis'
        )
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Simulation step')
        ax.set_ylabel('Velocity')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label('PDF')

    path = os.path.join(outdir, "distribution_evolution.png")
    plt.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def plot_macro_evolution(filepaths, outdir, field_files=None):
    """Track particle anisotropy and magnetic fluctuations versus time."""
    print("\nBuilding anisotropy and magnetic-fluctuation evolution plot...")

    sampled_paths = sample_filepaths(filepaths)
    particle_rows = [summarize_particle_snapshot(path) for path in sampled_paths]

    steps = np.asarray([row['step'] for row in particle_rows], dtype=float)
    ion_anis = np.asarray([row['ion_anisotropy'] for row in particle_rows], dtype=float)
    elec_anis = np.asarray([row['electron_anisotropy'] for row in particle_rows], dtype=float)

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
            delta_b_total.append(metrics['delta_b_total'])
            delta_b_parallel.append(metrics['delta_b_parallel'])
            delta_b_perp.append(metrics['delta_b_perp'])

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    fig.patch.set_facecolor('white')
    fig.suptitle('Temporal Evolution of Anisotropy and Magnetic Fluctuations', fontsize=17, fontweight='bold')

    ax = axes[0]
    ax.plot(steps, ion_anis, marker='o', linewidth=2.0, color='#c0392b', label='Ions')
    ax.plot(steps, elec_anis, marker='s', linewidth=2.0, color='#2980b9', label='Electrons')
    ax.axhline(1.0, color='black', linestyle=':', linewidth=1.0, alpha=0.8)
    ax.set_ylabel(r'$T_\perp / T_\parallel$')
    ax.set_xlabel('Simulation step')
    ax.set_title('Particle anisotropy')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    if field_steps:
        field_steps = np.asarray(field_steps, dtype=float)
        ax.plot(field_steps, delta_b_total, marker='o', linewidth=2.0, color='#8e44ad', label=r'$\delta B_{\rm rms} / B_0$')
        ax.plot(field_steps, delta_b_parallel, marker='^', linewidth=1.8, color='#16a085', label=r'$\delta B_{\parallel,\rm rms} / B_0$')
        ax.plot(field_steps, delta_b_perp, marker='d', linewidth=1.8, color='#f39c12', label=r'$\delta B_{\perp,\rm rms} / B_0$')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No matching pfd.*.h5 field files found for these steps.', ha='center', va='center', transform=ax.transAxes)
    ax.set_ylabel('Normalized fluctuation amplitude')
    ax.set_xlabel('Simulation step')
    ax.set_title('Magnetic-field fluctuations')
    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, "anisotropy_and_field_evolution.png")
    plt.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ── Main ────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = os.path.join(os.path.dirname(__file__),
                                  "..", "build", "src", "prt.000000000.h5")

    try:
        filepaths = resolve_particle_files(input_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Usage: python plot_prt.py [path_to_prt_file.h5|directory|glob]")
        sys.exit(1)

    filepath = filepaths[-1]
    outdir = os.path.join(os.path.dirname(os.path.abspath(filepath)), OUTPUT_DIR)
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}\n")

    if len(filepaths) > 1:
        print(f"Resolved {len(filepaths)} particle files. Using latest snapshot for static plots: {os.path.basename(filepath)}")
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

    print(f"\nAll plots saved to: {outdir}")


if __name__ == "__main__":
    main()
