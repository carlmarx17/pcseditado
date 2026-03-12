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
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from scipy.special import gamma as gamma_func

# ── Configuration ───────────────────────────────────────────────────
OUTPUT_DIR = "prt_plots"
DPI = 200

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
def load_particles(filepath):
    """Load particle data from a PSC prt.*.h5 file."""
    print(f"Loading particles from: {filepath}")
    with h5py.File(filepath, 'r') as f:
        data = f['particles']['p0']['1d'][:]
    print(f"  Total particles: {len(data):,}")
    return data


# ── Helper: Separate species ────────────────────────────────────────
def separate_species(data):
    """Separate particles by species using charge (q field)."""
    ions = data[data['q'] > 0]
    electrons = data[data['q'] < 0]
    print(f"  Ions: {len(ions):,},  Electrons: {len(electrons):,}")
    return ions, electrons


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


# ── Main ────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = os.path.join(os.path.dirname(__file__),
                                "..", "build", "src", "prt.000000000.h5")

    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print("Usage: python plot_prt.py [path_to_prt_file.h5]")
        sys.exit(1)

    outdir = os.path.join(os.path.dirname(os.path.abspath(filepath)), OUTPUT_DIR)
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}\n")

    data = load_particles(filepath)
    ions, electrons = separate_species(data)

    print_summary(ions, electrons)

    print("\nGenerating plots...")
    plot_vdf(ions, electrons, outdir)
    plot_kappa_comparison(ions, outdir)

    print(f"\nAll plots saved to: {outdir}")


if __name__ == "__main__":
    main()
