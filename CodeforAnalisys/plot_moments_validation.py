#!/usr/bin/env python3
"""
plot_moments_validation.py
==========================
Generates 2D spatial maps of the macroscopic moments (density, velocity, temperature)
computed from the particle data per grid cell. This visually validates that the
particle distributions consistently match the expected simulation parameters
(e.g., density = 1, v = 0, and anisotropic temperatures) across the entire domain.

Usage:
    python plot_moments_validation.py [path_to_prt_file.h5]
"""

import sys
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Expected Parameters (from psc_temp_aniso.cxx) ────────────────────────
BB = 1.0
Zi = 1.0
MASS_RATIO = 64.0
VA_OVER_C = 0.1
BETA_E_PAR = 1.0
BETA_I_PAR = 10.0
TI_PERP_OVER_TI_PAR = 3.5
TE_PERP_OVER_TE_PAR = 1.0
N_DENSITY = 1.0

B0 = VA_OVER_C
TE_PAR = BETA_E_PAR * B0**2 / 2.0         # 0.005
TE_PERP = TE_PERP_OVER_TE_PAR * TE_PAR    # 0.005
TI_PAR = BETA_I_PAR * B0**2 / 2.0         # 0.05
TI_PERP = TI_PERP_OVER_TI_PAR * TI_PAR    # 0.175

M_ION = MASS_RATIO * Zi    # 64.0
M_ELECTRON = 1.0

N_GRID_Y = 384
N_GRID_Z = 512
NICELL = 250
CORI = 1.0 / NICELL

OUTPUT_DIR = "moment_validation_plots"
DPI = 150

# ──────────────────────────────────────────────────────────────────────────

def add_colorbar(im, ax, title):
    """Add a perfectly sized colorbar next to an axes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(title, fontsize=10)
    return cbar

def plot_moment_map(y, z, values, statistic, bins, drange, expected,
                    title, cmap, ax, vmin=None, vmax=None):
    """Calculate and plot a 2D heatmap of a moment."""
    stat_grid, _, _, _ = binned_statistic_2d(
        y, z, values,
        statistic=statistic,
        bins=bins,
        range=drange
    )
    
    # NaN out cells with no particles
    stat_grid = np.where(stat_grid == 0, np.nan, stat_grid)
    
    # Transpose so Y goes along X-axis, Z along Y-axis
    im = ax.imshow(stat_grid.T, origin='lower', extent=[0, bins[0], 0, bins[1]],
                   cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    
    ax.set_title(f"{title}\nMeasured Mean: {np.nanmean(stat_grid):.5f} | Expected: {expected:.5f}", fontsize=11)
    ax.set_xlabel('Y (cells)', fontsize=10)
    ax.set_ylabel('Z (cells)', fontsize=10)
    
    add_colorbar(im, ax, title)
    return stat_grid

def validate_species(y, z, px, py, pz, m, species_name, T_perp_exp, T_par_exp, outdir):
    """Generate 2D moment validation maps for a given species."""
    print(f"\nProcessing {species_name} moments (N={len(y):,})...")
    
    vx = px / m
    vy = py / m
    vz = pz / m

    bins = [N_GRID_Y, N_GRID_Z]
    drange = [[0, N_GRID_Y], [0, N_GRID_Z]]

    # 1. Density Map
    fig, ax = plt.subplots(figsize=(7, 6))
    counts, _, _, _ = binned_statistic_2d(y, z, None, statistic='count', bins=bins, range=drange)
    density = counts * CORI
    im = ax.imshow(density.T, origin='lower', extent=[0, bins[0], 0, bins[1]],
                   cmap='viridis', aspect='equal', vmin=0.8, vmax=1.2, interpolation='nearest')
    ax.set_title(f"Density ($n$) : {species_name}\nMeasured Mean: {np.mean(density):.5f} | Expected: {N_DENSITY:.5f}", fontsize=11)
    ax.set_xlabel('Y (cells)', fontsize=10)
    ax.set_ylabel('Z (cells)', fontsize=10)
    add_colorbar(im, ax, "Density")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{species_name.lower()}_density.png"), dpi=DPI)
    plt.close()
    
    # 2. Macroscopic Velocity Maps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Macroscopic Velocity ($<v>$) : {species_name}", fontsize=14, fontweight='bold')
    
    # Small vmin/vmax tight around 0
    vbound = 0.05 * np.sqrt(T_perp_exp / m)  # 5% of thermal speed
    
    plot_moment_map(y, z, vx, 'mean', bins, drange, 0.0, r"$<v_x>$", 'coolwarm', axes[0], vmin=-vbound, vmax=vbound)
    plot_moment_map(y, z, vy, 'mean', bins, drange, 0.0, r"$<v_y>$", 'coolwarm', axes[1], vmin=-vbound, vmax=vbound)
    plot_moment_map(y, z, vz, 'mean', bins, drange, 0.0, r"$<v_z>$", 'coolwarm', axes[2], vmin=-vbound, vmax=vbound)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{species_name.lower()}_velocity.png"), dpi=DPI)
    plt.close()
    
    # 3. Temperature Maps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Temperature ($T = m \cdot Var(v)$) : {species_name}", fontsize=14, fontweight='bold')

    # Temperature is m * Var(v). We use the 'std' statistic from binned_statistic_2d and square it.
    std_vx, _, _, _ = binned_statistic_2d(y, z, vx, statistic='std', bins=bins, range=drange)
    std_vy, _, _, _ = binned_statistic_2d(y, z, vy, statistic='std', bins=bins, range=drange)
    std_vz, _, _, _ = binned_statistic_2d(y, z, vz, statistic='std', bins=bins, range=drange)
    
    var_vx = std_vx**2
    var_vy = std_vy**2
    var_vz = std_vz**2
    
    Tx = var_vx * m
    Ty = var_vy * m
    Tz = var_vz * m
    
    T_perp = 0.5 * (Tx + Ty)
    
    # Plot T_x (perp)
    im0 = axes[0].imshow(Tx.T, origin='lower', extent=[0, bins[0], 0, bins[1]], cmap='plasma', aspect='equal',
                         vmin=T_perp_exp*0.8, vmax=T_perp_exp*1.2, interpolation='nearest')
    axes[0].set_title(f"$T_x$ (Perpendicular)\nMeasured: {np.nanmean(Tx):.5f} | Expected: {T_perp_exp:.5f}", fontsize=11)
    add_colorbar(im0, axes[0], "$T_x$")

    # Plot T_y (perp)
    im1 = axes[1].imshow(Ty.T, origin='lower', extent=[0, bins[0], 0, bins[1]], cmap='plasma', aspect='equal',
                         vmin=T_perp_exp*0.8, vmax=T_perp_exp*1.2, interpolation='nearest')
    axes[1].set_title(f"$T_y$ (Perpendicular)\nMeasured: {np.nanmean(Ty):.5f} | Expected: {T_perp_exp:.5f}", fontsize=11)
    add_colorbar(im1, axes[1], "$T_y$")

    # Plot T_z (par)
    im2 = axes[2].imshow(Tz.T, origin='lower', extent=[0, bins[0], 0, bins[1]], cmap='plasma', aspect='equal',
                         vmin=T_par_exp*0.8, vmax=T_par_exp*1.2, interpolation='nearest')
    axes[2].set_title(f"$T_z$ (Parallel)\nMeasured: {np.nanmean(Tz):.5f} | Expected: {T_par_exp:.5f}", fontsize=11)
    add_colorbar(im2, axes[2], "$T_z$")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{species_name.lower()}_temperature.png"), dpi=DPI)
    plt.close()
    
    # 4. Anisotropy Map (T_perp / T_par)
    fig, ax = plt.subplots(figsize=(7, 6))
    aniso = T_perp / Tz
    expected_aniso = T_perp_exp / T_par_exp
    
    im = ax.imshow(aniso.T, origin='lower', extent=[0, bins[0], 0, bins[1]],
                   cmap='magma', aspect='equal', vmin=expected_aniso*0.8, vmax=expected_aniso*1.2, interpolation='nearest')
    ax.set_title(f"Anisotropy ($T_{{\perp}} / T_{{\parallel}}$) : {species_name}\nMeasured Mean: {np.nanmean(aniso):.5f} | Expected: {expected_aniso:.5f}", fontsize=11)
    ax.set_xlabel('Y (cells)', fontsize=10)
    ax.set_ylabel('Z (cells)', fontsize=10)
    add_colorbar(im, ax, "$T_{\perp} / T_{\parallel}$")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{species_name.lower()}_anisotropy.png"), dpi=DPI)
    plt.close()


def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = os.path.join(os.path.dirname(__file__), "..", "build", "src", "prt.000000000.h5")

    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    outdir = os.path.join(os.path.dirname(os.path.abspath(filepath)), OUTPUT_DIR)
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}")

    print("Loading particle data...")
    with h5py.File(filepath, 'r') as f:
        data = f['particles']['p0']['1d'][:]

    y, z = data['y'], data['z']
    px, py, pz = data['px'], data['py'], data['pz']
    m, q = data['m'], data['q']

    ion_idx = q > 0
    elec_idx = q < 0

    print("\nValidating IONS...")
    validate_species(y[ion_idx], z[ion_idx], px[ion_idx], py[ion_idx], pz[ion_idx],
                     M_ION, "IONS", TI_PERP, TI_PAR, outdir)

    print("\nValidating ELECTRONS...")
    validate_species(y[elec_idx], z[elec_idx], px[elec_idx], py[elec_idx], pz[elec_idx],
                     M_ELECTRON, "ELECTRONS", TE_PERP, TE_PAR, outdir)

    print(f"\nAll moment validation plots saved to: {outdir}")

if __name__ == "__main__":
    main()
