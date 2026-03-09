#!/usr/bin/env python3
"""
plot_prt.py — Visualize the first particle output from PSC simulation.
Reads the HDF5 particle file and creates diagnostic plots:
  1. Phase-space (position vs momentum) for each species
  2. Momentum distribution histograms (px, py, pz)
  3. 2D spatial scatter of particles colored by energy
  4. Velocity distribution comparison with Kappa and Maxwellian fits

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
from scipy.special import gamma as gamma_func

# ── Configuration ───────────────────────────────────────────────────
OUTPUT_DIR = "prt_plots"
DPI = 150

# ── Helper: Load particles from PSC HDF5 file ──────────────────────
def load_particles(filepath):
    """Load particle data from a PSC prt.*.h5 file."""
    print(f"Loading particles from: {filepath}")
    with h5py.File(filepath, 'r') as f:
        print(f"  HDF5 file structure:")
        def print_structure(name, obj):
            print(f"    {name}: {type(obj).__name__}", end="")
            if isinstance(obj, h5py.Dataset):
                print(f"  shape={obj.shape}  dtype={obj.dtype}")
            else:
                print()
        f.visititems(print_structure)

        # Navigate to particle data
        # PSC stores particles under particles/p0/1d
        grp = f['particles']['p0']
        data = grp['1d'][:]

    # Extract fields from the compound dataset
    fields = data.dtype.names
    print(f"\n  Available fields: {fields}")
    print(f"  Total particles: {len(data)}")

    return data, fields


# ── Helper: Separate species ────────────────────────────────────────
def separate_species(data):
    """Separate particles by species using charge (q field)."""
    # Ions have q > 0, electrons have q < 0
    ions = data[data['q'] > 0]
    electrons = data[data['q'] < 0]
    print(f"  Ions: {len(ions)},  Electrons: {len(electrons)}")
    return ions, electrons


# ── Plot 1: Phase Space (2D Heatmap) ───────────────────────────────
def plot_phase_space(ions, electrons, outdir):
    """Phase-space diagrams as 2D density heatmaps."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Phase Space — 2D Density (t=0)", fontsize=16, fontweight='bold')

    coords = [('y', 'py'), ('z', 'pz'), ('y', 'pz')]
    labels = [('Y', r'$p_y$'), ('Z', r'$p_z$'), ('Y', r'$p_z$')]
    cmaps  = ['inferno', 'inferno', 'inferno']

    for col, ((pos_name, mom_name), (pos_label, mom_label)) in enumerate(zip(coords, labels)):
        for row, (species, name, cmap_base) in enumerate([
            (ions, "Ions", 'YlOrRd'),
            (electrons, "Electrons", 'YlGnBu')
        ]):
            ax = axes[row, col]
            pos_data = species[pos_name]
            mom_data = species[mom_name]

            # Compute percentiles to crop outliers for better color contrast
            mom_lo, mom_hi = np.percentile(mom_data, [0.5, 99.5])

            h = ax.hist2d(pos_data, mom_data, bins=[256, 256],
                          range=[[pos_data.min(), pos_data.max()], [mom_lo, mom_hi]],
                          cmap=cmap_base, norm=LogNorm(), rasterized=True)
            plt.colorbar(h[3], ax=ax, label='Counts', pad=0.02)
            ax.set_xlabel(pos_label, fontsize=12)
            ax.set_ylabel(mom_label, fontsize=12)
            ax.set_title(f"{name}: {pos_label} vs {mom_label}", fontsize=13)

    plt.tight_layout()
    path = os.path.join(outdir, "phase_space.png")
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 2: Momentum Distributions ─────────────────────────────────
def plot_momentum_distributions(ions, electrons, outdir):
    """Histogram of px, py, pz for each species."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Momentum Distributions (t=0)", fontsize=16, fontweight='bold')

    mom_fields = ['px', 'py', 'pz']
    mom_labels = [r'$p_x$', r'$p_y$ (perp)', r'$p_z$ (parallel)']

    for col, (field, label) in enumerate(zip(mom_fields, mom_labels)):
        # Ions
        ax = axes[0, col]
        ion_data = ions[field]
        bins = np.linspace(np.percentile(ion_data, 0.5),
                           np.percentile(ion_data, 99.5), 200)
        ax.hist(ion_data, bins=bins, density=True, alpha=0.7,
                color='#e74c3c', label='Ions')
        ax.set_xlabel(label, fontsize=13)
        ax.set_ylabel('f(p)', fontsize=13)
        ax.set_title(f"Ions: {label}", fontsize=13)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Electrons
        ax = axes[1, col]
        ele_data = electrons[field]
        bins = np.linspace(np.percentile(ele_data, 0.5),
                           np.percentile(ele_data, 99.5), 200)
        ax.hist(ele_data, bins=bins, density=True, alpha=0.7,
                color='#3498db', label='Electrons')
        ax.set_xlabel(label, fontsize=13)
        ax.set_ylabel('f(p)', fontsize=13)
        ax.set_title(f"Electrons: {label}", fontsize=13)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(outdir, "momentum_distributions.png")
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 3: Kappa vs Maxwellian fit comparison ──────────────────────
def kappa_1d(v, kappa, v_th):
    """1D Kappa velocity distribution function.
    
    Derived from the sampling in createKappaMultivariate:
      Y ~ Gamma(kappa-0.5, 1), S = sqrt((kappa-1.5)/Y), p = Z*S*v_th
    This produces a scaled Student's t with nu=2*kappa-1 d.o.f.
    The 1D marginal PDF is:
      f(v) = Gamma(kappa) / (Gamma(kappa-0.5) * sqrt(pi*(2k-3)) * v_th)
             * (1 + v^2 / ((2k-3)*v_th^2))^(-kappa)
    """
    A = (1.0 / (np.sqrt(np.pi * (2*kappa - 3)) * v_th)) * \
        (gamma_func(kappa) / (gamma_func(kappa - 0.5)))
    return A * (1 + v**2 / ((2*kappa - 3) * v_th**2))**(-kappa)

def maxwellian_1d(v, v_th):
    """1D Maxwellian velocity distribution function."""
    return (1.0 / (np.sqrt(2 * np.pi) * v_th)) * np.exp(-v**2 / (2 * v_th**2))

def plot_kappa_comparison(ions, electrons, outdir):
    """Compare measured distributions to theoretical Kappa and Maxwellian."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(r"Distribution Comparison: Data vs Kappa ($\kappa$=3) vs Maxwellian",
                 fontsize=15, fontweight='bold')

    mom_fields = ['px', 'py', 'pz']
    titles = [r'$p_x$ (perp)', r'$p_y$ (perp)', r'$p_z$ (parallel)']
    kappa = 3.0

    for col, (field, title) in enumerate(zip(mom_fields, titles)):
        ax = axes[col]

        # Ion data
        ion_p = ions[field]
        v_th = np.std(ion_p)

        # Histogram of data — filled bars
        bins = np.linspace(np.percentile(ion_p, 0.1),
                           np.percentile(ion_p, 99.9), 200)
        hist, bin_edges = np.histogram(ion_p, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        # Replace zeros with NaN so log scale doesn't break
        hist_plot = hist.copy().astype(float)
        hist_plot[hist_plot == 0] = np.nan

        ax.bar(bin_centers, hist_plot, width=bin_width, alpha=0.5,
               color='#e74c3c', edgecolor='#c0392b', linewidth=0.3,
               label='Ion data', log=True)

        # Theoretical curves
        v_range = np.linspace(bin_centers[0], bin_centers[-1], 1000)
        ax.semilogy(v_range, kappa_1d(v_range, kappa, v_th),
                    '-', color='#2ecc71', linewidth=2.5,
                    label=rf'Kappa ($\kappa$={kappa})')
        ax.semilogy(v_range, maxwellian_1d(v_range, v_th),
                    '--', color='#9b59b6', linewidth=2.5,
                    label='Maxwellian')

        ax.set_xlabel(title, fontsize=13)
        ax.set_ylabel('f(p)', fontsize=13)
        ax.set_title(f"Ions: {title}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        valid = hist[hist > 0]
        ax.set_ylim(bottom=max(valid.min() * 0.1, 1e-8) if len(valid) else 1e-8)

    plt.tight_layout()
    path = os.path.join(outdir, "kappa_vs_maxwellian.png")
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 4: Spatial distribution colored by energy ──────────────────
def plot_spatial_energy(ions, electrons, outdir):
    """2D scatter of particle positions colored by kinetic energy."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Spatial Distribution Colored by Kinetic Energy (t=0)",
                 fontsize=15, fontweight='bold')

    for ax, species, name, cmap in [
        (axes[0], ions, "Ions", "hot"),
        (axes[1], electrons, "Electrons", "cool")
    ]:
        n_sample = min(len(species), 80000)
        idx = np.random.choice(len(species), n_sample, replace=False)

        # Kinetic energy ~ 0.5 * m * (px^2 + py^2 + pz^2)
        KE = 0.5 * np.abs(species['m'][idx]) * (
            species['px'][idx]**2 +
            species['py'][idx]**2 +
            species['pz'][idx]**2
        )

        sc = ax.scatter(species['y'][idx], species['z'][idx],
                        c=KE, s=0.2, alpha=0.4, cmap=cmap,
                        norm=LogNorm(vmin=max(KE[KE > 0].min(), 1e-10)),
                        rasterized=True)
        plt.colorbar(sc, ax=ax, label='Kinetic Energy')
        ax.set_xlabel('Y', fontsize=12)
        ax.set_ylabel('Z', fontsize=12)
        ax.set_title(name, fontsize=14)
        ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(outdir, "spatial_energy.png")
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 5: 2D Velocity Distribution Function (VDF) ────────────────
def plot_vdf(ions, electrons, outdir):
    """2D VDF: f(v_perp, v_parallel) heatmap for ions and electrons.
    v_perp = sqrt(px^2 + py^2), v_parallel = pz.
    Same colormap and shared color limits for direct comparison."""

    # ── Pre-compute both histograms to find shared limits ───────────
    nbins = 300
    species_data = {}
    for label, species in [("Ions", ions), ("Electrons", electrons)]:
        px, py, pz = species['px'], species['py'], species['pz']
        p_perp = np.sqrt(px**2 + py**2)
        perp_hi = np.percentile(p_perp, 99.5)
        par_lo, par_hi = np.percentile(pz, [0.5, 99.5])

        h, xedges, yedges = np.histogram2d(
            pz, p_perp,
            bins=[nbins, nbins // 2],
            range=[[par_lo, par_hi], [0, perp_hi]],
            density=True
        )
        species_data[label] = (h, xedges, yedges)

    # Shared color limits across both panels
    all_nonzero = np.concatenate([
        species_data["Ions"][0][species_data["Ions"][0] > 0],
        species_data["Electrons"][0][species_data["Electrons"][0] > 0]
    ])
    global_vmin = np.percentile(all_nonzero, 1)   # cut bottom 1% noise
    global_vmax = all_nonzero.max()

    # ── Plot ────────────────────────────────────────────────────────
    cmap_name = 'inferno'
    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color='#1a1a2e')  # solid dark background for masked regions

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(r"Velocity Distribution Function $f(v_\perp, v_\parallel)$  (t=0)",
                 fontsize=16, fontweight='bold')

    for ax, label in [(axes[0], "Ions"), (axes[1], "Electrons")]:
        h, xedges, yedges = species_data[label]

        # Mask bins below threshold to eliminate visual noise
        h_masked = np.ma.masked_where(h < global_vmin, h)

        ax.set_facecolor('#1a1a2e')
        im = ax.pcolormesh(
            xedges, yedges, h_masked.T,
            cmap=cmap,
            norm=LogNorm(vmin=global_vmin, vmax=global_vmax),
            rasterized=True, shading='flat'
        )
        plt.colorbar(im, ax=ax, label=r'$f(v_\perp, v_\parallel)$', pad=0.02)

        ax.set_xlabel(r'$v_\parallel$ ($p_z$)', fontsize=13)
        ax.set_ylabel(r'$v_\perp$ ($\sqrt{p_x^2 + p_y^2}$)', fontsize=13)
        ax.set_title(label, fontsize=14)

        # Contour lines (on the valid data)
        x_centers = 0.5 * (xedges[:-1] + xedges[1:])
        y_centers = 0.5 * (yedges[:-1] + yedges[1:])
        valid_nonzero = h[h > 0]
        if len(valid_nonzero) > 0:
            levels = np.logspace(
                np.log10(max(valid_nonzero.min(), global_vmin)),
                np.log10(valid_nonzero.max()), 8
            )
            ax.contour(x_centers, y_centers, h_masked.T,
                       levels=levels, colors='white', linewidths=0.6, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(outdir, "vdf_2d.png")
    plt.savefig(path, dpi=DPI, facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 6: Summary statistics ──────────────────────────────────────
def print_summary(ions, electrons):
    """Print summary statistics of the particle data."""
    print("\n" + "="*60)
    print("PARTICLE DATA SUMMARY (t=0)")
    print("="*60)

    for name, sp in [("IONS", ions), ("ELECTRONS", electrons)]:
        print(f"\n  {name}:")
        print(f"    Count: {len(sp)}")
        print(f"    Mass (m): {sp['m'][0]:.4f}")
        print(f"    Charge (q): {sp['q'][0]:.4f}")
        print(f"    Weight (w): mean={sp['w'].mean():.6f}")
        print(f"    Position Y: [{sp['y'].min():.2f}, {sp['y'].max():.2f}]")
        print(f"    Position Z: [{sp['z'].min():.2f}, {sp['z'].max():.2f}]")
        for p in ['px', 'py', 'pz']:
            vals = sp[p]
            print(f"    {p}: mean={vals.mean():.6e}  std={vals.std():.6e}  "
                  f"min={vals.min():.6e}  max={vals.max():.6e}")

        # Temperature proxy: <p^2> / m for each direction
        m = np.abs(sp['m'][0])
        for p, label in [('px', 'T_perp_x'), ('py', 'T_perp_y'), ('pz', 'T_par')]:
            T = np.mean(sp[p]**2) / m
            print(f"    {label} ~ <{p}^2>/m = {T:.6e}")

        # Anisotropy
        T_perp = 0.5 * (np.mean(sp['px']**2) + np.mean(sp['py']**2)) / m
        T_par = np.mean(sp['pz']**2) / m
        print(f"    T_perp/T_par = {T_perp/T_par:.4f}")

    print("\n" + "="*60)


# ── Main ────────────────────────────────────────────────────────────
def main():
    # Determine input file
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = os.path.join(os.path.dirname(__file__),
                                "..", "build", "src", "prt.000000000.h5")

    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print("Usage: python plot_prt.py [path_to_prt_file.h5]")
        sys.exit(1)

    # Create output directory
    outdir = os.path.join(os.path.dirname(os.path.abspath(filepath)), OUTPUT_DIR)
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}\n")

    # Load data
    data, fields = load_particles(filepath)
    ions, electrons = separate_species(data)

    # Print summary
    print_summary(ions, electrons)

    # Generate plots
    print("\nGenerating plots...")
    plot_phase_space(ions, electrons, outdir)
    plot_momentum_distributions(ions, electrons, outdir)
    plot_kappa_comparison(ions, electrons, outdir)
    plot_spatial_energy(ions, electrons, outdir)
    plot_vdf(ions, electrons, outdir)

    print(f"\nAll plots saved to: {outdir}")


if __name__ == "__main__":
    main()
