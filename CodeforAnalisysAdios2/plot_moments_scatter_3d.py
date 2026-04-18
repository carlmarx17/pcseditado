#!/usr/bin/env python3
"""
plot_moments_scatter_3d.py
==========================
Genera histogramas 2D en espacio de momentos (Phase Space) y un
scatter 3D con corte de cuadrante para ver la estructura interna de la VDF.

Plots generados:
  - px vs py, px vs pz, py vs pz  (histogramas 2D LogNorm)
  - p_par vs p_perp               (half-space, asimetría paralela vs perp)
  - 3D scatter con corte de octal y capas de densidad (LayeredCutaway)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from particle_reader import build_structured_particles

# ── Estética clara para publicación ──────────────────────────────────
# Removemos plt.style.use('dark_background')
matplotlib.rcParams['axes.facecolor'] = 'white'
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['grid.color'] = '#cccccc'
matplotlib.rcParams['grid.linestyle'] = '--'

OUTPUT_DIR = "phase_space_plots"
DPI = 200


def load_particle_data(filepath):
    """Carga partículas desde un archivo ADIOS2 prt.*.bp."""
    return build_structured_particles(
        filepath,
        include_position=False,
        include_weight=False,
        include_mass=False,
        verbose=False,
    )


def set_light_3d_axes(ax):
    """Aplica estética light al panel 3D."""
    ax.set_facecolor('white')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.xaxis._axinfo["grid"].update({"color": "#dddddd", "linewidth": 0.5, "linestyle": "--"})
    ax.yaxis._axinfo["grid"].update({"color": "#dddddd", "linewidth": 0.5, "linestyle": "--"})
    ax.zaxis._axinfo["grid"].update({"color": "#dddddd", "linewidth": 0.5, "linestyle": "--"})


def plot_2d_hist(x, y, xlabel, ylabel, title, outpath, symmetric_x=True, symmetric_y=True):
    """2D Histogram with LogNorm, white background, and expanded limits."""
    # Use standard deviation to set physical axis limits (~6 * v_th)
    std_x = np.std(x)
    std_y = np.std(y)
    
    x_hi = 6.0 * std_x
    y_hi = 6.0 * std_y

    x_range = [-x_hi, x_hi] if symmetric_x else [0.0, x_hi]
    y_range = [-y_hi, y_hi] if symmetric_y else [0.0, y_hi]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 'viridis' is excellent for white backgrounds. Set empty bins to white.
    import matplotlib.cm as cm
    cmap = getattr(cm, 'viridis').copy() if hasattr(cm, 'viridis') else plt.get_cmap('viridis').copy()
    cmap.set_bad('white')

    h = ax.hist2d(x, y, bins=300, cmap=cmap, norm=LogNorm(),
                  range=[x_range, y_range])

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    
    ax.grid(True, linestyle=':', color='#dddddd')

    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Particle Count', fontsize=12)

    plt.tight_layout()
    plt.savefig(outpath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_3d_layered_cutaway(px, py, pz, title, outpath, subsample=300000):
    """
    3D scatter de la VDF con:
      - Submuestra para no saturar memoria
      - Corte de 1 octante (el que mira hacia la cámara) para ver el interior
      - Capas concéntricas según |p| para revelar la estructura esférica
      - Colormap 'turbo' coloreado por magnitud del vector p
    """
    # 1. Submuestreo aleatorio
    n_total = len(px)
    n = min(subsample, n_total)
    idx = np.random.choice(n_total, n, replace=False)
    sx, sy, sz = px[idx], py[idx], pz[idx]

    p_mag = np.sqrt(sx**2 + sy**2 + sz**2)

    # 2. Reference scale: expand to ~6 standard deviations for better tail visibility
    std_p = np.sqrt(np.var(px) + np.var(py) + np.var(pz)) / np.sqrt(3)
    max_p_display = 6.0 * std_p
    if max_p_display == 0:
        max_p_display = 1.0

    # 3. Octant cut: remove the quadrant (+x, +y, +z) facing the camera
    mask_visible = ~((sx > 0) & (sy > 0) & (sz > 0))
    sx, sy, sz, p_mag = sx[mask_visible], sy[mask_visible], sz[mask_visible], p_mag[mask_visible]

    # 4. Concentric shells — only particles near certain radii
    shells_mask = np.zeros(len(p_mag), dtype=bool)
    dr = 0.045 * max_p_display
    for ratio in [0.20, 0.45, 0.70, 0.92]:
        r_c = ratio * max_p_display
        shells_mask |= (p_mag >= r_c - dr) & (p_mag <= r_c + dr)

    sx = sx[shells_mask]
    sy = sy[shells_mask]
    sz = sz[shells_mask]
    p_mag = p_mag[shells_mask]

    if len(sx) == 0:
        print(f"  [WARN] No particles in shells for {title}")
        return

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    set_light_3d_axes(ax)

    sc = ax.scatter(sx, sy, sz,
                    c=p_mag, cmap='viridis',
                    alpha=0.35, s=2.0,
                    edgecolors='none',
                    vmin=0, vmax=max_p_display)

    lim = max_p_display * 1.1
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    ax.set_xlabel(r'$p_x$', fontsize=12, fontstyle='italic', labelpad=10)
    ax.set_ylabel(r'$p_y$', fontsize=12, fontstyle='italic', labelpad=10)
    ax.set_zlabel(r'$p_z$', fontsize=12, fontstyle='italic', labelpad=10)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.view_init(elev=20, azim=45)

    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label(r'Momentum Magnitude $|p|$', fontsize=12)

    plt.tight_layout()
    plt.savefig(outpath, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def compute_and_plot_vdf(data, species_name, outdir):
    print(f"\nGenerating phase space plots for {species_name} ({len(data):,} particles)...")

    px, py, pz = data['px'], data['py'], data['pz']

    p_par = pz
    p_perp = np.sqrt(px**2 + py**2)

    print(f"  -> px vs py")
    plot_2d_hist(px, py, r'$p_x$', r'$p_y$',
                 f'{species_name}: $p_x$ vs $p_y$',
                 os.path.join(outdir, f'{species_name.lower()}_px_py.png'),
                 symmetric_x=True, symmetric_y=True)

    print(f"  -> px vs pz")
    plot_2d_hist(px, pz, r'$p_x$', r'$p_z$',
                 f'{species_name}: $p_x$ vs $p_z$',
                 os.path.join(outdir, f'{species_name.lower()}_px_pz.png'),
                 symmetric_x=True, symmetric_y=True)

    print(f"  -> py vs pz")
    plot_2d_hist(py, pz, r'$p_y$', r'$p_z$',
                 f'{species_name}: $p_y$ vs $p_z$',
                 os.path.join(outdir, f'{species_name.lower()}_py_pz.png'),
                 symmetric_x=True, symmetric_y=True)

    print(f"  -> p_par vs p_perp")
    plot_2d_hist(p_par, p_perp,
                 r'$p_{\parallel}$', r'$p_{\perp}$',
                 f'{species_name}: ' + r'$p_{\parallel}$ vs $p_{\perp}$',
                 os.path.join(outdir, f'{species_name.lower()}_ppar_pperp.png'),
                 symmetric_x=True, symmetric_y=False)

    print(f"  -> 3D Layered Cutaway")
    plot_3d_layered_cutaway(
        px, py, pz,
        f'{species_name}: 3D VDF — Momentum Shells',
        os.path.join(outdir, f'{species_name.lower()}_3d_scatter.png'),
        subsample=300000
    )


def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = os.path.join(os.path.dirname(__file__), "..", "build", "src", "prt.000000000.bp")

    if not os.path.exists(filepath):
        print(f"ERROR: Archivo '{filepath}' no encontrado.")
        return

    outdir = os.path.join(os.path.dirname(os.path.abspath(filepath)), OUTPUT_DIR)
    os.makedirs(outdir, exist_ok=True)
    print(f"Directorio de salida: {outdir}")

    print(f"Cargando partículas desde {os.path.basename(filepath)}...")
    data = load_particle_data(filepath)

    ion_idx = data['q'] > 0
    elec_idx = data['q'] < 0

    compute_and_plot_vdf(data[ion_idx], "Iones", outdir)
    compute_and_plot_vdf(data[elec_idx], "Electrones", outdir)

    print(f"\nTodos los gráficos guardados en: {outdir}")


if __name__ == "__main__":
    main()
