#!/usr/bin/env python3
"""
plot_vdf_3d.py
==============
Genera una superficie 3D de la VDF: f(v_parallel, v_perp).
Muestra la forma topográfica de la distribución para visualizar
las colas de Kappa y la anisotropía.

Un colorbar en el lateral indica la escala de densidad.
"""

import sys
import os
import adios2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LightSource
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter

# --- Configuración ---
OUTPUT_DIR = "fancy_vdf_plots"
DPI = 200
NBINS = 150


def plot_vdf_3d(species_name, pz, p_perp, out_path, cmap_name='magma'):
    """Representa la VDF 2D como superficie 3D con colorbar."""
    print(f"  Generando superficie 3D VDF para {species_name}...")

    # 1. Histograma 2D
    perp_hi = np.percentile(p_perp, 99.5)
    par_lo, par_hi = np.percentile(pz, [0.5, 99.5])

    h, xedges, yedges = np.histogram2d(
        pz, p_perp,
        bins=[NBINS, NBINS // 2],
        range=[[par_lo, par_hi], [0, perp_hi]],
        density=True
    )

    # Suavizar para mejor renderizado de superficie
    h_smooth = gaussian_filter(h, sigma=1.5)

    # Grid de ploteo
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Escala logarítmica en color, altura lineal
    h_max = np.max(h_smooth)
    h_floor = h_max * 1e-4
    norm = LogNorm(vmin=h_floor, vmax=h_max)

    ls = LightSource(azdeg=135, altdeg=45)
    rgb = ls.shade(h_smooth, cmap=plt.get_cmap(cmap_name), norm=norm, vert_exag=0.15)

    ax.plot_surface(X, Y, h_smooth,
                    facecolors=rgb,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)

    # Proyección en el piso (contour relleno) — solo capas principales, sin ruido
    offset = -0.05 * h_max
    ax.contourf(X, Y, h_smooth,
                zdir='z', offset=offset,
                cmap=cmap_name, alpha=0.5, norm=norm, levels=20)

    ax.set_title(
        f"Velocity Distribution Function: {species_name}\n" + r"$f(v_\parallel, v_\perp)$",
        fontsize=16, fontweight='bold', pad=20
    )
    ax.set_xlabel(r'$v_\parallel$ ($p_z$)', fontsize=12, labelpad=10)
    ax.set_ylabel(r'$v_\perp$ ($\sqrt{p_x^2 + p_y^2}$)', fontsize=12, labelpad=10)
    ax.set_zlabel('Densidad f', fontsize=12, labelpad=10)

    ax.set_zlim(offset, h_max * 1.1)
    ax.view_init(elev=30, azim=-55)

    # Colorbar explícito mediante ScalarMappable
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_name)
    mappable.set_array(h_smooth)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.1, shrink=0.6, aspect=15)
    cbar.set_label(r'$f(v_\parallel, v_\perp)$  [densidad]', fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    [Output] Guardado: {os.path.basename(out_path)}")


def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = os.path.join(os.path.dirname(__file__), "..", "build", "src", "prt.000000000.h5")

    if not os.path.exists(filepath):
        print(f"ERROR: Archivo '{filepath}' no encontrado.")
        return

    outdir = os.path.join(os.path.dirname(os.path.abspath(filepath)), OUTPUT_DIR)
    os.makedirs(outdir, exist_ok=True)

    print(f"Cargando VDF desde {os.path.basename(filepath)}...")
    with h5py.File(filepath, 'r') as f:
        data = f['particles']['p0']['1d'][:]

    for name, q_cond in [("Iones", data['q'] > 0), ("Electrones", data['q'] < 0)]:
        sp = data[q_cond]
        p_perp = np.sqrt(sp['px']**2 + sp['py']**2)
        pz = sp['pz']
        out = os.path.join(outdir, f"{name.lower()}_vdf_3d.png")
        plot_vdf_3d(name, pz, p_perp, out)

    print(f"\nFinalizado. Gráficos en: {outdir}")


if __name__ == "__main__":
    main()
