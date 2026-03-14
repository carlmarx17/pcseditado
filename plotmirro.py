#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from scipy.ndimage import gaussian_filter

def plot_mirror_physics(B0=0.1):
    # Buscar todos los archivos pfd disponibles
    files = sorted(glob.glob('pfd.*.h5'))
    if not files:
        print("No se encontraron archivos .h5")
        return

    print(f"Encontrados {len(files)} pasos temporales. Procesando...")

    for filename in files:
        step = re.search(r"\.(\d+)_", filename).group(1)
        
        try:
            with h5py.File(filename, 'r') as f:
                # Buscar el grupo base
                jeh_groups = [k for k in f.keys() if k.startswith('jeh-')]
                if not jeh_groups: continue
                grp = f[jeh_groups[0]]
                
                # Extraer Campo Z y Corriente X directamente (sabemos que existen por el explorador)
                bz3d = grp['hz_fc/p0/3d'][()]
                jx3d = grp['jx_ec/p0/3d'][()]
                
        except Exception as e:
            print(f"Error leyendo {filename}: {e}")
            continue

        # La forma es (512, 384, 1). Tomamos el índice 0 del último eje para aplanar a 2D.
        bz2d = bz3d[:, :, 0]
        jx2d = jx3d[:, :, 0]

        # Normalizar el campo magnético al campo de fondo
        bz2d = bz2d / B0
        
        # Suavizar levemente para quitar el ruido PIC de las partículas
        bz2d = gaussian_filter(bz2d, sigma=1.0)
        jx2d = gaussian_filter(jx2d, sigma=1.5)

        # === GRAFICAR ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Panel 1: El Campo Magnético (Los Agujeros)
        # Usamos vmin y vmax centrados en 1.0 para ver claramente dónde cae el campo
        im1 = ax1.imshow(bz2d, origin='lower', cmap='viridis', vmin=0.5, vmax=1.5, aspect='auto')
        ax1.set_title(r'Campo Magnético $B_z / B_0$ (Agujeros Mirror)', fontsize=14)
        ax1.set_xlabel('Eje Y', fontsize=12)
        ax1.set_ylabel('Eje Z (Paralelo a B0)', fontsize=12)
        fig.colorbar(im1, ax=ax1, label=r'$B_z / B_0$')

        # Panel 2: La Corriente (Las Paredes de los agujeros)
        # Usamos un mapa divergente centrado en 0 para ver corrientes positivas y negativas
        limit = np.percentile(np.abs(jx2d), 99) # Ajuste dinámico de escala
        im2 = ax2.imshow(jx2d, origin='lower', cmap='seismic', vmin=-limit, vmax=limit, aspect='auto')
        ax2.set_title(r'Corriente Diamagnética $J_x$', fontsize=14)
        ax2.set_xlabel('Eje Y', fontsize=12)
        fig.colorbar(im2, ax=ax2, label=r'Densidad de Corriente $J_x$')

        plt.tight_layout()
        outname = f"mirror_physics_step{step}.png"
        plt.savefig(outname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Generado: {outname}")

if __name__ == '__main__':
    # Usar backend de clúster
    plt.switch_backend('Agg') 
    plot_mirror_physics(B0=0.1)
