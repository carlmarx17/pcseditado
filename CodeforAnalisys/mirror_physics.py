#!/usr/bin/env python3
"""
mirror_physics.py
=================
Visualizes the magnetic field and diamagnetic current for mirror instability.
Refactored to use data_reader.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path
from data_reader import PICDataReader

class MirrorPhysicsPlotter:
    def __init__(self, B0: float = 0.1, output_dir: str = "mirror_plots"):
        self.B0 = B0
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.switch_backend('Agg')

    def run(self):
        file_map = PICDataReader.find_files('pfd.*.h5')
        if not file_map:
            print("No .h5 files found in the current directory.")
            return
            
        print(f"Found {len(file_map)} timesteps. Processing...")
        for step, filename in file_map.items():
            self.process_file(step, filename)

    def process_file(self, step: int, filename: str):
        try:
            fields = PICDataReader.read_multiple_fields_3d(
                filename, 'jeh-', ['hz_fc/p0/3d', 'jx_ec/p0/3d']
            )
            bz3d = fields['hz_fc/p0/3d']
            jx3d = fields['jx_ec/p0/3d']
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return

        bz2d = PICDataReader.flatten_2d_slice(bz3d) / self.B0
        jx2d = PICDataReader.flatten_2d_slice(jx3d)

        bz2d = gaussian_filter(bz2d, sigma=1.0)
        jx2d = gaussian_filter(jx2d, sigma=1.5)

        self._generate_plot(bz2d, jx2d, step)

    def _generate_plot(self, bz2d: np.ndarray, jx2d: np.ndarray, step: int):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        im1 = ax1.imshow(bz2d.T, origin='lower', cmap='viridis', vmin=0.5, vmax=1.5, aspect='auto')
        ax1.set_title(r'Magnetic Field $B_z / B_0$ (Mirror Holes)', fontsize=14)
        ax1.set_xlabel('Z Axis (Parallel to B0)', fontsize=12)
        ax1.set_ylabel('Y Axis', fontsize=12)
        fig.colorbar(im1, ax=ax1, label=r'$B_z / B_0$')

        limit = np.percentile(np.abs(jx2d), 99)
        limit = limit if limit > 0 else 1e-5
        
        im2 = ax2.imshow(jx2d.T, origin='lower', cmap='seismic', vmin=-limit, vmax=limit, aspect='auto')
        ax2.set_title(r'Diamagnetic Current $J_x$', fontsize=14)
        ax2.set_xlabel('Z Axis (Parallel to B0)', fontsize=12)
        ax2.set_ylabel('Y Axis', fontsize=12)
        fig.colorbar(im2, ax=ax2, label=r'Current Density $J_x$')

        plt.tight_layout()
        outname = self.output_dir / f"mirror_physics_step{step}.png"
        plt.savefig(outname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[✓] Generated: {outname}")

if __name__ == '__main__':
    plotter = MirrorPhysicsPlotter(B0=0.1)
    plotter.run()
