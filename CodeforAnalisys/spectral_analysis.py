#!/usr/bin/env python3
"""
spectral_analysis.py
====================
Spectral Analysis of Magnetic Field Fluctuations.
Computes the power spectral density (PSD) of the magnetic field from PIC simulation HDF5 files.
Generates 1D isotropic energy spectra E(k) to study plasma turbulence.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fft import fft2, fftshift
import warnings

from data_reader import PICDataReader

warnings.filterwarnings('ignore')

matplotlib_backend = 'Agg'
plt.switch_backend(matplotlib_backend)

class SpectralAnalyzer:
    def __init__(self, dx: float=1.0, dy: float=1.0, B0_ref: float=1.0, outdir: str="spectral_plots"):
        """
        dx, dy : grid spacing in x and y
        B0_ref : reference magnetic field for normalization
        """
        self.dx = dx
        self.dy = dy
        self.B0_ref = B0_ref
        self.outdir = Path(outdir)
        self.outdir.mkdir(exist_ok=True)
    
    def process_snapshot(self, filepath: str, plane: str='xy', slice_idx: int=None) -> dict:
        """
        Reads magnetic field data, extracts a 2D slice, and computes the 2D power spectrum.
        """
        try:
            b_fields = PICDataReader.read_multiple_fields_3d(
                filepath, 'jeh-', ['hx_fc/p0/3d', 'hy_fc/p0/3d', 'hz_fc/p0/3d']
            )
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return None
        
        bx_3d = b_fields['hx_fc/p0/3d'] * self.B0_ref
        by_3d = b_fields['hy_fc/p0/3d'] * self.B0_ref
        bz_3d = b_fields['hz_fc/p0/3d'] * self.B0_ref
        
        shape = bx_3d.shape
        
        # Extract 2D slice
        if plane == 'xy':
            idx = slice_idx if slice_idx is not None else shape[2] // 2
            bx = bx_3d[:, :, idx]
            by = by_3d[:, :, idx]
            bz = bz_3d[:, :, idx]
        elif plane == 'xz':
            idx = slice_idx if slice_idx is not None else shape[1] // 2
            bx = bx_3d[:, idx, :]
            by = by_3d[:, idx, :]
            bz = bz_3d[:, idx, :]
        elif plane == 'yz':
            idx = slice_idx if slice_idx is not None else shape[0] // 2
            bx = bx_3d[idx, :, :]
            by = by_3d[idx, :, :]
            bz = bz_3d[idx, :, :]
        else:
            raise ValueError(f"Unknown plane {plane}")
            
        return self.compute_power_spectrum(bx, by, bz)

    def compute_power_spectrum(self, bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> dict:
        """
        Computes the 2D power spectrum and the isotropic 1D spectrum E(k).
        """
        Nx, Ny = bx.shape
        # Remove mean (DC component)
        bx = bx - np.mean(bx)
        by = by - np.mean(by)
        bz = bz - np.mean(bz)
        
        # 2D FFT
        bx_k = fftshift(fft2(bx))
        by_k = fftshift(fft2(by))
        bz_k = fftshift(fft2(bz))
        
        # Power Spectral Density (PSD)
        psd_2d = (np.abs(bx_k)**2 + np.abs(by_k)**2 + np.abs(bz_k)**2) / (Nx * Ny)**2
        
        # Wavenumbers
        kx = fftshift(np.fft.fftfreq(Nx, d=self.dx)) * 2 * np.pi
        ky = fftshift(np.fft.fftfreq(Ny, d=self.dy)) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K = np.sqrt(KX**2 + KY**2)
        
        # Binning for 1D isotropic spectrum
        kx_pos = kx[kx > 0]
        ky_pos = ky[ky > 0]
        
        if len(kx_pos) == 0 or len(ky_pos) == 0:
            # Fallback for 1D-like slices
            k_min = np.max(K) / 100.0 if np.max(K) > 0 else 1e-1
        else:
            k_min = min(np.min(kx_pos), np.min(ky_pos))
            
        k_max = np.max(K)
        if k_max <= k_min:
             k_max = k_min * 10.0
        # Create bins
        num_bins = max(int(min(Nx, Ny) // 2), 10)
        bins = np.linspace(k_min, k_max, num_bins + 1)
        k_centers = 0.5 * (bins[1:] + bins[:-1])
        
        # Digitize
        bin_indices = np.digitize(K.flat, bins)
        
        E_k = np.zeros(num_bins)
        
        for i in range(1, num_bins + 1):
            mask = (bin_indices == i)
            if np.any(mask):
                # E(k) dk = sum_{shelL} PSD / dk
                # We roughly integrate the power in each circular shell.
                E_k[i-1] = np.sum(psd_2d.flat[mask])
                
        # Filter zero bins
        valid = (E_k > 0)
        k_centers = k_centers[valid]
        E_k = E_k[valid]
        
        return {
            'k': k_centers,
            'E_k': E_k,
            'psd_2d': psd_2d,
            'KX': KX,
            'KY': KY
        }
        
    def analyze_simulation(self, fields_pattern: str="pfd.*.h5", plane: str='xy', slice_idx: int=None, steps_to_process: list=None):
        print("Starting Spectral Analysis...")
        
        b_files = PICDataReader.find_files(fields_pattern)
        
        steps = sorted(list(b_files.keys()))
        if steps_to_process:
            steps = [s for s in steps if s in steps_to_process]
            
        if not steps:
            print("No snapshot files found.")
            return
            
        print(f"Found {len(steps)} snapshots to process.")
        
        for step in steps:
            print(f"Processing snapshot step {step}...")
            data = self.process_snapshot(b_files[step], plane=plane, slice_idx=slice_idx)
            if data is None:
                continue
            
            self.plot_1d_spectrum(data['k'], data['E_k'], step, plane)
            self.plot_2d_spectrum(data['KX'], data['KY'], data['psd_2d'], step, plane)
            
        print("Analysis completed.")

    def plot_1d_spectrum(self, k: np.ndarray, E_k: np.ndarray, step: int, plane: str, save: bool=True):
        plt.figure(figsize=(8, 6))
        
        plt.loglog(k, E_k, 'b-', label='Magnetic Energy $E(k)$', linewidth=2)
        
        # Add a reference slope -5/3 (Kolmogorov)
        if len(k) > 10:
            k_ref = k[len(k)//4:3*len(k)//4]
            E_ref = E_k[len(k)//4] * (k_ref / k_ref[0])**(-5/3)
            plt.loglog(k_ref, E_ref, 'r--', label='-5/3 slope')
            
        plt.xlabel(r'Wavenumber $k$')
        plt.ylabel(r'Energy Spectrum $E(k)$')
        plt.title(f'1D Magnetic Energy Spectrum - Step {step} ({plane} plane)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        
        if save:
            out_file = self.outdir / f"spectrum_1d_step{step}_{plane}.png"
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            print(f"Saved 1D Spectrum Plot: {out_file}")
        plt.close()

    def plot_2d_spectrum(self, KX: np.ndarray, KY: np.ndarray, psd_2d: np.ndarray, step: int, plane: str, save: bool=True):
        plt.figure(figsize=(8, 7))
        
        psd_log = np.log10(psd_2d + 1e-16)
        
        p = plt.pcolormesh(KX, KY, psd_log, shading='auto', cmap='inferno')
        cb = plt.colorbar(p)
        cb.set_label(r'$\log_{10}$(PSD)')
        
        plt.xlabel(r'$k_x$')
        plt.ylabel(r'$k_y$')
        plt.title(f'2D PSD spectrum - Step {step} ({plane} plane)')
        
        if save:
            out_file = self.outdir / f"spectrum_2d_step{step}_{plane}.png"
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            print(f"Saved 2D Spectrum Plot: {out_file}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral Analysis of Magnetic Field Fluctuations.")
    parser.add_argument("--fields", type=str, default="pfd.*.h5", help="Glob pattern for field files.")
    parser.add_argument("--plane", type=str, default="xy", choices=['xy', 'xz', 'yz'], help="Plane to extract slice.")
    parser.add_argument("--B0", type=float, default=1.0, help="Reference B0 field normalization.")
    parser.add_argument("--dx", type=float, default=1.0, help="Grid spacing in x-direction.")
    parser.add_argument("--dy", type=float, default=1.0, help="Grid spacing in y-direction.")
    parser.add_argument("--steps", nargs="*", type=int, default=None, help="Steps to process.")
    args = parser.parse_args()
    
    analyzer = SpectralAnalyzer(dx=args.dx, dy=args.dy, B0_ref=args.B0, outdir="spectral_plots")
    analyzer.analyze_simulation(fields_pattern=args.fields, plane=args.plane, steps_to_process=args.steps)
