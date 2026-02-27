#!/usr/bin/env python3
"""
anisotropy_analysis.py
======================
Plasma Anisotropy Analyzer.
Generates Brazil plots (Anisotropy vs Parallel Beta).
Refactored to use data_reader.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import warnings
from data_reader import PICDataReader

warnings.filterwarnings('ignore')

class PlasmaAnisotropyAnalyzer:
    def __init__(self, n0: float=1.0, T0: float=1.0, B0_ref: float=0.1, outdir: str="anisotropy_plots"):
        self.n0 = n0
        self.T0 = T0
        self.B0_ref = B0_ref
        self.kB = 1.0
        self.mu0 = 1.0
        self.outdir = Path(outdir)
        self.outdir.mkdir(exist_ok=True)
        self.reset_data()
        plt.switch_backend('Agg')
        
    def reset_data(self):
        self.all_anisotropy = []
        self.all_beta_par = []
        self.snapshots = []
    
    def process_snapshot(self, mom_file: str, bz_file: str) -> dict:
        try:
            moments = PICDataReader.read_multiple_fields_3d(
                mom_file, 'all_1st', ['txx_i/p0/3d', 'tyy_i/p0/3d', 'tzz_i/p0/3d', 'rho_i/p0/3d']
            )
            b_fields = PICDataReader.read_multiple_fields_3d(
                bz_file, 'jeh-', ['hz_fc/p0/3d']
            )
        except Exception as e:
            print(f"Error reading files {mom_file}, {bz_file}: {e}")
            return None
            
        Txx = moments['txx_i/p0/3d'] * self.T0
        Tyy = moments['tyy_i/p0/3d'] * self.T0
        Tzz = moments['tzz_i/p0/3d'] * self.T0
        n = moments['rho_i/p0/3d'] * self.n0
        Bz = b_fields['hz_fc/p0/3d'] * self.B0_ref
        
        Tpar = Tzz.ravel()
        Tperp = 0.5 * (Txx.ravel() + Tyy.ravel())
        n_flat = n.ravel()
        Bz_flat = Bz.ravel()
        
        anisotropy = Tperp / (Tpar + 1e-30)
        
        Ppar = n_flat * self.kB * Tpar
        Pmag = (Bz_flat**2) / (2.0 * self.mu0)
        beta_par = Ppar / (Pmag + 1e-30)
        
        mask = (Tpar > 0) & (Tperp > 0) & (n_flat > 0) & \
               (np.abs(Bz_flat) > 1e-30) & np.isfinite(beta_par) & \
               np.isfinite(anisotropy)
        
        return {
            'anisotropy': anisotropy[mask],
            'beta_par': beta_par[mask]
        }
    
    def analyze_simulation(self, mom_pattern: str="pfd_moments.*.h5", bz_pattern: str="pfd.*.h5"):
        print("Starting comprehensive plasma simulation analysis...")
        
        mom_files = PICDataReader.find_files(mom_pattern)
        bz_files = PICDataReader.find_files(bz_pattern)
        
        common_steps = sorted(set(mom_files.keys()).intersection(set(bz_files.keys())))
        
        if not common_steps:
            print("Missing matched field and moment files.")
            return
            
        print(f"Discovered {len(common_steps)} matching snapshots.")
        self.reset_data()
        
        for i, step in enumerate(common_steps):
            print(f"Processing snapshot step {step} ({i+1}/{len(common_steps)})...")
            data = self.process_snapshot(mom_files[step], bz_files[step])
            if data is None: continue
                
            self.all_anisotropy.extend(data['anisotropy'])
            self.all_beta_par.extend(data['beta_par'])
            self.snapshots.append(step)
                
        self.all_anisotropy = np.array(self.all_anisotropy)
        self.all_beta_par = np.array(self.all_beta_par)
        print("Analysis completed.")

    def plot_brazil_plot(self, save=True):
        plt.figure(figsize=(10, 7))
        if len(self.all_beta_par) == 0:
            print("No data available to plot.")
            return
        
        size = len(self.all_beta_par)
        sub_sample = 50000 if size > 50000 else size
        indices = np.random.choice(size, sub_sample, replace=False)
        
        scatter_beta = self.all_beta_par[indices]
        scatter_aniso = self.all_anisotropy[indices]
        
        plt.hist2d(scatter_beta, scatter_aniso, bins=100, cmap='viridis', norm=plt.cm.colors.LogNorm())
        cbar = plt.colorbar()
        cbar.set_label('Point Density')
        
        beta_range = np.logspace(-1, 3, 100)
        plt.plot(beta_range, 1 + 1 / beta_range, color='red', linestyle='--', linewidth=2, label=r'Mirror Threshold')
        
        beta_range_fh = np.logspace(np.log10(2.1), 3, 100)
        plt.plot(beta_range_fh, 1 - 2 / beta_range_fh, color='orange', linestyle='--', linewidth=2, label=r'Firehose Threshold')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e-1, 1e2)
        plt.ylim(1e-1, 1e1)
        plt.axhline(1, color='white', alpha=0.5, linestyle=':')
        
        plt.xlabel(r'$\beta_\parallel$')
        plt.ylabel(r'$T_\perp / T_\parallel$')
        plt.title('Brazil Plot: Anisotropy vs $\beta_\parallel$')
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        if save:
            out_file = self.outdir / "brazil_plot_anisotropy.png"
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            print(f"Saved Brazil Plot: {out_file}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Plasma Anisotropy and Generate Brazil Plots.")
    parser.add_argument("--moments", type=str, default="pfd_moments.*.h5", help="Glob pattern for moment files.")
    parser.add_argument("--fields", type=str, default="pfd.*.h5", help="Glob pattern for field files.")
    parser.add_argument("--B0", type=float, default=0.1, help="Reference B0 field normalization.")
    args = parser.parse_args()
    
    analyzer = PlasmaAnisotropyAnalyzer(B0_ref=args.B0, outdir="anisotropy_plots")
    analyzer.analyze_simulation(mom_pattern=args.moments, bz_pattern=args.fields)
    analyzer.plot_brazil_plot(save=True)
