#!/usr/bin/env python3
"""
anisotropy_analysis.py
======================
Plasma Anisotropy Analyzer.
Genera Brazil plots (Anisotropía vs Beta paralelo).

Umbrales de inestabilidad implementados (bi-Maxwelliana):
  Mirror:       T⊥/T‖ = 1 + 1/β‖      (Hasegawa 1969)
  Firehose:     T⊥/T‖ = 1 - 1/β‖      (Alfvén wave, Gary 1993)
  Ion-cyclotron: T⊥/T‖ ≈ 1 + 0.43/β‖^0.42  (Gary et al. 1994)

Unidades:
  Espacial  →  dᵢ = c/ωₚᵢ = √(mᵢ/n₀)
  Temporal  →  Ωcᵢ = qB₀/mᵢ
  Velocidad →  vA = B₀/√(n₀mᵢ)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import warnings
from data_reader import PICDataReader

# ── Parámetros de la simulación (consistente con psc_temp_aniso.cxx) ─────────────
MASS_RATIO = 64.0   # mi/me (masa artificial)
KAPPA      = 3.0    # parámetro kappa de la distribución iónica

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
        """
        NOTAS IMPORTANTES sobre las unidades de PSC:
        - 'txx_i/p0/3d' es el TENSOR DE PRESIÓN P_xx = <m vx vx * n> (no temperatura)
          En PSC el momento de segundo orden almacenado es: tij = n * m * <vi vj>
          Por eso NO se vuelve a multiplicar por n para calcular beta.
        - Los campos magnéticos se almacenan normalizados. El valor físico es: B = B_stored * B0
        - Para beta correctamente calculado se usa |B|^2 = Bx^2 + By^2 + Bz^2
        """
        try:
            moments = PICDataReader.read_multiple_fields_3d(
                mom_file, 'all_1st', ['txx_i/p0/3d', 'tyy_i/p0/3d', 'tzz_i/p0/3d', 'rho_i/p0/3d']
            )
            b_fields = PICDataReader.read_multiple_fields_3d(
                bz_file, 'jeh-', ['hx_fc/p0/3d', 'hy_fc/p0/3d', 'hz_fc/p0/3d']
            )
        except Exception as e:
            print(f"Error reading files {mom_file}, {bz_file}: {e}")
            return None

        # txx_i es P_xx = n * m * <vx vx>  →  temperatura: T_xx = P_xx / n
        # El factor T0 y n0 escalan desde las unidades de PSC a físicas.
        Pxx = moments['txx_i/p0/3d'].ravel()  # presión (adimensional PIC)
        Pyy = moments['tyy_i/p0/3d'].ravel()
        Pzz = moments['tzz_i/p0/3d'].ravel()
        n   = moments['rho_i/p0/3d'].ravel()

        # B total (física): B_stored es normalizado a B0 en los archivos HDF5
        Bx = b_fields['hx_fc/p0/3d'].ravel() * self.B0_ref
        By = b_fields['hy_fc/p0/3d'].ravel() * self.B0_ref
        Bz = b_fields['hz_fc/p0/3d'].ravel() * self.B0_ref
        B2 = Bx**2 + By**2 + Bz**2  # |B|^2 total

        # Temperatura: T = P / n  (presión / densidad)
        safe_n = np.where(n > 1e-10, n, np.nan)
        Tpar  = Pzz / safe_n  # T_paralelo (paralelo a B0 = Z)
        Tperp = 0.5 * (Pxx + Pyy) / safe_n  # T_perpendicular (avg de X, Y)

        anisotropy = Tperp / (Tpar + 1e-30)

        # β_‖ = n k_B T_‖ / (B²/2μ₀) = P_‖ / (B²/2μ₀)
        # En PIC las unidades son tales que μ₀ = 1
        Ppar_phys = Pzz  # ya es la presión paralela (P = n*T)
        Pmag = B2 / (2.0 * self.mu0)
        beta_par = Ppar_phys / (Pmag + 1e-30)

        mask = (Pzz   > 0) & (Pyy > 0) & (n > 1e-10) & \
               (B2    > 1e-30) & np.isfinite(beta_par) & \
               np.isfinite(anisotropy) & (anisotropy > 0)

        return {
            'anisotropy': anisotropy[mask],
            'beta_par':   beta_par[mask]
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
        """
        Brazil plot: Anisotropía T⊥/T‖ vs β‖.

        Umbrales de inestabilidad (bi-Maxwelliana):
          Mirror:   T⊥/T‖ = 1 + 1/β‖           (Hasegawa 1969)
          Firehose: T⊥/T‖ = 1 - 1/β‖           (instabilidad de onda Alfvén paralela)
          Ion-Cyclotron: T⊥/T‖ = 1 + 0.43/β‖⁰·⁴² (Gary et al. 1994, approx)

        NOTA: El umbral de firehose correcto para onda Alfvén paralela es  1 - 1/β‖
              (no 1 - 2/β‖ que corresponde a la inestabilidad de onda magnetosónica).
        """
        import matplotlib.colors as mcolors
        
        # ── Setup Figure (White / Publication style) ───────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Sub-sampling
        size  = len(self.all_beta_par)
        n_samples = 200_000 # Max number of points to plot
        n_sub = min(n_samples, size)
        idx   = np.random.choice(size, n_sub, replace=False)

        beta_sub  = self.all_beta_par[idx]
        aniso_sub = self.all_anisotropy[idx]

        # 2D Histogram (much better than scatter for large N)
        valid = (beta_sub > 0) & (aniso_sub > 0) & \
                np.isfinite(beta_sub) & np.isfinite(aniso_sub)
        
        # Use reversed magma or viridis for white background
        h = ax.hist2d(beta_sub[valid], aniso_sub[valid],
                      bins=120, cmap='viridis',
                      norm=mcolors.LogNorm(vmin=1),
                      cmin=1,
                      range=[[1e-1, 1e4], [1e-1, 1e1]])
        
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label('Point Density [a.u.]', fontsize=12)

        # ── Instability Thresholds ─────────────────────────────────
        b = np.logspace(-1, 4, 500)

        # Mirror: T_perp/T_par > 1 + 1/beta_par  (Hasegawa 1969)
        ax.plot(b, 1 + 1.0/b, '--', color='#cc0000', linewidth=2.2, zorder=5,
                label=r'Mirror  $T_\perp/T_\parallel = 1 + 1/\beta_\parallel$')

        # Firehose (Alfvén wave): T_perp/T_par < 1 - 1/beta_par
        b_fh = b[b > 1.0]
        ax.plot(b_fh, 1 - 1.0/b_fh, '--', color='#0055cc', linewidth=2.2, zorder=5,
                label=r'Firehose  $T_\perp/T_\parallel = 1 - 1/\beta_\parallel$')

        # Ion-cyclotron (approx Gary et al. 1994):
        ax.plot(b, 1 + 0.43 / b**0.42, ':', color='#008822', linewidth=2.2, zorder=5,
                label=r'Ion-cyclotron  $\approx 1 + 0.43/\beta_\parallel^{0.42}$')

        # Isotropy line
        ax.axhline(1.0, color='black', alpha=0.35, linewidth=1.0, linestyle=':')

        # Simulation initial operating point from psc_temp_aniso.cxx:
        # T_perp_i = 0.175,  T_par_i = 0.05,  B0 = 0.1,  n0 = 1.0
        # β_i‖ = n0 * T_par / (B0^2 / 2) = 1.0 * 0.05 / 0.005 = 10
        # T_perp/T_par = 0.175/0.05 = 3.5
        bi_par = (1.0 * 0.05) / (0.1**2 / 2.0)  # = 10.0
        ai_0   = 0.175 / 0.05                     # = 3.5
        ax.plot(bi_par, ai_0, '*', color='#ffaa00', markeredgecolor='black',
                markersize=20, zorder=10,
                label=rf'Init  ($\beta_{{i\parallel}}={bi_par:.0f},\; T_\perp/T_\parallel={ai_0:.1f}$)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-1, 1e4)
        ax.set_ylim(1e-1, 1e1)

        ax.set_xlabel(r'$\beta_{\parallel}$  (Ion Parallel Pressure / Magnetic Pressure)',
                      fontsize=13)
        ax.set_ylabel(r'$T_\perp / T_\parallel$', fontsize=13)
        ax.set_title(
            r'Brazil Plot: Anisotropy vs $\beta_\parallel$' + '\n' +
            fr'PSC  ($m_i/m_e = {int(MASS_RATIO)}$, $\kappa = {KAPPA}$)',
            fontsize=15, fontweight='bold'
        )

        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.grid(True, alpha=0.3, color='gray', linestyle=':')

        # Shaded regions
        b_fill = np.logspace(-1, 2, 300)
        ax.fill_between(b_fill, 1 + 1.0/b_fill, 1e1,
                        alpha=0.08, color='#cc0000')  # Mirror zone
                        
        b_fh2 = b_fill[b_fill > 1.0]
        ax.fill_between(b_fh2, 1e-1, 1 - 1.0/b_fh2,
                        alpha=0.08, color='#0055cc')  # Firehose zone

        ax.legend(fontsize=10.5, framealpha=0.85,
                  facecolor='white', edgecolor='#cccccc')

        # Meta-information (bottom right)
        n_snaps = len(self.snapshots) if self.snapshots else '?'
        ax.text(0.99, 0.02,
                f'{n_sub:,} points  |  {n_snaps} snapshots',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, color='#666666')

        if save:
            out_file = self.outdir / "brazil_plot_anisotropy.png"
            plt.savefig(out_file, dpi=200, bbox_inches='tight', facecolor='white')
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
