#!/usr/bin/env python3
"""
heat_flux_analysis.py
=====================
Ion Heat Flux Analysis for PIC Simulations (PSC).

Generates:
  - 2D maps of parallel & perpendicular heat flux components.
  - Time evolution of spatially-averaged heat flux.

The ion heat flux is computed from the third-order velocity moment:
  q_parallel = 0.5 * n * m * <(v_par - <v_par>)^3>  (simplified diagnostic)
  
In practice, we approximate q_par from the moment data:
  q_par ≈ (p_z - n * <v_z>) * v_th_par   (energy flux along B0)

For a more direct diagnostic, we use the pressure-velocity correlations:
  q_x = sum_j P_xj * v_j  → perpendicular heat flux
  q_z = sum_j P_zj * v_j  → parallel heat flux

PSC normalised units:
  - Pressure: P_ij = n m <v_i v_j>  (PIC)
  - Heat flux: q = n m <v_i v_j v_k>

References:
  - Gary et al. (1998) - Heat flux constraints in the solar wind
  - Bale et al. (2013) - Electron heat flux in the solar wind
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    def gaussian_filter(values, sigma=0):
        if sigma:
            raise RuntimeError(
                "SciPy is required when --sigma is greater than zero."
            )
        return values

from data_reader import PICDataReader
from psc_units import (
    B0,
    DI,
    DOMAIN_DI_Y,
    DOMAIN_DI_Z,
    DT_CODE,
    FIELD_FILE_PATTERN,
    INSTABILITY,
    KAPPA,
    MASS_RATIO,
    M_ION,
    MOMENT_FILE_PATTERN,
    N_GRID_Y,
    N_GRID_Z,
    OMEGA_CI,
    PROFILE_LABEL,
    TI_PAR,
    TI_PERP,
    VA,
    step_to_omegaci,
)

plt.switch_backend("Agg")

# ── Dark-theme colour palette ────────────────────────────────────────────────
DARK_BG  = "#0c0e14"
PANEL_BG = "#12151f"
TEXT_CLR  = "#dde2f0"
GRID_CLR = "#232840"


class HeatFluxAnalyzer:
    """Compute and visualise ion heat flux from PSC PIC outputs."""

    def __init__(
        self,
        moment_pattern: str = MOMENT_FILE_PATTERN,
        field_pattern: str = FIELD_FILE_PATTERN,
        sigma: float = 3.0,
        outdir: str = "heatflux_plots",
    ):
        self.moment_pattern = moment_pattern
        self.field_pattern = field_pattern
        self.sigma = sigma
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def compute_heat_flux(self, mom_file: str, fld_file: str) -> dict:
        """
        Compute proxy heat flux from moment data.

        For ions, the heat flux proxy is:
          q_perp = P_perp * v_perp  (perpendicular energy transport)
          q_par  = P_par  * v_par   (parallel energy transport)
        
        where v_i = p_i / (n * m_i) is the bulk velocity.
        """
        moments = PICDataReader.read_multiple_fields_3d(
            mom_file, "all_1st",
            [
                "txx_i/p0/3d", "tyy_i/p0/3d", "tzz_i/p0/3d",
                "txy_i/p0/3d", "tyz_i/p0/3d", "tzx_i/p0/3d",
                "px_i/p0/3d", "py_i/p0/3d", "pz_i/p0/3d",
                "rho_i/p0/3d",
            ],
        )
        fields = PICDataReader.read_multiple_fields_3d(
            fld_file, "jeh-",
            ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"],
        )

        # Flatten to 2D
        flat = lambda key: PICDataReader.flatten_2d_slice(moments[key])
        flat_f = lambda key: PICDataReader.flatten_2d_slice(fields[key])

        n_i = flat("rho_i/p0/3d").astype(float)
        Mxx = flat("txx_i/p0/3d").astype(float)
        Myy = flat("tyy_i/p0/3d").astype(float)
        Mzz = flat("tzz_i/p0/3d").astype(float)
        Mxy = flat("txy_i/p0/3d").astype(float)
        Myz = flat("tyz_i/p0/3d").astype(float)
        Mzx = flat("tzx_i/p0/3d").astype(float)

        px = flat("px_i/p0/3d").astype(float)
        py = flat("py_i/p0/3d").astype(float)
        pz = flat("pz_i/p0/3d").astype(float)

        Bx = flat_f("hx_fc/p0/3d").astype(float)
        By = flat_f("hy_fc/p0/3d").astype(float)
        Bz = flat_f("hz_fc/p0/3d").astype(float)

        # Apply smoothing
        sm = lambda arr: gaussian_filter(arr, sigma=self.sigma)
        n_i = sm(n_i)
        Mxx, Myy, Mzz = sm(Mxx), sm(Myy), sm(Mzz)
        Mxy, Myz, Mzx = sm(Mxy), sm(Myz), sm(Mzx)
        px, py, pz = sm(px), sm(py), sm(pz)
        Bx, By, Bz = sm(Bx), sm(By), sm(Bz)

        # Bulk velocity: v = p / (n * m_i)  
        # (px_i in PSC = n * m_i * v_x for ions)
        safe_n = np.where(n_i > 1e-10, n_i, np.nan)
        vx = px / (safe_n * M_ION)
        vy = py / (safe_n * M_ION)
        vz = pz / (safe_n * M_ION)

        # PSC writes raw second moments. Remove the bulk-flow contribution
        # before interpreting the tensor as thermal pressure.
        Pxx = Mxx - px * px / (safe_n * M_ION)
        Pyy = Myy - py * py / (safe_n * M_ION)
        Pzz = Mzz - pz * pz / (safe_n * M_ION)
        Pxy = Mxy - px * py / (safe_n * M_ION)
        Pyz = Myz - py * pz / (safe_n * M_ION)
        Pzx = Mzx - pz * px / (safe_n * M_ION)

        # B-field unit vector
        B_mag = np.sqrt(Bx**2 + By**2 + Bz**2 + 1e-40)
        bhat_x = Bx / B_mag
        bhat_y = By / B_mag
        bhat_z = Bz / B_mag

        # Parallel velocity: v_par = v · b_hat
        v_par = vx * bhat_x + vy * bhat_y + vz * bhat_z

        # Pressure tensor: P_ij * b_hat_j gives force along B
        # Heat flux proxy along B: q_par = sum_ij P_ij * b_hat_i * v_j
        # Simplified: q_par ≈ P_par * v_par where P_par = b·P·b
        P_par = (Pxx * bhat_x**2 + Pyy * bhat_y**2 + Pzz * bhat_z**2
                 + 2 * Pxy * bhat_x * bhat_y
                 + 2 * Pyz * bhat_y * bhat_z
                 + 2 * Pzx * bhat_z * bhat_x)
        
        P_perp = 0.5 * (Pxx + Pyy + Pzz - P_par)

        # Temperature
        T_par = P_par / safe_n
        T_perp = P_perp / safe_n

        # Heat flux proxy: energy transport rate
        # q_par = P_par * v_par (parallel energy flux density)
        q_par = P_par * v_par
        # q_perp uses the perpendicular velocity magnitude
        v_perp_x = vx - v_par * bhat_x
        v_perp_y = vy - v_par * bhat_y
        v_perp_z = vz - v_par * bhat_z
        v_perp_mag = np.sqrt(v_perp_x**2 + v_perp_y**2 + v_perp_z**2)
        q_perp = P_perp * v_perp_mag

        # Anisotropy 
        anisotropy = T_perp / (T_par + 1e-30)

        # beta_par = P_par / P_mag
        P_mag = B_mag**2 / 2.0
        beta_par = P_par / (P_mag + 1e-30)

        return {
            "q_par": q_par,
            "q_perp": q_perp,
            "P_par": P_par,
            "P_perp": P_perp,
            "T_par": T_par,
            "T_perp": T_perp,
            "anisotropy": anisotropy,
            "beta_par": beta_par,
            "B_mag": B_mag,
            "v_par": v_par,
        }

    def analyze_time_evolution(self):
        """Compute spatially-averaged heat flux and anisotropy vs time."""
        moment_files = PICDataReader.find_files(self.moment_pattern)
        field_files = PICDataReader.find_files(self.field_pattern)
        common_steps = sorted(set(moment_files) & set(field_files))

        if not common_steps:
            print("[ERROR] No matching moment/field file pairs found.")
            return

        print(f"Processing {len(common_steps)} snapshots for time evolution...")

        times = []
        avg_q_par = []
        avg_q_perp = []
        avg_aniso = []
        avg_beta_par = []
        avg_T_par = []
        avg_T_perp = []

        for i, step in enumerate(common_steps):
            if i % 10 == 0:
                print(f"  Step {step} ({i+1}/{len(common_steps)})...")
            data = self.compute_heat_flux(moment_files[step], field_files[step])
            if data is None:
                continue

            t = step_to_omegaci(step)
            times.append(t)
            avg_q_par.append(np.nanmean(np.abs(data["q_par"])))
            avg_q_perp.append(np.nanmean(np.abs(data["q_perp"])))
            mean_p_par = np.nanmean(data["P_par"])
            mean_p_perp = np.nanmean(data["P_perp"])
            avg_aniso.append(mean_p_perp / mean_p_par)
            avg_beta_par.append(
                2.0 * mean_p_par / np.nanmean(data["B_mag"] ** 2)
            )
            avg_T_par.append(np.nanmean(data["T_par"]))
            avg_T_perp.append(np.nanmean(data["T_perp"]))

        times = np.array(times)
        avg_q_par = np.array(avg_q_par)
        avg_q_perp = np.array(avg_q_perp)
        avg_aniso = np.array(avg_aniso)
        avg_beta_par = np.array(avg_beta_par)
        avg_T_par = np.array(avg_T_par)
        avg_T_perp = np.array(avg_T_perp)

        self._plot_time_evolution(
            times, avg_q_par, avg_q_perp, avg_aniso,
            avg_beta_par, avg_T_par, avg_T_perp
        )

    def _plot_time_evolution(
        self, times, q_par, q_perp, aniso, beta_par, T_par, T_perp
    ):
        """Generate a 4-panel time evolution figure."""
        fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
        fig.patch.set_facecolor(DARK_BG)

        for ax in axes:
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_CLR, direction="in", which="both", top=True, right=True)
            ax.grid(True, alpha=0.18, color=GRID_CLR, linestyle=":")
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_CLR)

        t_arr = np.asarray(times)

        # ── Panel 1: Anisotropy ──────────────────────────────────────────
        A0 = TI_PERP / TI_PAR  # initial anisotropy
        beta_arr = np.maximum(np.asarray(beta_par), 1e-12)
        if INSTABILITY == "firehose":
            threshold = 1.0 - 2.0 / beta_arr
            threshold_label = r"Firehose threshold $1-2/\beta_\parallel$"
        elif INSTABILITY == "mirror":
            threshold = 1.0 + 1.0 / beta_arr
            threshold_label = r"Mirror threshold $1+1/\beta_\parallel$"
        else:
            threshold = 1.0 + 0.21 / beta_arr**0.6
            threshold_label = r"Whistler threshold $1+0.21/\beta_\parallel^{0.6}$"
        axes[0].plot(t_arr, aniso, color="#ff6666", linewidth=1.8,
                     label=r"$\langle P_\perp\rangle/\langle P_\parallel\rangle_i$")
        axes[0].plot(t_arr, threshold, color="#ff9999", linewidth=1.0,
                     linestyle="--", alpha=0.7, label=threshold_label)
        axes[0].axhline(1.0, color=TEXT_CLR, alpha=0.3, linestyle=":", linewidth=0.8)
        axes[0].axhline(A0, color="#ffd700", alpha=0.45, linestyle="--", linewidth=1.0,
                        label=rf"$A_0 = {A0:.1f}$")
        axes[0].set_ylabel(r"$\langle T_\perp / T_\parallel \rangle_i$", color=TEXT_CLR, fontsize=12)
        finite_a = np.concatenate([
            np.asarray(aniso, dtype=float),
            np.asarray(threshold, dtype=float),
            np.array([1.0, A0]),
        ])
        finite_a = finite_a[np.isfinite(finite_a)]
        axes[0].set_ylim(
            max(0.02, float(np.min(finite_a)) * 0.85),
            max(1.1, float(np.max(finite_a)) * 1.15),
        )
        axes[0].legend(fontsize=9, facecolor="#1a1f30", edgecolor="#3a3f55",
                       labelcolor=TEXT_CLR, loc="upper right", ncol=2)
        axes[0].set_title(
            f"Ion Heat Flux & Anisotropy Evolution — {PROFILE_LABEL}\n"
            rf"PSC ($m_i/m_e = {int(MASS_RATIO)}$"
            + (rf", $\kappa = {KAPPA}$" if KAPPA is not None else ", Maxwellian")
            + ")",
            fontsize=14, fontweight="bold", color=TEXT_CLR, pad=10,
        )

        # ── Panel 2: Temperature normalized to T₀ ───────────────────────────
        # Normalize by initial temperature for clearer display
        T_par_norm  = np.asarray(T_par)  / TI_PAR
        T_perp_norm = np.asarray(T_perp) / TI_PERP
        axes[1].plot(t_arr, T_par_norm,  color="#66aaff", linewidth=1.8,
                     label=r"$\langle T_\parallel \rangle_i / T_{\parallel,0}$")
        axes[1].plot(t_arr, T_perp_norm, color="#ff6666", linewidth=1.8,
                     label=r"$\langle T_\perp \rangle_i / T_{\perp,0}$")
        axes[1].axhline(1.0, color=TEXT_CLR, alpha=0.25, linestyle=":", linewidth=0.8)
        axes[1].set_ylabel(r"$T_i / T_{i,0}$ (normalized)", color=TEXT_CLR, fontsize=12)
        axes[1].legend(fontsize=10, facecolor="#1a1f30", edgecolor="#3a3f55", labelcolor=TEXT_CLR)
        # Annotate heating rate
        if len(T_par_norm) > 2:
            axes[1].text(0.01, 0.92, rf"Heating: $T_\parallel/T_{{\parallel,0}}$ grows {T_par_norm[-1]:.0f}×",
                         transform=axes[1].transAxes, fontsize=9, color="#aab0cc")

        # ── Panel 3: Parallel beta (log scale) ────────────────────────────
        axes[2].plot(t_arr, beta_par, color="#44ffaa", linewidth=1.8,
                     label=r"$\langle \beta_\parallel \rangle_i$")
        # Mirror threshold: beta needs to stay < A-1 for stability
        axes[2].axhline(float(BETA_I_PAR := TI_PAR * 2 / B0**2),
                        color="#ffd700", linestyle="--", alpha=0.4, linewidth=1.0,
                        label=rf"$\beta_{{\parallel,0}} = {TI_PAR*2/B0**2:.0f}$")
        axes[2].set_ylabel(r"$\langle \beta_{\parallel,i} \rangle$", color=TEXT_CLR, fontsize=12)
        axes[2].set_yscale("log")
        axes[2].legend(fontsize=10, facecolor="#1a1f30", edgecolor="#3a3f55", labelcolor=TEXT_CLR)

        # ── Panel 4: Heat flux (log scale) ──────────────────────────────
        axes[3].plot(t_arr, q_par,  color="#ff8844", linewidth=1.8,
                     label=r"$\langle |q_\parallel| \rangle_i$")
        axes[3].plot(t_arr, q_perp, color="#8866ff", linewidth=1.8,
                     label=r"$\langle |q_\perp| \rangle_i$")
        axes[3].set_ylabel(r"Heat flux proxy [code]" , color=TEXT_CLR, fontsize=12)
        axes[3].set_xlabel(r"Time $[\Omega_{ci}^{-1}]$", color=TEXT_CLR, fontsize=13)
        axes[3].legend(fontsize=10, facecolor="#1a1f30", edgecolor="#3a3f55", labelcolor=TEXT_CLR)
        axes[3].set_yscale("log")

        fig.subplots_adjust(hspace=0.08)
        out_file = self.outdir / "heatflux_anisotropy_evolution.png"
        fig.savefig(out_file, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"Saved time evolution → {out_file}")

    def plot_snapshot_maps(self, step: int, mom_file: str, fld_file: str):
        """Generate 2D maps of heat flux and anisotropy for a single snapshot."""
        data = self.compute_heat_flux(mom_file, fld_file)

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.patch.set_facecolor(DARK_BG)

        t_omega = step_to_omegaci(step)
        extent = [0, DOMAIN_DI_Z, 0, DOMAIN_DI_Y]

        # Clamp anisotropy to physically meaningful range (0.5-5)
        aniso_map = np.clip(data["anisotropy"], 0.5, 5.0)
        # Clamp beta to reasonable range
        beta_map = np.clip(data["beta_par"], 0.5, np.nanpercentile(data["beta_par"], 99))

        configs = [
            (aniso_map, "RdYlBu_r",  r"$T_\perp / T_\parallel$ (ions)",   axes[0, 0], None),
            (beta_map,  "plasma",     r"$\beta_{\parallel,i}$",              axes[0, 1], None),
            (data["q_par"],  "seismic", r"$q_\parallel$ [code units]",       axes[1, 0], "sym"),
            (data["q_perp"], "inferno", r"$|q_\perp|$ [code units]",         axes[1, 1], None),
        ]

        for field, cmap, label, ax, norm_type in configs:
            ax.set_facecolor(PANEL_BG)

            if norm_type == "sym":
                vmax = np.nanpercentile(np.abs(field), 99)
                vmax = max(vmax, 1e-10)
                norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            else:
                vmin = np.nanpercentile(field, 1)
                vmax = np.nanpercentile(field, 99)
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            im = ax.imshow(
                field.T, origin="lower", cmap=cmap,
                norm=norm, aspect="auto", extent=extent,
            )
            cb = fig.colorbar(im, ax=ax, pad=0.02)
            cb.set_label(label, fontsize=11, color=TEXT_CLR)
            cb.ax.yaxis.set_tick_params(color=TEXT_CLR, labelsize=9)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_CLR)

            ax.set_xlabel(r"Z [$d_i$]", fontsize=11, color=TEXT_CLR)
            ax.set_ylabel(r"Y [$d_i$]", fontsize=11, color=TEXT_CLR)
            ax.tick_params(colors=TEXT_CLR, direction="in", which="both")
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_CLR)

        fig.suptitle(
            rf"Ion Heat Flux & Anisotropy — $t \approx {t_omega:.2f}\,\Omega_{{ci}}^{{-1}}$ (step {step})"
            "\n"
            rf"PSC ($m_i/m_e = {int(MASS_RATIO)}$"
            + (rf", $\kappa = {KAPPA}$" if KAPPA is not None else ", Maxwellian")
            + ")",
            fontsize=14, color=TEXT_CLR, fontweight="bold", y=1.01,
        )

        plt.tight_layout()
        out_file = self.outdir / f"heatflux_maps_step{step:06d}.png"
        fig.savefig(out_file, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Saved: {out_file}")


    def run(self, steps: list[int] | None = None, time_evolution: bool = True):
        """Full analysis: time evolution + individual snapshot maps."""
        moment_files = PICDataReader.find_files(self.moment_pattern)
        field_files = PICDataReader.find_files(self.field_pattern)
        common_steps = sorted(set(moment_files) & set(field_files))

        if not common_steps:
            print("[ERROR] No matching moment/field file pairs found.")
            return

        # Time evolution
        if time_evolution:
            self.analyze_time_evolution()

        # Snapshot maps
        selected = steps or [0, common_steps[len(common_steps)//4],
                             common_steps[len(common_steps)//2],
                             common_steps[3*len(common_steps)//4],
                             common_steps[-1]]
        selected = [s for s in selected if s in moment_files and s in field_files]

        print(f"\nGenerating snapshot maps for steps: {selected}")
        for step in selected:
            self.plot_snapshot_maps(step, moment_files[step], field_files[step])

        print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ion heat flux diagnostics from PSC outputs."
    )
    parser.add_argument("--moments", default=MOMENT_FILE_PATTERN,
                        help="Glob pattern for pfd_moments files.")
    parser.add_argument("--fields", default=FIELD_FILE_PATTERN,
                        help="Glob pattern for pfd field files.")
    parser.add_argument("--outdir", default="heatflux_plots",
                        help="Directory for output plots.")
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="Gaussian smoothing width in cells.")
    parser.add_argument("--steps", nargs="*", type=int,
                        help="Steps for snapshot maps.")
    parser.add_argument("--no-evolution", action="store_true",
                        help="Skip time evolution plot.")
    args = parser.parse_args()

    HeatFluxAnalyzer(
        moment_pattern=args.moments,
        field_pattern=args.fields,
        sigma=args.sigma,
        outdir=args.outdir,
    ).run(steps=args.steps, time_evolution=not args.no_evolution)
