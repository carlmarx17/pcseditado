#!/usr/bin/env python3
"""
anisotropy_analysis.py
======================
Plasma Anisotropy Analyzer — Brazil Plot Generator.

Generates Brazil plots (Anisotropy T_perp/T_par vs. parallel beta) from
PIC simulation output (PSC HDF5 files).

Implemented instability thresholds (bi-Maxwellian):
    Mirror:        T_perp/T_par = 1 + 1/beta_par          (Hasegawa, 1969)
    Firehose:      T_perp/T_par = 1 - 1/beta_par          (Gary, 1993)
    Ion-cyclotron: T_perp/T_par ≈ 1 + 0.43/beta_par^0.42  (Gary et al., 1994)

Normalised units (PSC code):
    Spatial  → d_i = c / omega_pi
    Temporal → Omega_ci = q B0 / m_i
    Velocity → v_A = B0 / sqrt(n0 m_i)

NOTE: Magnetic fields in PSC HDF5 files are stored in PIC-normalised units
where B0 already equals B0_ref numerically. No additional rescaling is
applied — doing so would introduce a spurious factor of ~1/B0_ref^2 in beta.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from pathlib import Path
import warnings
from data_reader import PICDataReader

# ── Simulation parameters (consistent with psc_temp_aniso.cxx) ──────────────
MASS_RATIO: float = 64.0   # m_i / m_e (artificial mass ratio)
KAPPA: float = 3.0          # Kappa index for the ion distribution

# Initial conditions (from psc_temp_aniso.cxx)
T_PERP_I: float = 0.175     # Ion perpendicular temperature
T_PAR_I: float = 0.05       # Ion parallel temperature

warnings.filterwarnings("ignore")


class PlasmaAnisotropyAnalyzer:
    """Compute and visualise anisotropy vs. parallel beta (Brazil plot)."""

    def __init__(
        self,
        n0: float = 1.0,
        T0: float = 1.0,
        B0_ref: float = 0.1,
        outdir: str = "anisotropy_plots",
    ):
        self.n0 = n0
        self.T0 = T0
        self.B0_ref = B0_ref
        self.mu0 = 1.0  # PIC normalised permeability
        self.outdir = Path(outdir)
        self.outdir.mkdir(exist_ok=True)
        self._reset()
        plt.switch_backend("Agg")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _reset(self):
        """Clear accumulated snapshot data."""
        self.all_anisotropy = np.array([])
        self.all_beta_par = np.array([])
        self.all_steps = np.array([])
        self.snapshots: list[int] = []

    # ── Data processing ──────────────────────────────────────────────────────

    def process_snapshot(self, mom_file: str, bz_file: str) -> dict | None:
        """
        Extract per-cell anisotropy and parallel beta from one time step.

        The PSC moment datasets 'txx_i', 'tyy_i', 'tzz_i' store pressure
        tensor components P_ij = n m <v_i v_j> (NOT temperature).
        Beta is computed as P_par / P_mag, where P_mag = B^2 / (2 mu_0).
        """
        try:
            moments = PICDataReader.read_multiple_fields_3d(
                mom_file,
                "all_1st",
                ["txx_i/p0/3d", "tyy_i/p0/3d", "tzz_i/p0/3d", "rho_i/p0/3d"],
            )
            b_fields = PICDataReader.read_multiple_fields_3d(
                bz_file,
                "jeh-",
                ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"],
            )
        except Exception as exc:
            print(f"[WARNING] Error reading {mom_file}, {bz_file}: {exc}")
            return None

        Pxx = moments["txx_i/p0/3d"].ravel()
        Pyy = moments["tyy_i/p0/3d"].ravel()
        Pzz = moments["tzz_i/p0/3d"].ravel()
        n = moments["rho_i/p0/3d"].ravel()

        Bx = b_fields["hx_fc/p0/3d"].ravel()
        By = b_fields["hy_fc/p0/3d"].ravel()
        Bz = b_fields["hz_fc/p0/3d"].ravel()
        B2 = Bx**2 + By**2 + Bz**2

        safe_n = np.where(n > 1e-10, n, np.nan)
        T_par = Pzz / safe_n
        T_perp = 0.5 * (Pxx + Pyy) / safe_n
        anisotropy = T_perp / (T_par + 1e-30)

        # beta_par = P_par / P_mag,  P_mag = B^2 / (2 mu_0)
        P_mag = B2 / (2.0 * self.mu0)
        beta_par = Pzz / (P_mag + 1e-30)

        mask = (
            (Pzz > 0)
            & (Pyy > 0)
            & (n > 1e-10)
            & (B2 > 1e-30)
            & np.isfinite(beta_par)
            & np.isfinite(anisotropy)
            & (anisotropy > 0)
        )

        return {"anisotropy": anisotropy[mask], "beta_par": beta_par[mask]}

    def analyze_simulation(
        self,
        mom_pattern: str = "pfd_moments.*.h5",
        bz_pattern: str = "pfd.*.h5",
    ):
        """Process all matched snapshots and accumulate results."""
        print("Starting plasma anisotropy analysis...")

        mom_files = PICDataReader.find_files(mom_pattern)
        bz_files = PICDataReader.find_files(bz_pattern)
        common_steps = sorted(set(mom_files) & set(bz_files))

        if not common_steps:
            print("[ERROR] No matching moment/field file pairs found.")
            return

        print(f"Found {len(common_steps)} matching snapshots.")
        self._reset()

        all_aniso, all_beta, all_idx = [], [], []

        for i, step in enumerate(common_steps):
            print(f"  Processing step {step} ({i + 1}/{len(common_steps)})...")
            data = self.process_snapshot(mom_files[step], bz_files[step])
            if data is None:
                continue

            n_pts = len(data["anisotropy"])
            all_aniso.extend(data["anisotropy"])
            all_beta.extend(data["beta_par"])
            all_idx.extend([i] * n_pts)
            self.snapshots.append(step)

        self.all_anisotropy = np.asarray(all_aniso)
        self.all_beta_par = np.asarray(all_beta)
        self.all_steps = np.asarray(all_idx)
        print("Analysis completed.\n")

    # ── Instability threshold curves ─────────────────────────────────────────

    @staticmethod
    def _mirror_threshold(beta: np.ndarray) -> np.ndarray:
        """Mirror instability: T_perp/T_par = 1 + 1/beta_par (Hasegawa 1969)."""
        return 1.0 + 1.0 / beta

    @staticmethod
    def _firehose_threshold(beta: np.ndarray) -> np.ndarray:
        """Firehose instability: T_perp/T_par = 1 - 1/beta_par (Gary 1993)."""
        return 1.0 - 1.0 / beta

    @staticmethod
    def _ion_cyclotron_threshold(beta: np.ndarray) -> np.ndarray:
        """Ion-cyclotron instability: ≈ 1 + 0.43/beta_par^0.42 (Gary+ 1994)."""
        return 1.0 + 0.43 / beta**0.42

    # ── Plotting ─────────────────────────────────────────────────────────────

    def plot_brazil_plot(self, save: bool = True):
        """
        Generate a publication-quality Brazil plot.

        Design:
          - Dark background for contrast.
          - 2D histogram with 'plasma' colormap for point density.
          - Instability threshold curves with shaded unstable regions.
          - Golden star marking the initial simulation conditions.
        """
        if len(self.all_beta_par) == 0:
            print("[WARNING] No data to plot.")
            return

        # ── Color scheme ─────────────────────────────────────────────────────
        DARK_BG = "#0f1117"
        PANEL_BG = "#181c27"
        TEXT_CLR = "#e8eaf0"
        GRID_CLR = "#2a2f3f"

        fig, ax = plt.subplots(figsize=(11, 9))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(PANEL_BG)

        # ── Sub-sampling ─────────────────────────────────────────────────────
        n_total = len(self.all_beta_par)
        n_sub = min(200_000, n_total)
        idx = np.random.choice(n_total, n_sub, replace=False)

        beta_sub = self.all_beta_par[idx]
        aniso_sub = self.all_anisotropy[idx]

        valid = (
            (beta_sub > 0)
            & (aniso_sub > 0)
            & np.isfinite(beta_sub)
            & np.isfinite(aniso_sub)
        )
        beta_v = beta_sub[valid]
        aniso_v = aniso_sub[valid]

        # ── Axis range ───────────────────────────────────────────────────────
        if len(beta_v) > 0:
            bp1, bp99 = np.percentile(beta_v, [0.5, 99.5])
            ap1, ap99 = np.percentile(aniso_v, [0.5, 99.5])
            xmin = min(0.3, bp1 * 0.5)
            xmax = max(300.0, bp99 * 2.0)
            ymin = min(0.3, ap1 * 0.5)
            ymax = max(12.0, ap99 * 1.5)
        else:
            xmin, xmax = 0.3, 300.0
            ymin, ymax = 0.3, 12.0

        # ── 2D histogram ────────────────────────────────────────────────────
        h = ax.hist2d(
            beta_v,
            aniso_v,
            bins=130,
            cmap="plasma",
            norm=mcolors.LogNorm(vmin=1),
            cmin=1,
            range=[[xmin, xmax], [ymin, ymax]],
        )
        cbar = plt.colorbar(h[3], ax=ax, pad=0.02)
        cbar.set_label("Point density [a.u.]", fontsize=12, color=TEXT_CLR)
        cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_CLR)

        # ── Instability thresholds ───────────────────────────────────────────
        b = np.logspace(
            np.log10(max(5e-2, xmin * 0.3)),
            np.log10(min(5e3, xmax * 5.0)),
            800,
        )

        # Mirror threshold
        mirror = self._mirror_threshold(b)
        in_range = (mirror >= ymin * 0.8) & (mirror <= ymax * 1.2)
        ax.plot(
            b[in_range],
            mirror[in_range],
            "--",
            color="#ff6666",
            linewidth=2.8,
            zorder=8,
            alpha=0.9,
            label=r"Mirror  $T_\perp/T_\parallel = 1 + 1/\beta_\parallel$",
        )
        ax.fill_between(
            b,
            np.clip(mirror, ymin, ymax * 2),
            ymax * 2,
            alpha=0.10,
            color="#ff4444",
            zorder=3,
        )

        # Firehose threshold (only valid for beta > 1)
        b_fh = b[b > 1.05]
        fh = self._firehose_threshold(b_fh)
        ax.plot(
            b_fh,
            fh,
            "--",
            color="#66aaff",
            linewidth=2.8,
            zorder=8,
            alpha=0.9,
            label=r"Firehose  $T_\perp/T_\parallel = 1 - 1/\beta_\parallel$",
        )
        ax.fill_between(
            b_fh,
            ymin * 0.5,
            np.clip(fh, ymin * 0.5, ymax),
            alpha=0.10,
            color="#4488ff",
            zorder=3,
        )

        # Ion-cyclotron threshold
        ic = self._ion_cyclotron_threshold(b)
        in_ic = (ic >= ymin * 0.8) & (ic <= ymax * 1.2)
        ax.plot(
            b[in_ic],
            ic[in_ic],
            ":",
            color="#44ffaa",
            linewidth=2.4,
            zorder=8,
            alpha=0.9,
            label=r"Ion-cyclotron  $\approx 1 + 0.43/\beta_\parallel^{0.42}$",
        )

        # Isotropy reference line
        ax.axhline(1.0, color=TEXT_CLR, alpha=0.30, linewidth=1.0, linestyle=":")

        # ── Initial conditions marker ────────────────────────────────────────
        # beta_par,i = n0 * T_par_i / (B0^2 / 2) = 0.05 / 0.005 = 10
        # T_perp / T_par = 0.175 / 0.05 = 3.5
        beta_init = (self.n0 * T_PAR_I) / (self.B0_ref**2 / 2.0)
        aniso_init = T_PERP_I / T_PAR_I
        ax.plot(
            beta_init,
            aniso_init,
            "*",
            color="#ffd700",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=22,
            zorder=10,
            label=(
                rf"Initial conditions "
                rf"($\beta_{{i\parallel}}={beta_init:.0f}$, "
                rf"$T_\perp/T_\parallel={aniso_init:.1f}$)"
            ),
        )

        # ── Axes & labels ────────────────────────────────────────────────────
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel(
            r"$\beta_{\parallel}$ — Ion parallel pressure / magnetic pressure",
            fontsize=13,
            color=TEXT_CLR,
            labelpad=8,
        )
        ax.set_ylabel(
            r"$T_\perp / T_\parallel$",
            fontsize=14,
            color=TEXT_CLR,
            labelpad=8,
        )
        ax.set_title(
            r"Brazil Plot: Anisotropy vs $\beta_\parallel$"
            + "\n"
            + rf"PSC  ($m_i/m_e = {int(MASS_RATIO)}$,  $\kappa = {KAPPA}$)",
            fontsize=15,
            fontweight="bold",
            color=TEXT_CLR,
            pad=14,
        )

        ax.tick_params(
            which="both", colors=TEXT_CLR, direction="in", top=True, right=True
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.grid(True, which="both", alpha=0.2, color=GRID_CLR, linestyle=":")

        ax.legend(
            fontsize=10.5,
            framealpha=0.6,
            facecolor="#1a1f30",
            edgecolor="#3a3f55",
            labelcolor=TEXT_CLR,
        )

        # ── Metadata annotation ──────────────────────────────────────────────
        n_snaps = len(self.snapshots) if self.snapshots else "?"
        ax.text(
            0.99,
            0.02,
            f"{n_sub:,} points  |  {n_snaps} snapshots",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            color="#8890aa",
        )

        if save:
            out_file = self.outdir / "brazil_plot_anisotropy.png"
            plt.savefig(out_file, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
            print(f"Saved Brazil Plot → {out_file}")
        plt.close()


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze plasma anisotropy and generate Brazil plots."
    )
    parser.add_argument(
        "--moments",
        type=str,
        default="pfd_moments.*.h5",
        help="Glob pattern for moment files.",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default="pfd.*.h5",
        help="Glob pattern for field files.",
    )
    parser.add_argument(
        "--B0",
        type=float,
        default=0.1,
        help="Reference B0 (for initial-condition marker).",
    )
    args = parser.parse_args()

    analyzer = PlasmaAnisotropyAnalyzer(B0_ref=args.B0, outdir="anisotropy_plots")
    analyzer.analyze_simulation(mom_pattern=args.moments, bz_pattern=args.fields)
    analyzer.plot_brazil_plot(save=True)
