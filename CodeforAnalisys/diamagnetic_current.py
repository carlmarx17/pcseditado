#!/usr/bin/env python3
"""
diamagnetic_current.py
======================
Ion and electron diamagnetic current analysis for PIC simulations (PSC).

Generates:
  - Individual 2D maps for selected simulation snapshots.
  - Animated GIF of the temporal evolution.

Diamagnetic current definition:
  J_d = (nabla P_perp × B_hat) / B

In 2D (YZ plane, B0 || z-hat):
  J_dx =  (dP_perp/dy · Bz  -  dP_perp/dz · By) / B^2
  J_dy, J_dz → 0  in the 2D YZ plane

The dominant out-of-plane component is J_dx.

PSC normalised units:
  - Pressure: P = n m <v^2>  (PIC), with mu_0 = 1
  - Magnetic field: normalised so B0 = B0_ref at t = 0
  - Resulting diamagnetic current in PIC units
"""

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import gaussian_filter

from data_reader import PICDataReader
from psc_units import (
    B0,
    FIELD_FILE_PATTERN,
    KAPPA,
    MASS_RATIO,
    MOMENT_FILE_PATTERN,
    step_to_omegaci,
)

try:
    from PIL import Image
except ImportError:
    Image = None

plt.switch_backend("Agg")

# ── Dark-theme colour palette ────────────────────────────────────────────────
DARK_BG   = "#0c0e14"
PANEL_BG  = "#12151f"
TEXT_CLR  = "#dde2f0"
GRID_CLR  = "#232840"

# Steps to plot individually (representative subset)
PLOT_STEPS: list[int] = [0, 500, 1000, 2000, 3000, 5000, 7000, 9000, 11600]

# GIF decimation factor
GIF_STRIDE: int = 5


class DiamagneticCurrentAnalyzer:
    """Compute and visualise ion/electron/total diamagnetic current maps."""

    def __init__(
        self,
        moment_pattern: str = MOMENT_FILE_PATTERN,
        field_pattern: str = FIELD_FILE_PATTERN,
        sigma: float = 4.0,
        outdir: str = "diamagnetic_plots",
    ):
        self.moment_pattern = moment_pattern
        self.field_pattern  = field_pattern
        self.sigma  = sigma
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def run(self, steps: list[int] | None = None, make_gif: bool = True):
        """Run full analysis: individual plots + optional animated GIF."""
        moment_files = PICDataReader.find_files(self.moment_pattern)
        field_files  = PICDataReader.find_files(self.field_pattern)
        common_steps = sorted(set(moment_files) & set(field_files))

        if not common_steps:
            print("No matched moment/field files found.")
            return

        selected_steps = (
            common_steps if not steps
            else [s for s in steps if s in moment_files and s in field_files]
        )
        if not selected_steps:
            print("No matching steps found for the requested selection.")
            return

        # ── 1. Compute global colour range ───────────────────────────────────
        print(f"\n[1/3] Computing global colour range from {len(common_steps)} snapshots...")
        sample_steps = common_steps[:: max(1, len(common_steps) // 12)]
        vmax_i_all, vmax_e_all, vmax_tot_all = [], [], []

        for s in sample_steps:
            d = self.compute_diamagnetic_current(moment_files[s], field_files[s])
            vmax_i_all.append(np.percentile(np.abs(d["Jdia_i"]), 99.5))
            vmax_e_all.append(np.percentile(np.abs(d["Jdia_e"]), 99.5))
            vmax_tot_all.append(np.percentile(np.abs(d["Jdia_total"]), 99.5))

        vmax_i   = float(np.percentile(vmax_i_all, 90))
        vmax_e   = float(np.percentile(vmax_e_all, 90))
        vmax_tot = float(np.percentile(vmax_tot_all, 90))
        print(f"   vmax_i={vmax_i:.4f}  vmax_e={vmax_e:.4f}  vmax_tot={vmax_tot:.4f}")

        # ── 2. Individual plots ───────────────────────────────────────────────
        print(f"\n[2/3] Generating individual plots for {len(selected_steps)} snapshots...")
        for step in selected_steps:
            data = self.compute_diamagnetic_current(moment_files[step], field_files[step])
            self.plot_snapshot(step, data, vmax_i, vmax_e, vmax_tot)

        if not make_gif or Image is None:
            if make_gif and Image is None:
                print("  [WARNING] Pillow not installed — skipping GIF generation.")
            print("\nDone.")
            return

        # ── 3. Animated GIF ───────────────────────────────────────────────────
        gif_steps = common_steps[::GIF_STRIDE]
        print(f"\n[3/3] Generating GIF with {len(gif_steps)} frames (stride={GIF_STRIDE})...")

        frames = []
        for i, s in enumerate(gif_steps):
            print(f"  frame {i + 1}/{len(gif_steps)}  step={s}", end="\r")
            data = self.compute_diamagnetic_current(moment_files[s], field_files[s])
            img  = self._render_frame_to_pil(s, data, vmax_i, vmax_e, vmax_tot)
            frames.append(img)

        gif_path = self.outdir / "jdia_evolution.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=120,
            loop=0,
            optimize=True,
        )
        print(f"\n  GIF saved: {gif_path}  ({len(frames)} frames)")
        print("\nDone.")

    def compute_diamagnetic_current(
        self, mom_file: str, fld_file: str
    ) -> dict[str, np.ndarray]:
        """
        Compute diamagnetic current density for ions and electrons.

        Gaussian smoothing (sigma cells) is applied to pressure and B fields
        before computing gradients — essential to suppress PIC shot noise.
        """
        moments = PICDataReader.read_multiple_fields_3d(
            mom_file, "all_1st",
            ["txx_i/p0/3d", "tyy_i/p0/3d", "txx_e/p0/3d", "tyy_e/p0/3d"],
        )
        fields = PICDataReader.read_multiple_fields_3d(
            fld_file, "jeh-",
            ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"],
        )

        pperp_i = 0.5 * (
            PICDataReader.flatten_2d_slice(moments["txx_i/p0/3d"])
            + PICDataReader.flatten_2d_slice(moments["tyy_i/p0/3d"])
        )
        pperp_e = 0.5 * (
            PICDataReader.flatten_2d_slice(moments["txx_e/p0/3d"])
            + PICDataReader.flatten_2d_slice(moments["tyy_e/p0/3d"])
        )

        bx = PICDataReader.flatten_2d_slice(fields["hx_fc/p0/3d"]) * B0
        by = PICDataReader.flatten_2d_slice(fields["hy_fc/p0/3d"]) * B0
        bz = PICDataReader.flatten_2d_slice(fields["hz_fc/p0/3d"]) * B0

        smooth = lambda arr: gaussian_filter(arr.astype(float), sigma=self.sigma)
        pperp_i, pperp_e = smooth(pperp_i), smooth(pperp_e)
        bx, by, bz       = smooth(bx), smooth(by), smooth(bz)

        b2 = bx**2 + by**2 + bz**2 + 1e-40

        # Pressure gradients: array layout (Nz, Ny) → axis 0 = z, axis 1 = y
        dyi, dzi = np.gradient(pperp_i, axis=1), np.gradient(pperp_i, axis=0)
        dye, dze = np.gradient(pperp_e, axis=1), np.gradient(pperp_e, axis=0)

        # J_dx = (dP_perp/dy · Bz  -  dP_perp/dz · By) / B^2
        jdia_i     = (dyi * bz - dzi * by) / b2
        jdia_e     = (dye * bz - dze * by) / b2
        jdia_total = jdia_i + jdia_e

        return {
            "Jdia_i":     jdia_i,
            "Jdia_e":     jdia_e,
            "Jdia_total": jdia_total,
            "Bmag":       np.sqrt(b2),
        }

    def plot_snapshot(
        self,
        step: int,
        data: dict[str, np.ndarray],
        vmax_i: float | None = None,
        vmax_e: float | None = None,
        vmax_tot: float | None = None,
    ):
        """Render a 3-panel figure and save as PNG."""
        fig = self._make_figure(step, data, vmax_i, vmax_e, vmax_tot)
        out_file = self.outdir / f"jdia_step{step:06d}.png"
        fig.savefig(out_file, dpi=160, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Saved: {out_file}")

    def _render_frame_to_pil(
        self,
        step: int,
        data: dict[str, np.ndarray],
        vmax_i: float | None,
        vmax_e: float | None,
        vmax_tot: float | None,
    ):
        """Render figure to an in-memory PIL Image (for GIF assembly)."""
        if Image is None:
            raise ImportError("Pillow is required for GIF generation.")
        fig = self._make_figure(step, data, vmax_i, vmax_e, vmax_tot)
        buf = BytesIO()
        fig.savefig(buf, dpi=100, bbox_inches="tight", facecolor=DARK_BG)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        plt.close(fig)
        return img

    def _make_figure(
        self,
        step: int,
        data: dict[str, np.ndarray],
        vmax_i: float | None,
        vmax_e: float | None,
        vmax_tot: float | None,
    ):
        """Build the 3-panel matplotlib figure (ions / electrons / total)."""
        Ji    = data["Jdia_i"]
        Je    = data["Jdia_e"]
        Jtot  = data["Jdia_total"]
        Bmod  = data["Bmag"]

        vmax_i   = vmax_i   or np.percentile(np.abs(Ji),   99.5)
        vmax_e   = vmax_e   or np.percentile(np.abs(Je),   99.5)
        vmax_tot = vmax_tot or np.percentile(np.abs(Jtot), 99.5)

        fig, axes = plt.subplots(1, 3, figsize=(19, 7), constrained_layout=True)
        fig.patch.set_facecolor(DARK_BG)

        configs = [
            (Ji,   "RdBu_r",  vmax_i,   r"$J^{(d)}_x$ ions",      r"$J_d^{(\rm i)}$ [a.u.]"),
            (Je,   "PuOr_r",  vmax_e,   r"$J^{(d)}_x$ electrons", r"$J_d^{(\rm e)}$ [a.u.]"),
            (Jtot, "seismic", vmax_tot, r"$J^{(d)}_x$ total",     r"$J_d^{(\rm tot)}$ [a.u.]"),
        ]

        for ax, (field, cmap, vm, title, lbl) in zip(axes, configs):
            ax.set_facecolor(PANEL_BG)
            im = ax.imshow(
                field.T, origin="lower", cmap=cmap,
                vmin=-vm, vmax=vm, aspect="auto",
            )
            vmin_bmod = float(np.percentile(Bmod, 10))
            vmax_bmod = float(np.percentile(Bmod, 95))
            if vmax_bmod > vmin_bmod + 1e-8:
                lvls = np.linspace(vmin_bmod, vmax_bmod, 7)
                ax.contour(Bmod.T, levels=lvls, colors="white", linewidths=0.5, alpha=0.4)
            cb = fig.colorbar(im, ax=ax, pad=0.01, aspect=30)
            cb.set_label(lbl, fontsize=10, color=TEXT_CLR)
            cb.ax.yaxis.set_tick_params(color=TEXT_CLR, labelsize=8)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_CLR)

            ax.set_xlabel("Z Axis", fontsize=11, color=TEXT_CLR)
            ax.set_ylabel("Y Axis", fontsize=11, color=TEXT_CLR)
            ax.set_title(title, fontsize=13, color=TEXT_CLR, pad=8)
            ax.tick_params(colors=TEXT_CLR, direction="in", which="both", top=True, right=True)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_CLR)

        fig.suptitle(
            rf"Diamagnetic Current  —  $t \approx {step_to_omegaci(step):.2f}\ \Omega_{{ci}}^{{-1}}$  (step {step})"
            "\n"
            rf"PSC  ($m_i/m_e = {int(MASS_RATIO)}$,  $\kappa = {KAPPA}$,  $B_0 = {B0}$)",
            fontsize=14, color=TEXT_CLR, y=1.02, fontweight="bold",
        )
        return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute diamagnetic current diagnostics from PSC outputs."
    )
    parser.add_argument("--moments", default=MOMENT_FILE_PATTERN,
                        help="Glob pattern for pfd_moments files.")
    parser.add_argument("--fields",  default=FIELD_FILE_PATTERN,
                        help="Glob pattern for pfd field files.")
    parser.add_argument("--outdir",  default="diamagnetic_plots",
                        help="Directory for output plots.")
    parser.add_argument("--sigma",   type=float, default=4.0,
                        help="Gaussian smoothing width in cells.")
    parser.add_argument("--steps",   nargs="*", type=int,
                        help="Optional list of steps to process.")
    parser.add_argument("--no-gif",  action="store_true",
                        help="Skip animated GIF generation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DiamagneticCurrentAnalyzer(
        moment_pattern=args.moments,
        field_pattern=args.fields,
        sigma=args.sigma,
        outdir=args.outdir,
    ).run(steps=args.steps, make_gif=not args.no_gif)
