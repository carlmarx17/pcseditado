#!/usr/bin/env python3
"""
diamagnetic_current.py
======================
Ion and electron diamagnetic current analysis for PIC simulations (PSC).

Generates:
  - Individual 2D maps for selected simulation snapshots.
  - Animated GIF of the temporal evolution.

Diamagnetic current definition (force balance nabla P = J × B):
  J_d = (B × nabla P_perp) / B^2

In 2D (YZ plane, B0 || z-hat, d/dx = 0):
  J_dx =  (By · dP_perp/dz  -  Bz · dP_perp/dy) / B^2

The in-plane components, -Bx dP/dz and Bx dP/dy, are second order in the
fluctuation for B ~ B0 z-hat; the out-of-plane J_dx is the one mapped here.

P_perp is the *thermal* pressure: PSC's raw second moment t_ab = n m <u v>
minus the bulk-flow part, projected on the local field (plasma_physics.
central_pressure_tensor / field_aligned_pressures). Gradients are taken in
code lengths (d_e), periodic, so J is in code current-density units; the
same formula (plasma_physics.diamagnetic_current_x) is used by
physical_diagnostics.py.
"""

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import plot_style as ps
from data_reader import PICDataReader
from plasma_physics import (
    central_pressure_tensor,
    diamagnetic_current_x,
    field_aligned_pressures,
)
from psc_units import (
    DOMAIN_DI_Y,
    DOMAIN_DI_Z,
    DX_DE,
    DX_DI,
    FIELD_FILE_PATTERN,
    M_ELEC,
    M_ION,
    MOMENT_FILE_PATTERN,
    PROFILE_LABEL,
    step_to_omegaci,
)

try:
    from PIL import Image
except ImportError:
    Image = None

ps.apply()
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

# Theme colours (paper: white background; dark: screen) from plot_style.
DARK_BG   = ps.FIG_BG
PANEL_BG  = ps.PANEL_BG
TEXT_CLR  = ps.TEXT_CLR
GRID_CLR  = ps.GRID_CLR
CONTOUR_CLR = ps.MUTED_CLR

# Default smoothing scale in d_i. The diamagnetic current comes from moment
# gradients: PIC shot noise (nicell ~ 1000) dominates any gradient computed
# below the ion kinetic scales, so the filter must live at a fraction of
# rho_i ~ d_i, NOT at a few cells (dx ~ 0.035 d_i). sigma = 0.5 d_i keeps the
# mirror/firehose structures (several d_i) while killing the grid-scale noise.
SIGMA_DI_DEFAULT: float = 0.5

# Production runs write snapshots every 500 steps; individual maps every
# snapshot are useless bulk. Default cadence for the per-step PNG maps:
STEP_EVERY_DEFAULT: int = 100_000

# GIF frames are rendered in memory (never written to disk), so the animation
# can run at a much finer cadence than the saved PNGs.
GIF_EVERY_DEFAULT: int = 10_000
# See fluctuationofmagneticfiel.py: dpi=80 keeps a 1.2M-step GIF near 30 MB.
GIF_DPI_DEFAULT: int = 80


class DiamagneticCurrentAnalyzer:
    """Compute and visualise ion/electron/total diamagnetic current maps."""

    def __init__(
        self,
        moment_pattern: str = MOMENT_FILE_PATTERN,
        field_pattern: str = FIELD_FILE_PATTERN,
        sigma: float | None = None,
        sigma_di: float = SIGMA_DI_DEFAULT,
        outdir: str = "diamagnetic_plots",
    ):
        self.moment_pattern = moment_pattern
        self.field_pattern  = field_pattern
        # sigma (cells) overrides sigma_di (physical); default is physical.
        self.sigma  = sigma if sigma is not None else sigma_di / DX_DI
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        print(f"Gaussian smoothing: sigma = {self.sigma:.1f} cells "
              f"({self.sigma * DX_DI:.2f} d_i)")

    def run(self, steps: list[int] | None = None, make_gif: bool = True,
            every: int = STEP_EVERY_DEFAULT, gif_every: int = GIF_EVERY_DEFAULT):
        """Run full analysis: individual plots + optional animated GIF."""
        moment_files = PICDataReader.find_files(self.moment_pattern)
        field_files  = PICDataReader.find_files(self.field_pattern)
        common_steps = sorted(set(moment_files) & set(field_files))

        if not common_steps:
            print("No matched moment/field files found.")
            return

        if steps:
            selected_steps = [s for s in steps if s in moment_files and s in field_files]
        else:
            # Keep only every `every` steps, always including first and last.
            selected_steps = [s for s in common_steps if s % every == 0]
            for endpoint in (common_steps[0], common_steps[-1]):
                if endpoint not in selected_steps:
                    selected_steps.append(endpoint)
            selected_steps.sort()
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
        gif_steps = [s for s in common_steps if s % gif_every == 0]
        for endpoint in (common_steps[0], common_steps[-1]):
            if endpoint not in gif_steps:
                gif_steps.append(endpoint)
        gif_steps.sort()
        print(f"\n[3/3] Generating GIF with {len(gif_steps)} frames "
              f"(one every {gif_every} steps)...")

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
        names = ["rho", "txx", "tyy", "tzz", "txy", "tyz", "tzx", "px", "py", "pz"]
        fields = PICDataReader.read_multiple_fields_3d(
            fld_file, "jeh-",
            ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"],
        )
        bx = PICDataReader.flatten_2d_slice(fields["hx_fc/p0/3d"]).astype(float)
        by = PICDataReader.flatten_2d_slice(fields["hy_fc/p0/3d"]).astype(float)
        bz = PICDataReader.flatten_2d_slice(fields["hz_fc/p0/3d"]).astype(float)
        smooth = lambda arr: gaussian_filter(arr.astype(float), sigma=self.sigma, mode="wrap")

        currents = {}
        for suffix, mass in (("i", M_ION), ("e", M_ELEC)):
            raw = PICDataReader.read_multiple_fields_3d(
                mom_file, "all_1st", [f"{n}_{suffix}/p0/3d" for n in names])
            mom = {k.split("/")[0]: PICDataReader.flatten_2d_slice(v).astype(float)
                   for k, v in raw.items()}
            t = central_pressure_tensor(mom, suffix, mass)
            _, pperp, _ = field_aligned_pressures(
                t["Pxx"], t["Pyy"], t["Pzz"], t["Pxy"], t["Pyz"], t["Pzx"], bx, by, bz)
            pperp = np.nan_to_num(pperp, nan=float(np.nanmean(pperp)))
            currents[suffix] = diamagnetic_current_x(
                smooth(pperp), smooth(by), smooth(bz), DX_DE, DX_DE)
        jdia_i, jdia_e = currents["i"], currents["e"]
        jdia_total = jdia_i + jdia_e
        b2 = smooth(bx) ** 2 + smooth(by) ** 2 + smooth(bz) ** 2

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
        """Render separated ion/electron/total maps and save as PNG."""
        Ji = data["Jdia_i"]
        Je = data["Jdia_e"]
        Jtot = data["Jdia_total"]
        Bmod = data["Bmag"]

        vmax_i = vmax_i or np.percentile(np.abs(Ji), 99.5)
        vmax_e = vmax_e or np.percentile(np.abs(Je), 99.5)
        vmax_tot = vmax_tot or np.percentile(np.abs(Jtot), 99.5)

        configs = [
            (Ji, ps.CMAP_DIVERGING, vmax_i, r"$J^{(d)}_x$ ions", r"$J_d^{(\rm i)}$ [code units]", "ions"),
            (Je, ps.CMAP_DIVERGING, vmax_e, r"$J^{(d)}_x$ electrons", r"$J_d^{(\rm e)}$ [code units]", "electrons"),
            (Jtot, ps.CMAP_DIVERGING, vmax_tot, r"$J^{(d)}_x$ total", r"$J_d^{(\rm tot)}$ [code units]", "total"),
        ]

        for field, cmap, vm, title, lbl, slug in configs:
            fig, ax = plt.subplots(figsize=(8.2, 6.2))
            fig.patch.set_facecolor(DARK_BG)
            ax.set_facecolor(PANEL_BG)
            im = ax.imshow(
                field.T, origin="lower", cmap=cmap,
                vmin=-vm, vmax=vm, aspect="auto",
                extent=[0, DOMAIN_DI_Z, 0, DOMAIN_DI_Y],
            )
            vmin_bmod = float(np.percentile(Bmod, 10))
            vmax_bmod = float(np.percentile(Bmod, 95))
            if vmax_bmod > vmin_bmod + 1e-8:
                lvls = np.linspace(vmin_bmod, vmax_bmod, 7)
                ax.contour(Bmod.T, levels=lvls, colors=CONTOUR_CLR, linewidths=0.5, alpha=0.6,
                           extent=[0, DOMAIN_DI_Z, 0, DOMAIN_DI_Y])
            cb = fig.colorbar(im, ax=ax, pad=0.01, aspect=30)
            cb.set_label(lbl, fontsize=14, color=TEXT_CLR)
            cb.ax.yaxis.set_tick_params(color=TEXT_CLR, labelsize=13)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_CLR)
            ps.spatial_axes(ax, fontsize=15, color=TEXT_CLR)
            ax.set_title(
                rf"{title} — $t\Omega_{{ci}} = {step_to_omegaci(step):.1f}$ (step {step})",
                fontsize=16, color=TEXT_CLR,
            )
            ax.tick_params(colors=TEXT_CLR, direction="in", which="both", top=True, right=True)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_CLR)
            out_file = self.outdir / f"jdia_{slug}_step{step:06d}.png"
            ps.save(fig, out_file)
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
        fig.savefig(buf, dpi=GIF_DPI_DEFAULT, bbox_inches="tight", facecolor=DARK_BG)
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
            (Ji,   ps.CMAP_DIVERGING, vmax_i,   r"$J^{(d)}_x$ ions",      r"$J_d^{(\rm i)}$ [code units]"),
            (Je,   ps.CMAP_DIVERGING, vmax_e,   r"$J^{(d)}_x$ electrons", r"$J_d^{(\rm e)}$ [code units]"),
            (Jtot, ps.CMAP_DIVERGING, vmax_tot, r"$J^{(d)}_x$ total",     r"$J_d^{(\rm tot)}$ [code units]"),
        ]

        for ax, (field, cmap, vm, title, lbl) in zip(axes, configs):
            ax.set_facecolor(PANEL_BG)
            im = ax.imshow(
                field.T, origin="lower", cmap=cmap,
                vmin=-vm, vmax=vm, aspect="auto",
                extent=[0, DOMAIN_DI_Z, 0, DOMAIN_DI_Y],
            )
            vmin_bmod = float(np.percentile(Bmod, 10))
            vmax_bmod = float(np.percentile(Bmod, 95))
            if vmax_bmod > vmin_bmod + 1e-8:
                lvls = np.linspace(vmin_bmod, vmax_bmod, 7)
                ax.contour(Bmod.T, levels=lvls, colors=CONTOUR_CLR, linewidths=0.5, alpha=0.6,
                           extent=[0, DOMAIN_DI_Z, 0, DOMAIN_DI_Y])
            cb = fig.colorbar(im, ax=ax, pad=0.01, aspect=30)
            cb.set_label(lbl, fontsize=14, color=TEXT_CLR)
            cb.ax.yaxis.set_tick_params(color=TEXT_CLR, labelsize=13)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_CLR)

            ps.spatial_axes(ax, fontsize=15, color=TEXT_CLR)
            ax.set_title(title, fontsize=16, color=TEXT_CLR)
            ax.tick_params(colors=TEXT_CLR, direction="in", which="both", top=True, right=True)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_CLR)

        fig.suptitle(
            rf"Diamagnetic current — $t\Omega_{{ci}} = {step_to_omegaci(step):.1f}$ (step {step})"
            "\n"
            rf"{PROFILE_LABEL}  (contours: $|B|$)",
            fontsize=17, color=TEXT_CLR, fontweight="bold",
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
    parser.add_argument("--sigma",   type=float, default=None,
                        help="Gaussian smoothing width in cells (overrides --sigma-di).")
    parser.add_argument("--sigma-di", type=float, default=SIGMA_DI_DEFAULT,
                        help="Gaussian smoothing width in ion inertial lengths "
                             f"(default {SIGMA_DI_DEFAULT} d_i).")
    parser.add_argument("--every",   type=int, default=STEP_EVERY_DEFAULT,
                        help="Save one PNG every this many simulation steps "
                             f"(default {STEP_EVERY_DEFAULT}). Ignored if --steps is given.")
    parser.add_argument("--gif-every", type=int, default=GIF_EVERY_DEFAULT,
                        help="GIF frame cadence in simulation steps "
                             f"(default {GIF_EVERY_DEFAULT}). Frames are rendered "
                             "in memory, never written to disk.")
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
        sigma_di=args.sigma_di,
        outdir=args.outdir,
    ).run(steps=args.steps, make_gif=not args.no_gif, every=args.every,
          gif_every=args.gif_every)
