#!/usr/bin/env python3
"""
mirror_physics.py  —  Mirror structure visualization (build2)
==============================================================
Genera mapas 2D de Bz/B0 (mirror holes) y Jx (corriente)
para snapshots seleccionados de la simulación build2.

Datos: build2/all_h5_feynman  (128x128, step 500, hasta 75000)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from data_reader import PICDataReader
from psc_units import (
    B0, DOMAIN_DI_Y, DOMAIN_DI_Z,
    FIELD_FILE_PATTERN, MASS_RATIO, step_to_omegaci,
)

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

# ── Paleta oscura ─────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
TEXT_CLR = "#e6edf3"
GRID_CLR = "#21262d"

# Steps por defecto — distribuidos a lo largo de los 151 snapshots del build2
DEFAULT_STEPS = [0, 2500, 5000, 10000, 15000, 25000, 35000, 50000, 65000, 75000]


class MirrorPhysicsPlotter:
    def __init__(
        self,
        pattern: str = FIELD_FILE_PATTERN,
        b0: float = B0,
        output_dir: str = "mirror_plots",
        sigma_b: float = 1.0,
        sigma_j: float = 1.5,
    ):
        self.pattern = pattern
        self.b0 = b0
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sigma_b = sigma_b
        self.sigma_j = sigma_j

    def run(self, steps: list[int] | None = None):
        file_map = PICDataReader.find_files(self.pattern)
        if not file_map:
            print("No field files found."); return

        # Si no se piden steps, usar los defaults que existan
        if steps is None:
            steps = [s for s in DEFAULT_STEPS if s in file_map]
        selected = [s for s in steps if s in file_map]
        if not selected:
            print("No matching steps."); return

        print(f"Processing {len(selected)} mirror snapshots...")
        for step in selected:
            self.process_file(step, file_map[step])

    def process_file(self, step: int, filename: str):
        try:
            fields = PICDataReader.read_multiple_fields_3d(
                filename, "jeh",
                ["hz_fc/p0/3d", "jx_ec/p0/3d"],
            )
        except Exception as exc:
            print(f"  [WARN] step {step}: {exc}"); return

        bz2d = PICDataReader.flatten_2d_slice(fields["hz_fc/p0/3d"]) / self.b0
        jx2d = PICDataReader.flatten_2d_slice(fields["jx_ec/p0/3d"])

        bz2d = gaussian_filter(bz2d, sigma=self.sigma_b)
        jx2d = gaussian_filter(jx2d, sigma=self.sigma_j)

        self._generate_plot(bz2d, jx2d, step)

    def _generate_plot(self, bz2d: np.ndarray, jx2d: np.ndarray, step: int):
        ext = [0, DOMAIN_DI_Z, 0, DOMAIN_DI_Y]
        toci = step_to_omegaci(step)

        # ── Bz / B0 ──────────────────────────────────────────────────────────
        fig, ax1 = plt.subplots(figsize=(8.2, 6.2))
        fig.patch.set_facecolor(DARK_BG)
        ax1.set_facecolor(PANEL_BG)
        bz_p5  = float(np.percentile(bz2d, 5))
        bz_p95 = float(np.percentile(bz2d, 95))
        vmin_b = max(0.3, bz_p5 - 0.1)
        vmax_b = min(2.5, bz_p95 + 0.1)
        im1 = ax1.imshow(bz2d.T, origin="lower", cmap="viridis",
                         vmin=vmin_b, vmax=vmax_b, aspect="auto", extent=ext)
        ax1.set_xlabel(r"Z  [$d_i$]", fontsize=15, color=TEXT_CLR)
        ax1.set_ylabel(r"Y  [$d_i$]", fontsize=15, color=TEXT_CLR)
        cb1 = fig.colorbar(im1, ax=ax1, pad=0.01, aspect=30)
        cb1.set_label(r"$B_z / B_0$", fontsize=14, color=TEXT_CLR)
        cb1.ax.yaxis.set_tick_params(color=TEXT_CLR, labelsize=13)
        plt.setp(cb1.ax.yaxis.get_ticklabels(), color=TEXT_CLR)
        ax1.tick_params(colors=TEXT_CLR, direction="in", which="both",
                        top=True, right=True)
        for sp in ax1.spines.values():
            sp.set_edgecolor(GRID_CLR)
        ax1.set_title(
            rf"$B_z/B_0$ mirror structures - step {step}, $t \approx {toci:.2f}\,\Omega_{{ci}}^{{-1}}$",
            fontsize=16, color=TEXT_CLR, pad=8,
        )
        outname = self.output_dir / f"mirror_bz_step{step:06d}.png"
        fig.savefig(outname, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Saved: {outname}")

        # ── Jx ────────────────────────────────────────────────────────────────
        fig, ax2 = plt.subplots(figsize=(8.2, 6.2))
        fig.patch.set_facecolor(DARK_BG)
        ax2.set_facecolor(PANEL_BG)
        jlim = float(np.percentile(np.abs(jx2d), 99))
        jlim = jlim if jlim > 0 else 1e-5
        im2 = ax2.imshow(jx2d.T, origin="lower", cmap="seismic",
                         vmin=-jlim, vmax=jlim, aspect="auto", extent=ext)
        ax2.set_xlabel(r"Z  [$d_i$]", fontsize=15, color=TEXT_CLR)
        ax2.set_ylabel(r"Y  [$d_i$]", fontsize=15, color=TEXT_CLR)
        cb2 = fig.colorbar(im2, ax=ax2, pad=0.01, aspect=30)
        cb2.set_label(r"$J_x$", fontsize=14, color=TEXT_CLR)
        cb2.ax.yaxis.set_tick_params(color=TEXT_CLR, labelsize=13)
        plt.setp(cb2.ax.yaxis.get_ticklabels(), color=TEXT_CLR)
        ax2.tick_params(colors=TEXT_CLR, direction="in", which="both",
                        top=True, right=True)
        for sp in ax2.spines.values():
            sp.set_edgecolor(GRID_CLR)
        ax2.set_title(
            rf"$J_x$ current density - step {step}, $t \approx {toci:.2f}\,\Omega_{{ci}}^{{-1}}$",
            fontsize=16, color=TEXT_CLR, pad=8,
        )
        outname = self.output_dir / f"mirror_jx_step{step:06d}.png"
        fig.savefig(outname, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Saved: {outname}")


def parse_args():
    parser = argparse.ArgumentParser(description="Mirror structure plots (build2).")
    parser.add_argument("--fields", default=FIELD_FILE_PATTERN)
    parser.add_argument("--B0", type=float, default=B0)
    parser.add_argument("--outdir", default="mirror_plots")
    parser.add_argument("--steps", nargs="*", type=int)
    parser.add_argument("--sigma-b", type=float, default=1.0)
    parser.add_argument("--sigma-j", type=float, default=1.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MirrorPhysicsPlotter(
        pattern=args.fields, b0=args.B0, output_dir=args.outdir,
        sigma_b=args.sigma_b, sigma_j=args.sigma_j,
    ).run(steps=args.steps)
