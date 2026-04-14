#!/usr/bin/env python3
"""
mirror_physics.py
=================
Visualize mirror structures from PSC field outputs.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from data_reader import PICDataReader
from psc_units import B0, FIELD_FILE_PATTERN, KAPPA, MASS_RATIO, step_to_omegaci

plt.switch_backend("Agg")


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
            print("No field files found.")
            return

        selected_steps = sorted(file_map) if not steps else [step for step in steps if step in file_map]
        if not selected_steps:
            print("No matching steps found for the requested selection.")
            return

        print(f"Processing {len(selected_steps)} mirror snapshots...")
        for step in selected_steps:
            self.process_file(step, file_map[step])

    def process_file(self, step: int, filename: str):
        try:
            fields = PICDataReader.read_multiple_fields_3d(
                filename,
                "jeh-",
                ["hz_fc/p0/3d", "jx_ec/p0/3d"],
            )
        except Exception as exc:
            print(f"Error reading {filename}: {exc}")
            return

        bz2d = PICDataReader.flatten_2d_slice(fields["hz_fc/p0/3d"]) / self.b0
        jx2d = PICDataReader.flatten_2d_slice(fields["jx_ec/p0/3d"])

        bz2d = gaussian_filter(bz2d, sigma=self.sigma_b)
        jx2d = gaussian_filter(jx2d, sigma=self.sigma_j)

        self._generate_plot(bz2d, jx2d, step)

    def _generate_plot(self, bz2d: np.ndarray, jx2d: np.ndarray, step: int):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

        im1 = ax1.imshow(bz2d.T, origin="lower", cmap="viridis", vmin=0.5, vmax=1.5, aspect="auto")
        ax1.set_title(r"Magnetic Field $B_z / B_0$ (Mirror Holes)", fontsize=14)
        ax1.set_xlabel("Z Axis (Parallel to B0)", fontsize=12)
        ax1.set_ylabel("Y Axis", fontsize=12)
        fig.colorbar(im1, ax=ax1, label=r"$B_z / B_0$")

        limit = np.percentile(np.abs(jx2d), 99)
        limit = limit if limit > 0 else 1e-5

        im2 = ax2.imshow(jx2d.T, origin="lower", cmap="seismic", vmin=-limit, vmax=limit, aspect="auto")
        ax2.set_title(r"Current Density $J_x$", fontsize=14)
        ax2.set_xlabel("Z Axis (Parallel to B0)", fontsize=12)
        ax2.set_ylabel("Y Axis", fontsize=12)
        fig.colorbar(im2, ax=ax2, label=r"$J_x$")

        time_omegaci = step_to_omegaci(step)
        fig.suptitle(
            rf"Mirror structures | step {step} | $t \approx {time_omegaci:.2f}\ \Omega_{{ci}}^{{-1}}$"
            "\n"
            rf"PSC ($m_i/m_e = {int(MASS_RATIO)}$, $\kappa = {KAPPA}$)",
            fontsize=14,
            fontweight="bold",
        )

        outname = self.output_dir / f"mirror_physics_step{step}.png"
        fig.savefig(outname, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {outname}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot mirror structures from PSC field outputs.")
    parser.add_argument("--fields", default=FIELD_FILE_PATTERN, help="Glob pattern for pfd field files.")
    parser.add_argument("--B0", type=float, default=B0, help="Background magnetic field normalization.")
    parser.add_argument("--outdir", default="mirror_plots", help="Directory for generated plots.")
    parser.add_argument("--steps", nargs="*", type=int, help="Optional list of steps to plot.")
    parser.add_argument("--sigma-b", type=float, default=1.0, help="Gaussian smoothing for Bz.")
    parser.add_argument("--sigma-j", type=float, default=1.5, help="Gaussian smoothing for Jx.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MirrorPhysicsPlotter(
        pattern=args.fields,
        b0=args.B0,
        output_dir=args.outdir,
        sigma_b=args.sigma_b,
        sigma_j=args.sigma_j,
    ).run(steps=args.steps)
