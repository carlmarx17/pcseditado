#!/usr/bin/env python3
"""
spectral_analysis.py
====================
Spectral analysis of magnetic-field fluctuations for PSC outputs.

Features:
  - Correct 2D slice extraction for PSC ordering (Nz, Ny, Nx)
  - Spectra of deltaB, deltaB_parallel, deltaB_perp
  - 2D anisotropic PSD in physical axes
  - 1D isotropic E(k)
  - Automatic power-law fit on the 1D spectrum
  - Tracking of dominant modes across snapshots
"""

import argparse
import csv
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, fftshift
from scipy.stats import linregress

from data_reader import PICDataReader

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")


class SpectralAnalyzer:
    def __init__(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        B0_ref: float = 1.0,
        outdir: str = "spectral_plots",
        parallel_axis: str = "z",
        top_modes: int = 8,
    ):
        self.dx = dx
        self.dy = dy
        self.B0_ref = B0_ref
        self.parallel_axis = parallel_axis.lower()
        self.top_modes = top_modes
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.summary_rows = []

    def _get_plane_slice(self, bx_3d: np.ndarray, by_3d: np.ndarray, bz_3d: np.ndarray, plane: str, slice_idx: int = None):
        """
        PSC stores arrays as (Nz, Ny, Nx).
        Returns two in-plane coordinates plus metadata for physical labeling.
        """
        nz, ny, nx = bx_3d.shape

        if plane == "xy":
            idx = slice_idx if slice_idx is not None else nz // 2
            return {
                "bx": bx_3d[idx, :, :].squeeze(),
                "by": by_3d[idx, :, :].squeeze(),
                "bz": bz_3d[idx, :, :].squeeze(),
                "axes": ("y", "x"),
                "spacing": (self.dy, self.dx),
                "normal_axis": "z",
                "slice_idx": idx,
            }
        if plane == "xz":
            idx = slice_idx if slice_idx is not None else ny // 2
            return {
                "bx": bx_3d[:, idx, :].squeeze(),
                "by": by_3d[:, idx, :].squeeze(),
                "bz": bz_3d[:, idx, :].squeeze(),
                "axes": ("z", "x"),
                "spacing": (self.dy, self.dx),
                "normal_axis": "y",
                "slice_idx": idx,
            }
        if plane == "yz":
            idx = slice_idx if slice_idx is not None else nx // 2
            return {
                "bx": bx_3d[:, :, idx].squeeze(),
                "by": by_3d[:, :, idx].squeeze(),
                "bz": bz_3d[:, :, idx].squeeze(),
                "axes": ("z", "y"),
                "spacing": (self.dy, self.dx),
                "normal_axis": "x",
                "slice_idx": idx,
            }
        raise ValueError(f"Unknown plane {plane}")

    def _window_2d(self, field: np.ndarray) -> np.ndarray:
        win_y = np.hanning(field.shape[0])
        win_x = np.hanning(field.shape[1])
        return field * np.outer(win_y, win_x)

    def _component_fluctuations(self, bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> dict:
        bx_fluct = bx - np.mean(bx)
        by_fluct = by - np.mean(by)
        bz_fluct = bz - np.mean(bz)

        components = {
            "parallel": bz_fluct,
            "perp": np.sqrt(bx_fluct**2 + by_fluct**2),
            "total": np.sqrt(bx_fluct**2 + by_fluct**2 + bz_fluct**2),
        }
        return {
            "bx": bx_fluct,
            "by": by_fluct,
            "bz": bz_fluct,
            **components,
        }

    def _compute_fft_psd(self, field: np.ndarray) -> np.ndarray:
        field_win = self._window_2d(field)
        nx, ny = field_win.shape
        field_k = fftshift(fft2(field_win))
        return np.abs(field_k) ** 2 / (nx * ny) ** 2

    def _compute_k_grids(self, shape: tuple[int, int], spacing: tuple[float, float], axes: tuple[str, str]) -> dict:
        n0, n1 = shape
        d0, d1 = spacing
        k0 = fftshift(np.fft.fftfreq(n0, d=d0)) * 2 * np.pi
        k1 = fftshift(np.fft.fftfreq(n1, d=d1)) * 2 * np.pi
        K0, K1 = np.meshgrid(k0, k1, indexing="ij")

        axis0, axis1 = axes
        if axis0 == self.parallel_axis:
            k_par = K0
            k_perp = K1
        elif axis1 == self.parallel_axis:
            k_par = K1
            k_perp = K0
        else:
            k_par = np.zeros_like(K0)
            k_perp = np.sqrt(K0**2 + K1**2)

        return {
            "k0": k0,
            "k1": k1,
            "K0": K0,
            "K1": K1,
            "k_par": k_par,
            "k_perp": k_perp,
            "k_mag": np.sqrt(K0**2 + K1**2),
        }

    def _radial_spectrum(self, psd_2d: np.ndarray, k_mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positive_k = np.unique(np.sort(k_mag[k_mag > 0]))
        if positive_k.size == 0:
            return np.array([]), np.array([])

        k_min = positive_k[0]
        k_max = np.max(k_mag)
        if k_max <= k_min:
            return np.array([]), np.array([])

        num_bins = max(int(min(psd_2d.shape) // 2), 10)
        bins = np.linspace(k_min, k_max, num_bins + 1)
        k_centers = 0.5 * (bins[1:] + bins[:-1])
        indices = np.digitize(k_mag.ravel(), bins)
        e_k = np.zeros(num_bins)

        for i in range(1, num_bins + 1):
            mask = indices == i
            if np.any(mask):
                e_k[i - 1] = np.sum(psd_2d.ravel()[mask])

        valid = e_k > 0
        return k_centers[valid], e_k[valid]

    def _fit_power_law(self, k: np.ndarray, e_k: np.ndarray) -> dict | None:
        if len(k) < 6:
            return None

        fit_slice = slice(len(k) // 4, max(3 * len(k) // 4, len(k) // 4 + 3))
        k_fit = k[fit_slice]
        e_fit = e_k[fit_slice]
        valid = (k_fit > 0) & (e_fit > 0)
        if np.count_nonzero(valid) < 3:
            return None

        result = linregress(np.log10(k_fit[valid]), np.log10(e_fit[valid]))
        return {
            "slope": result.slope,
            "intercept": result.intercept,
            "rvalue": result.rvalue,
            "k_fit": k_fit[valid],
        }

    def _extract_top_modes(self, psd_2d: np.ndarray, k_grids: dict, step: int, component: str) -> list[dict]:
        flat_indices = np.argsort(psd_2d.ravel())[::-1]
        modes = []

        for flat_idx in flat_indices:
            i, j = np.unravel_index(flat_idx, psd_2d.shape)
            k0 = k_grids["K0"][i, j]
            k1 = k_grids["K1"][i, j]
            if np.isclose(k0, 0.0) and np.isclose(k1, 0.0):
                continue

            modes.append(
                {
                    "step": step,
                    "component": component,
                    "mode_rank": len(modes) + 1,
                    "k_axis0": k0,
                    "k_axis1": k1,
                    "k_parallel": k_grids["k_par"][i, j],
                    "k_perp": k_grids["k_perp"][i, j],
                    "k_mag": k_grids["k_mag"][i, j],
                    "power": psd_2d[i, j],
                }
            )
            if len(modes) >= self.top_modes:
                break

        return modes

    def process_snapshot(self, filepath: str, plane: str = "xy", slice_idx: int = None) -> dict | None:
        try:
            b_fields = PICDataReader.read_multiple_fields_3d(
                filepath, "jeh-", ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"]
            )
        except Exception as exc:
            print(f"Error reading file {filepath}: {exc}")
            return None

        bx_3d = b_fields["hx_fc/p0/3d"] * self.B0_ref
        by_3d = b_fields["hy_fc/p0/3d"] * self.B0_ref
        bz_3d = b_fields["hz_fc/p0/3d"] * self.B0_ref

        plane_data = self._get_plane_slice(bx_3d, by_3d, bz_3d, plane, slice_idx)
        bx = np.atleast_2d(plane_data["bx"])
        by = np.atleast_2d(plane_data["by"])
        bz = np.atleast_2d(plane_data["bz"])

        fluctuations = self._component_fluctuations(bx, by, bz)
        k_grids = self._compute_k_grids(bx.shape, plane_data["spacing"], plane_data["axes"])

        spectra = {}
        for component in ("total", "parallel", "perp"):
            psd_2d = self._compute_fft_psd(fluctuations[component])
            k, e_k = self._radial_spectrum(psd_2d, k_grids["k_mag"])
            spectra[component] = {
                "psd_2d": psd_2d,
                "k": k,
                "E_k": e_k,
                "fit": self._fit_power_law(k, e_k),
            }

        return {
            "axes": plane_data["axes"],
            "normal_axis": plane_data["normal_axis"],
            "slice_idx": plane_data["slice_idx"],
            "k_grids": k_grids,
            "spectra": spectra,
        }

    def analyze_simulation(
        self,
        fields_pattern: str = "pfd.*.bp",
        plane: str = "xy",
        slice_idx: int = None,
        steps_to_process: list[int] = None,
    ):
        print("Starting Spectral Analysis...")
        b_files = PICDataReader.find_files(fields_pattern)
        steps = sorted(b_files.keys())
        if steps_to_process:
            steps = [step for step in steps if step in steps_to_process]

        if not steps:
            print("No snapshot files found.")
            return

        print(f"Found {len(steps)} snapshots to process.")
        all_modes = []

        for step in steps:
            print(f"Processing snapshot step {step}...")
            data = self.process_snapshot(b_files[step], plane=plane, slice_idx=slice_idx)
            if data is None:
                continue

            axes = data["axes"]
            k_grids = data["k_grids"]
            for component, spec in data["spectra"].items():
                self.plot_1d_spectrum(spec["k"], spec["E_k"], spec["fit"], step, plane, component)
                self.plot_2d_spectrum(k_grids, spec["psd_2d"], step, plane, component, axes)
                all_modes.extend(self._extract_top_modes(spec["psd_2d"], k_grids, step, component))

                fit = spec["fit"]
                self.summary_rows.append(
                    {
                        "step": step,
                        "plane": plane,
                        "component": component,
                        "axis0": axes[0],
                        "axis1": axes[1],
                        "slice_normal": data["normal_axis"],
                        "slice_idx": data["slice_idx"],
                        "peak_power": float(np.max(spec["psd_2d"])),
                        "peak_k_parallel": float(k_grids["k_par"].ravel()[np.argmax(spec["psd_2d"])]),
                        "peak_k_perp": float(k_grids["k_perp"].ravel()[np.argmax(spec["psd_2d"])]),
                        "fit_slope": "" if fit is None else fit["slope"],
                        "fit_rvalue": "" if fit is None else fit["rvalue"],
                    }
                )

        self._write_csv(self.outdir / f"spectral_summary_{plane}.csv", self.summary_rows)
        self._write_csv(self.outdir / f"dominant_modes_{plane}.csv", all_modes)
        if all_modes:
            self.plot_mode_evolution(all_modes, plane)
        print("Analysis completed.")

    def _write_csv(self, filepath: Path, rows: list[dict]):
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with filepath.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved data table: {filepath}")

    def plot_1d_spectrum(self, k: np.ndarray, e_k: np.ndarray, fit: dict | None, step: int, plane: str, component: str, save: bool = True):
        plt.figure(figsize=(8, 6))
        plt.loglog(k, e_k, linewidth=2, label=f"{component} $E(k)$")

        if fit is not None and len(fit["k_fit"]) >= 3:
            e_ref = 10 ** (fit["intercept"] + fit["slope"] * np.log10(fit["k_fit"]))
            plt.loglog(
                fit["k_fit"],
                e_ref,
                "k--",
                linewidth=1.5,
                label=fr"fit slope = {fit['slope']:.2f}",
            )

        if len(k) > 10:
            k_ref = k[len(k) // 4 : 3 * len(k) // 4]
            if len(k_ref) > 1:
                e_ref = e_k[len(k) // 4] * (k_ref / k_ref[0]) ** (-5 / 3)
                plt.loglog(k_ref, e_ref, "r:", label="-5/3 ref")

        plt.xlabel(r"Wavenumber $k$")
        plt.ylabel(r"Energy Spectrum $E(k)$")
        plt.title(f"1D Magnetic Spectrum - Step {step} ({plane}, {component})")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()

        if save:
            out_file = self.outdir / f"spectrum_1d_step{step}_{plane}_{component}.png"
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"Saved 1D Spectrum Plot: {out_file}")
        plt.close()

    def plot_2d_spectrum(self, k_grids: dict, psd_2d: np.ndarray, step: int, plane: str, component: str, axes: tuple[str, str], save: bool = True):
        plt.figure(figsize=(8, 7))
        psd_log = np.log10(psd_2d + 1e-16)
        p = plt.pcolormesh(k_grids["K0"], k_grids["K1"], psd_log, shading="auto", cmap="inferno")
        cb = plt.colorbar(p)
        cb.set_label(r"$\log_{10}$(PSD)")

        plt.xlabel(rf"$k_{{{axes[0]}}}$")
        plt.ylabel(rf"$k_{{{axes[1]}}}$")
        plt.title(f"2D PSD - Step {step} ({plane}, {component})")

        if save:
            out_file = self.outdir / f"spectrum_2d_step{step}_{plane}_{component}.png"
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"Saved 2D Spectrum Plot: {out_file}")
        plt.close()

        plt.figure(figsize=(8, 7))
        anis_psd = np.log10(psd_2d + 1e-16)
        p = plt.pcolormesh(k_grids["k_par"], k_grids["k_perp"], anis_psd, shading="auto", cmap="magma")
        cb = plt.colorbar(p)
        cb.set_label(r"$\log_{10}$(PSD)")
        plt.xlabel(r"$k_{\parallel}$")
        plt.ylabel(r"$k_{\perp}$")
        plt.title(f"Anisotropic PSD - Step {step} ({plane}, {component})")

        if save:
            out_file = self.outdir / f"spectrum_anisotropic_step{step}_{plane}_{component}.png"
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"Saved anisotropic PSD Plot: {out_file}")
        plt.close()

    def plot_mode_evolution(self, modes: list[dict], plane: str):
        plt.figure(figsize=(10, 6))
        for component in sorted({mode["component"] for mode in modes}):
            component_modes = [mode for mode in modes if mode["component"] == component and mode["mode_rank"] == 1]
            component_modes.sort(key=lambda row: row["step"])
            if not component_modes:
                continue
            plt.semilogy(
                [row["step"] for row in component_modes],
                [row["power"] for row in component_modes],
                marker="o",
                linewidth=1.8,
                label=f"{component} dominant mode",
            )

        plt.xlabel("Step")
        plt.ylabel("Mode Power")
        plt.title(f"Dominant-Mode Evolution ({plane})")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        out_file = self.outdir / f"dominant_mode_evolution_{plane}.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved dominant mode evolution plot: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral analysis of magnetic-field fluctuations.")
    parser.add_argument("--fields", type=str, default="pfd.*.bp", help="Glob pattern for field files.")
    parser.add_argument("--plane", type=str, default="xy", choices=["xy", "xz", "yz"], help="Plane to extract slice.")
    parser.add_argument("--slice", type=int, default=None, help="Fixed slice index along the plane normal.")
    parser.add_argument("--B0", type=float, default=1.0, help="Reference B0 field normalization.")
    parser.add_argument("--dx", type=float, default=1.0, help="Grid spacing for the second axis in the slice.")
    parser.add_argument("--dy", type=float, default=1.0, help="Grid spacing for the first axis in the slice.")
    parser.add_argument("--parallel-axis", type=str, default="z", choices=["x", "y", "z"], help="Direction of the guide field / parallel axis.")
    parser.add_argument("--top-modes", type=int, default=8, help="Number of strongest spectral modes to store per snapshot.")
    parser.add_argument("--steps", nargs="*", type=int, default=None, help="Specific steps to process.")
    args = parser.parse_args()

    analyzer = SpectralAnalyzer(
        dx=args.dx,
        dy=args.dy,
        B0_ref=args.B0,
        outdir="spectral_plots",
        parallel_axis=args.parallel_axis,
        top_modes=args.top_modes,
    )
    analyzer.analyze_simulation(
        fields_pattern=args.fields,
        plane=args.plane,
        slice_idx=args.slice,
        steps_to_process=args.steps,
    )
