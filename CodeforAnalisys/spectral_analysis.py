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
from numpy.fft import fft2, fftshift

try:
    from scipy.stats import linregress
except ImportError:  # NumPy fallback keeps spectra available without SciPy.
    def linregress(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        rvalue = np.corrcoef(x, y)[0, 1]
        return type(
            "LinearRegression",
            (),
            {
                "slope": float(slope),
                "intercept": float(intercept),
                "rvalue": float(rvalue),
            },
        )()

from data_reader import PICDataReader

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})


class SpectralAnalyzer:
    def __init__(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float | None = None,
        B0_ref: float = 1.0,
        outdir: str = "spectral_plots",
        parallel_axis: str = "z",
        top_modes: int = 8,
    ):
        self.spacing = {
            "x": float(dx),
            "y": float(dy),
            "z": float(dy if dz is None else dz),
        }
        self.B0_ref = B0_ref
        self.parallel_axis = parallel_axis.lower()
        self.top_modes = top_modes
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.summary_rows = []

    @staticmethod
    def _detect_plane(shape: tuple[int, int, int]) -> str:
        """Select the physical non-degenerate plane for PSC ordering (Nz, Ny, Nx)."""
        nz, ny, nx = shape
        if nx == 1:
            return "yz"
        if ny == 1:
            return "xz"
        if nz == 1:
            return "xy"
        # For full 3D data, use a central yz slice so z remains represented.
        return "yz"

    def _get_plane_slice(self, bx_3d: np.ndarray, by_3d: np.ndarray, bz_3d: np.ndarray, plane: str, slice_idx: int = None):
        """
        PSC stores arrays as (Nz, Ny, Nx).
        Returns two in-plane coordinates plus metadata for physical labeling.
        """
        if bx_3d.shape != by_3d.shape or bx_3d.shape != bz_3d.shape:
            raise ValueError(
                "Magnetic-field components have different shapes: "
                f"Bx={bx_3d.shape}, By={by_3d.shape}, Bz={bz_3d.shape}"
            )
        if bx_3d.ndim != 3:
            raise ValueError(f"Expected PSC 3D arrays, received shape {bx_3d.shape}")

        nz, ny, nx = bx_3d.shape
        if plane == "auto":
            plane = self._detect_plane(bx_3d.shape)

        if plane == "xy":
            idx = slice_idx if slice_idx is not None else nz // 2
            if not 0 <= idx < nz:
                raise IndexError(f"xy slice {idx} is outside z range [0, {nz})")
            return {
                "bx": bx_3d[idx, :, :].squeeze(),
                "by": by_3d[idx, :, :].squeeze(),
                "bz": bz_3d[idx, :, :].squeeze(),
                "axes": ("y", "x"),
                "spacing": (self.spacing["y"], self.spacing["x"]),
                "plane": plane,
                "normal_axis": "z",
                "slice_idx": idx,
            }
        if plane == "xz":
            idx = slice_idx if slice_idx is not None else ny // 2
            if not 0 <= idx < ny:
                raise IndexError(f"xz slice {idx} is outside y range [0, {ny})")
            return {
                "bx": bx_3d[:, idx, :].squeeze(),
                "by": by_3d[:, idx, :].squeeze(),
                "bz": bz_3d[:, idx, :].squeeze(),
                "axes": ("z", "x"),
                "spacing": (self.spacing["z"], self.spacing["x"]),
                "plane": plane,
                "normal_axis": "y",
                "slice_idx": idx,
            }
        if plane == "yz":
            idx = slice_idx if slice_idx is not None else nx // 2
            if not 0 <= idx < nx:
                raise IndexError(f"yz slice {idx} is outside x range [0, {nx})")
            return {
                "bx": bx_3d[:, :, idx].squeeze(),
                "by": by_3d[:, :, idx].squeeze(),
                "bz": bz_3d[:, :, idx].squeeze(),
                "axes": ("z", "y"),
                "spacing": (self.spacing["z"], self.spacing["y"]),
                "plane": plane,
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

        return {
            "bx": bx_fluct,
            "by": by_fluct,
            "bz": bz_fluct,
        }

    def _compute_fft_psd(self, field: np.ndarray) -> np.ndarray:
        field_win = self._window_2d(field)
        nx, ny = field_win.shape
        field_k = fftshift(fft2(field_win))
        window_power = np.sum(np.outer(np.hanning(nx), np.hanning(ny)) ** 2)
        if window_power == 0:
            window_power = float(nx * ny)
        return np.abs(field_k) ** 2 / (window_power * nx * ny)

    def _component_psds(self, fluctuations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Combine vector-component powers without FFTing field magnitudes."""
        component_psd = {
            axis: self._compute_fft_psd(fluctuations[f"b{axis}"])
            for axis in ("x", "y", "z")
        }
        perpendicular_axes = [axis for axis in ("x", "y", "z") if axis != self.parallel_axis]
        return {
            "parallel": component_psd[self.parallel_axis],
            "perp": sum(component_psd[axis] for axis in perpendicular_axes),
            "total": sum(component_psd.values()),
        }

    def _compute_k_grids(self, shape: tuple[int, int], spacing: tuple[float, float], axes: tuple[str, str]) -> dict:
        n0, n1 = shape
        d0, d1 = spacing
        k0 = fftshift(np.fft.fftfreq(n0, d=d0)) * 2 * np.pi
        k1 = fftshift(np.fft.fftfreq(n1, d=d1)) * 2 * np.pi
        K0, K1 = np.meshgrid(k0, k1, indexing="ij")

        axis0, axis1 = axes
        if axis0 == self.parallel_axis:
            k_par = K0
            k_perp = np.abs(K1)
        elif axis1 == self.parallel_axis:
            k_par = K1
            k_perp = np.abs(K0)
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
        seen_wavevectors = set()

        for flat_idx in flat_indices:
            i, j = np.unravel_index(flat_idx, psd_2d.shape)
            if not np.isfinite(psd_2d[i, j]) or psd_2d[i, j] <= 0:
                break
            k0 = k_grids["K0"][i, j]
            k1 = k_grids["K1"][i, j]
            if np.isclose(k0, 0.0) and np.isclose(k1, 0.0):
                continue
            wavevector_key = (
                round(abs(float(k_grids["k_par"][i, j])), 12),
                round(abs(float(k_grids["k_perp"][i, j])), 12),
                round(float(k_grids["k_mag"][i, j]), 12),
            )
            if wavevector_key in seen_wavevectors:
                continue
            seen_wavevectors.add(wavevector_key)

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

    @staticmethod
    def _peak_index(psd_2d: np.ndarray, k_grids: dict) -> tuple[int, int] | None:
        nonzero = k_grids["k_mag"] > 0
        if not np.any(nonzero) or not np.any(np.isfinite(psd_2d[nonzero])):
            return None
        candidates = np.where(nonzero, psd_2d, -np.inf)
        peak = np.unravel_index(int(np.argmax(candidates)), psd_2d.shape)
        if not np.isfinite(psd_2d[peak]) or psd_2d[peak] <= 0:
            return None
        return peak

    def process_snapshot(self, filepath: str, plane: str = "auto", slice_idx: int = None) -> dict | None:
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
        component_psds = self._component_psds(fluctuations)

        spectra = {}
        for component in ("total", "parallel", "perp"):
            psd_2d = component_psds[component]
            k, e_k = self._radial_spectrum(psd_2d, k_grids["k_mag"])
            spectra[component] = {
                "psd_2d": psd_2d,
                "k": k,
                "E_k": e_k,
                "fit": self._fit_power_law(k, e_k),
            }

        return {
            "plane": plane_data["plane"],
            "axes": plane_data["axes"],
            "spacing": plane_data["spacing"],
            "normal_axis": plane_data["normal_axis"],
            "slice_idx": plane_data["slice_idx"],
            "k_grids": k_grids,
            "spectra": spectra,
        }

    def analyze_simulation(
        self,
        fields_pattern: str = "pfd.*.h5",
        plane: str = "auto",
        slice_idx: int = None,
        steps_to_process: list[int] = None,
    ) -> int:
        print("Starting Spectral Analysis...")
        self.summary_rows = []
        b_files = PICDataReader.find_files(fields_pattern)
        steps = sorted(b_files.keys())
        if steps_to_process:
            steps = [step for step in steps if step in steps_to_process]

        if not steps:
            print("No snapshot files found.")
            return 0

        print(f"Found {len(steps)} snapshots to process.")
        all_modes = []

        for step in steps:
            print(f"Processing snapshot step {step}...")
            data = self.process_snapshot(b_files[step], plane=plane, slice_idx=slice_idx)
            if data is None:
                continue

            resolved_plane = data["plane"]
            axes = data["axes"]
            k_grids = data["k_grids"]
            for component, spec in data["spectra"].items():
                self.plot_1d_spectrum(spec["k"], spec["E_k"], spec["fit"], step, resolved_plane, component)
                self.plot_2d_spectrum(k_grids, spec["psd_2d"], step, resolved_plane, component, axes)
                all_modes.extend(self._extract_top_modes(spec["psd_2d"], k_grids, step, component))

                fit = spec["fit"]
                peak = self._peak_index(spec["psd_2d"], k_grids)
                if peak is None:
                    peak_power = np.nan
                    peak_k_parallel = np.nan
                    peak_k_perp = np.nan
                else:
                    peak_i, peak_j = peak
                    peak_power = float(spec["psd_2d"][peak_i, peak_j])
                    peak_k_parallel = float(k_grids["k_par"][peak_i, peak_j])
                    peak_k_perp = float(k_grids["k_perp"][peak_i, peak_j])
                self.summary_rows.append(
                    {
                        "step": step,
                        "plane": resolved_plane,
                        "component": component,
                        "axis0": axes[0],
                        "axis1": axes[1],
                        "delta_axis0": data["spacing"][0],
                        "delta_axis1": data["spacing"][1],
                        "slice_normal": data["normal_axis"],
                        "slice_idx": data["slice_idx"],
                        "peak_power": peak_power,
                        "peak_k_parallel": peak_k_parallel,
                        "peak_k_perp": peak_k_perp,
                        "fit_slope": "" if fit is None else fit["slope"],
                        "fit_rvalue": "" if fit is None else fit["rvalue"],
                    }
                )

        if not self.summary_rows:
            print("No snapshots could be analyzed.")
            return 0

        output_plane = self.summary_rows[0]["plane"]
        self._write_csv(self.outdir / f"spectral_summary_{output_plane}.csv", self.summary_rows)
        self._write_csv(self.outdir / f"dominant_modes_{output_plane}.csv", all_modes)
        if all_modes:
            self.plot_mode_evolution(all_modes, output_plane)
        print("Analysis completed.")
        return len({row["step"] for row in self.summary_rows})

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
        p = plt.pcolormesh(k_grids["K1"], k_grids["K0"], psd_log, shading="auto", cmap="inferno")
        cb = plt.colorbar(p)
        cb.set_label(r"$\log_{10}$(PSD)")

        plt.xlabel(rf"$k_{{{axes[1]}}}\,d_i$")
        plt.ylabel(rf"$k_{{{axes[0]}}}\,d_i$")
        plt.title(f"2D PSD - Step {step} ({plane}, {component})")

        if save:
            out_file = self.outdir / f"spectrum_2d_step{step}_{plane}_{component}.png"
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"Saved 2D Spectrum Plot: {out_file}")
        plt.close()

        plt.figure(figsize=(8, 7))
        if np.any(np.abs(k_grids["k_par"]) > 0):
            kpar = np.abs(k_grids["k_par"]).ravel()
            kperp = np.abs(k_grids["k_perp"]).ravel()
            weights = psd_2d.ravel()
            npar = max(8, psd_2d.shape[0] // 2)
            nperp = max(8, psd_2d.shape[1] // 2)
            hist, par_edges, perp_edges = np.histogram2d(
                kpar,
                kperp,
                bins=(npar, nperp),
                range=((0, kpar.max()), (0, kperp.max())),
                weights=weights,
            )
            p = plt.pcolormesh(
                par_edges,
                perp_edges,
                np.log10(hist.T + 1e-30),
                shading="auto",
                cmap="magma",
            )
            cb = plt.colorbar(p)
            cb.set_label(r"$\log_{10}$(binned PSD)")
        else:
            plt.text(
                0.5,
                0.5,
                f"The {plane} plane does not contain the parallel "
                f"{self.parallel_axis}-axis.",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
        plt.xlabel(r"$|k_{\parallel}|\,d_i$")
        plt.ylabel(r"$|k_{\perp}|\,d_i$")
        plt.title(f"Anisotropic PSD - Step {step} ({plane}, {component})")

        if save:
            out_file = self.outdir / f"spectrum_anisotropic_step{step}_{plane}_{component}.png"
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"Saved anisotropic PSD Plot: {out_file}")
        plt.close()

    def plot_mode_evolution(self, modes: list[dict], plane: str):
        first_step = min(mode["step"] for mode in modes)
        components = sorted({mode["component"] for mode in modes})
        if any(component != "total" for component in components):
            components = [component for component in components if component != "total"]

        fig, ax = plt.subplots(figsize=(10.5, 6.4))
        for component in components:
            component_modes = [mode for mode in modes if mode["component"] == component and mode["mode_rank"] == 1]
            component_modes = [mode for mode in component_modes if mode["step"] != first_step]
            component_modes.sort(key=lambda row: row["step"])
            if not component_modes:
                continue
            ax.semilogy(
                [row["step"] for row in component_modes],
                [row["power"] for row in component_modes],
                marker="o",
                linewidth=2.2,
                markersize=6.5,
                label=f"{component} dominant mode",
            )

        ax.set_xlabel("Step")
        ax.set_ylabel("Mode Power")
        ax.set_title(f"Dominant-Mode Evolution ({plane})")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend()
        out_file = self.outdir / f"dominant_mode_evolution_{plane}.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved dominant mode evolution plot: {out_file}")


if __name__ == "__main__":
    try:
        from psc_units import DX_DI as PROFILE_DX_DI
    except (ImportError, ValueError):
        PROFILE_DX_DI = 1.0

    parser = argparse.ArgumentParser(description="Spectral analysis of magnetic-field fluctuations.")
    parser.add_argument("--fields", type=str, default="pfd.*.h5", help="Glob pattern for field files.")
    parser.add_argument("--plane", type=str, default="auto", choices=["auto", "xy", "xz", "yz"], help="Plane to extract; auto selects the non-degenerate PSC plane.")
    parser.add_argument("--slice", type=int, default=None, help="Fixed slice index along the plane normal.")
    parser.add_argument("--B0", type=float, default=1.0, help="Reference B0 field normalization.")
    parser.add_argument("--dx", type=float, default=PROFILE_DX_DI, help="Grid spacing along x in d_i.")
    parser.add_argument("--dy", type=float, default=PROFILE_DX_DI, help="Grid spacing along y in d_i.")
    parser.add_argument("--dz", type=float, default=PROFILE_DX_DI, help="Grid spacing along z in d_i.")
    parser.add_argument("--parallel-axis", type=str, default="z", choices=["x", "y", "z"], help="Direction of the guide field / parallel axis.")
    parser.add_argument("--top-modes", type=int, default=8, help="Number of strongest spectral modes to store per snapshot.")
    parser.add_argument("--steps", nargs="*", type=int, default=None, help="Specific steps to process.")
    parser.add_argument("--outdir", type=str, default="spectral_plots", help="Directory for spectral plots and CSV outputs.")
    args = parser.parse_args()

    analyzer = SpectralAnalyzer(
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        B0_ref=args.B0,
        outdir=args.outdir,
        parallel_axis=args.parallel_axis,
        top_modes=args.top_modes,
    )
    processed = analyzer.analyze_simulation(
        fields_pattern=args.fields,
        plane=args.plane,
        slice_idx=args.slice,
        steps_to_process=args.steps,
    )
    if processed == 0:
        raise SystemExit(1)
