#!/usr/bin/env python3
"""
spectral_analysis.py
====================
Spectral analysis of magnetic-field fluctuations for PSC outputs.

Features:
  - Correct 2D slice extraction for PSC ordering (Nz, Ny, Nx)
  - 2D anisotropic PSD in physical axes (perp/parallel/total), reused by
    physical_diagnostics.py for the single-snapshot P(k) diagnostic
  - Mode-resolved growth rate gamma(k): E(k, Omega_ci t) is accumulated over
    every snapshot and each k-shell is fit with a log-linear growth rate,
    following the gamma(k) technique described for PIC instability studies
    (e.g. Hellinger et al. 2018) instead of one static spectrum per snapshot.
  - Reduced magnetic helicity sigma_m(k) and parallel/perpendicular
    compressibility, used to discriminate mirror/EMIC/firehose branches.
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
from psc_units import DX_DI, DI, DOMAIN_DI, RHO_I, step_to_omegaci

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


def _fit_growth_rate(time: np.ndarray, amplitude: np.ndarray) -> dict:
    """Log-linear growth rate of a mode amplitude ~ exp(gamma * t).

    Mirrors the linear-phase window selection used for the box-integrated
    delta_B_rms(t) fit, applied here per k-shell so every mode gets its own
    gamma instead of a single global number.
    """
    valid = np.isfinite(time) & np.isfinite(amplitude) & (amplitude > 0)
    if np.count_nonzero(valid) < 4:
        return {"gamma": np.nan, "rvalue": np.nan, "n_points": int(np.count_nonzero(valid))}

    t = time[valid]
    y = np.log(amplitude[valid])
    slope_local = np.gradient(y, t)
    positive = slope_local > 0
    if np.count_nonzero(positive) >= 3:
        idx_valid = np.where(positive)[0]
        lo = idx_valid[max(0, int(0.15 * len(idx_valid)))]
        hi = idx_valid[min(len(idx_valid) - 1, int(0.85 * len(idx_valid)))]
        fit_slice = slice(lo, hi + 1)
    else:
        fit_slice = slice(1, -1)
    if len(t[fit_slice]) < 3:
        fit_slice = slice(None)

    coeff = np.polyfit(t[fit_slice], y[fit_slice], 1)
    fitted = np.polyval(coeff, t[fit_slice])
    ss_res = float(np.sum((y[fit_slice] - fitted) ** 2))
    ss_tot = float(np.sum((y[fit_slice] - np.mean(y[fit_slice])) ** 2))
    rvalue = float(np.sqrt(max(0.0, 1.0 - ss_res / ss_tot))) if ss_tot > 0 else np.nan
    return {
        "gamma": float(coeff[0]),
        "rvalue": rvalue,
        "n_points": int(len(t[fit_slice])),
    }


def report_box_resolution(k_ratio_range: tuple[float, float] = (0.3, 0.5)) -> str:
    """L_box/rho_i for the active PSC_PROFILE, and whether the box's longest
    mode (k_min) is long enough to contain the fastest-growing mirror
    wavelength (k*rho_i ~ 0.3-0.5, Hellinger et al. 2006). A box with only a
    few rho_i per side cannot fit that wavelength even once."""
    rho_i_di = RHO_I / DI
    l_over_rho = DOMAIN_DI / rho_i_di
    k_min_di = 2.0 * np.pi / DOMAIN_DI
    k_min_rho = k_min_di * rho_i_di
    lo, hi = k_ratio_range
    lines = [
        f"L_box = {DOMAIN_DI:.1f} d_i = {l_over_rho:.2f} rho_i "
        f"(rho_i = {rho_i_di:.3f} d_i)",
        f"Box fundamental mode: k_min d_i = {k_min_di:.3f}, "
        f"k_min rho_i = {k_min_rho:.3f}",
    ]
    if l_over_rho < 20.0:
        lines.append(
            f"[WARN] L_box/rho_i = {l_over_rho:.1f} is below the ~20-40 rho_i "
            "per side usually needed to resolve mirror; the most unstable "
            f"wavelength (k rho_i ~ {lo}-{hi}) needs L_box >= "
            f"{2.0 * np.pi / hi:.1f} rho_i just to fit once."
        )
    if k_min_rho > hi:
        lines.append(
            f"[WARN] k_min rho_i = {k_min_rho:.3f} is already above the "
            f"peak-growth range (k rho_i ~ {lo}-{hi}); the box may be too "
            "small to contain the fastest-growing mirror mode at all."
        )
    return "\n".join(lines)


class SpectralAnalyzer:
    def __init__(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float | None = None,
        B0_ref: float = 1.0,
        outdir: str = "spectral_plots",
        parallel_axis: str = "z",
    ):
        self.spacing = {
            "x": float(dx),
            "y": float(dy),
            "z": float(dy if dz is None else dz),
        }
        self.B0_ref = B0_ref
        self.parallel_axis = parallel_axis.lower()
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

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

    def _compute_fft_complex(self, field: np.ndarray) -> np.ndarray:
        field_win = self._window_2d(field)
        return fftshift(fft2(field_win))

    def _compute_fft_psd(self, field: np.ndarray) -> np.ndarray:
        nx, ny = field.shape
        field_k = self._compute_fft_complex(field)
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
        """Single-snapshot isotropic E(k), kept for physical_diagnostics.py's P(k) plot."""
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

    @staticmethod
    def _radial_bin_edges(k_mag: np.ndarray, num_bins: int, k_max_di: float | None = None) -> np.ndarray:
        """Fixed, log-spaced bin edges computed once so every snapshot bins onto
        the same k axis; capped at ``k_max_di`` to keep resolution on the
        physically relevant scales instead of wasting bins on grid noise."""
        positive_k = k_mag[k_mag > 0]
        k_min = float(np.min(positive_k))
        k_max = float(np.max(k_mag))
        if k_max_di is not None:
            k_max = min(k_max, k_max_di)
        return np.geomspace(k_min, k_max, num_bins + 1)

    @staticmethod
    def _bin_radial_sum(values: np.ndarray, k_mag: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
        """Sum ``values`` on fixed k-shells; unlike _radial_spectrum this keeps a
        constant-length output (including empty/negative shells) so it can be
        stacked across snapshots into an E(k, t) matrix."""
        sums, _ = np.histogram(
            k_mag.ravel(), bins=bin_edges, weights=np.asarray(values, dtype=float).ravel()
        )
        return sums

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
        # Complex spectra (phase kept) of the raw component fluctuations, used to
        # build the reduced magnetic helicity sigma_m(k) between the two axes
        # perpendicular to the guide field.
        component_ffts = {
            axis: self._compute_fft_complex(fluctuations[f"b{axis}"])
            for axis in ("x", "y", "z")
        }

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
            "component_ffts": component_ffts,
        }

    def analyze_simulation(
        self,
        fields_pattern: str = "pfd.*.h5",
        plane: str = "auto",
        slice_idx: int = None,
        steps_to_process: list[int] = None,
        k_max_di: float = 2.0,
    ) -> int:
        """Mode-resolved growth rate gamma(k): accumulate E(k, Omega_ci t) across
        every snapshot and fit a log-linear growth rate per k-shell, instead of
        rendering one static P(k) image per snapshot."""
        print("Starting spectral growth-rate analysis...")
        print(report_box_resolution())
        b_files = PICDataReader.find_files(fields_pattern)
        steps = sorted(b_files.keys())
        if steps_to_process:
            steps = [step for step in steps if step in steps_to_process]

        # The first snapshot's "fluctuation" is delta = field - mean(field) on a
        # single, barely-relaxed quiet-start snapshot; it is not a meaningful
        # sample of the growing mode and would bias the t=0 end of every fit.
        steps = steps[1:]

        if len(steps) < 4:
            print(f"Need at least four snapshots (after dropping the first) for a growth-rate fit; found {len(steps)}.")
            return 0

        print(f"Found {len(steps)} snapshots to process (first snapshot dropped).")
        times = np.array([step_to_omegaci(step) for step in steps], dtype=float)
        perpendicular_axes = [axis for axis in ("x", "y", "z") if axis != self.parallel_axis]

        bin_edges = None
        resolved_plane = None
        energy_perp_rows = []
        energy_par_rows = []
        helicity_num_rows = []
        helicity_den_rows = []
        valid_times = []

        for step, t in zip(steps, times):
            print(f"Processing snapshot step {step}...")
            data = self.process_snapshot(b_files[step], plane=plane, slice_idx=slice_idx)
            if data is None:
                continue

            k_mag = data["k_grids"]["k_mag"]
            if bin_edges is None:
                resolved_plane = data["plane"]
                if self.parallel_axis == data["normal_axis"]:
                    print(
                        f"[WARN] parallel_axis='{self.parallel_axis}' is the plane's "
                        f"collapsed axis ('{data['normal_axis']}'): B0 would be "
                        f"perpendicular to the analyzed {resolved_plane} plane, so no "
                        "oblique/mirror mode can be represented here by construction."
                    )
                num_bins = max(int(min(k_mag.shape) // 2), 10)
                bin_edges = self._radial_bin_edges(k_mag, num_bins, k_max_di=k_max_di)

            energy_perp_rows.append(
                self._bin_radial_sum(data["spectra"]["perp"]["psd_2d"], k_mag, bin_edges)
            )
            energy_par_rows.append(
                self._bin_radial_sum(data["spectra"]["parallel"]["psd_2d"], k_mag, bin_edges)
            )

            b1 = data["component_ffts"][perpendicular_axes[0]]
            b2 = data["component_ffts"][perpendicular_axes[1]]
            helicity_num_rows.append(
                self._bin_radial_sum(np.imag(np.conj(b1) * b2), k_mag, bin_edges)
            )
            helicity_den_rows.append(
                self._bin_radial_sum(np.abs(b1) ** 2 + np.abs(b2) ** 2, k_mag, bin_edges)
            )
            valid_times.append(t)

        if bin_edges is None or len(valid_times) < 4:
            print("No snapshots could be analyzed.")
            return 0

        valid_times = np.asarray(valid_times, dtype=float)
        k_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric center of log-spaced bins
        energy_perp = np.asarray(energy_perp_rows).T  # (n_k, n_t)
        energy_par = np.asarray(energy_par_rows).T
        helicity_den = np.asarray(helicity_den_rows).T
        helicity_num = np.asarray(helicity_num_rows).T

        growth_rows = []
        for i, k in enumerate(k_centers):
            fit_perp = _fit_growth_rate(valid_times, np.sqrt(np.clip(energy_perp[i], 0, None)))
            fit_par = _fit_growth_rate(valid_times, np.sqrt(np.clip(energy_par[i], 0, None)))
            weight = helicity_den[i]
            sigma_m_k = (
                float(np.sum(helicity_num[i] * weight) / np.sum(weight))
                if np.sum(weight) > 0 else np.nan
            )
            growth_rows.append({
                "k": float(k),
                "gamma_perp": fit_perp["gamma"],
                "gamma_perp_rvalue": fit_perp["rvalue"],
                "gamma_perp_npoints": fit_perp["n_points"],
                "gamma_parallel": fit_par["gamma"],
                "gamma_parallel_rvalue": fit_par["rvalue"],
                "gamma_parallel_npoints": fit_par["n_points"],
                "sigma_m": sigma_m_k,
                "final_energy_perp": float(energy_perp[i, -1]),
                "final_energy_parallel": float(energy_par[i, -1]),
            })

        total_par = energy_par.sum(axis=0)
        total_perp = energy_perp.sum(axis=0)
        compressibility = np.divide(
            total_par, total_par + total_perp,
            out=np.full_like(total_par, np.nan), where=(total_par + total_perp) > 0,
        )
        compressibility_rows = [
            {"step": step, "omega_ci_t": float(t), "compressibility": float(c)}
            for step, t, c in zip(steps, valid_times, compressibility)
        ]

        energy_kt_rows = [
            {
                "step": step, "omega_ci_t": float(t), "k": float(k),
                "E_perp": float(energy_perp[i, j]), "E_parallel": float(energy_par[i, j]),
            }
            for j, (step, t) in enumerate(zip(steps, valid_times))
            for i, k in enumerate(k_centers)
        ]

        self._write_csv(self.outdir / f"growth_rate_by_k_{resolved_plane}.csv", growth_rows)
        self._write_csv(self.outdir / f"compressibility_vs_time_{resolved_plane}.csv", compressibility_rows)
        self._write_csv(self.outdir / f"field_energy_kt_{resolved_plane}.csv", energy_kt_rows)

        self.plot_energy_kt(k_centers, valid_times, energy_perp, "perp", resolved_plane)
        self.plot_energy_kt(k_centers, valid_times, energy_par, "parallel", resolved_plane)
        self.plot_growth_rate_vs_k(growth_rows, resolved_plane)
        self.plot_compressibility(valid_times, compressibility, resolved_plane)
        self.plot_helicity_vs_k(growth_rows, resolved_plane)

        # Weak-power bins can still fit a clean exponential (spectral leakage
        # tracks whatever mode dominates the box), so the headline peak is
        # restricted to shells carrying a non-negligible share of the energy
        # to avoid reporting a numerical-noise/leakage bin as "the" mode.
        max_final_perp = max((row["final_energy_perp"] for row in growth_rows), default=0.0)
        max_final_par = max((row["final_energy_parallel"] for row in growth_rows), default=0.0)
        significant_perp = [
            row for row in growth_rows
            if np.isfinite(row["gamma_perp"]) and row["final_energy_perp"] >= 1e-3 * max_final_perp
        ]
        significant_par = [
            row for row in growth_rows
            if np.isfinite(row["gamma_parallel"]) and row["final_energy_parallel"] >= 1e-3 * max_final_par
        ]
        if significant_perp:
            peak = max(significant_perp, key=lambda row: row["gamma_perp"])
            print(
                f"  Max gamma_perp = {peak['gamma_perp']:.4g} Omega_ci at "
                f"k d_i = {peak['k']:.3g} (r={peak['gamma_perp_rvalue']:.2f})"
            )
        if significant_par:
            peak = max(significant_par, key=lambda row: row["gamma_parallel"])
            print(
                f"  Max gamma_parallel = {peak['gamma_parallel']:.4g} Omega_ci at "
                f"k d_i = {peak['k']:.3g} (r={peak['gamma_parallel_rvalue']:.2f})"
            )
        print("Analysis completed.")
        return len(valid_times)

    def _write_csv(self, filepath: Path, rows: list[dict]):
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with filepath.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved data table: {filepath}")

    def plot_energy_kt(self, k_centers: np.ndarray, times: np.ndarray, energy: np.ndarray, component: str, plane: str):
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        log_energy = np.log10(np.clip(energy, np.finfo(float).tiny, None))
        mesh = ax.pcolormesh(times, k_centers, log_energy, shading="auto", cmap="inferno")
        cb = fig.colorbar(mesh, ax=ax)
        cb.set_label(r"$\log_{10} E(k,t)$")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\Omega_{ci} t$")
        ax.set_ylabel(r"$k\,d_i$")
        ax.set_title(f"Mode energy evolution E(k,t) — {component} ({plane})")
        out_file = self.outdir / f"energy_kt_{component}_{plane}.png"
        fig.savefig(out_file, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved E(k,t) map: {out_file}")

    def plot_growth_rate_vs_k(self, growth_rows: list[dict], plane: str):
        k = np.array([row["k"] for row in growth_rows])
        gamma_perp = np.array([row["gamma_perp"] for row in growth_rows])
        gamma_par = np.array([row["gamma_parallel"] for row in growth_rows])

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.axhline(0.0, color="0.6", lw=1.0)
        ax.plot(k, gamma_perp, "o-", markersize=4, label=r"$\gamma_\perp(k)$ (transverse, EMIC/firehose-like)")
        ax.plot(k, gamma_par, "s-", markersize=4, label=r"$\gamma_\parallel(k)$ (compressive, mirror — primary indicator)")
        ax.set_xscale("log")
        ax.set_xlabel(r"$k\,d_i$")
        ax.set_ylabel(r"$\gamma\ [\Omega_{ci}]$")
        ax.set_title(f"Mode-resolved growth rate ({plane})")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        out_file = self.outdir / f"growth_rate_vs_k_{plane}.png"
        fig.savefig(out_file, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved growth rate plot: {out_file}")

    def plot_compressibility(self, times: np.ndarray, compressibility: np.ndarray, plane: str):
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.plot(times, compressibility, lw=2.0, color="firebrick")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$\Omega_{ci} t$")
        ax.set_ylabel(r"$\delta B_\parallel^2 / (\delta B_\parallel^2+\delta B_\perp^2)$")
        ax.set_title(f"Magnetic compressibility ({plane})")
        ax.grid(alpha=0.3)
        out_file = self.outdir / f"compressibility_vs_time_{plane}.png"
        fig.savefig(out_file, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved compressibility plot: {out_file}")

    def plot_helicity_vs_k(self, growth_rows: list[dict], plane: str):
        k = np.array([row["k"] for row in growth_rows])
        sigma_m = np.array([row["sigma_m"] for row in growth_rows])

        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.axhline(0.0, color="0.6", lw=1.0)
        ax.plot(k, sigma_m, "o-", markersize=4, color="teal")
        ax.set_xscale("log")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r"$k\,d_i$")
        ax.set_ylabel(r"$\sigma_m(k)$")
        ax.set_title(f"Reduced magnetic helicity ({plane})")
        ax.grid(alpha=0.3, which="both")
        out_file = self.outdir / f"helicity_vs_k_{plane}.png"
        fig.savefig(out_file, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved helicity plot: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mode-resolved growth rate gamma(k) from magnetic-field fluctuations.")
    parser.add_argument("--fields", type=str, default="pfd.*.h5", help="Glob pattern for field files.")
    parser.add_argument("--plane", type=str, default="auto", choices=["auto", "xy", "xz", "yz"], help="Plane to extract; auto selects the non-degenerate PSC plane.")
    parser.add_argument("--slice", type=int, default=None, help="Fixed slice index along the plane normal.")
    parser.add_argument("--B0", type=float, default=1.0, help="Reference B0 field normalization.")
    parser.add_argument("--dx", type=float, default=DX_DI, help="Grid spacing along x in d_i.")
    parser.add_argument("--dy", type=float, default=DX_DI, help="Grid spacing along y in d_i.")
    parser.add_argument("--dz", type=float, default=DX_DI, help="Grid spacing along z in d_i.")
    parser.add_argument("--parallel-axis", type=str, default="z", choices=["x", "y", "z"], help="Direction of the guide field / parallel axis.")
    parser.add_argument("--k-max-di", type=float, default=2.0, help="Upper bound k*d_i for the gamma(k) reconstruction; higher bins are grid noise, not physical modes.")
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
    )
    processed = analyzer.analyze_simulation(
        fields_pattern=args.fields,
        plane=args.plane,
        slice_idx=args.slice,
        steps_to_process=args.steps,
        k_max_di=args.k_max_di,
    )
    if processed == 0:
        raise SystemExit(1)
