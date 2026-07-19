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
from psc_units import DX_DI, DI, DOMAIN_DI, RHO_I, INSTABILITY, PROFILE_LABEL, step_to_omegaci

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

# Same dark poster style used across anisotropy_analysis.py / physical_diagnostics.py,
# so plots from every stage of the pipeline (mirror, firehose, whistler, any
# strength) look like one system instead of one-off matplotlib defaults.
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
TEXT_CLR = "#e6edf3"
GRID_CLR = "#30363d"
POSTER_TITLE = 19
POSTER_TICK = 15
POSTER_LEGEND = 14

# Which channel carries the instability's growth signature depends on which
# case is active: mirror/oblique-firehose are compressive (parallel channel),
# EMIC/parallel-firehose/whistler are transverse (perp channel). Never assume
# "mirror" — this module runs unmodified for every reference case.
_CHANNEL_HINT = {
    "mirror": {"perp": "transverse", "parallel": "compressive — mirror signature"},
    "firehose": {"perp": "transverse — parallel-firehose/EMIC signature", "parallel": "compressive — oblique-firehose signature"},
    "whistler": {"perp": "transverse — whistler signature", "parallel": "compressive"},
}.get(INSTABILITY, {"perp": "transverse", "parallel": "compressive"})


def _style_axes(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, direction="in", which="both", top=True, right=True, labelsize=POSTER_TICK)
    ax.grid(True, color=GRID_CLR, alpha=0.25, linestyle=":")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)


def _new_dark_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    return fig, ax


def _fit_growth_rate(time: np.ndarray, amplitude: np.ndarray) -> dict:
    """Log-linear growth rate of a mode amplitude ~ exp(gamma * t).

    Mirrors the linear-phase window selection used for the box-integrated
    delta_B_rms(t) fit, applied here per k-shell so every mode gets its own
    gamma instead of a single global number.
    """
    valid = np.isfinite(time) & np.isfinite(amplitude) & (amplitude > 0)
    if np.count_nonzero(valid) < 4:
        return {
            "gamma": np.nan, "intercept": np.nan, "rvalue": np.nan,
            "n_points": int(np.count_nonzero(valid)), "fit_time_range": None,
        }

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
        "intercept": float(coeff[1]),
        "rvalue": rvalue,
        "n_points": int(len(t[fit_slice])),
        "fit_time_range": (float(t[fit_slice][0]), float(t[fit_slice][-1])),
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
            # num_bins is normally sized for the full Nyquist range; cutting
            # k_max down to the physically relevant range leaves far fewer
            # distinct k-shells available (e.g. only ~20 below k d_i = 2 on a
            # 1024^2 grid). Sizing bins for the full range here would make
            # almost every one empty (NaN growth rate, gaps in the plots).
            available = np.unique(positive_k[positive_k <= k_max])
            num_bins = int(np.clip(available.size // 2, 4, num_bins))
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
        half_plane = None
        bin_counts = None
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
                bin_counts, _ = np.histogram(k_mag.ravel(), bins=bin_edges)
                # Non-redundant half of k-space (k and -k carry conjugate, not
                # independent, information for a real field): binning the full
                # annulus makes Im(conj(b1)*b2) cancel between +k/-k exactly,
                # so sigma_m(k) would be ~0 everywhere by construction. Restrict
                # the helicity numerator/denominator to one half-plane instead.
                K0, K1 = data["k_grids"]["K0"], data["k_grids"]["K1"]
                half_plane = (K1 > 1e-12) | ((np.abs(K1) <= 1e-12) & (K0 > 1e-12))

            energy_perp_rows.append(
                self._bin_radial_sum(data["spectra"]["perp"]["psd_2d"], k_mag, bin_edges)
            )
            energy_par_rows.append(
                self._bin_radial_sum(data["spectra"]["parallel"]["psd_2d"], k_mag, bin_edges)
            )

            b1 = data["component_ffts"][perpendicular_axes[0]]
            b2 = data["component_ffts"][perpendicular_axes[1]]
            helicity_num_rows.append(
                self._bin_radial_sum(
                    np.where(half_plane, np.imag(np.conj(b1) * b2), 0.0), k_mag, bin_edges
                )
            )
            helicity_den_rows.append(
                self._bin_radial_sum(
                    np.where(half_plane, np.abs(b1) ** 2 + np.abs(b2) ** 2, 0.0), k_mag, bin_edges
                )
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

        # Drop k-shells that contain zero grid points (possible at this coarse,
        # k_max_di-capped resolution): they carry exact-zero energy/NaN gamma
        # and otherwise show up as a solid black gap in the E(k,t) maps and a
        # dead row in growth_rate_by_k.csv instead of being left out cleanly.
        occupied_bins = bin_counts > 0
        if not np.all(occupied_bins):
            dropped = k_centers[~occupied_bins]
            print(f"[INFO] Dropping {np.count_nonzero(~occupied_bins)} empty k-shell(s) "
                  f"(no grid points at k*d_i = {np.round(dropped, 3).tolist()}).")
            k_centers = k_centers[occupied_bins]
            energy_perp = energy_perp[occupied_bins]
            energy_par = energy_par[occupied_bins]
            helicity_den = helicity_den[occupied_bins]
            helicity_num = helicity_num[occupied_bins]

        growth_rows = []
        for i, k in enumerate(k_centers):
            fit_perp = _fit_growth_rate(valid_times, np.sqrt(np.clip(energy_perp[i], 0, None)))
            fit_par = _fit_growth_rate(valid_times, np.sqrt(np.clip(energy_par[i], 0, None)))
            # Power-weighted time average of the per-snapshot helicity fraction
            # sigma_m(t) = num_t/den_t, i.e. sum(w_t * x_t)/sum(w_t) with w_t =
            # den_t: since w_t * x_t = num_t, that reduces to sum(num_t)/sum(den_t)
            # (previously this multiplied num_t by den_t again, over-weighting
            # high-power snapshots quadratically instead of linearly).
            weight = helicity_den[i]
            sigma_m_k = (
                float(np.sum(helicity_num[i]) / np.sum(weight))
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
            i for i, row in enumerate(growth_rows)
            if np.isfinite(row["gamma_perp"]) and row["final_energy_perp"] >= 1e-3 * max_final_perp
        ]
        significant_par = [
            i for i, row in enumerate(growth_rows)
            if np.isfinite(row["gamma_parallel"]) and row["final_energy_parallel"] >= 1e-3 * max_final_par
        ]
        # Fall back to the loudest bin (even a decaying/flat one) so every case
        # still gets a growth-curve plot instead of silently skipping it.
        peak_perp_idx = max(significant_perp, key=lambda i: growth_rows[i]["gamma_perp"]) \
            if significant_perp else int(np.argmax(energy_perp[:, -1]))
        peak_par_idx = max(significant_par, key=lambda i: growth_rows[i]["gamma_parallel"]) \
            if significant_par else int(np.argmax(energy_par[:, -1]))

        if significant_perp:
            row = growth_rows[peak_perp_idx]
            print(
                f"  Max gamma_perp = {row['gamma_perp']:.4g} Omega_ci at "
                f"k d_i = {row['k']:.3g} (r={row['gamma_perp_rvalue']:.2f})"
            )
        if significant_par:
            row = growth_rows[peak_par_idx]
            print(
                f"  Max gamma_parallel = {row['gamma_parallel']:.4g} Omega_ci at "
                f"k d_i = {row['k']:.3g} (r={row['gamma_parallel_rvalue']:.2f})"
            )

        self.plot_growth_curve(
            valid_times, energy_perp[peak_perp_idx], k_centers[peak_perp_idx],
            "perp", resolved_plane,
        )
        self.plot_growth_curve(
            valid_times, energy_par[peak_par_idx], k_centers[peak_par_idx],
            "parallel", resolved_plane,
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
        fig, ax = _new_dark_fig((9.5, 6.5))
        # Relative log scale (like dispersion_analysis.py's density map): E(k,t)
        # spans many decades from noise floor to saturation, so an absolute
        # log10 colormap is dominated by the emptiest cells and reads as a
        # blank/"null" plot. Normalizing by the run's own peak and clipping the
        # bottom six decades keeps the growing region visible.
        peak = float(np.max(energy)) if np.any(np.isfinite(energy)) and np.max(energy) > 0 else 1.0
        log_rel = np.log10(np.clip(energy / peak, 1e-6, None))
        mesh = ax.pcolormesh(times, k_centers, log_rel, shading="auto", cmap="inferno", vmin=-6, vmax=0)
        cb = fig.colorbar(mesh, ax=ax)
        cb.set_label(r"$\log_{10}[E(k,t)/E_{\max}]$", color=TEXT_CLR)
        cb.ax.yaxis.set_tick_params(color=TEXT_CLR)
        plt.setp(plt.getp(cb.ax, "yticklabels"), color=TEXT_CLR)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\Omega_{ci} t$")
        ax.set_ylabel(r"$k\,d_i$")
        ax.set_title(f"E(k,t) — {component} ({_CHANNEL_HINT[component]}, {PROFILE_LABEL}, {plane})", fontsize=POSTER_TITLE)
        out_file = self.outdir / f"energy_kt_{component}_{plane}.png"
        fig.savefig(out_file, dpi=220, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"Saved E(k,t) map: {out_file}")

    def plot_growth_curve(self, times: np.ndarray, energy_k: np.ndarray, k: float, component: str, plane: str):
        """The classic growth-rate check plot: amplitude of a single k-shell vs
        time on a log axis, with the fitted exponential overlaid, so growth
        (or its absence) is visible directly instead of only as a derived
        gamma(k) number."""
        amplitude = np.sqrt(np.clip(energy_k, 0, None))
        fit = _fit_growth_rate(times, amplitude)

        fig, ax = _new_dark_fig((8.5, 5.5))
        ax.set_xlim(float(np.min(times)), float(np.max(times)))
        valid = amplitude > 0
        if np.count_nonzero(valid) < 2:
            # This channel never rose above zero/floating-point noise (e.g. a
            # non-fluctuating parallel field): say so instead of leaving a
            # blank plot with an arbitrary autoscaled axis.
            ax.text(
                0.5, 0.5, "no measurable amplitude in this channel\n(at or below numerical noise floor)",
                transform=ax.transAxes, ha="center", va="center", color=TEXT_CLR, fontsize=13,
            )
        else:
            ax.semilogy(times[valid], amplitude[valid], "o", color="#58a6ff", markersize=5, label="measured")
            if np.isfinite(fit["gamma"]):
                lo, hi = fit["fit_time_range"]
                ax.axvspan(lo, hi, color=GRID_CLR, alpha=0.4, label="fit window")
                fitted = np.exp(np.polyval([fit["gamma"], fit["intercept"]], times))
                ax.semilogy(times, fitted, "--", color="#f0883e", lw=2.0,
                            label=fr"fit $\gamma={fit['gamma']:.3g}\,\Omega_{{ci}}$ (r={fit['rvalue']:.2f})")
        ax.set_xlabel(r"$\Omega_{ci} t$")
        ax.set_ylabel(r"$\sqrt{E(k,t)}$  (field amplitude, a.u.)")
        ax.set_title(
            f"Growth curve — {component} ({_CHANNEL_HINT[component]}), "
            f"$k\\,d_i={k:.3g}$, {PROFILE_LABEL} ({plane})",
            fontsize=POSTER_TITLE - 2,
        )
        if ax.get_legend_handles_labels()[0]:
            ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
        out_file = self.outdir / f"growth_curve_{component}_{plane}.png"
        fig.savefig(out_file, dpi=220, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"Saved growth curve: {out_file}")

    def plot_growth_rate_vs_k(self, growth_rows: list[dict], plane: str, significance_threshold: float = 1e-3):
        """gamma(k), with k-shells whose final energy never rose above
        ``significance_threshold`` of the run's peak drawn as faint,
        unconnected points instead of a solid line. A log-linear fit to a
        numerical-noise-floor amplitude can still return a "significant"
        looking gamma with decent R^2, which otherwise reads as a real
        growing mode at that k (see the analyze_simulation peak-selection
        filter, which already applies this same threshold)."""
        k = np.array([row["k"] for row in growth_rows])
        gamma_perp = np.array([row["gamma_perp"] for row in growth_rows])
        gamma_par = np.array([row["gamma_parallel"] for row in growth_rows])
        final_perp = np.array([row["final_energy_perp"] for row in growth_rows])
        final_par = np.array([row["final_energy_parallel"] for row in growth_rows])

        max_perp = float(np.max(final_perp)) if np.any(np.isfinite(final_perp)) and np.max(final_perp) > 0 else 1.0
        max_par = float(np.max(final_par)) if np.any(np.isfinite(final_par)) and np.max(final_par) > 0 else 1.0
        sig_perp = np.isfinite(gamma_perp) & (final_perp >= significance_threshold * max_perp)
        sig_par = np.isfinite(gamma_par) & (final_par >= significance_threshold * max_par)

        fig, ax = _new_dark_fig((9, 6))
        ax.axhline(0.0, color=GRID_CLR, lw=1.2)
        ax.plot(k[sig_perp], gamma_perp[sig_perp], "o-", markersize=4, color="#58a6ff",
                label=fr"$\gamma_\perp(k)$ ({_CHANNEL_HINT['perp']})")
        ax.plot(k[sig_par], gamma_par[sig_par], "s-", markersize=4, color="#f0883e",
                label=fr"$\gamma_\parallel(k)$ ({_CHANNEL_HINT['parallel']})")
        if np.any(~sig_perp):
            ax.plot(k[~sig_perp], gamma_perp[~sig_perp], "o", markersize=4,
                    color="#58a6ff", alpha=0.25, markeredgecolor="none")
        if np.any(~sig_par):
            ax.plot(k[~sig_par], gamma_par[~sig_par], "s", markersize=4,
                    color="#f0883e", alpha=0.25, markeredgecolor="none")
        if np.any(~sig_perp) or np.any(~sig_par):
            ax.plot([], [], "x", color=TEXT_CLR, alpha=0.4,
                    label=f"below {significance_threshold:.0e}$\\times$peak energy (noise floor)")
        ax.set_xscale("log")
        ax.set_xlabel(r"$k\,d_i$")
        ax.set_ylabel(r"$\gamma\ [\Omega_{ci}]$")
        ax.set_title(f"Mode-resolved growth rate — {PROFILE_LABEL} ({plane})", fontsize=POSTER_TITLE)
        ax.legend(fontsize=POSTER_LEGEND, facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
        out_file = self.outdir / f"growth_rate_vs_k_{plane}.png"
        fig.savefig(out_file, dpi=220, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"Saved growth rate plot: {out_file}")

    def plot_compressibility(self, times: np.ndarray, compressibility: np.ndarray, plane: str):
        fig, ax = _new_dark_fig((9, 5.2))
        ax.plot(times, compressibility, lw=2.2, color="#f85149")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$\Omega_{ci} t$")
        ax.set_ylabel(r"$\delta B_\parallel^2 / (\delta B_\parallel^2+\delta B_\perp^2)$")
        ax.set_title(f"Magnetic compressibility — {PROFILE_LABEL} ({plane})", fontsize=POSTER_TITLE)
        out_file = self.outdir / f"compressibility_vs_time_{plane}.png"
        fig.savefig(out_file, dpi=220, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"Saved compressibility plot: {out_file}")

    def plot_helicity_vs_k(self, growth_rows: list[dict], plane: str):
        k = np.array([row["k"] for row in growth_rows])
        sigma_m = np.array([row["sigma_m"] for row in growth_rows])
        finite = np.isfinite(sigma_m)

        fig, ax = _new_dark_fig((9, 5.2))
        ax.set_xlim(float(np.min(k)), float(np.max(k)))
        ax.axhline(0.0, color=GRID_CLR, lw=1.2)
        if np.any(finite):
            ax.plot(k[finite], sigma_m[finite], "o-", markersize=4, color="#3fb950")
        else:
            ax.text(0.5, 0.5, "no measurable helicity (zero power in both transverse components)",
                    transform=ax.transAxes, ha="center", va="center", color=TEXT_CLR, fontsize=13)
        ax.set_xscale("log")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r"$k\,d_i$")
        ax.set_ylabel(r"$\sigma_m(k)$")
        ax.set_title(f"Reduced magnetic helicity — {PROFILE_LABEL} ({plane})", fontsize=POSTER_TITLE)
        out_file = self.outdir / f"helicity_vs_k_{plane}.png"
        fig.savefig(out_file, dpi=220, bbox_inches="tight", facecolor=DARK_BG)
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
