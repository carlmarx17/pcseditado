#!/usr/bin/env python3
"""Space-time spectral density in frequency--phase-velocity coordinates."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

import plot_style as ps

ps.apply()
import numpy as np

from data_reader import PICDataReader
from streaming_fields import SnapshotSeries, spatial_spectra
from dispersion_modes import characterize_modes, plot_mode_summary, plot_mode_fit
from spectral_analysis import SpectralAnalyzer

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


def _smooth_2d(values: np.ndarray, passes: int = 2) -> np.ndarray:
    """Small dependency-free Gaussian-like smoother for display density.

    Edges are replicated rather than zero-padded. ``np.convolve(mode="same")``
    implicitly assumes zeros outside the array, which costs a boundary row
    roughly half its kernel weight -- and the omega = 0 row *is* a boundary.
    Suppressing it is exactly the wrong bias for an aperiodic mode, whose
    entire signal sits there.
    """
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    pad = len(kernel) // 2

    def smooth_axis(row: np.ndarray) -> np.ndarray:
        if row.size <= pad:
            return row
        return np.convolve(np.pad(row, pad, mode="edge"), kernel, mode="valid")

    result = np.asarray(values, dtype=float)
    for _ in range(max(passes, 0)):
        result = np.apply_along_axis(smooth_axis, 0, result)
        result = np.apply_along_axis(smooth_axis, 1, result)
    return result


def _make_window(kind: str, n: int, alpha: float = 0.25) -> np.ndarray:
    """Analysis window of length ``n``.

    Windows exist to suppress the leakage caused by analysing a *non-periodic*
    record with a transform that assumes periodicity. When the record already
    is periodic the window has nothing to fix and only does harm, because it
    is a multiplication in the signal domain and therefore a convolution in
    the Fourier domain: it smears every mode into its neighbours.

    For a Hann window that smearing is exactly known. An exact box mode ends
    up with 2/3 of its power in the true bin and 1/6 in each adjacent bin --
    a third of the mode thrown one wavenumber away.

    ``"tukey"`` tapers only a fraction ``alpha`` of each end and leaves the
    middle untouched, which removes the edge discontinuity at a much smaller
    cost in main-lobe width than a full Hann (measured on a 772-sample record:
    worst sidelobe -33 dB against -48 dB for Hann, for ~20% broadening instead
    of 100%).
    """
    if n <= 0:
        return np.ones(0)
    if kind == "none":
        return np.ones(n)
    if kind == "hann":
        return np.hanning(n)
    if kind == "tukey":
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if alpha <= 0 or n < 3:
            return np.ones(n)
        window = np.ones(n)
        taper = int(np.floor(alpha * (n - 1) / 2.0)) + 1
        ramp_index = np.arange(taper)
        ramp = 0.5 * (
            1.0 + np.cos(np.pi * (2.0 * ramp_index / (alpha * (n - 1)) - 1.0))
        )
        window[:taper] = ramp
        window[n - taper:] = ramp[::-1]
        return window
    raise ValueError(f"Unknown window: {kind!r}")


def _parabolic_offset(values: np.ndarray, index: int) -> float:
    """Sub-bin position of a peak, in bins, from a parabola through log-power.

    With T ~ 50-150 Omega_ci^-1 the frequency bin is a sizeable fraction of the
    frequency being measured, so rounding the peak to the nearest bin is a real
    systematic. Interpolating recovers most of it; the estimate is still bounded
    by the linewidth, which is why the width is reported alongside it.
    """
    if index <= 0 or index >= len(values) - 1:
        return 0.0
    trio = np.asarray(values[index - 1: index + 2], dtype=float)
    if np.any(trio <= 0):
        return 0.0
    y0, y1, y2 = np.log(trio)
    curvature = y0 - 2.0 * y1 + y2
    if curvature >= 0 or not np.isfinite(curvature):
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / curvature, -0.5, 0.5))


def _smooth_along(values: np.ndarray, axis: int, passes: int = 2) -> np.ndarray:
    """Same smoother restricted to one axis.

    Used when the other axis carries the quantity being measured: smoothing a
    5-point kernel along omega blurs the peak by +-2 bins, which is the whole
    frequency resolution of a short window.
    """
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    pad = len(kernel) // 2

    def smooth_axis(row: np.ndarray) -> np.ndarray:
        if row.size <= pad:
            return row
        return np.convolve(np.pad(row, pad, mode="edge"), kernel, mode="valid")

    result = np.asarray(values, dtype=float)
    for _ in range(max(passes, 0)):
        result = np.apply_along_axis(smooth_axis, axis, result)
    return result


def _autocrop_signal_extent(centers: np.ndarray, marginal: np.ndarray, exclude_fraction: float) -> np.ndarray | None:
    """Return the subset of ``centers`` needed to cover ``1-exclude_fraction``
    of the total marginal power, smallest-power bins dropped first.

    A per-pixel floor lets a long tail of individually-weak (but numerous)
    noise bins defeat autocropping, since each survives on its own even
    though collectively they carry little power. Ranking by cumulative
    contribution to the total is robust to that: a real ridge concentrates
    most of the power in comparatively few bins, so the crop still tracks it
    even when scattered noise extends much further out.
    """
    total = float(np.sum(marginal))
    if total <= 0:
        return None
    order = np.argsort(marginal)[::-1]
    cumulative = np.cumsum(marginal[order])
    cutoff = int(np.searchsorted(cumulative, (1.0 - exclude_fraction) * total)) + 1
    keep = np.zeros_like(marginal, dtype=bool)
    keep[order[:cutoff]] = True
    return centers[keep] if np.any(keep) else None


# ---------------------------------------------------------------------------
# Mode presets.
#
# omega_r_ci is the expected real frequency in Omega_ci (0.0 for aperiodic,
# purely growing modes); k_di is the expected range of k*d_i at maximum growth;
# theta_deg is the expected propagation angle relative to B0.
#
# The point of the table is not to pre-judge the answer but to let the run-time
# resolution report state, in physical units, whether the box and the output
# cadence can represent the mode at all before any spectrum is plotted.
# ---------------------------------------------------------------------------
MODE_PRESETS: dict[str, dict] = {
    "generic": {
        "omega_r_ci": None, "k_di": (0.1, 2.0), "theta_deg": (0.0, 90.0),
        "aperiodic": False, "species": "ion",
    },
    "mirror": {
        "omega_r_ci": 0.0, "k_di": (0.2, 1.0), "theta_deg": (50.0, 80.0),
        "aperiodic": True, "species": "ion",
    },
    "firehose-oblique": {
        "omega_r_ci": 0.0, "k_di": (0.2, 0.8), "theta_deg": (50.0, 80.0),
        "aperiodic": True, "species": "ion",
    },
    "firehose-parallel": {
        "omega_r_ci": 0.2, "k_di": (0.2, 0.6), "theta_deg": (0.0, 30.0),
        "aperiodic": False, "species": "ion",
    },
    "emic": {
        "omega_r_ci": 0.3, "k_di": (0.3, 1.0), "theta_deg": (0.0, 25.0),
        "aperiodic": False, "species": "ion",
    },
    "whistler": {
        # omega_r ~ 0.1-0.5 Omega_ce and k d_e ~ 0.3-0.6; both are rescaled to
        # ion units at run time using the actual mass ratio.
        "omega_r_ce": 0.3, "k_de": (0.3, 0.6), "theta_deg": (0.0, 25.0),
        "aperiodic": False, "species": "electron",
    },
}


def resolve_mode_expectations(mode: str, mass_ratio: float | None) -> dict:
    """Return the preset for ``mode`` with electron-scale entries mapped to ion units."""
    if mode not in MODE_PRESETS:
        raise ValueError(f"Unknown mode preset: {mode!r}")
    preset = dict(MODE_PRESETS[mode])
    if preset.get("species") == "electron":
        if mass_ratio is None or mass_ratio <= 0:
            raise ValueError(
                f"Mode preset {mode!r} is electron-scale and needs a mass ratio "
                "(pass --mass-ratio or make psc_units.MASS_RATIO importable)"
            )
        root = float(np.sqrt(mass_ratio))
        preset["omega_r_ci"] = preset.pop("omega_r_ce") * mass_ratio
        k_de_lo, k_de_hi = preset.pop("k_de")
        # k d_i = (k d_e) * sqrt(mi/me)
        preset["k_di"] = (k_de_lo * root, k_de_hi * root)
    return preset


def spectral_resolution_report(
    times: np.ndarray,
    spacing: tuple[float, float],
    grid_shape: tuple[int, int],
    *,
    mode: str = "generic",
    mass_ratio: float | None = None,
    gamma_ci: float | None = None,
) -> dict:
    """Quantify what the (box, cadence) pair can actually resolve, in physical units.

    Everything downstream of the FFT is bounded by four numbers that follow
    from the sampling alone, before any physics:

        dk   = 2*pi/L        smallest resolvable wavenumber separation
        k_Ny = pi/dx         largest representable wavenumber
        dw   = 2*pi/T        frequency resolution of the time window
        w_Ny = pi/dt_out     largest representable frequency

    A dispersion *branch* only exists as a measurable object when the mode is
    (a) sampled by enough discrete k in the physically interesting band and
    (b) narrow enough in omega to be distinguishable. For an unstable mode the
    intrinsic linewidth is ~2*gamma, so the branch is resolvable only when
    omega_r/gamma is large; a purely growing mode (omega_r = 0) has no branch
    at all and the omega-k diagram is the wrong diagnostic for it.
    """
    times = np.asarray(times, dtype=float)
    dt = float(np.median(np.diff(times)))
    duration = float(times[-1] - times[0])
    expectations = resolve_mode_expectations(mode, mass_ratio)

    per_axis = []
    for length_per_cell, n_cells in zip(spacing, grid_shape):
        box = float(length_per_cell) * int(n_cells)
        dk = 2.0 * np.pi / box
        per_axis.append({
            "box_di": box,
            "cells": int(n_cells),
            "dx_di": float(length_per_cell),
            "dk_di_inv": dk,
            "k_nyquist_di_inv": np.pi / float(length_per_cell),
            "modes_below_kdi_1": int(np.floor(1.0 / dk)),
        })

    dk_min = min(axis["dk_di_inv"] for axis in per_axis)
    k_lo, k_hi = expectations["k_di"]
    modes_in_band = int(np.floor(k_hi / dk_min) - np.ceil(k_lo / dk_min) + 1)

    report = {
        "mode": mode,
        "expectations": expectations,
        "n_snapshots": int(times.size),
        "window_oci": duration,
        "dt_out_oci": dt,
        "domega_ci": 2.0 * np.pi / duration,
        "omega_nyquist_ci": np.pi / dt,
        "independent_positive_frequencies": (int(times.size) - 1) // 2,
        "axes": per_axis,
        "modes_in_expected_k_band": max(modes_in_band, 0),
        "gamma_ci": gamma_ci,
        "checks": [],
    }

    def check(name, ok, detail):
        report["checks"].append({"name": name, "status": "PASS" if ok else "WARN",
                                 "detail": detail})

    check(
        "k_sampling",
        modes_in_band >= 8,
        f"{max(modes_in_band, 0)} discrete k modes inside the expected band "
        f"k d_i in [{k_lo:g}, {k_hi:g}] (dk d_i = {dk_min:.4f}). "
        f"A heuristic target of 8 samples across the band corresponds to L ~ "
        f"{2 * np.pi * 8 / max(k_hi - k_lo, 1e-9):.0f} d_i; this is not a hard detection limit.",
    )
    check(
        "k_nyquist",
        min(axis["k_nyquist_di_inv"] for axis in per_axis) > 2.0 * k_hi,
        f"k_Nyquist d_i = {min(a['k_nyquist_di_inv'] for a in per_axis):.1f} vs "
        f"expected k_max d_i = {k_hi:g}.",
    )

    omega_r = expectations["omega_r_ci"]
    if expectations["aperiodic"] or (omega_r is not None and omega_r == 0.0):
        check(
            "omega_branch_exists",
            False,
            f"Preset {mode!r} expects omega_r = 0. Retain the zero-frequency ridge "
            "with --omega-scale linear and characterize its gamma(k_par,k_perp) "
            "and geometry. A zero-frequency maximum alone does not prove an instability.",
        )
    elif omega_r is not None:
        n_bins = omega_r / report["domega_ci"]
        check(
            "omega_resolution",
            n_bins >= 5,
            f"expected omega_r = {omega_r:g} Omega_ci spans {n_bins:.1f} frequency bins "
            f"(d_omega = {report['domega_ci']:.4f} Omega_ci over T = {duration:.1f} "
            f"Omega_ci^-1). A conservative target of 5 bins requires "
            f"T >= {10 * np.pi / omega_r:.0f} Omega_ci^-1.",
        )
        check(
            "omega_nyquist",
            report["omega_nyquist_ci"] > omega_r,
            f"omega_Nyquist = {report['omega_nyquist_ci']:.2f} Omega_ci vs expected "
            f"omega_r = {omega_r:g} Omega_ci. Aliasing occurs at/above Nyquist. "
            f"A separate conservative target of 8 samples per period would use "
            f"dt_out <= {2 * np.pi / (8 * omega_r):.4f} Omega_ci^-1.",
        )
        if gamma_ci is not None and gamma_ci > 0:
            ratio = omega_r / gamma_ci
            check(
                "branch_sharpness",
                ratio >= 10.0,
                f"omega_r/gamma = {ratio:.1f}. Growth can broaden a finite-window "
                "spectrum on a scale of order gamma. The ratio >=10 check is a "
                "conservative sharpness heuristic; inspect the measured width and phase fit.",
            )

    if gamma_ci is not None and gamma_ci > 0:
        growth_factor = gamma_ci * duration
        check(
            "stationarity",
            growth_factor <= 3.0,
            f"gamma*T = {growth_factor:.1f} e-foldings inside the FFT window. Above ~3 the "
            "signal is strongly non-stationary and the temporal FFT measures the growth "
            "envelope rather than omega(k). Use --degrowth per-k, or shorten the window.",
        )

    return report


def _spatial_fft_sliced(
    component: np.ndarray,
    spatial_window: np.ndarray,
    slice0: slice,
    slice1: slice,
    block: int = 64,
) -> np.ndarray:
    """Windowed 2D spatial FFT, sliced to the retained k range, in time blocks.

    The full (nt, n0, n1) complex transform is never materialised: for a long
    run at 576^2 or 1152^2 that array alone is tens of GB, and only the small
    central k block survives the slice anyway.
    """
    nt = component.shape[0]
    out = np.empty(
        (nt, slice0.stop - slice0.start, slice1.stop - slice1.start),
        dtype=np.complex128,
    )
    for start in range(0, nt, max(block, 1)):
        stop = min(start + max(block, 1), nt)
        chunk = np.asarray(component[start:stop], dtype=float) * spatial_window
        transformed = np.fft.fftshift(np.fft.fft2(chunk, axes=(1, 2)), axes=(1, 2))
        out[start:stop] = transformed[:, slice0, slice1]
    return out


def _fit_growth_per_mode(
    mode_power: np.ndarray,
    times: np.ndarray,
    *,
    power_floor_fraction: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares gamma(k) from d/dt log|b(k,t)|, plus the fit R^2.

    ``mode_power`` is |b(k,t)|^2 summed over components, shape (nt, nk0, nk1).
    Returns ``(gamma, r_squared)`` each shaped (nk0, nk1). Modes whose power
    never rises above ``power_floor_fraction`` of the global peak get gamma = 0,
    so de-growth never amplifies pure PIC noise.
    """
    times = np.asarray(times, dtype=float)
    tiny = np.finfo(float).tiny
    peak = float(np.max(mode_power)) if mode_power.size else 0.0
    # Exact zero initial fluctuations have no logarithmic amplitude. Exclude
    # them instead of replacing log(0) by -354 and fitting artificial growth.
    valid = mode_power > np.maximum(np.max(mode_power, axis=0) * 1e-12, tiny)
    count = np.maximum(valid.sum(axis=0), 1)
    t_grid = times[:, None, None]
    mean_t = np.sum(t_grid * valid, axis=0) / count
    log_amp = 0.5 * np.log(np.maximum(mode_power, tiny))
    mean_y = np.sum(log_amp * valid, axis=0) / count
    tc = t_grid - mean_t
    yc = log_amp - mean_y
    denom = np.sum(tc**2 * valid, axis=0)
    gamma = np.sum(tc * yc * valid, axis=0) / np.maximum(denom, tiny)
    residual = yc - gamma[None, :, :] * tc
    ss_res = np.sum(residual**2 * valid, axis=0)
    ss_tot = np.sum(yc**2 * valid, axis=0)
    r_squared = np.clip(1.0 - ss_res / np.maximum(ss_tot, tiny), 0.0, 1.0)

    if peak > 0:
        alive = (np.max(mode_power, axis=0) > power_floor_fraction * peak) & (count >= 4)
        gamma = np.where(alive, gamma, 0.0)
        r_squared = np.where(alive, r_squared, 0.0)
    return gamma, r_squared


def compute_phase_velocity_density(
    field_series: np.ndarray,
    time_oci: np.ndarray,
    spacing: tuple[float, float],
    axes: tuple[str, str],
    parallel_axis: str = "z",
    velocity_min: float = 0.0,
    velocity_max: float = 12.0,
    velocity_bins: int = 240,
    frequency_bins: int = 180,
    absolute_velocity: bool = True,
    max_spatial_mode: int = 128,
    temporal_fft_size: int = 128,
    theta_max_deg: float | None = None,
    density_normalization: str = "global",
    power_floor_db: float = -40.0,
    time_detrend: str = "none",
    degrowth: str = "none",
    kmax_di: float | None = None,
    spatial_window: str = "none",
    temporal_window: str = "tukey",
    window_alpha: float = 0.25,
    component_axes: tuple[str, ...] | None = None,
    theta_min_deg: float | None = None,
) -> dict:
    """Transform B(t, axis0, axis1) into weighted (omega, v_phase) density.

    ``field_series`` has shape ``(n_components, nt, n0, n1)``. Time is
    normalized as ``Omega_ci t`` and spacing is measured in ``d_i``. Therefore
    ``(omega/Omega_ci) / (k_parallel d_i)`` is directly ``v_phase/v_A``.

    ``time_detrend``
        ``"none"`` (default) keeps the zero-frequency content. Subtracting the
        per-cell time mean -- the previous hard-wired behaviour -- deletes the
        omega = 0 row exactly, which is precisely the signal of an aperiodic
        mode (mirror, oblique firehose). ``"mean"`` restores the old behaviour
        and ``"linear"`` removes a per-cell linear trend; both are only
        appropriate for genuinely propagating modes.

    ``degrowth``
        Divide out the exponential envelope before the temporal FFT, so the
        transform sees a stationary signal and returns the real frequency
        instead of the growth-broadened envelope. ``"global"`` uses one gamma
        from the total fluctuation power, ``"per-k"`` fits gamma(k) for every
        retained mode independently. ``"none"`` disables it.

    ``kmax_di``
        Physical cap on the retained wavenumber, in units of 1/d_i. Preferred
        over ``max_spatial_mode``: it keeps the analysis inside the band where
        the fluctuations are physical rather than grid-scale particle noise,
        and it shrinks the transform by orders of magnitude.

    ``spatial_window``
        Default ``"none"``. The PSC anisotropy runs use periodic field and
        particle boundaries in all three directions, so a snapshot is already
        an exact period and the DFT basis is exact. Windowing it does not
        remove leakage -- there is none -- it *creates* leakage, scattering a
        third of every mode's power into the adjacent wavenumbers. With only
        a handful of resolved modes below k d_i = 1 that is a large fraction
        of the useful range.

    ``temporal_window``
        Default ``"tukey"``. Time is *not* periodic: the record starts and
        stops at arbitrary phase, so a window is genuinely needed here. Hann
        buries the sidelobes but doubles the main-lobe width, which these runs
        cannot afford; a Tukey taper keeps most of the resolution.
    """
    fields = field_series if isinstance(field_series, SnapshotSeries) else np.asarray(field_series)
    times = np.asarray(time_oci, dtype=float)
    if fields.ndim != 4:
        raise ValueError(
            "field_series must have shape (components, time, axis0, axis1)"
        )
    if fields.shape[1] != len(times) or len(times) < 4:
        raise ValueError("At least four time-aligned snapshots are required")
    if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("Snapshot times must be finite and strictly increasing")
    if len(spacing) != 2 or not np.all(np.isfinite(spacing)) or min(spacing) <= 0:
        raise ValueError("Plane spacing must contain two finite positive values")
    if max_spatial_mode < 1 or (kmax_di is not None and kmax_di <= 0):
        raise ValueError("Spatial mode count and kmax_di must be positive")
    if not 0 <= velocity_min < velocity_max:
        raise ValueError("Velocity range must satisfy 0 <= min < max")
    if parallel_axis not in axes:
        raise ValueError(
            f"Plane axes {axes} do not contain parallel axis '{parallel_axis}'"
        )

    delta_t = np.diff(times)
    if not np.allclose(delta_t, np.median(delta_t), rtol=1e-6, atol=1e-12):
        raise ValueError("Snapshot times must be uniformly spaced")
    dt = float(np.median(delta_t))

    if time_detrend not in ("none", "mean", "linear"):
        raise ValueError(f"Unknown time_detrend: {time_detrend!r}")

    nt, n0, n1 = fields.shape[1:]
    if min(n0, n1) < 4:
        # A one-cell axis carries no wavenumber information at all, and with a
        # Hann window it would additionally multiply the whole field by zero
        # (np.hanning(1) is [0.0]) so the failure would only surface much later
        # as an empty spectrum. Say so here instead.
        raise ValueError(
            f"Plane has a degenerate axis (shape {n0}x{n1}); the analysis plane "
            "or the parallel axis is probably wrong for this run"
        )
    spatial_window_name, temporal_window_name = spatial_window, temporal_window
    spatial_window = (
        _make_window(spatial_window_name, n0, window_alpha)[:, None]
        * _make_window(spatial_window_name, n1, window_alpha)[None, :]
    )
    temporal_window = _make_window(temporal_window_name, nt, window_alpha)[:, None, None]

    full_k0 = np.fft.fftshift(np.fft.fftfreq(n0, d=spacing[0])) * 2.0 * np.pi
    full_k1 = np.fft.fftshift(np.fft.fftfreq(n1, d=spacing[1])) * 2.0 * np.pi
    half0 = min(max_spatial_mode, (n0 - 1) // 2)
    half1 = min(max_spatial_mode, (n1 - 1) // 2)
    if kmax_di is not None and kmax_di > 0:
        # A physical cap beats a fixed mode-count cap: max_spatial_mode = 128
        # reaches k d_i ~ 40 in a 20 d_i box, two decades above anything the
        # ion-scale physics occupies, so most of the plotted range is PIC noise.
        dk0 = 2.0 * np.pi / (spacing[0] * n0)
        dk1 = 2.0 * np.pi / (spacing[1] * n1)
        half0 = min(half0, int(np.floor(kmax_di / dk0)))
        half1 = min(half1, int(np.floor(kmax_di / dk1)))
    center0 = n0 // 2
    center1 = n1 // 2
    slice0 = slice(center0 - half0, center0 + half0 + 1)
    slice1 = slice(center1 - half1, center1 + half1 + 1)
    k0 = full_k0[slice0]
    k1 = full_k1[slice1]

    if degrowth not in ("none", "global", "per-k"):
        raise ValueError(f"Unknown degrowth: {degrowth!r}")

    spectra = spatial_spectra(fields, spatial_window, slice0, slice1)
    # Apply temporal detrending to retained complex coefficients: linearity
    # gives the same result without materializing the full space-time cube.
    if time_detrend in ("mean", "linear"):
        spectra -= spectra.mean(axis=1, keepdims=True)
    if time_detrend == "linear":
        tc = times - times.mean()
        slope = np.einsum("t,ctij->cij", tc, spectra) / np.dot(tc, tc)
        spectra -= slope[:, None] * tc[None, :, None, None]

    if kmax_di is not None:
        retained = np.hypot(k0[:, None], k1[None, :]) <= kmax_di * (1 + 1e-12)
        for spectrum in spectra:
            spectrum[:, ~retained] = 0
    if component_axes is not None and len(component_axes) != len(spectra):
        raise ValueError("component_axes must label every field component")
    mode_candidates = characterize_modes(
        spectra, times, k0, k1, axes, parallel_axis, component_axes,
        theta_min_deg=theta_min_deg, theta_max_deg=theta_max_deg,
    )
    dominant_trace = None
    if mode_candidates:
        m = mode_candidates[0]
        coord = (m["k_parallel_d_i"], m["k_perp_signed_d_i"])
        if axes[0] != parallel_axis:
            coord = coord[::-1]
        i, j = np.argmin(abs(k0 - coord[0])), np.argmin(abs(k1 - coord[1]))
        coefficients = np.stack([s[:, i, j] for s in spectra])
        energy = np.sum(np.abs(coefficients)**2, axis=0)
        valid_trace = energy > max(float(energy.max()) * 1e-12, np.finfo(float).tiny)
        ref = coefficients[np.argmax(np.mean(np.abs(coefficients)**2, axis=1))]
        dominant_trace = {
            "times": times[valid_trace],
            "amplitude": np.sqrt(energy[valid_trace]),
            "phase": np.unwrap(np.angle(ref[valid_trace])),
        }
    mode_power = sum(np.abs(spectrum) ** 2 for spectrum in spectra)
    spatial_mode_power = np.mean(mode_power, axis=0)
    gamma_map, gamma_r2 = _fit_growth_per_mode(mode_power, times)
    gamma_global = float(_fit_growth_per_mode(
        np.sum(mode_power, axis=(1, 2))[:, None, None], times
    )[0][0, 0])
    del mode_power
    if degrowth != "none":
        envelope_gamma = (
            np.where(gamma_r2 >= 0.8, gamma_map, 0.0) if degrowth == "per-k"
            else np.full(gamma_map.shape, gamma_global, dtype=float)
        )
        # Clip the exponent, not gamma: a mode whose power is pure noise can fit
        # an arbitrarily steep slope, and exp() of that would overflow to inf.
        exponent = np.clip(
            envelope_gamma[None, :, :] * (times - times.mean())[:, None, None],
            -30.0, 30.0,
        )
        envelope = np.exp(exponent)
        spectra = [spectrum / envelope for spectrum in spectra]

    nfft = max(int(temporal_fft_size), nt)
    power = np.zeros((nfft, len(k0), len(k1)), dtype=float)
    for spectrum in spectra:
        transformed = np.fft.fftshift(
            # Spatial FFT uses exp(-ik.x); temporal transform must use
            # exp(+i omega t) for a physical exp[i(k.x-omega*t)] wave.
            nfft * np.fft.ifft(spectrum * temporal_window, n=nfft, axis=0),
            axes=0,
        )
        power += np.abs(transformed) ** 2
    del spectra

    omega = np.fft.fftshift(np.fft.fftfreq(nfft, d=dt)) * 2.0 * np.pi
    omega_grid, k0_grid, k1_grid = np.meshgrid(
        omega, k0, k1, indexing="ij"
    )
    k_parallel = k0_grid if axes[0] == parallel_axis else k1_grid

    k0_2d, k1_2d = np.meshgrid(k0, k1, indexing="ij")
    k_par_2d = k0_2d if axes[0] == parallel_axis else k1_2d
    k_perp_2d = k1_2d if axes[0] == parallel_axis else k0_2d
    theta_grid = np.degrees(np.arctan2(np.abs(k_perp_2d), np.abs(k_par_2d)))

    positive_frequency = omega_grid >= 0
    nonzero_k = np.abs(k_parallel) > 1e-12
    if theta_max_deg is not None:
        angle_ok = np.broadcast_to(theta_grid[None, :, :] <= theta_max_deg, omega_grid.shape)
    else:
        angle_ok = np.ones_like(omega_grid, dtype=bool)
    if theta_min_deg is not None:
        angle_ok = angle_ok & (theta_grid[None, :, :] >= theta_min_deg)
    phase_velocity = np.divide(
        omega_grid,
        k_parallel,
        out=np.full_like(omega_grid, np.nan),
        where=nonzero_k,
    )
    if absolute_velocity:
        phase_velocity = np.abs(phase_velocity)

    if absolute_velocity:
        velocity_range = (velocity_min, velocity_max)
        velocity_mask = (
            (phase_velocity >= velocity_min) & (phase_velocity <= velocity_max)
        )
    else:
        velocity_range = (-velocity_max, velocity_max)
        velocity_mask = (
            (np.abs(phase_velocity) >= velocity_min)
            & (np.abs(phase_velocity) <= velocity_max)
        )

    valid = (
        positive_frequency
        & nonzero_k
        & angle_ok
        & velocity_mask
        & np.isfinite(phase_velocity)
        & np.isfinite(power)
        & (power > 0)
    )
    omega_values = omega_grid[valid]
    velocity_values = phase_velocity[valid]
    # Histogram integrated native-bin power: multiplying discrete bins by
    # omega/k^2 biases their relative strength and deletes omega=0 power.
    weights = power[valid]
    omega_max = float(np.max(omega[omega > 0]))
    density, omega_edges, velocity_edges = np.histogram2d(
        omega_values,
        velocity_values,
        bins=(frequency_bins, velocity_bins),
        range=((0.0, omega_max), velocity_range),
        weights=weights,
    )
    # Display smoothing only. Detection never uses this interpolated density.
    density = _smooth_2d(density, passes=3)

    # A per-omega-column floor: cells too weak relative to the global peak are
    # zeroed out before normalizing, so noise-only frequency columns render as
    # blank instead of being rescaled up to look as strong as real signal.
    global_peak = float(np.max(density))
    power_floor = (10.0 ** (power_floor_db / 10.0)) * global_peak
    density = np.where(density >= power_floor, density, 0.0)

    if density_normalization == "conditional":
        # Legacy conditional density P(v_phase | omega): prevents frequencies
        # with larger total fluctuation power from hiding weaker but coherent
        # branches, at the cost of making low-power (mostly noise) columns
        # look as intense as high-power ones.
        frequency_power = np.sum(density, axis=1, keepdims=True)
        density = np.divide(
            density,
            frequency_power,
            out=np.zeros_like(density),
            where=frequency_power > 0,
        )
    elif density_normalization == "global":
        density = density / max(float(np.max(density)), np.finfo(float).tiny)
    else:
        raise ValueError(f"Unknown density_normalization: {density_normalization!r}")

    return {
        "density": density,
        "omega_edges": omega_edges,
        "velocity_edges": velocity_edges,
        "velocity_min": velocity_min,
        "velocity_max": velocity_max,
        "omega_samples": omega[omega > 0],
        "omega_grid": omega_grid,
        "phase_velocity": phase_velocity,
        "power": power,
        "k0": k0,
        "k1": k1,
        "omega": omega,
        "spacing": spacing,
        "axes": axes,
        "parallel_axis": parallel_axis,
        "jacobian_weight": np.divide(
            phase_velocity**2,
            np.abs(omega_grid),
            out=np.zeros_like(phase_velocity),
            where=np.isfinite(phase_velocity) & (np.abs(omega_grid) > 0),
        ),
        "valid": valid,
        "absolute_velocity": absolute_velocity,
        "independent_positive_frequencies": (nt - 1) // 2,
        "omega_resolution": 2.0 * np.pi / (times[-1] - times[0]),
        "omega_fft_bin_spacing": 2.0 * np.pi / (nfft * dt),
        "theta_grid": theta_grid,
        "theta_max_deg": theta_max_deg,
        "density_normalization": density_normalization,
        "time_detrend": time_detrend,
        "degrowth": degrowth,
        "spatial_window": spatial_window_name,
        "temporal_window": temporal_window_name,
        "window_alpha": window_alpha,
        "gamma_map": gamma_map,
        "gamma_r_squared": gamma_r2,
        "gamma_global": gamma_global,
        "times": times,
        "grid_shape": (n0, n1),
        "mode_candidates": mode_candidates,
        "spatial_mode_power": spatial_mode_power,
        "dominant_trace": dominant_trace,
    }


def _reduce_kperp(
    power: np.ndarray,
    k0: np.ndarray,
    k1: np.ndarray,
    axes: tuple[str, str],
    parallel_axis: str,
    kperp_reduction: str = "sum",
    theta_max_deg: float | None = None,
    theta_grid: np.ndarray | None = None,
    theta_min_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the perpendicular k axis of a (nfft, nk0, nk1) power cube.

    Returns ``(p2d, k_par)`` with ``p2d`` shaped ``(nfft, nk_par)``. Cells whose
    propagation angle (from ``theta_grid``, shape ``(nk0, nk1)``) falls outside
    ``[theta_min_deg, theta_max_deg]`` are zeroed before reducing, so the
    reduction reflects only the angular band of interest.

    An angular *band* rather than a ceiling matters for oblique instabilities:
    mirror and oblique firehose peak near theta_kB ~ 60-75 deg, and summing
    every k_perp onto the k_parallel axis smears that oblique peak into the
    quasi-parallel range where the mode does not live.
    """
    if axes[0] == parallel_axis:
        k_par = k0
        axis_perp = 2
    else:
        k_par = k1
        axis_perp = 1

    if theta_max_deg is not None or theta_min_deg is not None:
        if theta_grid is None:
            raise ValueError("theta_grid is required when a theta filter is set")
        keep = np.ones_like(theta_grid, dtype=bool)
        if theta_max_deg is not None:
            keep &= theta_grid <= theta_max_deg
        if theta_min_deg is not None:
            keep &= theta_grid >= theta_min_deg
        power = np.where(keep[None, :, :], power, 0.0)

    if kperp_reduction == "sum":
        p2d = np.sum(power, axis=axis_perp)
    elif kperp_reduction == "max":
        p2d = np.max(power, axis=axis_perp)
    else:  # perpendicular slice at k_perp ~ 0
        kperp = k1 if axis_perp == 2 else k0
        j0 = int(np.argmin(np.abs(kperp)))
        p2d = np.take(power, j0, axis=axis_perp)

    return p2d, k_par


def _fold_signed_kpar(p2d: np.ndarray, k_par: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold power at -k onto +k, since unsigned |v_phase| discards propagation
    direction: a mode's power can land on either sign of k_parallel depending
    on FFT sign convention, and naively slicing to k_parallel > 0 would silently
    drop real signal that happens to fall on the negative side.
    """
    if len(k_par) < 2:
        return np.zeros((p2d.shape[0], 0)), np.array([], dtype=float)
    dk = float(k_par[1] - k_par[0])
    n = len(k_par)
    pos_idx = np.where(k_par > 1e-12)[0]
    k_pos = k_par[pos_idx]
    folded = np.zeros((p2d.shape[0], len(pos_idx)), dtype=p2d.dtype)
    for out_col, i in enumerate(pos_idx):
        folded[:, out_col] = p2d[:, i]
        neg_index = int(round((-k_par[i] - k_par[0]) / dk))
        if 0 <= neg_index < n and neg_index != i:
            folded[:, out_col] += p2d[:, neg_index]
    return folded, k_pos


def _extract_ridges_v_phase(
    result: dict, ridge_count: int, min_power_db: float
) -> list[dict]:
    """Legacy per-omega-column argmax on the (already normalized) v_phase histogram."""
    rows: list[dict] = []
    omega_grid = result["omega_grid"]
    velocity = result["phase_velocity"]
    power = result["power"]
    jacobian_weight = result["jacobian_weight"]
    valid = result["valid"]
    velocity_edges = result["velocity_edges"]
    centers = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])

    histograms = []
    for omega in result["omega_samples"]:
        frequency_mask = valid & np.isclose(omega_grid, omega)
        if not np.any(frequency_mask):
            histograms.append(np.zeros_like(centers))
            continue
        histogram, _ = np.histogram(
            velocity[frequency_mask],
            bins=velocity_edges,
            weights=power[frequency_mask] * jacobian_weight[frequency_mask],
        )
        histograms.append(histogram.astype(float))
    global_peak = float(max((float(np.max(h)) for h in histograms), default=0.0))
    power_floor = (10.0 ** (min_power_db / 10.0)) * global_peak

    for omega, histogram in zip(result["omega_samples"], histograms):
        work = histogram.copy()
        exclusion = max(2, len(centers) // 40)
        for rank in range(1, ridge_count + 1):
            index = int(np.argmax(work))
            peak_power = float(work[index])
            if peak_power <= 0 or peak_power < power_floor:
                break
            rows.append(
                {
                    "omega_over_omega_ci": float(omega),
                    "phase_velocity_over_va": float(centers[index]),
                    "k_parallel_d_i": float("nan"),
                    "ridge_rank": rank,
                    "spectral_power": peak_power,
                }
            )
            lo = max(0, index - exclusion)
            hi = min(len(work), index + exclusion + 1)
            work[lo:hi] = 0.0
    return rows


def _extract_ridges_omega_k(
    result: dict,
    ridge_count: int,
    kperp_reduction: str,
    theta_max_deg: float | None,
    min_power_db: float,
    max_jump_fraction: float,
    apply_jacobian: bool = False,
    use_velocity_mask: bool = False,
    theta_min_deg: float | None = None,
    ridge_axis: str = "k",
    include_zero_omega: bool = True,
) -> list[dict]:
    """Continuity-constrained branch tracking on the native (omega, k_parallel) grid.

    ``apply_jacobian`` reweights by |omega|/k^2 before tracking. That factor
    belongs to the v_phase histogram, where it compensates the nonuniform
    sampling of the omega/k mapping. On the native (omega, k) grid there is no
    such mapping, so the factor is not a correction but a 1/k^2 bias: it pulls
    the tracked ridge onto the smallest resolved k at every frequency and
    produces a vertical stripe at k = 2*pi/L regardless of the physics. It is
    therefore off by default.

    ``use_velocity_mask`` restricts tracking to the v_phase window used by the
    density plot. That window has a lower bound, so it silently removes any
    mode with omega -> 0 at finite k -- exactly the aperiodic modes. Prefer a
    physical ``kmax_di`` cap to suppress grid-scale noise instead.

    ``ridge_axis`` selects which variable is treated as independent.
    ``"k"`` (default) reports one omega per resolved wavenumber, which is what
    a dispersion relation is: k is quantised by the box and is the controlled
    variable, omega is measured. ``"omega"`` is the legacy row-walk that scans
    frequencies and reports the strongest k at each; with only a handful of
    resolved k it returns hundreds of points stacked on the same wavenumber,
    which reads as a branch on paper but is a vertical stripe on the plot.
    """
    omega = result["omega"]
    k0 = result["k0"]
    k1 = result["k1"]
    axes = result["axes"]
    parallel_axis = result["parallel_axis"]
    absolute_velocity = result["absolute_velocity"]
    velocity_min = result["velocity_min"]
    velocity_max = result["velocity_max"]

    power = (
        np.where(result["valid"], result["power"], 0.0)
        if use_velocity_mask
        else result["power"]
    )

    p2d, k_par = _reduce_kperp(
        power, k0, k1, axes, parallel_axis, kperp_reduction,
        theta_max_deg=theta_max_deg, theta_grid=result["theta_grid"],
        theta_min_deg=theta_min_deg,
    )
    # Aperiodic modes live at omega = 0 exactly; dropping that row guarantees
    # they can never be reported.
    pos_w = omega >= 0 if include_zero_omega else omega > 0
    w = omega[pos_w]
    p2d = p2d[pos_w, :]
    if absolute_velocity:
        # Fold +k/-k power together: propagation direction is discarded, so a
        # mode's real power must not be dropped just because it happens to
        # land on the negative side of k_parallel for this FFT sign convention.
        p2d, kk = _fold_signed_kpar(p2d, k_par)
    else:
        pos_k = np.abs(k_par) > 1e-12
        kk = k_par[pos_k]
        p2d = p2d[:, pos_k]
    if w.size == 0 or kk.size == 0:
        return []

    if apply_jacobian:
        jacobian = np.abs(w)[:, None] / np.maximum(np.abs(kk[None, :]), 1e-12) ** 2
        p2d = p2d * jacobian

    # Measuring at a discrete box mode must never borrow power from another
    # k bin. Smoothing creates fictitious branches around an isolated wave.
    if ridge_axis != "k":
        p2d = _smooth_2d(p2d, passes=2)

    # Smoothing mixes power across neighboring omega rows, which can leak a
    # little power past the velocity_min/velocity_max boundary into cells
    # that were exactly zero before smoothing. Re-clip so accepted ridge
    # points always land inside the requested, physically-meaningful range.
    if use_velocity_mask:
        velocity_grid = w[:, None] / kk[None, :]
        in_velocity_range = (
            (np.abs(velocity_grid) >= velocity_min)
            & (np.abs(velocity_grid) <= velocity_max)
        )
        p2d = np.where(in_velocity_range, p2d, 0.0)

    global_peak = float(np.max(p2d)) if p2d.size else 0.0
    power_floor = (10.0 ** (min_power_db / 10.0)) * global_peak
    max_jump = max_jump_fraction * (float(np.max(kk)) - float(np.min(kk)))
    exclusion = max(2, len(kk) // 40)

    rows: list[dict] = []

    if ridge_axis == "k":
        # One measured omega per resolved k. The half-power width in omega is
        # reported alongside it: for an unstable mode that width is bounded
        # below by both the growth-rate broadening (~2*gamma) and the window
        # resolution (2*pi/T), so a peak whose width exceeds its own centre
        # frequency is not a branch and the CSV should say so.
        domega = float(w[1] - w[0]) if w.size > 1 else float("nan")
        resolution = float(result.get("omega_resolution", domega))
        omega_exclusion = max(1, int(np.ceil(resolution / domega)))
        sample_times = result["times"] - result["times"][0]
        taper = _make_window(result["temporal_window"], len(sample_times), result["window_alpha"])
        taper_sum = max(float(taper.sum()), np.finfo(float).tiny)
        for column, this_k in enumerate(kk):
            work_column = p2d[:, column]
            # Only local maxima, with contrast above the column noise floor.
            # Suppressing bins then taking argmax again selects shoulders of
            # the same peak and used to report them as additional branches.
            candidates = np.flatnonzero(
                (work_column > np.r_[-np.inf, work_column[:-1]])
                & (work_column >= np.r_[work_column[1:], -np.inf])
            )
            candidates = sorted(candidates, key=lambda i: work_column[i], reverse=True)
            accepted = []
            accepted_peaks = []
            for index in candidates:
                if len(accepted) >= ridge_count:
                    break
                if any(abs(index - previous) <= omega_exclusion for previous in accepted):
                    continue
                peak_power = float(work_column[index])
                if peak_power <= power_floor or not np.isfinite(peak_power):
                    break
                if peak_power < 8.0 * float(np.median(work_column)):
                    continue
                this_omega = float(w[index]) + _parabolic_offset(work_column, index) * domega
                # Test the actual taper's sidelobe response. Otherwise a
                # single Tukey-windowed sinusoid produces a spurious second
                # "branch" about 13 dB below its real peak after padding.
                leakage = 0.0
                for previous_omega, previous_power in accepted_peaks:
                    response = abs(np.sum(taper * np.exp(1j * (this_omega - previous_omega) * sample_times)) / taper_sum)**2
                    if absolute_velocity:
                        response += abs(np.sum(taper * np.exp(1j * (this_omega + previous_omega) * sample_times)) / taper_sum)**2
                    leakage += previous_power * response
                if accepted_peaks and peak_power <= 4 * leakage:
                    continue
                half = 0.5 * peak_power
                lo_index = index
                while lo_index > 0 and work_column[lo_index] > half:
                    lo_index -= 1
                hi_index = index
                while hi_index < len(work_column) - 1 and work_column[hi_index] > half:
                    hi_index += 1
                width = float(w[hi_index] - w[lo_index])
                velocity = this_omega / this_k
                if absolute_velocity:
                    velocity = abs(velocity)
                rows.append(
                    {
                        "k_parallel_d_i": float(this_k),
                        "omega_over_omega_ci": this_omega,
                        "omega_fwhm_over_omega_ci": width,
                        "omega_resolution_over_omega_ci": resolution,
                        "omega_fft_bin_spacing_over_omega_ci": domega,
                        "phase_velocity_over_va": float(velocity),
                        "resolved": int(
                            np.isfinite(width) and this_omega > max(width, resolution)
                            and index < len(w) - 1
                        ),
                        "ridge_rank": len(accepted) + 1,
                        "spectral_power": peak_power,
                    }
                )
                accepted.append(index)
                accepted_peaks.append((this_omega, peak_power))
        return rows

    work_rows = p2d.copy()
    for rank in range(1, ridge_count + 1):
        prev_k = None
        for row_index, this_omega in enumerate(w):
            row = work_rows[row_index]
            if not np.any(row > power_floor):
                continue
            if prev_k is not None:
                candidate_mask = np.abs(kk - prev_k) <= max_jump
                if np.any(candidate_mask & (row > power_floor)):
                    row = np.where(candidate_mask, row, -np.inf)
            index = int(np.argmax(row))
            peak_power = float(row[index])
            if peak_power <= power_floor or not np.isfinite(peak_power):
                prev_k = None
                continue
            prev_k = float(kk[index])
            velocity = this_omega / prev_k
            if absolute_velocity:
                velocity = abs(velocity)
            rows.append(
                {
                    "omega_over_omega_ci": float(this_omega),
                    "phase_velocity_over_va": float(velocity),
                    "k_parallel_d_i": prev_k,
                    "ridge_rank": rank,
                    "spectral_power": float(p2d[row_index, index]),
                }
            )
            lo = max(0, index - exclusion)
            hi = min(len(work_rows[row_index]), index + exclusion + 1)
            work_rows[row_index, lo:hi] = 0.0
    return rows


def extract_ridges(
    result: dict,
    ridge_count: int = 2,
    *,
    ridge_source: str = "omega-k",
    kperp_reduction: str = "sum",
    theta_max_deg: float | None = None,
    min_power_db: float = -30.0,
    max_jump: float = 0.12,
    theta_min_deg: float | None = None,
    ridge_axis: str = "k",
    apply_jacobian: bool = False,
    use_velocity_mask: bool = False,
    include_zero_omega: bool = True,
) -> list[dict]:
    """Find dispersion-branch peaks on the native (omega, k_parallel) grid.

    ``ridge_source="omega-k"`` (default) tracks branches on the native,
    globally-normalized (omega, k_parallel) grid with a minimum-power floor.
    With ``ridge_axis="k"`` it reports one omega per resolved wavenumber plus
    the half-power width, which is the form a dispersion measurement should
    take; with ``ridge_axis="omega"`` it reproduces the legacy continuity-
    constrained walk across frequencies.
    ``ridge_source="v-phase"`` reproduces the legacy independent per-omega
    argmax on the v_phase histogram, now with the same power floor applied.
    """
    if ridge_source == "omega-k":
        return _extract_ridges_omega_k(
            result, ridge_count, kperp_reduction, theta_max_deg,
            min_power_db, max_jump,
            apply_jacobian=apply_jacobian,
            use_velocity_mask=use_velocity_mask,
            theta_min_deg=theta_min_deg,
            ridge_axis=ridge_axis,
            include_zero_omega=include_zero_omega,
        )
    if ridge_source == "v-phase":
        return _extract_ridges_v_phase(result, ridge_count, min_power_db)
    raise ValueError(f"Unknown ridge_source: {ridge_source!r}")


def plot_density(
    result: dict,
    ridges: list[dict],
    output: Path,
    component: str,
    omega_max_ci: float | None = None,
    vph_max: float | None = None,
    autocrop: bool = True,
    autocrop_floor: float = 1e-2,
):
    density = result["density"]
    normalized = density / max(float(np.max(density)), np.finfo(float).tiny)
    log_density = np.log10(normalized + 1e-8)

    fig, axis = plt.subplots(figsize=(9.2, 7.0))
    image = axis.pcolormesh(
        result["omega_edges"],
        result["velocity_edges"],
        log_density.T,
        shading="auto",
        cmap=ps.CMAP_SEQUENTIAL,
        rasterized=True,
        vmin=-6,
        vmax=0,
    )
    levels = [-4.0, -3.0, -2.0, -1.0]
    omega_centers = 0.5 * (
        result["omega_edges"][:-1] + result["omega_edges"][1:]
    )
    velocity_centers = 0.5 * (
        result["velocity_edges"][:-1] + result["velocity_edges"][1:]
    )
    axis.contour(
        omega_centers,
        velocity_centers,
        log_density.T,
        levels=levels,
        colors="white",
        linewidths=0.7,
        linestyles="dotted",
        alpha=0.8,
    )

    for rank in sorted({row["ridge_rank"] for row in ridges}):
        selected = [row for row in ridges if row["ridge_rank"] == rank]
        axis.plot(
            [row["omega_over_omega_ci"] for row in selected],
            [row["phase_velocity_over_va"] for row in selected],
            "k.",
            markersize=5 if rank == 1 else 3.5,
            alpha=0.95 if rank == 1 else 0.65,
            label="dominant ridge" if rank == 1 else None,
        )

    velocity_label = (
        r"$|v_{\rm ph}|/v_A$"
        if result["absolute_velocity"]
        else r"$v_{\rm ph}/v_A$"
    )
    axis.set_xlabel(r"Angular frequency $\omega/\Omega_{ci}$")
    axis.set_ylabel(velocity_label)
    axis.set_title(
        f"Phase-velocity projection ({component} magnetic power)"
    )
    axis.grid(alpha=0.15)
    if ridges:
        axis.legend(loc="upper right")

    omega_marginal = np.sum(normalized, axis=1)
    velocity_marginal = np.sum(normalized, axis=0)
    omega_signal = _autocrop_signal_extent(omega_centers, omega_marginal, autocrop_floor)
    velocity_signal = _autocrop_signal_extent(velocity_centers, velocity_marginal, autocrop_floor)

    if omega_max_ci is not None:
        axis.set_xlim(0, omega_max_ci)
    elif autocrop and omega_signal is not None:
        axis.set_xlim(0, float(omega_signal.max()) * 1.3)

    if vph_max is not None:
        if result["absolute_velocity"]:
            axis.set_ylim(0, vph_max)
        else:
            axis.set_ylim(-vph_max, vph_max)
    elif autocrop and velocity_signal is not None:
        if result["absolute_velocity"]:
            axis.set_ylim(0, float(velocity_signal.max()) * 1.3)
        else:
            extent = max(abs(float(velocity_signal.min())), abs(float(velocity_signal.max()))) * 1.3
            axis.set_ylim(-extent, extent)

    colorbar = fig.colorbar(image, ax=axis)
    colorbar_label = (
        r"$\log_{10}[P(v_{\rm ph}\mid\omega)/P_{\max}]$"
        if result["density_normalization"] == "conditional"
        else r"$\log_{10}[P(v_{\rm ph},\omega)/P_{\max}]$"
    )
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    ps.save(fig, output)
    plt.close(fig)


def plot_omega_k_dispersion(
    result: dict,
    output: Path,
    component: str,
    va_over_c: float,
    kperp_reduction: str = "sum",
    theta_max_deg: float | None = None,
    omega_max_ci: float | None = None,
    kpar_max_di: float | None = None,
    autocrop: bool = True,
    autocrop_floor: float = 1e-2,
    theta_min_deg: float | None = None,
    omega_scale: str = "linear",
    ridges: list[dict] | None = None,
):
    """Dense omega-k dispersion diagram: log(omega/omega_p) vs log(k c/omega_p),
    or a linear k axis when ``result`` carries signed velocities.

    ``omega_scale="linear"`` keeps the omega = 0 row on the plot. Logarithmic
    axes cannot represent it at all, so for an aperiodic mode the log version
    shows only whatever leaks to finite frequency and never the mode itself.

    Uses the full ``power`` cube (nfft, nk0, nk1) already computed in
    ``compute_phase_velocity_density``. Because time is Omega_ci t and spacing is
    in d_i, we have k c/omega_pi = k d_i exactly, and omega/omega_pi =
    (omega/Omega_ci) * (v_A/c). Pass ``va_over_c`` (= Omega_ci/omega_pi).

    Sign convention is derived from ``result["absolute_velocity"]`` so this
    plot can never disagree with the companion v_phase density plot about
    whether forward/backward propagation is being distinguished.
    """
    power = result["power"]                      # (nfft, nk0, nk1)
    omega = result["omega"]                      # omega/Omega_ci, shape (nfft,)
    k0 = result["k0"]                            # k*d_i along axis0
    k1 = result["k1"]                            # k*d_i along axis1
    axes = result["axes"]
    parallel_axis = result["parallel_axis"]
    signed = not result["absolute_velocity"]

    p2d, k_par = _reduce_kperp(
        power, k0, k1, axes, parallel_axis, kperp_reduction,
        theta_max_deg=theta_max_deg, theta_grid=result["theta_grid"],
        theta_min_deg=theta_min_deg,
    )

    # A log frequency axis cannot show omega = 0, so that row is only kept when
    # the axis is linear; otherwise log10(0) would propagate -inf into the mesh.
    pos_w = omega >= 0 if omega_scale == "linear" else omega > 0
    w = omega[pos_w]                             # omega/Omega_ci
    p2d = p2d[pos_w, :]
    if signed:
        pos_k = np.abs(k_par) > 1e-12
        kk = k_par[pos_k]                         # k d_i = k c/omega_pi
        p2d = p2d[:, pos_k]
    else:
        # Fold +k/-k power together: a mode's power can land on either sign
        # of k_parallel depending on FFT sign convention, and slicing to
        # k_parallel > 0 alone would silently drop real signal.
        p2d, kk = _fold_signed_kpar(p2d, k_par)

    if not len(kk) or not len(w):
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.text(0.5, 0.5, "No nonzero parallel wavenumbers in the retained band.\nSee the full wavevector mode summary.",
                  transform=axis.transAxes, ha="center", va="center")
        ps.save(fig, output)
        return

    # Convert to requested units.
    w_wp = w * va_over_c                         # omega/omega_pi
    kc_wp = kk                                   # k c/omega_pi = k d_i (identity)

    pnorm = p2d / max(float(np.max(p2d)), np.finfo(float).tiny)
    log_p = np.log10(pnorm + 1e-8)

    # This plot renders the raw, unweighted native grid on purpose (that's the
    # point of showing it before any v_phase transform). But raw power here has
    # poor per-pixel SNR - a broadband, fairly uniform floor spread across many
    # k columns - which drags out the marginal-coverage crop even though the
    # real ridge is visually obvious. Smooth a copy purely to size the crop;
    # what actually gets rendered (log_p) stays untouched.
    pnorm_for_crop = _smooth_2d(pnorm, passes=1)
    k_marginal = np.sum(pnorm_for_crop, axis=0)
    omega_marginal = np.sum(pnorm_for_crop, axis=1)
    kc_signal = _autocrop_signal_extent(kc_wp, k_marginal, autocrop_floor)
    w_wp_signal = _autocrop_signal_extent(w_wp, omega_marginal, autocrop_floor)
    has_k_signal = kc_signal is not None
    has_w_signal = w_wp_signal is not None

    fig, axis = plt.subplots(figsize=(9.6, 7.4))
    if signed or omega_scale == "linear":
        x_coord, y_coord = kc_wp, w_wp
        image = axis.pcolormesh(
            x_coord, y_coord, log_p,
            shading="auto", cmap=ps.CMAP_SEQUENTIAL, vmin=-6, vmax=0, rasterized=True,
        )
        kline = np.linspace(kc_wp.min(), kc_wp.max(), 50)
        axis.plot(
            kline, np.abs(kline) * va_over_c,
            color="white", ls="--", lw=1.2, alpha=0.7, label=r"$v_{ph}=v_A$",
        )
        axis.set_xlabel(r"$k_\parallel d_i$" if signed else r"$|k_\parallel| d_i$")
        axis.set_ylabel(r"$\omega/\omega_{pi}$")
        # When the velocities are unsigned, k has already been folded onto the
        # positive half, so a symmetric x range would be half empty.
        k_low = -1.0 if signed else 0.0
        if kpar_max_di is not None:
            axis.set_xlim(k_low * kpar_max_di, kpar_max_di)
        elif autocrop and has_k_signal:
            extent = float(np.max(np.abs(kc_signal))) * 1.3
            axis.set_xlim(k_low * extent, extent)
        if omega_max_ci is not None:
            axis.set_ylim(0, omega_max_ci * va_over_c)
        elif autocrop and has_w_signal:
            axis.set_ylim(0, max(float(w_wp_signal.max()) * 1.3,
                                  2 * result["omega_resolution"] * va_over_c))
    else:
        x_coord, y_coord = np.log10(kc_wp), np.log10(w_wp)
        image = axis.pcolormesh(
            x_coord, y_coord, log_p,
            shading="auto", cmap=ps.CMAP_SEQUENTIAL, vmin=-6, vmax=0, rasterized=True,
        )
        # Reference line: Alfven phase speed v_ph = v_A  ->  omega/Omega_ci = k d_i
        #   -> omega/omega_pi = (k d_i) * va_over_c
        kline = np.logspace(np.log10(kc_wp.min()), np.log10(kc_wp.max()), 50)
        axis.plot(
            np.log10(kline), np.log10(kline * va_over_c),
            color="white", ls="--", lw=1.2, alpha=0.7, label=r"$v_{ph}=v_A$",
        )
        axis.set_xlabel(r"$\log_{10}(|k_\parallel| d_i)$")
        axis.set_ylabel(r"$\log_{10}(\omega/\omega_{pi})$")
        if kpar_max_di is not None:
            axis.set_xlim(right=np.log10(kpar_max_di))
        elif autocrop and has_k_signal:
            log_k_signal = np.log10(kc_signal)
            axis.set_xlim(log_k_signal.min() - 0.1, log_k_signal.max() + 0.3)
        if omega_max_ci is not None:
            axis.set_ylim(top=np.log10(omega_max_ci * va_over_c))
        elif autocrop and has_w_signal:
            log_w_signal = np.log10(w_wp_signal)
            axis.set_ylim(log_w_signal.min() - 0.2, log_w_signal.max() + 0.2)
    if ridges:
        measured = [r for r in ridges if np.isfinite(r["k_parallel_d_i"])
                    and (omega_scale == "linear" or r["omega_over_omega_ci"] > 0)]
        if measured:
            rk = np.array([r["k_parallel_d_i"] for r in measured])
            rw = np.array([r["omega_over_omega_ci"] for r in measured]) * va_over_c
            if not signed and omega_scale == "log":
                rk, rw = np.log10(rk), np.log10(rw)
            axis.scatter(rk, rw, s=24, facecolors="none", edgecolors="white",
                         linewidths=0.9, label="Native-bin peaks", zorder=4)
    axis.set_title(f"Dispersion diagram ({component} magnetic power)")
    if omega_scale == "linear":
        secondary = axis.secondary_yaxis("right", functions=(lambda x: x / va_over_c,
                                                             lambda x: x * va_over_c))
        secondary.set_ylabel(r"$\omega/\Omega_{ci}$")
    axis.legend(loc="lower right")
    colorbar = fig.colorbar(image, ax=axis, pad=0.16 if omega_scale == "linear" else 0.03)
    colorbar.set_label(r"$\log_{10}[P(\omega,k_\parallel)/P_{\max}]$")
    fig.tight_layout()
    ps.save(fig, output)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_series(
    pattern: str,
    plane: str,
    parallel_axis: str,
    component: str,
    dx: float,
    dy: float,
    dz: float,
    step_to_time,
    *,
    t_start: float | None = None,
    t_end: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    files = PICDataReader.find_files(pattern)
    total_found = len(files)
    if t_start is not None or t_end is not None:
        lo = -np.inf if t_start is None else t_start
        hi = np.inf if t_end is None else t_end
        files = {
            step: path
            for step, path in files.items()
            if lo <= step_to_time(step) <= hi
        }
    if len(files) < 4:
        raise ValueError(
            "Need at least four snapshots inside "
            f"[{t_start}, {t_end}]; found {len(files)} of {total_found} total"
        )

    slicer = SpectralAnalyzer(
        dx=dx, dy=dy, dz=dz, parallel_axis=parallel_axis, outdir="/tmp/psc-dispersion"
    )
    files = dict(sorted(files.items()))
    perpendicular_axes = [axis for axis in ("x", "y", "z") if axis != parallel_axis]
    metadata = {}

    def load_frame(filepath):
        fields = PICDataReader.read_multiple_fields_3d(
            filepath, "jeh-", ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"]
        )
        plane_data = slicer._get_plane_slice(
            fields["hx_fc/p0/3d"], fields["hy_fc/p0/3d"], fields["hz_fc/p0/3d"], plane
        )
        # Keep geometry only, not references to the full field arrays.
        for key in ("axes", "spacing", "plane", "normal_axis", "slice_idx"):
            metadata[key] = plane_data[key]
        components = {axis: np.atleast_2d(plane_data[f"b{axis}"]) for axis in "xyz"}
        selected = (perpendicular_axes if component == "perp" else
                    [parallel_axis] if component == "parallel" else list("xyz"))
        return np.stack([components[axis] for axis in selected])

    series = SnapshotSeries(files.values(), load_frame)
    metadata["source_files"] = list(files.values())
    metadata["storage"] = "streaming; one spatial snapshot at a time"
    return series, np.asarray([step_to_time(step) for step in files]), metadata


# Per-mode analysis defaults. These only fill in options the user did not set
# explicitly, so an explicit flag always wins.
MODE_ANALYSIS_DEFAULTS: dict[str, dict] = {
    "mirror": {
        # Aperiodic and oblique: keep omega = 0, look in the 45-85 deg band,
        # and take the maximum over k_perp rather than summing, so the oblique
        # peak is not diluted by the quasi-parallel range where it does not live.
        "omega_scale": "linear", "kmax_di": 2.0, "theta_min_deg": 45.0,
        "theta_max_deg": 85.0, "kperp_reduction": "max", "time_detrend": "none",
        "degrowth": "per-k", "velocity_min": 0.0,
    },
    "firehose-oblique": {
        "omega_scale": "linear", "kmax_di": 2.0, "theta_min_deg": 45.0,
        "theta_max_deg": 85.0, "kperp_reduction": "max", "time_detrend": "none",
        "degrowth": "per-k", "velocity_min": 0.0,
    },
    "firehose-parallel": {
        "omega_scale": "linear", "kmax_di": 1.5, "theta_max_deg": 30.0,
        "kperp_reduction": "sum", "time_detrend": "none", "degrowth": "per-k",
        "velocity_min": 0.0,
    },
    "emic": {
        "omega_scale": "linear", "kmax_di": 2.0, "theta_max_deg": 25.0,
        "kperp_reduction": "sum", "time_detrend": "none", "degrowth": "per-k",
        "velocity_min": 0.0,
    },
    "whistler": {
        # Electron scale: k d_i ~ sqrt(mi/me) * k d_e, so the useful band sits
        # an order of magnitude higher in k than for the ion modes.
        "omega_scale": "log", "kmax_di": 20.0, "theta_max_deg": 25.0,
        "kperp_reduction": "sum", "time_detrend": "none", "degrowth": "per-k",
        "velocity_min": 0.0,
    },
    "generic": {},
}


def main() -> int:
    try:
        from psc_units import DX_DI, VA_OVER_C, step_to_omegaci
    except (ImportError, ValueError):
        DX_DI = 1.0
        VA_OVER_C = None
        step_to_omegaci = lambda step: float(step)
    try:
        from psc_units import MASS_RATIO
    except (ImportError, ValueError):
        MASS_RATIO = None

    parser = argparse.ArgumentParser(
        description="Frequency--phase-velocity density map from PSC field snapshots."
    )
    parser.add_argument("--fields", default="pfd.*.h5")
    parser.add_argument("--plane", choices=["auto", "xy", "xz", "yz"], default="auto")
    parser.add_argument("--parallel-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--component", choices=["perp", "parallel", "total"], default="total",
                        help="Magnetic components to analyse; total permits compressibility/polarization measurements.")
    parser.add_argument("--dx", type=float, default=DX_DI)
    parser.add_argument("--dy", type=float, default=DX_DI)
    parser.add_argument("--dz", type=float, default=DX_DI)
    parser.add_argument("--mode", choices=sorted(MODE_PRESETS), default="generic",
                        help="Instability preset. Sets physically appropriate defaults for "
                             "k range, angular band, frequency axis and time handling, and "
                             "drives the resolution report. Explicit flags always win.")
    parser.add_argument("--mass-ratio", type=float, default=MASS_RATIO,
                        help="m_i/m_e, needed to express electron-scale presets (whistler) "
                             "in ion units. Defaults to psc_units.MASS_RATIO when available.")
    parser.add_argument("--velocity-min", type=float, default=None,
                        help="Lower |v_phase|/v_A bound for the density plot. Default 0. "
                             "A nonzero value removes every mode with omega -> 0 at finite k, "
                             "i.e. all aperiodic modes.")
    parser.add_argument("--velocity-max", type=float, default=12.0)
    parser.add_argument("--max-spatial-mode", type=int, default=128)
    parser.add_argument("--kmax-di", type=float, default=None,
                        help="Physical cap on retained k*d_i. Preferred over --max-spatial-mode: "
                             "a fixed mode count reaches k d_i ~ 40 in a 20 d_i box, far above "
                             "any ion-scale physics, so the extra range is PIC noise.")
    parser.add_argument("--spatial-window", choices=["none", "hann", "tukey"], default="none",
                        help="Window applied to each snapshot before the spatial FFT. Default "
                             "'none': these runs are periodic in space, so a snapshot is already "
                             "an exact period and windowing it scatters a third of every mode's "
                             "power into the two adjacent wavenumbers.")
    parser.add_argument("--temporal-window", choices=["tukey", "hann", "none"], default="tukey",
                        help="Window applied along time before the temporal FFT. Time is not "
                             "periodic so a window is needed here; 'tukey' tapers only the ends "
                             "and keeps most of the frequency resolution that 'hann' would spend.")
    parser.add_argument("--window-alpha", type=float, default=0.25,
                        help="Tapered fraction of a Tukey window (0 = rectangular, 1 = Hann).")
    parser.add_argument("--time-detrend", choices=["none", "mean", "linear"], default=None,
                        help="Per-cell temporal detrending before the FFT. Default 'none'. "
                             "'mean' deletes the omega = 0 row exactly and must not be used "
                             "for mirror or oblique firehose.")
    parser.add_argument("--degrowth", choices=["none", "global", "per-k"], default=None,
                        help="Divide out the exp(gamma t) envelope before the temporal FFT so "
                             "the transform sees a stationary signal. 'per-k' fits gamma "
                             "independently for every retained mode.")
    parser.add_argument("--omega-scale", choices=["log", "linear"], default=None,
                        help="Frequency axis of the omega-k diagram. 'linear' keeps omega = 0 "
                             "visible, which a log axis cannot represent.")
    parser.add_argument("--theta-min-deg", type=float, default=None,
                        help="Lower bound on propagation angle theta_kB (degrees). Combined with "
                             "--theta-max-deg this selects an angular band, which is what oblique "
                             "modes need.")
    parser.add_argument("--ridge-axis", choices=["k", "omega"], default="k",
                        help="'k' (default) reports one omega per resolved wavenumber plus its "
                             "half-power width. 'omega' is the legacy walk across frequencies, "
                             "which stacks many points on the same k when few k are resolved.")
    parser.add_argument("--ridge-jacobian", action="store_true",
                        help="Reweight the ridge tracker by |omega|/k^2. Off by default: on the "
                             "native (omega,k) grid this is a 1/k^2 bias that pins every ridge "
                             "to the smallest resolved wavenumber.")
    parser.add_argument("--ridge-velocity-mask", action="store_true",
                        help="Restrict ridge tracking to the v_phase window. Off by default "
                             "because its lower bound removes aperiodic modes.")
    parser.add_argument("--temporal-fft-size", type=int, default=128)
    parser.add_argument("--signed-velocity", action="store_true")
    parser.add_argument("--ridges", type=int, default=2)
    parser.add_argument("--va-over-c", type=float, default=VA_OVER_C,
                        help="v_A/c = Omega_ci/omega_pi. If set, also emit the omega-k dispersion diagram. "
                             "Defaults to psc_units.VA_OVER_C when available.")
    parser.add_argument("--kperp-reduction", choices=["sum", "max", "slice"], default="sum",
                        help="How to collapse the perpendicular k axis for the omega-k diagram.")
    parser.add_argument("--theta-max-deg", type=float, default=None,
                        help="Restrict to propagation angles theta_kB <= this value (degrees). "
                             "Default keeps all angles (no filtering).")
    parser.add_argument("--t-start", type=float, default=None,
                        help="Only use snapshots with Omega_ci*t >= this value.")
    parser.add_argument("--t-end", type=float, default=None,
                        help="Only use snapshots with Omega_ci*t <= this value.")
    parser.add_argument("--density-normalization", choices=["global", "conditional"], default="global",
                        help="Normalize the v_phase density globally (P/P_max, default) or per-omega-column "
                             "(legacy P(v_ph|omega), can make noise-only columns look as strong as signal).")
    parser.add_argument("--power-floor-db", type=float, default=-40.0,
                        help="Zero out v_phase density cells this many dB below the global peak, "
                             "before normalization.")
    parser.add_argument("--ridge-source", choices=["omega-k", "v-phase"], default="omega-k",
                        help="Track ridges on the native (omega,k_parallel) grid with continuity "
                             "(default) or on the legacy per-omega v_phase histogram.")
    parser.add_argument("--ridge-min-power-db", type=float, default=-30.0,
                        help="Minimum ridge-point power relative to the global peak, in dB.")
    parser.add_argument("--ridge-max-jump", type=float, default=0.12,
                        help="Maximum allowed step in k_parallel between consecutive ridge points, "
                             "as a fraction of the resolved k_parallel range (omega-k ridge source only).")
    parser.add_argument("--omega-max-ci", type=float, default=None,
                        help="Crop the displayed omega/Omega_ci axis range (cosmetic only). "
                             "Overrides autocrop when set.")
    parser.add_argument("--kpar-max-di", type=float, default=None,
                        help="Crop the displayed k_parallel*d_i axis range on the omega-k plot "
                             "(cosmetic only). Overrides autocrop when set.")
    parser.add_argument("--vph-max", type=float, default=None,
                        help="Crop the displayed v_phase/v_A axis range on the density plot "
                             "(cosmetic only). Overrides autocrop when set.")
    parser.add_argument("--no-autocrop", action="store_true",
                        help="Disable automatic cropping to the region carrying signal (by default, "
                             "both plots crop out the empty axis range above the fastest/slowest modes "
                             "actually present, unless --omega-max-ci/--kpar-max-di/--vph-max override it).")
    parser.add_argument("--autocrop-floor", type=float, default=1e-2,
                        help="Autocrop tolerance: the fraction of total marginal power allowed to be "
                             "excluded from the cropped range (smaller = tighter crop, more likely to "
                             "cut off a weak-but-real tail).")
    parser.add_argument("--outdir", default="spectral_plots")
    args = parser.parse_args()

    # Fill unset options from the mode preset; anything the user passed wins.
    for option, value in MODE_ANALYSIS_DEFAULTS.get(args.mode, {}).items():
        if getattr(args, option, None) is None:
            setattr(args, option, value)
    if args.velocity_min is None:
        args.velocity_min = 0.0
    if args.time_detrend is None:
        args.time_detrend = "none"
    if args.degrowth is None:
        args.degrowth = "none"
    if args.omega_scale is None:
        args.omega_scale = "linear"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    series, times, metadata = load_series(
        args.fields,
        args.plane,
        args.parallel_axis,
        args.component,
        args.dx,
        args.dy,
        args.dz,
        step_to_omegaci,
        t_start=args.t_start,
        t_end=args.t_end,
    )
    result = compute_phase_velocity_density(
        series,
        times,
        metadata["spacing"],
        metadata["axes"],
        parallel_axis=args.parallel_axis,
        velocity_min=args.velocity_min,
        velocity_max=args.velocity_max,
        absolute_velocity=not args.signed_velocity,
        max_spatial_mode=args.max_spatial_mode,
        temporal_fft_size=args.temporal_fft_size,
        theta_max_deg=args.theta_max_deg,
        density_normalization=args.density_normalization,
        power_floor_db=args.power_floor_db,
        time_detrend=args.time_detrend,
        degrowth=args.degrowth,
        kmax_di=args.kmax_di,
        spatial_window=args.spatial_window,
        temporal_window=args.temporal_window,
        window_alpha=args.window_alpha,
        component_axes=tuple(axis for axis in ("x", "y", "z")
                             if args.component == "total"
                             or (axis == args.parallel_axis) == (args.component == "parallel")),
        theta_min_deg=args.theta_min_deg,
    )
    ridges = extract_ridges(
        result,
        ridge_count=args.ridges,
        ridge_source=args.ridge_source,
        kperp_reduction=args.kperp_reduction,
        theta_max_deg=args.theta_max_deg,
        min_power_db=args.ridge_min_power_db,
        max_jump=args.ridge_max_jump,
        theta_min_deg=args.theta_min_deg,
        ridge_axis=args.ridge_axis,
        apply_jacobian=args.ridge_jacobian,
        use_velocity_mask=args.ridge_velocity_mask,
        include_zero_omega=True,
    )
    suffix = "absolute" if result["absolute_velocity"] else "signed"
    image_path = outdir / f"dispersion_density_{metadata['plane']}_{args.component}_{suffix}.png"
    csv_path = outdir / f"dispersion_ridges_{metadata['plane']}_{args.component}_{suffix}.csv"
    plot_density(result, ridges, image_path, args.component,
                omega_max_ci=args.omega_max_ci,
                vph_max=args.vph_max,
                autocrop=not args.no_autocrop,
                autocrop_floor=args.autocrop_floor)
    if args.va_over_c is not None:
        wk_path = outdir / f"dispersion_omega_k_{metadata['plane']}_{args.component}_{suffix}.png"
        plot_omega_k_dispersion(result, wk_path, args.component, args.va_over_c,
                                kperp_reduction=args.kperp_reduction,
                                theta_max_deg=args.theta_max_deg,
                                omega_max_ci=args.omega_max_ci,
                                kpar_max_di=args.kpar_max_di,
                                autocrop=not args.no_autocrop,
                                autocrop_floor=args.autocrop_floor,
                                theta_min_deg=args.theta_min_deg,
                                omega_scale=args.omega_scale, ridges=ridges)
        print(f"Saved omega-k dispersion diagram: {wk_path}")
    write_csv(csv_path, ridges)
    stem = f"dispersion_modes_{metadata['plane']}_{args.component}"
    mode_rows = result["mode_candidates"]
    write_csv(outdir / f"{stem}.csv", mode_rows)
    strongest = mode_rows[0] if mode_rows else None
    dominant = strongest if strongest is not None and strongest["accepted"] else None
    mode_report = {
        "status": "dominant_candidate_detected" if dominant is not None else "dominant_mode_not_confirmed",
        "dominant": dominant,
        "strongest_spatial_peak": strongest,
        "strongest_coherent_candidate": next((r for r in mode_rows if r["accepted"]), None),
        "candidates": mode_rows,
        "ranking": "time-averaged measured power of conjugate spatial pairs before de-growth",
        "frequency_convention": "B = Re[b exp(i k.x - i omega t)]; canonical k_parallel >= 0",
        "polarization_convention": "sigma_B = 2 Im(b1 conj(b2))/(|b1|^2+|b2|^2), right-handed transverse basis",
        "frame": "simulation frame; no bulk-flow Doppler correction",
        "time_range_omega_ci": [float(times[0]), float(times[-1])],
        "plane": metadata["plane"], "axes": metadata["axes"],
        "parallel_axis": args.parallel_axis, "component": args.component,
        "fields_pattern": args.fields,
        "source_files": metadata["source_files"],
        "psc_profile": os.environ.get("PSC_PROFILE", "mirror_bimaxwellian_strong"),
        "settings": vars(args),
        "limitations": "Single-complex-exponential characterization in the selected interval and retained k band; unresolved/aliased frequencies and multiple branches require further data. No instability species is inferred.",
    }
    (outdir / f"{stem}.json").write_text(json.dumps(mode_report, indent=2, allow_nan=False))
    plot_mode_summary(result, outdir / f"{stem}.png")
    plot_mode_fit(result, outdir / f"{stem}_fit.png")
    if dominant is not None:
        print(f"Dominant coherent candidate: k_parallel*d_i={dominant['k_parallel_d_i']:.4g}, "
              f"|k_perp|*d_i={dominant['k_perp_d_i']:.4g}, theta={dominant['theta_deg']:.1f} deg; "
              f"{dominant['status']}")
    else:
        print("The strongest spatial peak is not confirmed as a coherent single mode in this interval.")

    # The resolution report is written unconditionally: what a run can resolve
    # is a property of the box and the output cadence, and it should be on
    # record next to every figure produced from them.
    gamma_estimate = result.get("gamma_global")
    if gamma_estimate is None and result.get("gamma_map") is not None:
        gamma_estimate = float(np.max(result["gamma_map"]))
    report = spectral_resolution_report(
        times,
        metadata["spacing"],
        result["grid_shape"],
        mode=args.mode,
        mass_ratio=args.mass_ratio,
        gamma_ci=gamma_estimate,
    )
    report["settings"] = {
        "spatial_window": args.spatial_window,
        "temporal_window": args.temporal_window,
        "window_alpha": args.window_alpha,
        "time_detrend": args.time_detrend,
        "degrowth": args.degrowth,
        "kmax_di": args.kmax_di,
        "kperp_reduction": args.kperp_reduction,
        "theta_min_deg": args.theta_min_deg,
        "theta_max_deg": args.theta_max_deg,
        "ridge_axis": args.ridge_axis,
        "ridge_jacobian": bool(args.ridge_jacobian),
        "omega_scale": args.omega_scale,
    }
    report_path = outdir / f"dispersion_resolution_{metadata['plane']}_{args.component}.json"
    report_path.write_text(json.dumps(report, indent=2, default=float))

    print(f"Processed {series.shape[1]} snapshots on plane {metadata['plane']}, "
          f"t in [{times.min():.3f}, {times.max():.3f}] Omega_ci^-1.")
    print(
        "Independent positive frequencies: "
        f"{result['independent_positive_frequencies']}; "
        f"displayed FFT bins after zero-padding: {len(result['omega_samples'])}."
    )
    print(f"\nSpectral resolution ({args.mode}):")
    print(f"  d_k  d_i     = {min(a['dk_di_inv'] for a in report['axes']):.4f}   "
          f"(k modes below k d_i = 1: "
          f"{min(a['modes_below_kdi_1'] for a in report['axes'])})")
    print(f"  d_omega/Om_ci = {report['domega_ci']:.4f}   "
          f"(window T = {report['window_oci']:.1f} Om_ci^-1)")
    print(f"  omega_Nyquist = {report['omega_nyquist_ci']:.2f} Om_ci")
    if gamma_estimate is not None:
        print(f"  gamma/Om_ci   = {gamma_estimate:.4f} (fitted)")
    for entry in report["checks"]:
        print(f"  [{entry['status']}] {entry['name']}: {entry['detail']}")

    print(f"\nSaved density map: {image_path}")
    print(f"Saved modal ridges: {csv_path}")
    print(f"Saved resolution report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
