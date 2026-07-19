#!/usr/bin/env python3
"""
polarization_dispersion.py
===========================
omega-k_parallel dispersion maps and per-mode growth curves using the
circularly-polarized transverse combination psi_pm = (dBx +- i dBy)/B0,
following the diagnostic used by Shaaban et al. for EMIC / proton-firehose
mode identification (Fourier convention validated against a synthetic
signal, see ``validate_polarization_convention``).

B0 = B0 zhat in this simulation, so k_parallel = k_z and k_perp = k_y; the
transverse components are Bx, By and the compressive one is Bz. This module
covers the parallel/circularly-polarized diagnostic (EMIC, parallel
firehose, whistler). Mirror is fundamentally oblique/compressive, so it
gets its own delta_Bz(k_perp, k_parallel) spectrum instead of psi_pm.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from data_reader import PICDataReader
from spectral_analysis import (
    SpectralAnalyzer, _fit_growth_rate,
    DARK_BG, PANEL_BG, TEXT_CLR, GRID_CLR, _new_dark_fig,
)
import matplotlib.pyplot as plt  # noqa: E402  (Agg backend set by spectral_analysis import above)
from psc_units import (
    B0, DI, DX_DI, MASS_RATIO, KAPPA, INSTABILITY, PROFILE_LABEL,
    SIM_PROFILE, step_to_omegaci,
)

EPS = 1e-30

# Mirror is fundamentally oblique/compressive (see module docstring), but the
# Makefile runs this psi_pm diagnostic unconditionally for every instability.
# For a mirror case, spatial_fft_kpar0 below only keeps the k_perp=0 slice, a
# purely field-aligned fluctuation that is a *different* mode from the real
# oblique mirror mode (theta_kB != 0, see mirror_bparallel_spectrum output).
# Flag that on every psi_pm plot instead of letting it look like a duplicate
# or a broken +/- polarization split.
MIRROR_CAVEAT = (
    r"Note: $\psi_\pm$ only sees the $k_\perp=0$ slice; this mirror case's actual "
    "oblique instability is shown in mirror_bparallel_spectrum, not here."
)


def _add_mirror_caveat(fig) -> None:
    if INSTABILITY != "mirror":
        return
    fig.text(0.5, -0.02, MIRROR_CAVEAT, ha="center", va="top", fontsize=9.5, color="#8b949e")


# ── Data loading ──────────────────────────────────────────────────────────

def load_bxyz_series(pattern: str, plane: str, dx: float, dy: float, dz: float) -> dict:
    """Read Bx, By, Bz on one plane for every snapshot in ``pattern``."""
    files = PICDataReader.find_files(pattern)
    if len(files) < 4:
        raise ValueError(f"Need at least four snapshots; found {len(files)}")

    slicer = SpectralAnalyzer(dx=dx, dy=dy, dz=dz, outdir="/tmp/psc-polarization")
    steps = sorted(files)
    bx_list, by_list, bz_list = [], [], []
    metadata = None
    for step in steps:
        fields = PICDataReader.read_multiple_fields_3d(
            files[step], "jeh-", ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"]
        )
        plane_data = slicer._get_plane_slice(
            fields["hx_fc/p0/3d"], fields["hy_fc/p0/3d"], fields["hz_fc/p0/3d"], plane
        )
        bx_list.append(np.atleast_2d(plane_data["bx"]))
        by_list.append(np.atleast_2d(plane_data["by"]))
        bz_list.append(np.atleast_2d(plane_data["bz"]))
        metadata = plane_data

    return {
        "bx": np.asarray(bx_list, dtype=np.float64),
        "by": np.asarray(by_list, dtype=np.float64),
        "bz": np.asarray(bz_list, dtype=np.float64),
        "steps": np.asarray(steps, dtype=int),
        "axes": metadata["axes"],
        "spacing": metadata["spacing"],
        "plane": metadata["plane"],
        "normal_axis": metadata["normal_axis"],
    }


# ── Core transforms ───────────────────────────────────────────────────────

def remove_spatial_mean(field_t: np.ndarray) -> np.ndarray:
    """delta_B(y,z,t) = B(y,z,t) - <B(y,z,t)>_{y,z}, removed every instant so
    the k=0 mode never leaks into the spectrum."""
    return field_t - np.mean(field_t, axis=(1, 2), keepdims=True)


def build_psi(bx_t: np.ndarray, by_t: np.ndarray, b0: float) -> tuple[np.ndarray, np.ndarray]:
    dbx = remove_spatial_mean(bx_t)
    dby = remove_spatial_mean(by_t)
    psi_plus = (dbx + 1j * dby) / b0
    psi_minus = (dbx - 1j * dby) / b0
    return psi_plus, psi_minus


def spatial_fft_kpar0(
    psi: np.ndarray, spacing: tuple[float, float], axes: tuple[str, str],
    parallel_axis: str, apply_window: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """2D spatial FFT per snapshot, then select the k_perp = 0 bin.

    Returns ``A(t, k_parallel)`` (complex) and the parallel-k axis, both
    already expressed in whatever length unit ``spacing`` was given in
    (feed spacing in d_i to get k*d_i, or spacing*DI to get k*d_e).
    """
    nt, n0, n1 = psi.shape
    if apply_window:
        window = np.hanning(n0)[:, None] * np.hanning(n1)[None, :]
        psi = psi * window[None, :, :]

    psi_k = np.fft.fftshift(np.fft.fft2(psi, axes=(1, 2)), axes=(1, 2)) / (n0 * n1)
    k0 = np.fft.fftshift(np.fft.fftfreq(n0, d=spacing[0])) * 2.0 * np.pi
    k1 = np.fft.fftshift(np.fft.fftfreq(n1, d=spacing[1])) * 2.0 * np.pi

    if axes[0] == parallel_axis:
        k_par = k0
        perp_idx = int(np.argmin(np.abs(k1)))
        A = psi_k[:, :, perp_idx]
    elif axes[1] == parallel_axis:
        k_par = k1
        perp_idx = int(np.argmin(np.abs(k0)))
        A = psi_k[:, perp_idx, :]
    else:
        raise ValueError(f"Plane axes {axes} do not contain parallel axis '{parallel_axis}'")
    return A, k_par


def temporal_dispersion(
    A: np.ndarray, times_norm: np.ndarray, nfft: int | None = None, detrend: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """FFT_t[A(k_parallel, t)] -> P(omega, k_parallel) = |S|^2.

    ``times_norm`` must already be the normalized time axis (Omega t); the
    resulting omega axis then comes out directly in units of that Omega.
    """
    nt = A.shape[0]
    dt = float(np.median(np.diff(times_norm)))
    if not np.allclose(np.diff(times_norm), dt, rtol=1e-6, atol=1e-9):
        raise ValueError("Snapshot times must be uniformly spaced for the omega-k map")

    A0 = A - np.mean(A, axis=0, keepdims=True)
    if detrend:
        t_idx = np.arange(nt, dtype=float)
        coeffs_r = np.polyfit(t_idx, A0.real, 1)
        coeffs_i = np.polyfit(t_idx, A0.imag, 1)
        trend = (
            coeffs_r[0][None, :] * t_idx[:, None] + coeffs_r[1][None, :]
            + 1j * (coeffs_i[0][None, :] * t_idx[:, None] + coeffs_i[1][None, :])
        )
        A0 = A0 - trend

    A0 = A0 * np.hanning(nt)[:, None]
    nfft = nfft or nt
    # Space uses the forward-FFT convention exp(-i k z) (spatial_fft_kpar0,
    # same as numpy's default). To land a physical wave exp[i(kz-wt)] on the
    # standard (+k,+omega) quadrant instead of (+k,-omega), time must use the
    # opposite sign, i.e. ifft's exp(+i omega t) kernel rather than fft's.
    # Confirmed empirically by validate_polarization_convention().
    S = np.fft.fftshift(np.fft.ifft(A0, n=nfft, axis=0), axes=0)
    omega = np.fft.fftshift(np.fft.fftfreq(nfft, d=dt)) * 2.0 * np.pi
    return np.abs(S) ** 2, omega


def sigma_m(power_plus: np.ndarray, power_minus: np.ndarray, min_relative_power: float = 1e-4) -> np.ndarray:
    """(P+-P-)/(P++P-), masked to NaN wherever P++P- is below
    ``min_relative_power`` of its own peak. Outside the real signal ridge both
    channels sit at the numerical noise floor, so the raw ratio is noise
    divided by noise (visible as uniform +/-1 speckle covering the whole
    map); masking those pixels leaves only the bins with enough power for the
    ratio to mean anything."""
    total = power_plus + power_minus
    sigma = (power_plus - power_minus) / (total + EPS)
    max_total = float(np.max(total)) if np.max(total) > 0 else 1.0
    return np.where(total >= min_relative_power * max_total, sigma, np.nan)


# ── Convention check (spec section 8) ────────────────────────────────────

def validate_polarization_convention(n: int = 64, nt: int = 32, k_index: int = 3, omega_index: int = 2) -> dict:
    """Confirm which quadrant of (k, omega) each of psi_+ / psi_- lands on
    for a known traveling wave, instead of assuming a sign convention."""
    z = np.arange(n, dtype=float)
    t = np.arange(nt, dtype=float)
    k = 2.0 * np.pi * k_index / n
    omega = 2.0 * np.pi * omega_index / nt
    Z, T = np.meshgrid(z, t, indexing="xy")  # shape (nt, n)

    results = {}
    for by_sign, label in ((1.0, "Bx=cos(kz-wt), By=+sin(kz-wt)"), (-1.0, "Bx=cos(kz-wt), By=-sin(kz-wt)")):
        bx = np.cos(k * Z - omega * T)[:, :, None]
        by = (by_sign * np.sin(k * Z - omega * T))[:, :, None]
        psi_plus, psi_minus = build_psi(bx, by, 1.0)
        entry = {}
        for name, psi in (("psi_plus", psi_plus), ("psi_minus", psi_minus)):
            A, k_par = spatial_fft_kpar0(psi, (1.0, 1.0), ("z", "y"), "z")
            power, omega_axis = temporal_dispersion(A, t)
            idx = np.unravel_index(int(np.argmax(power)), power.shape)
            entry[name] = {
                "peak_k": float(k_par[idx[1]]),
                "peak_omega": float(omega_axis[idx[0]]),
            }
        results[label] = entry
    return results


# ── Theory overlay ────────────────────────────────────────────────────────

def load_theory(path: str) -> dict:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    kdi = np.array([float(r["kdi"]) for r in rows])
    omega_r = np.array([float(r["omega_r_over_Omegai"]) for r in rows])
    gamma = np.array([float(r["gamma_over_Omegai"]) for r in rows])
    order = np.argsort(kdi)
    return {"kdi": kdi[order], "omega_r": omega_r[order], "gamma": gamma[order]}


def interp_theory(theory: dict | None, k_query: float) -> tuple[float, float]:
    if theory is None or len(theory["kdi"]) == 0:
        return float("nan"), float("nan")
    lo, hi = theory["kdi"][0], theory["kdi"][-1]
    if k_query < lo or k_query > hi:
        return float("nan"), float("nan")
    omega_r = float(np.interp(k_query, theory["kdi"], theory["omega_r"]))
    gamma = float(np.interp(k_query, theory["kdi"], theory["gamma"]))
    return omega_r, gamma


# ── Mode selection & growth-rate fitting ─────────────────────────────────

def select_modes(k_par: np.ndarray, final_power: np.ndarray, k_target: float | None) -> list[int]:
    """Pick the discrete k-bin closest to a theoretical k_max plus one
    neighbor, or (if no target given) the dominant positive-k mode plus
    its neighbor."""
    positive = np.where(k_par > 1e-12)[0]
    if positive.size == 0:
        return []
    if k_target is not None:
        center = positive[int(np.argmin(np.abs(k_par[positive] - k_target)))]
    else:
        center = positive[int(np.argmax(final_power[positive]))]
    center_pos = int(np.where(positive == center)[0][0])
    neighbor_pos = center_pos + 1 if center_pos + 1 < len(positive) else center_pos - 1
    modes = [center]
    if 0 <= neighbor_pos < len(positive) and positive[neighbor_pos] != center:
        modes.append(positive[neighbor_pos])
    return modes


def _slope_standard_error(times: np.ndarray, amplitude: np.ndarray, fit: dict) -> float:
    if fit["fit_time_range"] is None:
        return float("nan")
    lo, hi = fit["fit_time_range"]
    mask = (times >= lo) & (times <= hi) & (amplitude > 0)
    t = times[mask]
    n = t.size
    if n < 3:
        return float("nan")
    y = np.log(amplitude[mask])
    yhat = fit["gamma"] * t + fit["intercept"]
    dof = n - 2
    s2 = float(np.sum((y - yhat) ** 2) / dof)
    sxx = float(np.sum((t - np.mean(t)) ** 2))
    if sxx <= 0:
        return float("nan")
    return float(np.sqrt(s2 / sxx))


def growth_rate_rows(
    case: str, distribution: str, instability: str, polarization: str,
    times_norm: np.ndarray, A: np.ndarray, k_par: np.ndarray, modes: list[int],
    length_unit: str, theory: dict | None,
) -> list[dict]:
    rows = []
    for mode_idx in modes:
        amplitude = np.abs(A[:, mode_idx])
        fit = _fit_growth_rate(times_norm, amplitude)
        gamma_error = _slope_standard_error(times_norm, amplitude, fit)
        k_val = float(k_par[mode_idx])
        omega_theory, gamma_theory = interp_theory(theory, k_val) if length_unit == "d_i" else (float("nan"), float("nan"))
        relative_diff = (
            100.0 * (fit["gamma"] - gamma_theory) / gamma_theory
            if np.isfinite(fit["gamma"]) and np.isfinite(gamma_theory) and gamma_theory != 0
            else float("nan")
        )
        rows.append({
            "case": case,
            "distribution": distribution,
            "instability": instability,
            "polarization": polarization,
            "mode_index": mode_idx,
            "kdi": k_val if length_unit == "d_i" else float("nan"),
            "kde": k_val if length_unit == "d_e" else float("nan"),
            "fit_t_start": fit["fit_time_range"][0] if fit["fit_time_range"] else float("nan"),
            "fit_t_end": fit["fit_time_range"][1] if fit["fit_time_range"] else float("nan"),
            "gamma_pic": fit["gamma"],
            "gamma_pic_error": gamma_error,
            "gamma_theory": gamma_theory,
            "relative_difference_pct": relative_diff,
            "R2": fit["rvalue"] ** 2 if np.isfinite(fit["rvalue"]) else float("nan"),
        })
    return rows


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_dispersion_map(
    power: np.ndarray, k_par: np.ndarray, omega: np.ndarray, title: str, outpath: Path,
    k_label: str, omega_label: str, theory: dict | None = None, vmin: float = -8.0, vmax: float = 0.0,
):
    normalized = power / max(float(np.max(power)), np.finfo(float).tiny)
    log_power = np.log10(normalized + 1e-12)

    fig, ax = _new_dark_fig((9.2, 7.2))
    mesh = ax.pcolormesh(k_par, omega, log_power, shading="auto", cmap="turbo", vmin=vmin, vmax=vmax)
    if theory is not None and len(theory["kdi"]):
        ax.plot(theory["kdi"], theory["omega_r"], "--", color="white", lw=1.6, label="linear theory")
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label(r"$\log_{10}(P/P_{\max})$", color=TEXT_CLR)
    cb.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(plt.getp(cb.ax, "yticklabels"), color=TEXT_CLR)
    ax.set_xlabel(k_label)
    ax.set_ylabel(omega_label)
    ax.set_title(title, fontsize=17)
    _add_mirror_caveat(fig)
    fig.savefig(outpath, dpi=220, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"Saved dispersion map: {outpath}")


def plot_sigma_map(sigma: np.ndarray, k_par: np.ndarray, omega: np.ndarray, title: str, outpath: Path, k_label: str, omega_label: str):
    fig, ax = _new_dark_fig((9.2, 7.2))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(PANEL_BG)  # masked (below-noise-floor) bins blend into the background
    mesh = ax.pcolormesh(k_par, omega, sigma, shading="auto", cmap=cmap, vmin=-1.0, vmax=1.0)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label(r"$\sigma_m = (P_+-P_-)/(P_++P_-)$", color=TEXT_CLR)
    cb.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(plt.getp(cb.ax, "yticklabels"), color=TEXT_CLR)
    ax.set_xlabel(k_label)
    ax.set_ylabel(omega_label)
    ax.set_title(title, fontsize=17)
    _add_mirror_caveat(fig)
    fig.savefig(outpath, dpi=220, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"Saved helicity map: {outpath}")


def plot_mode_growth(
    times_norm: np.ndarray, A: np.ndarray, mode_idx: int, k_val: float, polarization: str,
    length_unit: str, time_unit: str, outpath: Path, theory: dict | None,
):
    amplitude = np.abs(A[:, mode_idx])
    power = amplitude ** 2
    fit = _fit_growth_rate(times_norm, amplitude)

    fig, ax = _new_dark_fig((8.5, 5.5))
    valid = power > 0
    if np.count_nonzero(valid) < 2:
        ax.text(0.5, 0.5, "no measurable power in this mode", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_CLR, fontsize=13)
    else:
        ax.semilogy(times_norm[valid], power[valid], "o", color="#58a6ff", markersize=5, label="PIC $P_n(t)$")
        if np.isfinite(fit["gamma"]):
            lo, hi = fit["fit_time_range"]
            ax.axvspan(lo, hi, color=GRID_CLR, alpha=0.4, label="fit window")
            fitted_amp = np.exp(np.polyval([fit["gamma"], fit["intercept"]], times_norm))
            ax.semilogy(times_norm, fitted_amp ** 2, "--", color="#f0883e", lw=2.0,
                        label=fr"PIC fit $\gamma={fit['gamma']:.3g}$ (R$^2$={fit['rvalue']**2:.2f})")
            _, gamma_theory = interp_theory(theory, k_val) if length_unit == "d_i" else (None, float("nan"))
            if np.isfinite(gamma_theory):
                t0 = times_norm[valid][0]
                p0 = power[valid][0]
                theory_curve = p0 * np.exp(2.0 * gamma_theory * (times_norm - t0))
                ax.semilogy(times_norm, theory_curve, ":", color="#a371f7", lw=2.0,
                            label=fr"theory $\gamma={gamma_theory:.3g}$")
    ax.set_xlabel(fr"${time_unit}$")
    ax.set_ylabel(r"$P_n(t) = |\psi_" + polarization + r"(k,t)|^2$")
    ax.set_title(fr"Mode growth $\psi_{polarization}$, $k\,{length_unit}={k_val:.3g}$, {PROFILE_LABEL}", fontsize=15)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=11)
    _add_mirror_caveat(fig)
    fig.savefig(outpath, dpi=220, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"Saved mode-growth curve: {outpath}")


def plot_mirror_bparallel(
    avg_power: np.ndarray, k0: np.ndarray, k1: np.ndarray, peak_idx: tuple[int, int], outpath: Path,
):
    normalized = avg_power / max(float(np.max(avg_power)), np.finfo(float).tiny)
    log_power = np.log10(normalized + 1e-12)
    fig, ax = _new_dark_fig((8.5, 7.0))
    mesh = ax.pcolormesh(k1, k0, log_power, shading="auto", cmap="inferno", vmin=-6, vmax=0)
    ax.plot(k1[peak_idx[1]], k0[peak_idx[0]], "x", color="#3fb950", markersize=12, mew=2.5, label="peak mode")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label(r"$\log_{10}(|\delta B_z/B_0|^2 / \max)$", color=TEXT_CLR)
    cb.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(plt.getp(cb.ax, "yticklabels"), color=TEXT_CLR)
    ax.set_xlabel(r"$k_\perp\,d_i$")
    ax.set_ylabel(r"$k_\parallel\,d_i$")
    ax.set_title(f"Mirror $\\delta B_z$ oblique spectrum, {PROFILE_LABEL}", fontsize=16)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    # The mirror mode peak usually sits within a few k_min of the origin, which
    # is invisible against the full +/-Nyquist axis range shown above. Add a
    # zoomed inset around the peak so the actual oblique structure is legible
    # instead of only a fading haze around k=0.
    k_par_peak = float(k0[peak_idx[0]])
    k_perp_peak = float(k1[peak_idx[1]])
    zoom = max(8.0 * max(abs(k_par_peak), abs(k_perp_peak)), 5.0 * float(np.median(np.abs(np.diff(k0)))))
    if zoom < 0.9 * max(float(np.max(np.abs(k0))), float(np.max(np.abs(k1)))):
        axins = ax.inset_axes([0.60, 0.60, 0.38, 0.38])
        axins.set_facecolor(PANEL_BG)
        axins.pcolormesh(k1, k0, log_power, shading="auto", cmap="inferno", vmin=-6, vmax=0)
        axins.plot(k_perp_peak, k_par_peak, "x", color="#3fb950", markersize=10, mew=2.2)
        axins.set_xlim(-zoom, zoom)
        axins.set_ylim(-zoom, zoom)
        axins.tick_params(colors=TEXT_CLR, labelsize=8)
        for spine in axins.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.indicate_inset_zoom(axins, edgecolor=TEXT_CLR)

    fig.savefig(outpath, dpi=220, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"Saved mirror oblique spectrum: {outpath}")


# ── Mirror-specific diagnostic (spec section 13) ─────────────────────────

def mirror_bparallel_spectrum(
    bz_t: np.ndarray, spacing: tuple[float, float], axes: tuple[str, str], parallel_axis: str, b0: float,
) -> dict:
    dbz = remove_spatial_mean(bz_t) / b0
    nt, n0, n1 = dbz.shape
    window = np.hanning(n0)[:, None] * np.hanning(n1)[None, :]
    dbz_k = np.fft.fftshift(np.fft.fft2(dbz * window[None, :, :], axes=(1, 2)), axes=(1, 2)) / (n0 * n1)
    power_kt = np.abs(dbz_k) ** 2

    k0 = np.fft.fftshift(np.fft.fftfreq(n0, d=spacing[0])) * 2.0 * np.pi
    k1 = np.fft.fftshift(np.fft.fftfreq(n1, d=spacing[1])) * 2.0 * np.pi

    avg_power = power_kt[nt // 2:].mean(axis=0)
    avg_power_excl_dc = avg_power.copy()
    avg_power_excl_dc[n0 // 2, n1 // 2] = 0.0
    peak_idx = np.unravel_index(int(np.argmax(avg_power_excl_dc)), avg_power_excl_dc.shape)

    k_par_axis, k_perp_axis = (k0, k1) if axes[0] == parallel_axis else (k1, k0)
    k_par_peak = float(k_par_axis[peak_idx[0] if axes[0] == parallel_axis else peak_idx[1]])
    k_perp_peak = float(k_perp_axis[peak_idx[1] if axes[0] == parallel_axis else peak_idx[0]])
    theta_kb = float(np.degrees(np.arctan2(abs(k_perp_peak), abs(k_par_peak) + EPS)))

    amplitude_t = dbz_k[:, peak_idx[0], peak_idx[1]]
    return {
        "power_kt": power_kt, "avg_power": avg_power, "k0": k0, "k1": k1,
        "peak_idx": peak_idx, "k_par_peak": k_par_peak, "k_perp_peak": k_perp_peak,
        "theta_kb_deg": theta_kb, "amplitude_t": amplitude_t,
    }


# ── CLI ────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved growth-rate table: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="omega-k_parallel dispersion maps (P+, P-, sigma_m) and mode-growth "
                    "curves from the circular polarization psi_pm = (dBx +- i dBy)/B0, "
                    "plus the mirror delta_Bz(k_perp,k_parallel) oblique spectrum."
    )
    parser.add_argument("--fields", default="pfd.*.h5")
    parser.add_argument("--plane", choices=["auto", "xy", "xz", "yz"], default="auto")
    parser.add_argument("--parallel-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--dx", type=float, default=DX_DI)
    parser.add_argument("--dy", type=float, default=DX_DI)
    parser.add_argument("--dz", type=float, default=DX_DI)
    parser.add_argument("--b0", type=float, default=B0)
    parser.add_argument("--normalization", choices=["auto", "ion", "electron"], default="auto",
                        help="ion: k*d_i, omega/Omega_ci. electron: k*d_e, omega/|Omega_ce| "
                            "(auto picks electron for whistler, ion otherwise).")
    parser.add_argument("--t-start", type=float, default=None, help="Omega_ci*t lower bound for the omega-k window.")
    parser.add_argument("--t-end", type=float, default=None, help="Omega_ci*t upper bound for the omega-k window.")
    parser.add_argument("--temporal-fft-size", type=int, default=None)
    parser.add_argument("--detrend", action="store_true")
    parser.add_argument("--k-target-di", type=float, default=None,
                        help="Theoretical k_max*d_i to center the per-mode growth-curve selection on.")
    parser.add_argument("--theory-csv", default=None,
                        help="CSV with columns kdi,omega_r_over_Omegai,gamma_over_Omegai.")
    parser.add_argument("--mirror", action="store_true", default=None,
                        help="Also produce the mirror delta_Bz oblique spectrum (default: on for mirror cases).")
    parser.add_argument("--outdir", default="spectral_plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    normalization = args.normalization
    if normalization == "auto":
        normalization = "electron" if INSTABILITY == "whistler" else "ion"
    length_unit = "d_e" if normalization == "electron" else "d_i"
    time_unit = r"|\Omega_{ce}|\,t" if normalization == "electron" else r"\Omega_{ci}\,t"
    k_label = r"$k_\parallel\,d_e$" if normalization == "electron" else r"$k_\parallel\,d_i$"
    omega_label = r"$\omega/|\Omega_{ce}|$" if normalization == "electron" else r"$\omega/\Omega_{ci}$"

    convention = validate_polarization_convention()
    print("Polarization convention check (synthetic Bx=cos(kz-wt), By=+-sin(kz-wt)):")
    for label, entry in convention.items():
        print(f"  {label}: {entry}")

    if INSTABILITY == "mirror":
        print(
            "[NOTE] instability='mirror': the psi_pm dispersion/mode-growth plots below "
            "only cover the k_perp=0 slice, not the oblique mirror mode itself (see "
            "mirror_bparallel_spectrum for that). psi_plus/psi_minus commonly come out "
            "nearly identical here because a k_perp=0 fluctuation isn't preferentially "
            "circularly polarized -- that is expected, not a duplicate/bug."
        )

    series = load_bxyz_series(args.fields, args.plane, args.dx, args.dy, args.dz)
    times_step = series["steps"]
    times_omegaci = np.array([step_to_omegaci(step) for step in times_step], dtype=float)
    times_norm = times_omegaci * MASS_RATIO if normalization == "electron" else times_omegaci

    if args.t_start is not None or args.t_end is not None:
        lo = args.t_start if args.t_start is not None else times_omegaci.min()
        hi = args.t_end if args.t_end is not None else times_omegaci.max()
        mask = (times_omegaci >= lo) & (times_omegaci <= hi)
        if np.count_nonzero(mask) < 4:
            raise ValueError("Requested [--t-start, --t-end] window contains fewer than four snapshots")
    else:
        mask = np.ones_like(times_omegaci, dtype=bool)

    bx_t, by_t, bz_t = series["bx"][mask], series["by"][mask], series["bz"][mask]
    times_norm_win = times_norm[mask]
    spacing_di = series["spacing"]
    spacing = tuple(s * DI for s in spacing_di) if normalization == "electron" else spacing_di

    psi_plus, psi_minus = build_psi(bx_t, by_t, args.b0)
    A_plus, k_par = spatial_fft_kpar0(psi_plus, spacing, series["axes"], args.parallel_axis)
    A_minus, _ = spatial_fft_kpar0(psi_minus, spacing, series["axes"], args.parallel_axis)

    power_plus, omega = temporal_dispersion(A_plus, times_norm_win, nfft=args.temporal_fft_size, detrend=args.detrend)
    power_minus, _ = temporal_dispersion(A_minus, times_norm_win, nfft=args.temporal_fft_size, detrend=args.detrend)
    sigma = sigma_m(power_plus, power_minus)

    theory = load_theory(args.theory_csv) if args.theory_csv else None
    if theory is not None and length_unit != "d_i":
        print("[WARN] --theory-csv is given in k*d_i; skipping theory overlay for electron normalization.")
        theory = None

    plot_dispersion_map(
        power_plus, k_par, omega, fr"$P_+(k_\parallel,\omega)$ — {PROFILE_LABEL}",
        outdir / f"polarization_dispersion_plus_{series['plane']}.png", k_label, omega_label, theory,
    )
    plot_dispersion_map(
        power_minus, k_par, omega, fr"$P_-(k_\parallel,\omega)$ — {PROFILE_LABEL}",
        outdir / f"polarization_dispersion_minus_{series['plane']}.png", k_label, omega_label, theory,
    )
    plot_sigma_map(
        sigma, k_par, omega, fr"Reduced helicity $\sigma_m(k_\parallel,\omega)$ — {PROFILE_LABEL}",
        outdir / f"polarization_sigma_m_{series['plane']}.png", k_label, omega_label,
    )

    k_target = args.k_target_di
    if k_target is not None and normalization == "electron":
        k_target = k_target * DI
    final_power = np.abs(A_plus[-1]) ** 2 + np.abs(A_minus[-1]) ** 2
    modes = select_modes(k_par, final_power, k_target)

    distribution = "Bi-Maxwellian" if KAPPA is None else f"Bi-Kappa (kappa={KAPPA})"
    growth_rows = []
    for mode_idx in modes:
        k_val = float(k_par[mode_idx])
        for polarization, A in (("plus", A_plus), ("minus", A_minus)):
            growth_rows += growth_rate_rows(
                SIM_PROFILE, distribution, INSTABILITY, polarization,
                times_norm_win, A, k_par, [mode_idx], length_unit, theory,
            )
            plot_mode_growth(
                times_norm_win, A, mode_idx, k_val, "+" if polarization == "plus" else "-",
                length_unit, time_unit,
                outdir / f"mode_growth_{polarization}_n{mode_idx}_{series['plane']}.png", theory,
            )
    _write_csv(outdir / "polarization_growth_rates.csv", growth_rows)

    run_mirror = args.mirror if args.mirror is not None else (INSTABILITY == "mirror")
    mirror_result = None
    if run_mirror:
        mirror_result = mirror_bparallel_spectrum(bz_t, spacing_di, series["axes"], args.parallel_axis, args.b0)
        plot_mirror_bparallel(
            mirror_result["avg_power"], mirror_result["k0"], mirror_result["k1"], mirror_result["peak_idx"],
            outdir / f"mirror_bparallel_spectrum_{series['plane']}.png",
        )
        mirror_amp = np.abs(mirror_result["amplitude_t"])
        mirror_fit = _fit_growth_rate(times_norm_win, mirror_amp)
        mirror_error = _slope_standard_error(times_norm_win, mirror_amp, mirror_fit)
        print(
            f"Mirror oblique peak: k_perp d_i = {mirror_result['k_perp_peak']:.3g}, "
            f"k_par d_i = {mirror_result['k_par_peak']:.3g}, theta_kB = {mirror_result['theta_kb_deg']:.1f} deg, "
            f"gamma_pic/Omega_ci = {mirror_fit['gamma']:.3g} +/- {mirror_error:.2g}"
        )
        _write_csv(outdir / "mirror_growth_rate.csv", [{
            "case": SIM_PROFILE, "distribution": distribution, "instability": INSTABILITY,
            "kdi_perp": mirror_result["k_perp_peak"], "kdi_parallel": mirror_result["k_par_peak"],
            "theta_kB_deg": mirror_result["theta_kb_deg"],
            "fit_t_start": mirror_fit["fit_time_range"][0] if mirror_fit["fit_time_range"] else float("nan"),
            "fit_t_end": mirror_fit["fit_time_range"][1] if mirror_fit["fit_time_range"] else float("nan"),
            "gamma_pic": mirror_fit["gamma"], "gamma_pic_error": mirror_error,
            "R2": mirror_fit["rvalue"] ** 2 if np.isfinite(mirror_fit["rvalue"]) else float("nan"),
        }])

    params = {
        "case": SIM_PROFILE, "instability": INSTABILITY, "distribution": distribution,
        "plane": series["plane"], "parallel_axis": args.parallel_axis,
        "normalization": normalization, "b0": args.b0,
        "n_snapshots_used": int(len(times_norm_win)),
        "t_omegaci_range": [float(times_omegaci[mask].min()), float(times_omegaci[mask].max())],
        "selected_modes_kdi": [float(k_par[m] / (DI if normalization == "electron" else 1.0)) for m in modes],
        "polarization_convention_check": convention,
    }
    with (outdir / "polarization_fft_parameters.json").open("w") as handle:
        json.dump(params, handle, indent=2)
    print(f"Saved run parameters: {outdir / 'polarization_fft_parameters.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
