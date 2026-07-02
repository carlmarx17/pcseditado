#!/usr/bin/env python3
"""
physical_diagnostics.py
=======================
Integrated PSC diagnostics for temperature-anisotropy instabilities.

This module complements the specialized plotting scripts with standard CSV
tables and publication-ready figures requested for mirror, firehose and kappa
comparisons. It is deliberately tolerant of partial data directories: particle,
field and moment diagnostics are enabled only when the corresponding files are
present.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import os
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm

try:
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import curve_fit
except ImportError:  # pragma: no cover - requirements include scipy
    gaussian_filter = None
    curve_fit = None

from data_reader import PICDataReader
from spectral_analysis import SpectralAnalyzer
from psc_units import (
    B0,
    BETA_I_PAR,
    BETA_I_PERP_OVER_PAR,
    DI,
    DOMAIN_DI_Y,
    DOMAIN_DI_Z,
    DRIVEN_SPECIES,
    INSTABILITY,
    KAPPA,
    MASS_RATIO,
    M_ELEC,
    M_ION,
    PROFILE_LABEL,
    TE_PAR,
    TE_PERP,
    TI_PAR,
    TI_PERP,
    VA,
    step_to_omegaci,
)


DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
TEXT_CLR = "#e6edf3"
GRID_CLR = "#30363d"
RNG = np.random.default_rng(20260623)
POSTER_FONT = 15
POSTER_LABEL = 18
POSTER_TITLE = 19
POSTER_TICK = 15
POSTER_LEGEND = 14

plt.rcParams.update({
    "font.size": POSTER_FONT,
    "axes.labelsize": POSTER_LABEL,
    "axes.titlesize": POSTER_TITLE,
    "xtick.labelsize": POSTER_TICK,
    "ytick.labelsize": POSTER_TICK,
    "legend.fontsize": POSTER_LEGEND,
    "figure.titlesize": POSTER_TITLE + 1,
})


@dataclass
class ParticleSnapshot:
    step: int
    time: float
    q: np.ndarray
    m: np.ndarray
    px: np.ndarray
    py: np.ndarray
    pz: np.ndarray
    w: np.ndarray


def _safe_div(num, den, fill=np.nan):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), fill, dtype=float)
    return np.divide(num, den, out=out, where=np.abs(den) > 1e-30)


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _style_axes(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(
        colors=TEXT_CLR,
        direction="in",
        which="both",
        top=True,
        right=True,
        labelsize=POSTER_TICK,
    )
    ax.grid(True, color=GRID_CLR, alpha=0.22, linestyle=":")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)


def _parallel_workers(requested: int, task_count: int) -> int:
    if task_count <= 1:
        return 1
    available = os.cpu_count() or 1
    workers = requested if requested > 0 else available
    return max(1, min(workers, task_count))


def _run_step_tasks(worker, tasks: list[tuple], jobs: int, label: str) -> list:
    if not tasks:
        return []
    workers = _parallel_workers(jobs, len(tasks))
    if workers == 1:
        return [worker(task) for task in tasks]

    print(f"{label}: {len(tasks)} steps with {workers} processes")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, task): task[0] for task in tasks}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            step = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"\n[ERROR] {label} step {step} failed: {exc}")
            print(f"  {label} step {step} ({index}/{len(tasks)})", end="\r")
    print()
    return sorted(results, key=lambda item: item[0])


def _savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _savefig_many(fig, paths: Iterable[Path]):
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _extract_step(path: str | Path) -> int:
    step = PICDataReader.get_step_from_filename(str(path))
    if step is None:
        raise ValueError(f"Could not extract step from {path}")
    return step


def _select_steps(steps: list[int], requested: list[int] | None, max_count: int) -> list[int]:
    if requested:
        requested_set = set(requested)
        return [s for s in steps if s in requested_set]
    if len(steps) <= max_count:
        return steps
    idx = np.unique(np.linspace(0, len(steps) - 1, max_count, dtype=int))
    return [steps[i] for i in idx]


def _read_particle_snapshot(path: str, max_particles: int) -> ParticleSnapshot:
    step = _extract_step(path)
    q, m, px, py, pz, w = PICDataReader.read_particles_snapshot(
        path, max_particles=max_particles, rng=RNG
    )
    return ParticleSnapshot(step, step_to_omegaci(step), q, m, px, py, pz, w)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    if not np.isfinite(mean):
        return float("nan")
    return _weighted_mean((np.asarray(values, dtype=float) - mean) ** 2, weights)


def _species_mask(snapshot: ParticleSnapshot, species: str) -> np.ndarray:
    return snapshot.q > 0 if species == "ion" else snapshot.q < 0


def particle_temperatures(snapshot: ParticleSnapshot, species: str = "ion") -> dict:
    mask = _species_mask(snapshot, species)
    if not np.any(mask):
        return {}
    mass = abs(_weighted_mean(snapshot.m[mask], snapshot.w[mask]))
    vx = snapshot.px[mask]
    vy = snapshot.py[mask]
    vz = snapshot.pz[mask]
    weights = snapshot.w[mask]
    tpar = mass * _weighted_var(vz, weights)
    tperp = 0.5 * mass * (_weighted_var(vx, weights) + _weighted_var(vy, weights))
    return {
        "T_parallel": float(tpar),
        "T_perp": float(tperp),
        "A": float(tperp / max(tpar, 1e-30)),
        "R": float(tpar / max(tperp, 1e-30)),
        "mass": float(mass),
        "count": int(np.count_nonzero(mask)),
    }


def load_fields(path: str) -> dict[str, np.ndarray]:
    data = PICDataReader.read_multiple_fields_3d(
        path, "jeh-", ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"]
    )
    return {
        "Bx": np.asarray(data["hx_fc/p0/3d"], dtype=float),
        "By": np.asarray(data["hy_fc/p0/3d"], dtype=float),
        "Bz": np.asarray(data["hz_fc/p0/3d"], dtype=float),
    }


def load_moments(path: str, suffix: str = "i") -> dict[str, np.ndarray]:
    names = [
        f"rho_{suffix}/p0/3d",
        f"txx_{suffix}/p0/3d",
        f"tyy_{suffix}/p0/3d",
        f"tzz_{suffix}/p0/3d",
        f"px_{suffix}/p0/3d",
        f"py_{suffix}/p0/3d",
        f"pz_{suffix}/p0/3d",
    ]
    optional = [f"jx_{suffix}/p0/3d", f"jy_{suffix}/p0/3d", f"jz_{suffix}/p0/3d"]
    data = PICDataReader.read_multiple_fields_3d(path, "all_1st", names)
    result = {key.split("/")[0]: PICDataReader.flatten_2d_slice(value).astype(float)
              for key, value in data.items()}
    try:
        currents = PICDataReader.read_multiple_fields_3d(path, "all_1st", optional)
        result.update({key.split("/")[0]: PICDataReader.flatten_2d_slice(value).astype(float)
                       for key, value in currents.items()})
    except Exception:
        pass
    return result


def moment_thermal_maps(moment_file: str, field_file: str | None = None,
                        species: str = "ion") -> dict[str, np.ndarray]:
    suffix = "i" if species == "ion" else "e"
    mass = M_ION if species == "ion" else M_ELEC
    mom = load_moments(moment_file, suffix)
    rho = mom[f"rho_{suffix}"]
    n = rho if species == "ion" else np.abs(rho)
    safe_n = np.where(n > 1e-12, n, np.nan)

    pxx = mom[f"txx_{suffix}"] - mom[f"px_{suffix}"] ** 2 / (safe_n * mass)
    pyy = mom[f"tyy_{suffix}"] - mom[f"py_{suffix}"] ** 2 / (safe_n * mass)
    pzz = mom[f"tzz_{suffix}"] - mom[f"pz_{suffix}"] ** 2 / (safe_n * mass)

    if field_file:
        fld = load_fields(field_file)
        bx = PICDataReader.flatten_2d_slice(fld["Bx"])
        by = PICDataReader.flatten_2d_slice(fld["By"])
        bz = PICDataReader.flatten_2d_slice(fld["Bz"])
        bmag = np.sqrt(bx**2 + by**2 + bz**2)
    else:
        bmag = np.full_like(pzz, B0)

    tpar = pzz / safe_n
    tperp = 0.5 * (pxx + pyy) / safe_n
    anisotropy = tperp / (tpar + 1e-30)
    beta_par = 2.0 * pzz / (bmag**2 + 1e-30)
    return {
        "n": n,
        "Pxx": pxx,
        "Pyy": pyy,
        "Pzz": pzz,
        "T_parallel": tpar,
        "T_perp": tperp,
        "A": anisotropy,
        "R": 1.0 / np.maximum(anisotropy, 1e-30),
        "beta_parallel": beta_par,
        "B_magnitude": bmag,
    }


def field_metrics(field_file: str, b0: float = B0) -> dict:
    fld = load_fields(field_file)
    bx, by, bz = fld["Bx"], fld["By"], fld["Bz"]
    bmag = np.sqrt(bx**2 + by**2 + bz**2)
    delta_b = bmag - b0
    dbx = bx - np.nanmean(bx)
    dby = by - np.nanmean(by)
    dbz = bz - b0
    sigma_b = np.nanstd(bmag)
    b0_abs = max(abs(b0), 1e-30)
    return {
        "B_magnitude": bmag,
        "delta_B": delta_b,
        "delta_B_over_B0": delta_b / b0_abs,
        "delta_B_rms": float(np.sqrt(np.nanmean(delta_b**2))),
        "delta_B_rms_over_B0": float(np.sqrt(np.nanmean(delta_b**2)) / b0_abs),
        "delta_B_parallel_rms": float(np.sqrt(np.nanmean(dbz**2))),
        "delta_B_parallel_rms_over_B0": float(np.sqrt(np.nanmean(dbz**2)) / b0_abs),
        "delta_B_perp_rms": float(np.sqrt(np.nanmean(dbx**2 + dby**2))),
        "delta_B_perp_rms_over_B0": float(np.sqrt(np.nanmean(dbx**2 + dby**2)) / b0_abs),
        "B_min": float(np.nanmin(bmag)),
        "mirror_depth": float(1.0 - np.nanmin(bmag) / b0_abs),
        "mirror_area_fraction": float(np.nanmean(bmag < (b0 - sigma_b))),
        "magnetic_energy_fluct": float(0.5 * np.nanmean(delta_b**2)),
    }


def _spectral_plane(shape: tuple[int, ...]) -> str:
    """Select the physical 2D plane from the singleton PSC dimension."""
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D PSC field, got shape {shape}")
    if shape[2] == 1:
        return "yz"
    if shape[1] == 1:
        return "xz"
    if shape[0] == 1:
        return "xy"
    # Full 3D output: analyze the central yz plane, matching B0 || z.
    return "yz"


def magnetic_perpendicular_spectrum(field_file: str) -> dict:
    """Compute the transverse magnetic spectrum PSD(Bx) + PSD(By)."""
    fields = load_fields(field_file)
    bx = np.asarray(fields["Bx"], dtype=float)
    by = np.asarray(fields["By"], dtype=float)
    bz = np.asarray(fields["Bz"], dtype=float)
    plane = _spectral_plane(bx.shape)

    axis_lengths = {"x": 1.0, "y": DOMAIN_DI_Y, "z": DOMAIN_DI_Z}
    probe = SpectralAnalyzer(outdir="/tmp/psc_spectral_probe", parallel_axis="z")
    plane_data = probe._get_plane_slice(bx, by, bz, plane)
    axes = plane_data["axes"]
    plane_shape = np.atleast_2d(plane_data["bx"]).shape
    spacing = (
        axis_lengths[axes[0]] / max(plane_shape[0], 1),
        axis_lengths[axes[1]] / max(plane_shape[1], 1),
    )
    analyzer = SpectralAnalyzer(
        dx=spacing[1],
        dy=spacing[0],
        outdir="/tmp/psc_spectral_probe",
        parallel_axis="z",
    )
    plane_data = analyzer._get_plane_slice(bx, by, bz, plane)
    bx_2d = np.atleast_2d(plane_data["bx"])
    by_2d = np.atleast_2d(plane_data["by"])
    dbx = bx_2d - np.nanmean(bx_2d)
    dby = by_2d - np.nanmean(by_2d)
    psd_2d = analyzer._compute_fft_psd(dbx) + analyzer._compute_fft_psd(dby)
    k_grids = analyzer._compute_k_grids(
        psd_2d.shape, plane_data["spacing"], plane_data["axes"]
    )
    k, power = analyzer._radial_spectrum(psd_2d, k_grids["k_mag"])
    fit = analyzer._fit_power_law(k, power)
    if power.size:
        peak_idx = int(np.nanargmax(power))
        peak_k = float(k[peak_idx])
        peak_power = float(power[peak_idx])
    else:
        peak_k = np.nan
        peak_power = np.nan
    return {
        "plane": plane,
        "axes": axes,
        "spacing": spacing,
        "k": k,
        "power": power,
        "fit": fit,
        "peak_k": peak_k,
        "peak_power": peak_power,
    }


def plot_magnetic_spectrum(spectrum: dict, step: int, outdir: Path):
    k = np.asarray(spectrum["k"], dtype=float)
    power = np.asarray(spectrum["power"], dtype=float)
    if k.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.loglog(k, power, color="#58a6ff", lw=2.0, label=r"$E_{B_\perp}(k)$")
    fit = spectrum["fit"]
    if fit is not None and len(fit["k_fit"]) >= 3:
        fit_power = 10 ** (
            fit["intercept"] + fit["slope"] * np.log10(fit["k_fit"])
        )
        ax.loglog(
            fit["k_fit"],
            fit_power,
            "--",
            color="#ff7b72",
            lw=1.8,
            label=rf"fit: $k^{{{fit['slope']:.2f}}}$",
        )
    if len(k) > 8:
        ref = slice(len(k) // 4, 3 * len(k) // 4)
        k_ref = k[ref]
        p_ref = power[ref]
        valid = (k_ref > 0) & (p_ref > 0)
        if np.any(valid):
            first = np.flatnonzero(valid)[0]
            kolmogorov = p_ref[first] * (k_ref / k_ref[first]) ** (-5.0 / 3.0)
            ax.loglog(k_ref, kolmogorov, ":", color="#f2cc60", label=r"$k^{-5/3}$")
    ax.set_xlabel(r"$k\,[d_i^{-1}]$", color=TEXT_CLR)
    ax.set_ylabel(r"$E_{B_\perp}(k)$", color=TEXT_CLR)
    ax.set_title(
        f"Transverse magnetic spectrum - step {step} ({spectrum['plane']})",
        color=TEXT_CLR,
        fontweight="bold",
    )
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _savefig(fig, outdir / f"magnetic_spectrum_step_{step}.png")


def maxwellian_pdf(x: np.ndarray, amp: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    return amp * np.exp(-0.5 * (x / sigma) ** 2)


def kappa_pdf_shape(x: np.ndarray, amp: float, sigma: float, kappa: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    kappa = max(float(kappa), 1.501)
    return amp * (1.0 + x**2 / ((2.0 * kappa - 3.0) * sigma**2)) ** (-kappa)


def _least_squares_amplitude(y: np.ndarray, shape: np.ndarray) -> float:
    denom = float(np.sum(shape * shape))
    if denom <= 0:
        return 0.0
    return max(float(np.sum(y * shape) / denom), 0.0)


def _grid_fit_distribution(x: np.ndarray, y: np.ndarray, sigma0: float) -> tuple[tuple, tuple]:
    """Fallback fit that needs only NumPy."""
    sigma_grid = sigma0 * np.geomspace(0.35, 2.8, 50)
    best_m = (np.inf, (float(np.nanmax(y)), sigma0))
    best_k = (np.inf, (float(np.nanmax(y)), sigma0, 80.0))

    def log_error(model):
        return float(np.sqrt(np.nanmean((np.log10(y) - np.log10(model + 1e-300)) ** 2)))

    for sigma in sigma_grid:
        shape_m = np.exp(-0.5 * (x / sigma) ** 2)
        amp_m = _least_squares_amplitude(y, shape_m)
        err_m = log_error(amp_m * shape_m)
        if err_m < best_m[0]:
            best_m = (err_m, (amp_m, float(sigma)))

    kappa_grid = np.concatenate([
        np.linspace(1.6, 8.0, 55),
        np.linspace(8.5, 80.0, 40),
    ])
    for sigma in sigma_grid:
        for kappa in kappa_grid:
            shape_k = (1.0 + x**2 / ((2.0 * kappa - 3.0) * sigma**2)) ** (-kappa)
            amp_k = _least_squares_amplitude(y, shape_k)
            err_k = log_error(amp_k * shape_k)
            if err_k < best_k[0]:
                best_k = (err_k, (amp_k, float(sigma), float(kappa)))
    return best_m[1], best_k[1]


def fit_distribution(snapshot: ParticleSnapshot, species: str = "ion") -> dict:
    mask = _species_mask(snapshot, species)
    if not np.any(mask):
        return {}

    vx = snapshot.px[mask]
    vy = snapshot.py[mask]
    vz = snapshot.pz[mask]
    weights = snapshot.w[mask]
    centered = vz - _weighted_mean(vz, weights)
    sigma0 = math.sqrt(max(_weighted_var(centered, weights), 1e-30))
    v_abs = np.abs(centered)
    vmax = np.nanpercentile(v_abs, 99.7)
    if not np.isfinite(vmax) or vmax <= 0:
        return {}

    hist, edges = np.histogram(centered, bins=160, range=(-vmax, vmax),
                               weights=weights, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = np.isfinite(hist) & (hist > 0)
    if np.count_nonzero(valid) < 12:
        return {}

    x = centers[valid]
    y = hist[valid]
    amp0 = float(np.nanmax(y))
    if curve_fit is None:
        popt_m, popt_k = _grid_fit_distribution(x, y, sigma0)
    else:
        try:
            popt_m, _ = curve_fit(
                maxwellian_pdf, x, y, p0=(amp0, sigma0),
                bounds=([0.0, sigma0 * 0.05], [np.inf, sigma0 * 20.0]),
                maxfev=20000,
            )
            popt_k, _ = curve_fit(
                kappa_pdf_shape, x, y, p0=(amp0, sigma0, KAPPA or 5.0),
                bounds=([0.0, sigma0 * 0.05, 1.51], [np.inf, sigma0 * 20.0, 80.0]),
                maxfev=30000,
            )
        except Exception:
            popt_m, popt_k = _grid_fit_distribution(x, y, sigma0)

    y_m = maxwellian_pdf(x, *popt_m)
    y_k = kappa_pdf_shape(x, *popt_k)
    tail = np.abs(x) > 3.0 * sigma0
    err_m = float(np.sqrt(np.nanmean((np.log10(y) - np.log10(y_m + 1e-300)) ** 2)))
    err_k = float(np.sqrt(np.nanmean((np.log10(y) - np.log10(y_k + 1e-300)) ** 2)))
    err_m_tail = float(np.sqrt(np.nanmean((np.log10(y[tail]) - np.log10(y_m[tail] + 1e-300)) ** 2))) if np.any(tail) else np.nan
    err_k_tail = float(np.sqrt(np.nanmean((np.log10(y[tail]) - np.log10(y_k[tail] + 1e-300)) ** 2))) if np.any(tail) else np.nan
    v3 = np.sqrt(vx**2 + vy**2 + vz**2)
    vth = float(np.sqrt(_weighted_var(vx, weights) + _weighted_var(vy, weights) + _weighted_var(vz, weights)))
    supra = float(np.sum(weights[v3 > 3.0 * vth]) / max(np.sum(weights), 1e-30))

    return {
        "step": snapshot.step,
        "omega_ci_t": snapshot.time,
        "kappa_fit": float(popt_k[2]),
        "maxwellian_sigma": float(popt_m[1]),
        "kappa_sigma": float(popt_k[1]),
        "error_maxwellian": err_m,
        "error_kappa": err_k,
        "error_tail_maxwellian": err_m_tail,
        "error_tail_kappa": err_k_tail,
        "suprathermal_fraction": supra,
        "hist_x": centers,
        "hist_y": hist,
        "fit_x": np.linspace(-vmax, vmax, 700),
        "maxwellian_params": tuple(float(v) for v in popt_m),
        "kappa_params": tuple(float(v) for v in popt_k),
    }


def particle_heat_flux(snapshot: ParticleSnapshot, species: str = "ion") -> dict:
    mask = _species_mask(snapshot, species)
    if not np.any(mask):
        return {}
    mass = abs(_weighted_mean(snapshot.m[mask], snapshot.w[mask]))
    weights = snapshot.w[mask]
    vx = snapshot.px[mask]
    vy = snapshot.py[mask]
    vz = snapshot.pz[mask]
    dvx = vx - _weighted_mean(vx, weights)
    dvy = vy - _weighted_mean(vy, weights)
    dvz = vz - _weighted_mean(vz, weights)
    dv2 = dvx**2 + dvy**2 + dvz**2
    dvperp = np.sqrt(dvx**2 + dvy**2)
    return {
        "q_parallel_particle": 0.5 * mass * _weighted_mean(dv2 * dvz, weights),
        "q_perp_particle": 0.5 * mass * _weighted_mean(dv2 * dvperp, weights),
    }


def particle_energy(snapshot: ParticleSnapshot, species: str = "ion") -> dict:
    mask = _species_mask(snapshot, species)
    if not np.any(mask):
        return {}
    mass = abs(_weighted_mean(snapshot.m[mask], snapshot.w[mask]))
    weights = snapshot.w[mask]
    vx = snapshot.px[mask]
    vy = snapshot.py[mask]
    vz = snapshot.pz[mask]
    ux = _weighted_mean(vx, weights)
    uy = _weighted_mean(vy, weights)
    uz = _weighted_mean(vz, weights)
    bulk = 0.5 * mass * (ux**2 + uy**2 + uz**2)
    thermal = 0.5 * mass * _weighted_mean((vx - ux) ** 2 + (vy - uy) ** 2 + (vz - uz) ** 2, weights)
    return {"E_kin_bulk": bulk, "E_kin_thermal": thermal}


def plot_validation(rows: list[dict], outdir: Path):
    if not rows:
        return
    first = rows[0]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    labels = [r"$T_{\parallel i}$", r"$T_{\perp i}$", r"$A_i$", r"$R_i$", r"$\beta_{\parallel i}$"]
    values = [
        first["T_parallel_i"],
        first["T_perp_i"],
        first["A_i"],
        first["R_i"],
        first["beta_parallel_i"],
    ]
    colors = ["#58a6ff", "#ff7b72", "#f2cc60", "#d2a8ff", "#56d364"]
    ax.bar(labels, values, color=colors, alpha=0.9)
    ax.axhline(1.0, color=TEXT_CLR, linestyle=":", alpha=0.45)
    ax.set_title(f"Initial validation - {PROFILE_LABEL}", color=TEXT_CLR, fontsize=15, fontweight="bold")
    ax.set_ylabel("code units / dimensionless", color=TEXT_CLR)
    _savefig(fig, outdir / "A_i_initial_check.png")


def plot_time_series(rows: list[dict], outdir: Path):
    if not rows:
        return
    t = np.array([r["omega_ci_t"] for r in rows], dtype=float)
    a = np.array([r["A_i"] for r in rows], dtype=float)
    r_inv = np.array([r["R_i"] for r in rows], dtype=float)
    tpar = np.array([r["T_parallel_i"] for r in rows], dtype=float)
    tperp = np.array([r["T_perp_i"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(t, a, "o-", color="#ff7b72", label=r"$A_i=T_\perp/T_\parallel$")
    ax.plot(t, r_inv, "s-", color="#f2cc60", label=r"$R_i=T_\parallel/T_\perp$")
    ax.axhline(1.0, color=TEXT_CLR, alpha=0.35, linestyle=":")
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel("anisotropy ratio", color=TEXT_CLR)
    ax.set_title("Ion anisotropy evolution", color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _savefig(fig, outdir / "anisotropy_vs_time.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(t, tpar, "o-", color="#58a6ff", label=r"$T_{\parallel i}$")
    ax.plot(t, tperp, "o-", color="#ff7b72", label=r"$T_{\perp i}$")
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel("temperature [code]", color=TEXT_CLR)
    ax.set_title("Parallel and perpendicular ion temperature", color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _savefig(fig, outdir / "temperature_parallel_perp_vs_time.png")


def plot_vdf2d(snapshot: ParticleSnapshot, outdir: Path, species: str = "ion") -> Path | None:
    mask = _species_mask(snapshot, species)
    if not np.any(mask):
        return None
    vx, vy, vz = snapshot.px[mask], snapshot.py[mask], snapshot.pz[mask]
    weights = snapshot.w[mask]
    vpar = (vz - _weighted_mean(vz, weights)) / VA
    vx_centered = (vx - _weighted_mean(vx, weights)) / VA
    vy_centered = (vy - _weighted_mean(vy, weights)) / VA
    vperp = np.sqrt(vx_centered**2 + vy_centered**2)
    par_abs = np.nanpercentile(np.abs(vpar), 99.7)
    perp_hi = np.nanpercentile(vperp, 99.7)
    if par_abs <= 0 or perp_hi <= 0:
        return None
    hist, xedges, yedges = np.histogram2d(
        vpar, vperp, bins=(220, 150), weights=weights,
        range=((-par_abs, par_abs), (0.0, perp_hi)), density=True,
    )
    if gaussian_filter is not None:
        hist = gaussian_filter(hist.astype(float), sigma=0.8)

    finite_hist = hist[np.isfinite(hist) & (hist > 0)]
    if finite_hist.size == 0:
        return None
    vmin = max(np.nanpercentile(finite_hist, 1.0), np.nanmax(finite_hist) * 1e-5)
    vmax = np.nanmax(finite_hist)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=13)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    pcm = ax.pcolormesh(
        xedges, yedges, hist.T, cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        shading="auto",
    )
    levels = vmax * np.array([1e-4, 1e-3, 1e-2, 1e-1])
    levels = levels[(levels > vmin) & (levels < vmax)]
    if levels.size:
        ax.contour(
            0.5 * (xedges[:-1] + xedges[1:]),
            0.5 * (yedges[:-1] + yedges[1:]),
            hist.T,
            levels=levels,
            colors="white",
            linewidths=0.65,
            alpha=0.75,
        )
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$f(v_\parallel,v_\perp)$ [PDF]", fontsize=13)
    cb.ax.tick_params(which="both", direction="in", labelsize=12)
    ax.axvline(0.0, color="white", lw=0.8, ls=":", alpha=0.8)
    ax.set_xlabel(r"$(v_\parallel-\langle v_\parallel\rangle)/v_A$", fontsize=14)
    ax.set_ylabel(r"$v_\perp/v_A$", fontsize=14)
    ax.set_title(
        rf"{species.capitalize()} VDF, step {snapshot.step}, $t\Omega_{{ci}}={snapshot.time:.2f}$",
        fontsize=14, fontweight="bold",
    )
    path = outdir / f"vdf_2d_step_{snapshot.step}.png"
    _savefig(fig, path)
    return path


def plot_fit_metrics(rows: list[dict], outdir: Path):
    if not rows:
        return
    t = np.array([r["omega_ci_t"] for r in rows], dtype=float)
    kfit = np.array([r["kappa_fit"] for r in rows], dtype=float)
    supra = np.array([r["suprathermal_fraction"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(t, kfit, "o-", color="#d2a8ff")
    if KAPPA:
        ax.axhline(KAPPA, color="#f2cc60", alpha=0.6, linestyle="--", label=rf"$\kappa_0={KAPPA:g}$")
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel(r"$\kappa_{\rm fit}$", color=TEXT_CLR)
    ax.set_title("Kappa fit vs time", color=TEXT_CLR, fontweight="bold")
    _savefig(fig, outdir / "kappa_fit_vs_time.png")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(t, supra, "o-", color="#ff7b72")
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel(r"$F_{\rm supra}(|v|>3v_{th})$", color=TEXT_CLR)
    ax.set_title("Suprathermal fraction vs time", color=TEXT_CLR, fontweight="bold")
    _savefig(fig, outdir / "suprathermal_fraction_vs_time.png")


def plot_distribution_fit(fit: dict, outdir: Path):
    if not fit:
        return
    x = fit["fit_x"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.step(fit["hist_x"], fit["hist_y"], where="mid", color="#ff7b72", lw=1.4, label="simulation")
    ax.plot(x, maxwellian_pdf(x, *fit["maxwellian_params"]), "--", color="#58a6ff", lw=2.0,
            label="Maxwellian fit")
    ax.plot(x, kappa_pdf_shape(x, *fit["kappa_params"]), "-", color="#d2a8ff", lw=2.0,
            label=rf"Kappa fit, $\kappa={fit['kappa_fit']:.2f}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$v_\parallel-\langle v_\parallel\rangle$", color=TEXT_CLR)
    ax.set_ylabel("PDF", color=TEXT_CLR)
    ax.set_title(f"Kappa vs Maxwellian - step {fit['step']}", color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _savefig(fig, outdir / f"kappa_vs_maxwellian_step_{fit['step']}.png")


def plot_map(field: np.ndarray, path: Path, title: str, label: str, cmap="viridis", symmetric=False):
    arr = np.asarray(field, dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    if symmetric:
        vmax = max(float(np.nanpercentile(np.abs(arr), 99)), 1e-12)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        lo, hi = np.nanpercentile(arr[np.isfinite(arr)], [1, 99]) if np.any(np.isfinite(arr)) else (0, 1)
        norm = None if hi <= lo else plt.Normalize(lo, hi)
    im = ax.imshow(arr.T, origin="lower", extent=[0, DOMAIN_DI_Z, 0, DOMAIN_DI_Y],
                   aspect="auto", cmap=cmap, norm=norm)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(label, color=TEXT_CLR)
    cb.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_CLR)
    ax.set_xlabel(r"Z [$d_i$]", color=TEXT_CLR)
    ax.set_ylabel(r"Y [$d_i$]", color=TEXT_CLR)
    ax.set_title(title, color=TEXT_CLR, fontweight="bold")
    _savefig(fig, path)


def plot_spatial_maps(rows: list[dict], outdir: Path):
    if not rows:
        return
    t = np.array([r["omega_ci_t"] for r in rows])
    mean_a = np.array([r["A_mean"] for r in rows])
    p10 = np.array([r["A_p10"] for r in rows])
    p90 = np.array([r["A_p90"] for r in rows])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.fill_between(t, p10, p90, color="#ff7b72", alpha=0.22, label="P10-P90")
    ax.plot(t, mean_a, "o-", color="#ff7b72", label=r"$\langle A_i\rangle$")
    ax.axhline(1.0, color=TEXT_CLR, linestyle=":", alpha=0.35)
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel(r"$A_i(x,y)$", color=TEXT_CLR)
    ax.set_title("Spatial anisotropy from moments", color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _savefig(fig, outdir / "brazil_plot_time_evolution.png")


def growth_rate(time: np.ndarray, delta_b: np.ndarray) -> dict:
    valid = np.isfinite(time) & np.isfinite(delta_b) & (delta_b > 0)
    if np.count_nonzero(valid) < 4:
        return {}
    t = time[valid]
    y = np.log(delta_b[valid])
    slope_local = np.gradient(y, t)
    positive = slope_local > 0
    if np.count_nonzero(positive) >= 3:
        idx_valid = np.where(positive)[0]
        # Use central 70 percent of positive-growth samples to avoid noise/saturation.
        lo = idx_valid[max(0, int(0.15 * len(idx_valid)))]
        hi = idx_valid[min(len(idx_valid) - 1, int(0.85 * len(idx_valid)))]
        fit_slice = slice(lo, hi + 1)
    else:
        fit_slice = slice(1, -1)
    if len(t[fit_slice]) < 3:
        fit_slice = slice(None)
    coeff = np.polyfit(t[fit_slice], y[fit_slice], 1)
    return {
        "gamma": float(coeff[0]),
        "intercept": float(coeff[1]),
        "linear_phase_start": float(t[fit_slice][0]),
        "linear_phase_end": float(t[fit_slice][-1]),
        "time": t,
        "ln_delta_b": y,
        "fit_time": t[fit_slice],
        "fit_ln_delta_b": np.polyval(coeff, t[fit_slice]),
    }


def plot_field_time(rows: list[dict], outdir: Path):
    if not rows:
        return
    t = np.array([r["omega_ci_t"] for r in rows], dtype=float)
    rms = np.array([r["delta_B_rms_over_B0"] for r in rows], dtype=float)
    par = np.array([r["delta_B_parallel_rms_over_B0"] for r in rows], dtype=float)
    perp = np.array([r["delta_B_perp_rms_over_B0"] for r in rows], dtype=float)
    depth = np.array([r["mirror_depth"] for r in rows], dtype=float)
    area = np.array([r["mirror_area_fraction"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(t, rms, "o-", color="#58a6ff")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel(r"$\delta B_{\rm rms}/B_0$", color=TEXT_CLR)
    ax.set_title("Magnetic fluctuation growth", color=TEXT_CLR, fontweight="bold")
    _savefig_many(fig, [outdir / "deltaB_rms_vs_time.png", outdir / "deltaB_over_B0_vs_time.png"])

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(t, par, "o-", color="#ff7b72", label=r"$\delta B_\parallel/B_0$")
    ax.plot(t, perp, "s-", color="#56d364", label=r"$\delta B_\perp/B_0$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel("RMS fluctuation", color=TEXT_CLR)
    ax.set_title("Parallel vs perpendicular magnetic fluctuations", color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _savefig_many(
        fig,
        [
            outdir / "deltaB_components_comparison.png",
            outdir / "deltaB_parallel_vs_time.png",
            outdir / "deltaB_perp_vs_time.png",
        ],
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(t, depth, "o-", color="#d2a8ff", label="depth")
    ax.plot(t, area, "s-", color="#f2cc60", label="area fraction")
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel("mirror-hole metric", color=TEXT_CLR)
    ax.set_title("Mirror-hole depth and area fraction", color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _savefig_many(fig, [outdir / "mirror_depth_vs_time.png", outdir / "mirror_area_fraction_vs_time.png"])


def plot_growth(growth: dict, outdir: Path):
    if not growth:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(growth["time"], growth["ln_delta_b"], "o-", color="#58a6ff", label=r"$\ln\delta B_{\rm rms}$")
    ax.plot(growth["fit_time"], growth["fit_ln_delta_b"], "--", color="#ff7b72",
            label=rf"$\gamma={growth['gamma']:.4g}$")
    ax.axvspan(growth["linear_phase_start"], growth["linear_phase_end"],
               color="#ff7b72", alpha=0.12)
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel(r"$\ln(\delta B_{\rm rms})$", color=TEXT_CLR)
    ax.set_title("Linear growth-rate fit", color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _savefig(fig, outdir / "growth_rate_fit.png")


def compute_jdia(moment_file: str, field_file: str) -> dict[str, np.ndarray]:
    mom_i = load_moments(moment_file, "i")
    mom_e = load_moments(moment_file, "e")
    fld = load_fields(field_file)
    flat_f = lambda key: PICDataReader.flatten_2d_slice(fld[key]).astype(float)
    bx, by, bz = flat_f("Bx"), flat_f("By"), flat_f("Bz")
    b2 = bx**2 + by**2 + bz**2 + 1e-30

    def pperp(mom, suffix):
        return 0.5 * (mom[f"txx_{suffix}"] + mom[f"tyy_{suffix}"])

    pi = pperp(mom_i, "i")
    pe = pperp(mom_e, "e")
    if gaussian_filter is not None:
        pi = gaussian_filter(pi, sigma=3.0)
        pe = gaussian_filter(pe, sigma=3.0)
        by = gaussian_filter(by, sigma=3.0)
        bz = gaussian_filter(bz, sigma=3.0)
    dpidz, dpidy = np.gradient(pi)
    dpedz, dpedy = np.gradient(pe)
    j_i = (dpidy * bz - dpidz * by) / b2
    j_e = (dpedy * bz - dpedz * by) / b2
    return {"J_dia_i": j_i, "J_dia_e": j_e, "J_dia_total": j_i + j_e}


def moment_heat_flux_maps(moment_file: str, field_file: str) -> dict[str, np.ndarray]:
    """Compute spatial ion heat-flux proxies from pressure and bulk velocity."""
    names = [
        "rho_i/p0/3d",
        "txx_i/p0/3d", "tyy_i/p0/3d", "tzz_i/p0/3d",
        "txy_i/p0/3d", "tyz_i/p0/3d", "tzx_i/p0/3d",
        "px_i/p0/3d", "py_i/p0/3d", "pz_i/p0/3d",
    ]
    raw = PICDataReader.read_multiple_fields_3d(
        moment_file, "all_1st", names
    )
    mom = {
        key.split("/")[0]: PICDataReader.flatten_2d_slice(value).astype(float)
        for key, value in raw.items()
    }
    fields = load_fields(field_file)
    bx = PICDataReader.flatten_2d_slice(fields["Bx"]).astype(float)
    by = PICDataReader.flatten_2d_slice(fields["By"]).astype(float)
    bz = PICDataReader.flatten_2d_slice(fields["Bz"]).astype(float)

    n = np.where(mom["rho_i"] > 1e-12, mom["rho_i"], np.nan)
    px, py, pz = mom["px_i"], mom["py_i"], mom["pz_i"]
    vx, vy, vz = px / (n * M_ION), py / (n * M_ION), pz / (n * M_ION)
    pxx = mom["txx_i"] - px * px / (n * M_ION)
    pyy = mom["tyy_i"] - py * py / (n * M_ION)
    pzz = mom["tzz_i"] - pz * pz / (n * M_ION)
    pxy = mom["txy_i"] - px * py / (n * M_ION)
    pyz = mom["tyz_i"] - py * pz / (n * M_ION)
    pzx = mom["tzx_i"] - pz * px / (n * M_ION)

    bmag = np.sqrt(bx**2 + by**2 + bz**2 + 1e-30)
    bhx, bhy, bhz = bx / bmag, by / bmag, bz / bmag
    vpar = vx * bhx + vy * bhy + vz * bhz
    ppar = (
        pxx * bhx**2 + pyy * bhy**2 + pzz * bhz**2
        + 2.0 * pxy * bhx * bhy
        + 2.0 * pyz * bhy * bhz
        + 2.0 * pzx * bhz * bhx
    )
    pperp = 0.5 * (pxx + pyy + pzz - ppar)
    vperp = np.sqrt(
        (vx - vpar * bhx) ** 2
        + (vy - vpar * bhy) ** 2
        + (vz - vpar * bhz) ** 2
    )
    return {
        "q_parallel": ppar * vpar,
        "q_perp": pperp * vperp,
        "P_parallel": ppar,
        "P_perp": pperp,
    }


def localized_heat_flux_rows(
    heat_flux: dict[str, np.ndarray], step: int
) -> list[dict]:
    """Summarize heat flux in four fixed spatial quadrants."""
    qpar = np.asarray(heat_flux["q_parallel"], dtype=float)
    qperp = np.asarray(heat_flux["q_perp"], dtype=float)
    mid0, mid1 = qpar.shape[0] // 2, qpar.shape[1] // 2
    slices = {
        "low_y_low_z": (slice(0, mid0), slice(0, mid1)),
        "low_y_high_z": (slice(0, mid0), slice(mid1, None)),
        "high_y_low_z": (slice(mid0, None), slice(0, mid1)),
        "high_y_high_z": (slice(mid0, None), slice(mid1, None)),
    }
    rows = []
    for region, selection in slices.items():
        local_par = qpar[selection]
        local_perp = qperp[selection]
        rows.append({
            "step": step,
            "omega_ci_t": step_to_omegaci(step),
            "region": region,
            "q_parallel_mean": float(np.nanmean(local_par)),
            "q_parallel_abs_mean": float(np.nanmean(np.abs(local_par))),
            "q_perp_mean": float(np.nanmean(local_perp)),
            "q_perp_abs_mean": float(np.nanmean(np.abs(local_perp))),
        })
    return rows


def correlations(a_map: np.ndarray, delta_b: np.ndarray, bmag: np.ndarray,
                 jdia: np.ndarray, rho: np.ndarray, step: int) -> dict:
    def corr(x, y):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(valid) < 3:
            return float("nan")
        return float(np.corrcoef(x[valid], y[valid])[0, 1])

    return {
        "step": step,
        "omega_ci_t": step_to_omegaci(step),
        "corr_A_deltaB": corr(a_map, delta_b),
        "corr_A_Bmag": corr(a_map, bmag),
        "corr_A_Jdia": corr(a_map, jdia),
        "corr_A_rho_i": corr(a_map, rho),
    }


def plot_scatter(x, y, path: Path, xlabel: str, ylabel: str, title: str):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        return
    if np.count_nonzero(valid) > 120_000:
        idx = RNG.choice(np.where(valid)[0], 120_000, replace=False)
    else:
        idx = np.where(valid)[0]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.scatter(x[idx], y[idx], s=2, alpha=0.18, color="#58a6ff")
    ax.set_xlabel(xlabel, color=TEXT_CLR)
    ax.set_ylabel(ylabel, color=TEXT_CLR)
    ax.set_title(title, color=TEXT_CLR, fontweight="bold")
    _savefig(fig, path)

def _process_particle_step_worker(args):
    """Worker function for run_particles to process a single step in parallel."""
    step, filepath, max_particles, outdir = args
    try:
        snap = _read_particle_snapshot(filepath, max_particles)
        if snap is None:
            return step, None, None, None
        
        ion = particle_temperatures(snap, "ion")
        elec = particle_temperatures(snap, "electron")
        
        row = None
        if ion:
            beta_par = 2.0 * ion["T_parallel"] / (B0**2 + 1e-30)
            row = {
                "step": step,
                "omega_ci_t": snap.time,
                "T_parallel_i": ion["T_parallel"],
                "T_perp_i": ion["T_perp"],
                "A_i": ion["A"],
                "R_i": ion["R"],
                "beta_parallel_i": beta_par,
                "ion_count": ion["count"],
                "T_parallel_e": elec.get("T_parallel", np.nan),
                "T_perp_e": elec.get("T_perp", np.nan),
                "A_e": elec.get("A", np.nan),
            }
            row.update(particle_heat_flux(snap, "ion"))
            row.update(particle_energy(snap, "ion"))

        path = plot_vdf2d(snap, outdir, "ion")
        fit = fit_distribution(snap, "ion")
        
        return step, row, path, fit
    except Exception as exc:
        print(f"\n[ERROR] Step {step} processing failed: {exc}")
        return step, None, None, None


def _field_metrics_worker(args):
    step, field_file, b0 = args
    metrics = field_metrics(field_file, b0)
    return step, {
        "step": step,
        "omega_ci_t": step_to_omegaci(step),
        **{k: v for k, v in metrics.items() if np.isscalar(v)},
    }


def _magnetic_spectrum_worker(args):
    step, field_file, outdir = args
    spectrum = magnetic_perpendicular_spectrum(field_file)
    plot_magnetic_spectrum(spectrum, step, outdir)
    fit = spectrum["fit"]
    return step, {
        "step": step,
        "omega_ci_t": step_to_omegaci(step),
        "plane": spectrum["plane"],
        "axis0": spectrum["axes"][0],
        "axis1": spectrum["axes"][1],
        "delta_axis0": spectrum["spacing"][0],
        "delta_axis1": spectrum["spacing"][1],
        "peak_k": spectrum["peak_k"],
        "peak_power": spectrum["peak_power"],
        "power_law_slope": np.nan if fit is None else fit["slope"],
        "power_law_rvalue": np.nan if fit is None else fit["rvalue"],
    }


def _moment_stats_worker(args):
    step, moment_file, field_file, species = args
    maps = moment_thermal_maps(moment_file, field_file, species)
    a = maps["A"]
    b = maps["beta_parallel"]
    return step, {
        "step": step,
        "omega_ci_t": step_to_omegaci(step),
        "A_mean": float(np.nanmean(a)),
        "A_median": float(np.nanmedian(a)),
        "A_p10": float(np.nanpercentile(a, 10)),
        "A_p90": float(np.nanpercentile(a, 90)),
        "T_parallel_mean": float(np.nanmean(maps["T_parallel"])),
        "T_perp_mean": float(np.nanmean(maps["T_perp"])),
        "beta_parallel_mean": float(np.nanmean(b)),
        "beta_parallel_median": float(np.nanmedian(b)),
    }


def _moment_correlation_worker(args):
    step, moment_file, field_file, species = args
    maps = moment_thermal_maps(moment_file, field_file, species)
    fmet = field_metrics(field_file, B0)
    jdia = compute_jdia(moment_file, field_file)
    heat_flux = moment_heat_flux_maps(moment_file, field_file)
    return step, correlations(
        maps["A"],
        PICDataReader.flatten_2d_slice(fmet["delta_B"]),
        PICDataReader.flatten_2d_slice(fmet["B_magnitude"]),
        jdia["J_dia_total"],
        maps["n"],
        step,
    ), localized_heat_flux_rows(heat_flux, step)


def _field_map_worker(args):
    step, field_file, outdir = args
    metrics = field_metrics(field_file, B0)
    plot_map(PICDataReader.flatten_2d_slice(metrics["delta_B_over_B0"]),
             outdir / f"deltaB_map_step_{step}.png",
             rf"$\delta B/B_0$ - step {step}", r"$\delta B/B_0$",
             cmap="RdBu_r", symmetric=True)
    plot_map(PICDataReader.flatten_2d_slice(metrics["B_magnitude"]),
             outdir / f"mirror_holes_map_step_{step}.png",
             rf"$|B|$ mirror structures - step {step}", r"$|B|$",
             cmap="magma")
    return step, True


def _thermal_map_worker(args):
    step, moment_file, field_file, species, outdir = args
    maps = moment_thermal_maps(moment_file, field_file, species)
    plot_map(maps["T_parallel"], outdir / f"T_parallel_map_step_{step}.png",
             rf"$T_{{\parallel i}}$ - step {step}", r"$T_{\parallel i}$", cmap="plasma")
    plot_map(maps["T_perp"], outdir / f"T_perp_map_step_{step}.png",
             rf"$T_{{\perp i}}$ - step {step}", r"$T_{\perp i}$", cmap="plasma")
    plot_map(maps["A"], outdir / f"A_i_map_step_{step}.png",
             rf"$A_i=T_\perp/T_\parallel$ - step {step}", r"$A_i$", cmap="RdYlBu_r")
    return step, True


class PhysicalDiagnostics:
    def __init__(
        self,
        data_dir: str,
        outdir: str,
        particle_pattern: str | None,
        field_pattern: str | None,
        moment_pattern: str | None,
        max_particles: int,
        max_particle_steps: int,
        max_map_steps: int,
        selected_steps: list[int] | None,
        jobs: int = 0,
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.outdir = Path(outdir)
        self.max_particles = max_particles
        self.max_particle_steps = max_particle_steps
        self.max_map_steps = max_map_steps
        self.selected_steps = selected_steps
        self.jobs = jobs
        discovered = PICDataReader.discover_outputs(str(self.data_dir))
        if particle_pattern:
            self.particle_files = PICDataReader.find_files(particle_pattern)
        else:
            particle_series = discovered["particles"]
            if len(particle_series) > 1:
                names = ", ".join(sorted(particle_series))
                raise ValueError(
                    f"Multiple particle series found ({names}). Pass --particles "
                    "to select exactly one case."
                )
            self.particle_files = (
                next(iter(particle_series.values())) if particle_series else {}
            )
        self.field_files = (
            PICDataReader.find_files(field_pattern)
            if field_pattern
            else discovered["fields"]
        )
        self.moment_files = (
            PICDataReader.find_files(moment_pattern)
            if moment_pattern
            else discovered["moments"]
        )

    def run(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        particle_rows = self.run_particles()
        field_rows = self.run_fields()
        spatial_rows = self.run_moments_and_correlations()
        self.run_energy_summary(particle_rows, field_rows)
        print(f"Physical diagnostics written to {self.outdir}")
        if not (particle_rows or field_rows or spatial_rows):
            print("[WARN] No diagnostics were generated; check input file patterns.")

    def run_particles(self) -> list[dict]:
        if not self.particle_files:
            print("[INFO] No particle files found; skipping particle diagnostics.")
            return []

        steps = sorted(self.particle_files)
        selected = _select_steps(steps, self.selected_steps, self.max_particle_steps)
        rows = []
        fit_rows = []
        vdf_paths = []
        first_fit = None

        tasks = [
            (step, self.particle_files[step], self.max_particles, self.outdir)
            for step in selected
        ]
        results = _run_step_tasks(
            _process_particle_step_worker, tasks, self.jobs, "Particle diagnostics"
        )
        for step, row, path, fit in results:
            if row:
                rows.append(row)
            if path:
                vdf_paths.append(path)
            if fit:
                if first_fit is None:
                    first_fit = fit
                fit_rows.append({
                    key: fit[key] for key in [
                        "step", "omega_ci_t", "kappa_fit", "maxwellian_sigma",
                        "kappa_sigma", "error_maxwellian", "error_kappa",
                        "error_tail_maxwellian", "error_tail_kappa",
                        "suprathermal_fraction",
                    ]
                })

        _write_csv(self.outdir / "validation_table.csv", rows)
        _write_csv(self.outdir / "anisotropy_table.csv", rows)
        _write_csv(self.outdir / "fit_metrics.csv", fit_rows)
        self.write_validation_summary(rows)
        plot_validation(rows, self.outdir)
        plot_time_series(rows, self.outdir)
        plot_fit_metrics(fit_rows, self.outdir)
        plot_distribution_fit(first_fit, self.outdir)
        return rows

    def write_validation_summary(self, rows: list[dict]):
        if not rows:
            return
        row = rows[0]
        expected = []
        if INSTABILITY == "mirror":
            expected.append("mirror: expected A_i > 1 and T_perp_i > T_parallel_i")
            ok = row["A_i"] > 1.0 and row["T_perp_i"] > row["T_parallel_i"]
        elif INSTABILITY == "firehose":
            expected.append("firehose: expected A_i < 1, R_i > 1 and T_parallel_i > T_perp_i")
            ok = row["A_i"] < 1.0 and row["R_i"] > 1.0 and row["T_parallel_i"] > row["T_perp_i"]
        else:
            expected.append("whistler/electron-driven case: ion validation is reported but not the active driver")
            ok = True
        text = [
            f"Profile: {PROFILE_LABEL}",
            f"Instability: {INSTABILITY}",
            f"Initial step: {row['step']}",
            f"T_parallel_i = {row['T_parallel_i']:.8g}",
            f"T_perp_i     = {row['T_perp_i']:.8g}",
            f"A_i          = {row['A_i']:.8g}",
            f"R_i          = {row['R_i']:.8g}",
            f"beta_parallel_i = {row['beta_parallel_i']:.8g}",
            "",
            *expected,
            f"Initial check: {'PASS' if ok else 'CHECK'}",
        ]
        (self.outdir / "validation_summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")

    def run_fields(self) -> list[dict]:
        if not self.field_files:
            print("[INFO] No field files found; skipping magnetic diagnostics.")
            return []
        steps = sorted(self.field_files)
        tasks = [(step, self.field_files[step], B0) for step in steps]
        results = _run_step_tasks(_field_metrics_worker, tasks, self.jobs, "Field diagnostics")
        rows = [row for _, row in results]
        _write_csv(self.outdir / "field_fluctuation_table.csv", rows)
        plot_field_time(rows, self.outdir)
        growth = growth_rate(
            np.array([r["omega_ci_t"] for r in rows]),
            np.array([r["delta_B_rms"] for r in rows]),
        )
        plot_growth(growth, self.outdir)
        if growth:
            _write_csv(self.outdir / "growth_rate_summary.csv", [{
                "gamma": growth["gamma"],
                "linear_phase_start": growth["linear_phase_start"],
                "linear_phase_end": growth["linear_phase_end"],
            }])
        self.plot_field_maps()
        self.run_magnetic_spectra()
        return rows

    def run_magnetic_spectra(self):
        """Generate transverse P(k) within the integrated diagnostics run."""
        steps = _select_steps(
            sorted(self.field_files), self.selected_steps, self.max_map_steps
        )
        rows = []
        tasks = [(step, self.field_files[step], self.outdir) for step in steps]
        results = _run_step_tasks(
            _magnetic_spectrum_worker, tasks, self.jobs, "Magnetic spectrum"
        )
        rows = [row for _, row in results]
        _write_csv(self.outdir / "magnetic_spectrum_table.csv", rows)

    def plot_field_maps(self):
        steps = _select_steps(sorted(self.field_files), self.selected_steps, self.max_map_steps)
        tasks = [(step, self.field_files[step], self.outdir) for step in steps]
        _run_step_tasks(_field_map_worker, tasks, self.jobs, "Field maps")

    def run_moments_and_correlations(self) -> list[dict]:
        if not self.moment_files:
            print("[INFO] No moment files found; skipping moment diagnostics.")
            return []
        rows = []
        corr_rows = []
        heat_flux_rows = []
        common = sorted(set(self.moment_files) & set(self.field_files))
        steps_for_maps = _select_steps(sorted(self.moment_files), self.selected_steps, self.max_map_steps)
        species = "ion"

        stats_tasks = [
            (step, self.moment_files[step], self.field_files.get(step), species)
            for step in sorted(self.moment_files)
        ]
        stats_results = _run_step_tasks(
            _moment_stats_worker, stats_tasks, self.jobs, "Moment statistics"
        )
        rows = [row for _, row in stats_results]

        map_tasks = [
            (step, self.moment_files[step], self.field_files.get(step), species, self.outdir)
            for step in steps_for_maps
        ]
        _run_step_tasks(_thermal_map_worker, map_tasks, self.jobs, "Thermal maps")

        corr_tasks = [
            (step, self.moment_files[step], self.field_files[step], species)
            for step in common
        ]
        corr_results = _run_step_tasks(
            _moment_correlation_worker, corr_tasks, self.jobs, "Moment correlations"
        )
        for step, corr_row, heat_rows in corr_results:
            corr_rows.append(corr_row)
            heat_flux_rows.extend(heat_rows)

        for step in [s for s in steps_for_maps if s in common]:
            maps = moment_thermal_maps(self.moment_files[step], self.field_files[step], species)
            fmet = field_metrics(self.field_files[step], B0)
            jdia = compute_jdia(self.moment_files[step], self.field_files[step])
            heat_flux = moment_heat_flux_maps(
                self.moment_files[step], self.field_files[step]
            )
            plot_map(jdia["J_dia_i"], self.outdir / f"J_dia_i_map_step_{step}.png",
                     rf"$J_{{dia,i}}$ - step {step}", r"$J_{dia,i}$", cmap="RdBu_r", symmetric=True)
            plot_map(jdia["J_dia_e"], self.outdir / f"J_dia_e_map_step_{step}.png",
                     rf"$J_{{dia,e}}$ - step {step}", r"$J_{dia,e}$", cmap="RdBu_r", symmetric=True)
            plot_map(jdia["J_dia_total"], self.outdir / f"J_dia_total_map_step_{step}.png",
                     rf"$J_{{dia,total}}$ - step {step}", r"$J_{dia,total}$", cmap="RdBu_r", symmetric=True)
            plot_map(
                heat_flux["q_parallel"],
                self.outdir / f"q_parallel_map_step_{step}.png",
                rf"$q_{{\parallel,i}}$ - step {step}",
                r"$q_{\parallel,i}$",
                cmap="RdBu_r",
                symmetric=True,
            )
            plot_map(
                heat_flux["q_perp"],
                self.outdir / f"q_perp_map_step_{step}.png",
                rf"$q_{{\perp,i}}$ - step {step}",
                r"$q_{\perp,i}$",
                cmap="inferno",
            )
            plot_scatter(maps["A"], PICDataReader.flatten_2d_slice(fmet["delta_B"]),
                         self.outdir / "A_vs_deltaB_scatter.png", r"$A_i$", r"$\delta B$",
                         "Spatial correlation: anisotropy vs delta B")
            plot_scatter(maps["A"], PICDataReader.flatten_2d_slice(fmet["B_magnitude"]),
                         self.outdir / "A_vs_B_scatter.png", r"$A_i$", r"$|B|$",
                         "Spatial correlation: anisotropy vs |B|")
            plot_scatter(maps["A"], jdia["J_dia_total"],
                         self.outdir / "A_vs_Jdia_scatter.png", r"$A_i$", r"$J_{dia}$",
                         "Spatial correlation: anisotropy vs Jdia")

        _write_csv(self.outdir / "anisotropy_spatial_stats.csv", rows)
        _write_csv(self.outdir / "spatial_correlations.csv", corr_rows)
        _write_csv(self.outdir / "localized_heat_flux_table.csv", heat_flux_rows)
        plot_spatial_maps(rows, self.outdir)
        self.plot_brazil_from_rows(rows)
        return rows

    def plot_brazil_from_rows(self, rows: list[dict]):
        if not rows:
            return
        beta = np.array([r["beta_parallel_mean"] for r in rows])
        a = np.array([r["A_mean"] for r in rows])
        t = np.array([r["omega_ci_t"] for r in rows])
        fig, ax = plt.subplots(figsize=(8, 6.5))
        fig.patch.set_facecolor(DARK_BG)
        _style_axes(ax)
        sc = ax.scatter(beta, a, c=t, cmap="plasma", s=40, edgecolors="white", linewidths=0.3)
        ax.plot(beta, a, color="white", alpha=0.35, lw=1.0)
        bgrid = np.logspace(np.log10(max(np.nanmin(beta) * 0.6, 0.05)),
                            np.log10(max(np.nanmax(beta) * 1.6, 0.2)), 300)
        ax.plot(bgrid, 1.0 + 1.0 / bgrid, "--", color="#ff7b72", label="mirror")
        fh = bgrid[bgrid > 2.0]
        if len(fh):
            ax.plot(fh, 1.0 - 2.0 / fh, "--", color="#58a6ff", label="firehose")
        ax.axhline(1.0, color=TEXT_CLR, alpha=0.35, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\beta_{\parallel i}$", color=TEXT_CLR)
        ax.set_ylabel(r"$A_i$", color=TEXT_CLR)
        ax.set_title("Brazil plot from moment averages", color=TEXT_CLR, fontweight="bold")
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(r"$t\Omega_{ci}$", color=TEXT_CLR)
        cb.ax.yaxis.set_tick_params(color=TEXT_CLR)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_CLR)
        _savefig_many(fig, [self.outdir / "brazil_plot_global.png", self.outdir / "brazil_plot_spatial.png"])

    def run_energy_summary(self, particle_rows: list[dict], field_rows: list[dict]):
        if not particle_rows and not field_rows:
            return
        by_step: dict[int, dict] = {}
        for row in particle_rows:
            by_step.setdefault(row["step"], {}).update(row)
        for row in field_rows:
            by_step.setdefault(row["step"], {}).update(row)
        rows = []
        for step in sorted(by_step):
            r = by_step[step]
            e_bulk = r.get("E_kin_bulk", np.nan)
            e_th = r.get("E_kin_thermal", np.nan)
            e_b = r.get("magnetic_energy_fluct", np.nan)
            if not np.all(np.isfinite([e_bulk, e_th, e_b])):
                continue
            total = e_bulk + e_th + e_b
            rows.append({
                "step": step,
                "omega_ci_t": step_to_omegaci(step),
                "E_kin_bulk": e_bulk,
                "E_kin_thermal": e_th,
                "E_internal_i": 1.5 * (r.get("T_parallel_i", np.nan) + 2.0 * r.get("T_perp_i", np.nan)) / 3.0,
                "E_internal_e": np.nan,
                "E_B": e_b,
                "E_total": total,
            })
        if rows and np.isfinite(rows[0]["E_total"]) and rows[0]["E_total"] != 0:
            e0 = rows[0]["E_total"]
            for row in rows:
                row["energy_error"] = (row["E_total"] - e0) / e0
        _write_csv(self.outdir / "energy_table.csv", rows)
        self.plot_energy(rows)

    def plot_energy(self, rows: list[dict]):
        if not rows:
            return
        t = np.array([r["omega_ci_t"] for r in rows])
        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        fig.patch.set_facecolor(DARK_BG)
        _style_axes(ax)
        for key, color, label in [
            ("E_kin_bulk", "#58a6ff", "bulk"),
            ("E_kin_thermal", "#ff7b72", "thermal"),
            ("E_B", "#56d364", "magnetic fluct."),
            ("E_total", "#f2cc60", "total"),
        ]:
            y = np.array([r.get(key, np.nan) for r in rows], dtype=float)
            if np.any(np.isfinite(y)):
                ax.plot(t, y, "o-", color=color, label=label)
        ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
        ax.set_ylabel("energy proxy [code]", color=TEXT_CLR)
        ax.set_title("Energy partition", color=TEXT_CLR, fontweight="bold")
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
        _savefig(fig, self.outdir / "energy_partition.png")

        err = np.array([r.get("energy_error", np.nan) for r in rows], dtype=float)
        if np.any(np.isfinite(err)):
            fig, ax = plt.subplots(figsize=(8.8, 5.4))
            fig.patch.set_facecolor(DARK_BG)
            _style_axes(ax)
            ax.plot(t, err, "o-", color="#d2a8ff")
            ax.axhline(0, color=TEXT_CLR, alpha=0.35, linestyle=":")
            ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
            ax.set_ylabel(r"$(E(t)-E(0))/E(0)$", color=TEXT_CLR)
            ax.set_title("Energy conservation error", color=TEXT_CLR, fontweight="bold")
            _savefig(fig, self.outdir / "energy_conservation_error.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run integrated physical diagnostics for PSC instability outputs."
    )
    parser.add_argument("--data-dir", default=".", help="Directory containing pfd, pfd_moments and prt files.")
    parser.add_argument("--outdir", default="physical_diagnostics", help="Output directory.")
    parser.add_argument(
        "--particles",
        help="Particle glob pattern. Defaults to auto-discovery of prt*.h5/prt*.bp.",
    )
    parser.add_argument(
        "--fields",
        help="Field glob pattern. Defaults to auto-discovery of pfd HDF5/BP snapshots.",
    )
    parser.add_argument(
        "--moments",
        help="Moment glob pattern. Defaults to auto-discovery of pfd_moments HDF5/BP snapshots.",
    )
    parser.add_argument("--max-particles", type=int, default=500_000,
                        help="Maximum particles read per snapshot.")
    parser.add_argument("--max-particle-steps", type=int, default=12,
                        help="Maximum particle snapshots to process.")
    parser.add_argument("--max-map-steps", type=int, default=5,
                        help="Maximum spatial-map snapshots to render.")
    parser.add_argument("--steps", nargs="*", type=int, help="Optional explicit steps.")
    parser.add_argument("--jobs", "-j", type=int, default=0,
                        help="Number of parallel processes to use (default: 0 = use all CPUs).")
    return parser.parse_args()


def main():
    args = parse_args()
    PhysicalDiagnostics(
        data_dir=args.data_dir,
        outdir=args.outdir,
        particle_pattern=args.particles,
        field_pattern=args.fields,
        moment_pattern=args.moments,
        max_particles=args.max_particles,
        max_particle_steps=args.max_particle_steps,
        max_map_steps=args.max_map_steps,
        selected_steps=args.steps,
        jobs=args.jobs,
    ).run()


if __name__ == "__main__":
    main()
