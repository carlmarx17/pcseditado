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
from matplotlib.patches import Rectangle

import plot_style as ps

ps.apply()

try:
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import curve_fit
except ImportError:  # pragma: no cover - requirements include scipy
    gaussian_filter = None
    curve_fit = None

from data_reader import PICDataReader
from vdf_validation import predictive_check
from analysis_contract import sample_rng, stable_rng, cylindrical_density, atomic_json, effective_sample_size
from growth_fit import BRANCH_SERIES, fit_exponential_growth
from plasma_physics import (
    central_pressure_tensor,
    central_uv,
    diamagnetic_current_x,
    field_aligned_pressures,
    reference_threshold,
    velocity_from_u,
)
from spectral_analysis import SpectralAnalyzer
from psc_units import (
    B0,
    BETA_I_PAR,
    DI,
    DOMAIN_DI_Y,
    DOMAIN_DI_Z,
    DRIVEN_SPECIES,
    DRIVEN_SUFFIX,
    DX_DE,
    INSTABILITY,
    KAPPA,
    K_MAX_DI_DEFAULT,
    M_ELEC,
    M_ION,
    N_GRID_Y,
    N_GRID_Z,
    PROFILE_LABEL,
    PRT_OUTPUT_HI,
    PRT_OUTPUT_LO,
    TI_PAR,
    TI_PERP,
    VA,
    step_to_omegaci,
)


DARK_BG = ps.c("#0d1117")
PANEL_BG = ps.c("#161b22")
TEXT_CLR = ps.c("#e6edf3")
GRID_CLR = ps.c("#30363d")

#: Upper bound of the fitted kappa. A fit that ends there found no measurable
#: tail: it is Maxwellian-consistent, not a measurement of kappa = KAPPA_FIT_MAX.
KAPPA_FIT_MAX = 80.0


# Dominio común de ajuste de la VDF, en unidades de sigma0 (velocidad térmica
# paralela medida). Fijarlo hace que error_maxwellian, error_kappa y sus
# versiones de cola se midan sobre el MISMO rango en todos los casos, que es
# la condición para poder comparar bi-kappa contra bi-Maxwelliana.
FIT_SIGMA_MAX = 6.0    # borde del histograma / ajuste
TAIL_SIGMA = 3.0       # inicio de la región de cola
MIN_BIN_COUNTS = 20    # cuentas mínimas por bin para entrar en el error log

# Criterios de validez de un ajuste de tasa de crecimiento (ver growth_rate).
MIN_AMPLITUDE_GAIN = 2.0   # la amplitud debe al menos duplicarse en la ventana
MAX_ONSET_T0 = 5.0         # t*Omega_ci máximo del primer punto de la serie
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


def _series_style(n: int, marker: str = "o") -> dict:
    """Estilo de línea que se adapta al número de puntos de la serie.

    Las series de campos tienen 1700-2400 snapshots. Con un marcador por
    muestra el marcador es más ancho que el espaciado entre puntos y la curva
    se convierte en una banda sólida: se pierde la forma, que es justo lo que
    se quiere leer (crecimiento, rodilla, saturación). Las series de partículas
    tienen 39-121 puntos y ahí el marcador sí informa de dónde hay dato.
    """
    if n <= 60:
        return {"linestyle": "-", "marker": marker, "markersize": 5}
    if n <= 400:
        return {"linestyle": "-", "marker": marker, "markersize": 3.5,
                "markevery": max(1, n // 40)}
    return {"linestyle": "-", "marker": "none", "linewidth": 1.8}


def _savefig(fig, path: Path, pad_inches: float = 0.1):
    ps.save(fig, path, pad_inches=pad_inches)


def _savefig_many(fig, paths: Iterable[Path]):
    ps.save_many(fig, paths)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(dict.fromkeys(k for row in rows for k in row))
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


def _select_steps_by_cadence(
    steps: list[int], requested: list[int] | None, cadence_omegaci: float,
    step_to_time, max_count: int | None = None,
) -> list[int]:
    """Pick one snapshot per ``cadence_omegaci`` interval of physical time.

    The previous particle-step selection (``_select_steps``) spread a fixed
    count evenly across *indices* of whatever dump files happened to exist,
    which has no relationship to Omega_ci*t -- the resulting cadence changed
    with how many particle dumps a given run happened to leave behind. This
    walks the available steps in time order and keeps the first one at or
    past each 10*n Omega_ci boundary, so the VDF evolution (and every other
    per-step particle diagnostic fed by the same selection) is spaced in
    physical time, never denser than the raw dumps allow and never fabricated
    between them.
    """
    if requested:
        requested_set = set(requested)
        return [s for s in steps if s in requested_set]
    if not steps:
        return []
    times = {s: step_to_time(s) for s in steps}
    ordered = sorted(steps, key=lambda s: times[s])
    if cadence_omegaci <= 0:
        selected = ordered
    else:
        selected = []
        next_target = times[ordered[0]]
        for s in ordered:
            if times[s] + 1e-9 >= next_target:
                selected.append(s)
                next_target = times[s] + cadence_omegaci
    if max_count is not None and len(selected) > max_count:
        idx = np.unique(np.linspace(0, len(selected) - 1, max_count, dtype=int))
        selected = [selected[i] for i in idx]
    return selected


def _read_particle_snapshot(path: str, max_particles: int) -> ParticleSnapshot:
    step = _extract_step(path)
    q, m, px, py, pz, w = PICDataReader.read_particles_snapshot(
        path, max_particles=max_particles, rng=sample_rng(path, "both", "physical-particles")
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
    ux, uy, uz = snapshot.px[mask], snapshot.py[mask], snapshot.pz[mask]
    weights = snapshot.w[mask]
    # PSC writes u = gamma v and deposits t_ab = n m <u_a v_b>. Building the
    # particle temperature with the same central <u v> moment makes it the
    # estimator the moment maps use, and it is the correct kinetic pressure
    # for the suprathermal tails of kappa = 3 (reduces to m var(v) if |u|<<1).
    vx, vy, vz, _ = velocity_from_u(ux, uy, uz)
    tpar = mass * central_uv(uz, vz, weights)
    tperp = 0.5 * mass * (central_uv(ux, vx, weights) + central_uv(uy, vy, weights))
    return {
        "T_parallel": float(tpar),
        "T_perp": float(tperp),
        "A": float(tperp / max(tpar, 1e-30)),
        "R": float(tpar / max(tperp, 1e-30)),
        "mass": float(mass),
        "count": int(np.count_nonzero(mask)),
    }


def particle_kinematics_validity(snapshot: ParticleSnapshot, species: str) -> dict:
    """Quantify u=gamma*v versus v for the nonrelativistic particle diagnostics.

    Compare diagonal central m<u*u> with m<u*v> in the window's lab frame.
    This checks an approximation; it is not a relativistic rest-frame temperature.
    """
    mask = _species_mask(snapshot, species)
    if not np.any(mask):
        return {}
    u = np.stack([snapshot.px[mask], snapshot.py[mask], snapshot.pz[mask]])
    weights = snapshot.w[mask]
    u2 = np.sum(u*u, axis=0)
    gamma = np.sqrt(1.0 + u2)
    velocity = u / gamma
    # u^2/(gamma+1) avoids cancellation in gamma-1 for cold particles.
    exact = _weighted_mean(u2 / (gamma + 1.0), weights)
    approximate = 0.5 * _weighted_mean(u2, weights)
    errors = []
    for uj, vj in zip(u, velocity):
        central_uv = _weighted_mean(uj*vj, weights) - _weighted_mean(uj, weights)*_weighted_mean(vj, weights)
        central_uu = _weighted_var(uj, weights)
        if central_uv > 0:
            errors.append(abs(central_uu-central_uv)/central_uv)
    return {
        "max_lorentz_gamma": float(np.max(gamma)),
        "nr_energy_relative_error": (approximate-exact)/exact if exact > 0 else 0.0,
        "max_nr_diagonal_pressure_relative_error": max(errors) if errors else float("nan"),
        "fraction_u_above_c": _weighted_mean(u2 > 1.0, weights),
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
        f"txy_{suffix}/p0/3d",
        f"tyz_{suffix}/p0/3d",
        f"tzx_{suffix}/p0/3d",
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
    pxy = mom[f"txy_{suffix}"] - mom[f"px_{suffix}"] * mom[f"py_{suffix}"] / (safe_n * mass)
    pyz = mom[f"tyz_{suffix}"] - mom[f"py_{suffix}"] * mom[f"pz_{suffix}"] / (safe_n * mass)
    pzx = mom[f"tzx_{suffix}"] - mom[f"pz_{suffix}"] * mom[f"px_{suffix}"] / (safe_n * mass)

    if field_file:
        fld = load_fields(field_file)
        bx = PICDataReader.flatten_2d_slice(fld["Bx"])
        by = PICDataReader.flatten_2d_slice(fld["By"])
        bz = PICDataReader.flatten_2d_slice(fld["Bz"])
        bmag = np.sqrt(bx**2 + by**2 + bz**2)
    else:
        bx = np.zeros_like(pzz)
        by = np.zeros_like(pzz)
        bz = np.full_like(pzz, B0)
        bmag = np.full_like(pzz, B0)

    ppar, pperp, _ = field_aligned_pressures(
        pxx, pyy, pzz, pxy, pyz, pzx, bx, by, bz
    )
    tpar = ppar / safe_n
    tperp = pperp / safe_n
    anisotropy = tperp / (tpar + 1e-30)
    beta_par = 2.0 * ppar / (bmag**2 + 1e-30)
    return {
        "n": n,
        "Pxx": pxx,
        "Pyy": pyy,
        "Pzz": pzz,
        "P_parallel": ppar,
        "P_perp": pperp,
        "T_parallel": tpar,
        "T_perp": tperp,
        "A": anisotropy,
        "R": 1.0 / np.maximum(anisotropy, 1e-30),
        "beta_parallel": beta_par,
        "B_magnitude": bmag,
    }


# ── Ventana prt (la region de la que se guardan las particulas / la VDF) ─────
#
# PSC solo escribe partículas dentro de la caja de celdas [lo, hi) fijada en
# `OutputParticlesParams`. Todas las VDF salen de ahí, así que las métricas de
# campo globales no son las que ve la VDF: si el hueco magnético que se está
# caracterizando cae fuera de la ventana, la correlación VDF-hueco no existe.
# Estas funciones dan la misma foto de B pero restringida a esa ventana, y la
# dibujan sobre los mapas para que cada figura diga de dónde salen las
# partículas.


def prt_window_bounds(particle_file: str | None) -> tuple[np.ndarray, np.ndarray, str]:
    """(lo, hi, origen) de la ventana prt, leída del archivo cuando se puede.

    La ventana no es constante entre corridas (hay datos escritos con el 40 %
    por eje y otros con el 20 %), por eso el fallback de `psc_units` solo se
    usa si no hay archivo: confiar en él ubicaría la caja en el lugar
    equivocado sin ningún síntoma visible.
    """
    if particle_file:
        try:
            lo, hi = PICDataReader.read_prt_window(particle_file)
            return np.asarray(lo, dtype=int), np.asarray(hi, dtype=int), "archivo prt"
        except (KeyError, OSError, ValueError) as exc:
            print(f"[WARN] no se pudo leer la ventana prt de {particle_file}: {exc}; "
                  "usando el fallback de psc_units.")
    return (np.asarray(PRT_OUTPUT_LO, dtype=int),
            np.asarray(PRT_OUTPUT_HI, dtype=int), "fallback psc_units")


def _prt_cell_slices(shape: tuple[int, ...], lo, hi) -> tuple[slice, slice] | None:
    """Traduce (lo, hi) en índices (x,y,z) a slices del mapa 2D aplanado.

    Los arrays PSC llegan como (Nz, Ny) en el plano yz, pero no se asume: se
    decide comparando la forma real contra la malla del perfil, porque una
    transposición silenciosa pondría la caja rotada 90 grados sobre el mapa.
    """
    iy0, iy1 = int(lo[1]), int(hi[1])
    iz0, iz1 = int(lo[2]), int(hi[2])
    if shape == (N_GRID_Z, N_GRID_Y):
        return slice(iz0, iz1), slice(iy0, iy1)
    if shape == (N_GRID_Y, N_GRID_Z):
        return slice(iy0, iy1), slice(iz0, iz1)
    print(f"[WARN] forma de campo {shape} no coincide con la malla "
          f"({N_GRID_Z}, {N_GRID_Y}); se omiten las métricas en la ventana prt.")
    return None


def prt_window_extent_di(lo, hi) -> dict:
    """Bordes de la ventana en d_i, en la convención de `plot_map` (Z, Y)."""
    z0 = float(lo[2]) / N_GRID_Z * DOMAIN_DI_Z
    z1 = float(hi[2]) / N_GRID_Z * DOMAIN_DI_Z
    y0 = float(lo[1]) / N_GRID_Y * DOMAIN_DI_Y
    y1 = float(hi[1]) / N_GRID_Y * DOMAIN_DI_Y
    return {"z_di": (z0, z1), "y_di": (y0, y1),
            "area_fraction": ((z1 - z0) * (y1 - y0)) / (DOMAIN_DI_Z * DOMAIN_DI_Y)}


def _window_field_scalars(bx, by, bz, lo, hi, b0: float) -> dict:
    """Las mismas métricas escalares de `field_metrics`, sólo dentro del prt."""
    flat = PICDataReader.flatten_2d_slice(np.asarray(bx))
    slices = _prt_cell_slices(flat.shape, lo, hi)
    if slices is None:
        return {}
    sz, sy = slices

    def _win(arr):
        return PICDataReader.flatten_2d_slice(np.asarray(arr))[sz, sy]

    wbx, wby, wbz = _win(bx), _win(by), _win(bz)
    if wbx.size == 0:
        print("[WARN] la ventana prt quedó vacía sobre la malla de campos.")
        return {}
    bmag = np.sqrt(wbx**2 + wby**2 + wbz**2)
    delta_b = bmag - b0
    # Las medias se toman DENTRO de la ventana: restar la media global metería
    # el offset del resto del dominio en una region que no lo comparte.
    dbx = wbx - np.nanmean(wbx)
    dby = wby - np.nanmean(wby)
    dbz = wbz - b0
    b0_abs = max(abs(b0), 1e-30)
    sigma_b = np.nanstd(bmag)
    return {
        "prt_B_mean_over_B0": float(np.nanmean(bmag) / b0_abs),
        "prt_delta_B_rms": float(np.sqrt(np.nanmean(delta_b**2))),
        "prt_delta_B_rms_over_B0": float(np.sqrt(np.nanmean(delta_b**2)) / b0_abs),
        "prt_delta_B_parallel_rms_over_B0": float(np.sqrt(np.nanmean(dbz**2)) / b0_abs),
        "prt_delta_B_perp_rms_over_B0": float(np.sqrt(np.nanmean(dbx**2 + dby**2)) / b0_abs),
        "prt_delta_B_vec_rms_over_B0": float(np.sqrt(np.nanmean(dbx**2 + dby**2 + dbz**2)) / b0_abs),
        "prt_B_min_over_B0": float(np.nanmin(bmag) / b0_abs),
        "prt_B_max_over_B0": float(np.nanmax(bmag) / b0_abs),
        "prt_mirror_depth": float(1.0 - np.nanmin(bmag) / b0_abs),
        # Scale-free: the threshold is the fluctuation's own std, so this is
        # ~16-25 % for any amplitude and is NOT a mirror-hole area
        # (hole populations with absolute thresholds: structures_analysis.py).
        "prt_fraction_below_B0_minus_std": float(np.nanmean(bmag < (b0 - sigma_b))),
        "prt_magnetic_energy_fluct": float(0.5 * np.nanmean(dbx**2 + dby**2 + dbz**2)),
        "prt_cells": int(bmag.size),
    }


def field_metrics(field_file: str, b0: float = B0,
                  prt_window: tuple | None = None,
                  modes: list[tuple[int, int]] | None = None) -> dict:
    """Métricas de |B| sobre todo el dominio.

    Con ``prt_window=(lo, hi)`` añade las mismas métricas escalares medidas
    sólo dentro de la ventana de salida de partículas, con prefijo ``prt_``.
    Con ``modes`` (índices FFT de ``mode_candidates``) añade ``mode_power``,
    el <|dB|^2> de cada modo separado en parte transversal y compresiva
    (2, n_modos), para el ajuste de gamma y la rama de cada modo.
    """
    fld = load_fields(field_file)
    bx, by, bz = fld["Bx"], fld["By"], fld["Bz"]
    power = (mode_power(_fluctuation_plane(bx, by, bz)[0], modes, split=True)
             if modes else np.zeros((2, 0)))
    bmag = np.sqrt(bx**2 + by**2 + bz**2)
    delta_b = bmag - b0
    dbx = bx - np.nanmean(bx)
    dby = by - np.nanmean(by)
    dbz = bz - b0
    sigma_b = np.nanstd(bmag)
    b0_abs = max(abs(b0), 1e-30)
    window_scalars = (
        _window_field_scalars(bx, by, bz, prt_window[0], prt_window[1], b0)
        if prt_window is not None else {}
    )
    return {
        **window_scalars,
        "mode_power": power,
        "B_magnitude": bmag,
        "delta_B": delta_b,
        "delta_B_over_B0": delta_b / b0_abs,
        # Mapas 2D de las componentes, para las correlaciones espaciales:
        # dbz aísla la parte compresiva (mirror) y b_perp la transversal
        # (firehose / EMIC).
        "delta_B_parallel_map": dbz,
        "B_perp_map": np.sqrt(dbx**2 + dby**2),
        "delta_B_rms": float(np.sqrt(np.nanmean(delta_b**2))),
        "delta_B_rms_over_B0": float(np.sqrt(np.nanmean(delta_b**2)) / b0_abs),
        "delta_B_parallel_rms": float(np.sqrt(np.nanmean(dbz**2))),
        "delta_B_parallel_rms_over_B0": float(np.sqrt(np.nanmean(dbz**2)) / b0_abs),
        "delta_B_perp_rms": float(np.sqrt(np.nanmean(dbx**2 + dby**2))),
        "delta_B_perp_rms_over_B0": float(np.sqrt(np.nanmean(dbx**2 + dby**2)) / b0_abs),
        # Full vector fluctuation |dB| = |B - <B>|. This, not |B| - B0, is the
        # amplitude whose log-slope is gamma for every branch: for transverse
        # modes (parallel firehose, EMIC, whistler) |B| - B0 ~ dB_perp^2/2B0
        # is second order and its log-slope is 2 gamma.
        "delta_B_vec_rms": float(np.sqrt(np.nanmean(dbx**2 + dby**2 + dbz**2))),
        "delta_B_vec_rms_over_B0": float(np.sqrt(np.nanmean(dbx**2 + dby**2 + dbz**2)) / b0_abs),
        "B_min": float(np.nanmin(bmag)),
        "mirror_depth": float(1.0 - np.nanmin(bmag) / b0_abs),
        # See prt_fraction_below_B0_minus_std: a statistic, not a hole area.
        "fraction_below_B0_minus_std": float(np.nanmean(bmag < (b0 - sigma_b))),
        # Full fluctuation energy density 0.5<|dB|^2>: using only the
        # magnitude fluctuation (|B|-B0) drops the perpendicular components,
        # which dominate by ~an order of magnitude in these runs.
        "magnetic_energy_fluct": float(0.5 * np.nanmean(dbx**2 + dby**2 + dbz**2)),
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


# ── Dominant Fourier mode: the reference growth rate ─────────────────────────
#
# The domain rms <|dB|^2>^1/2 is a poor growth amplitude: it sums the PIC noise
# of every k with the unstable mode, the noise itself grows with the secular
# heating, and its log-slope stays below gamma until the mode dwarfs the noise,
# which happens only near saturation. On the v5 mirror runs it returned
# gamma ~ 0.056-0.065 where the dominant mode grows at ~0.11-0.12. The
# reference gamma is therefore the fit of one Fourier mode of dB: the mode
# with the largest amplitude over the run, followed through every snapshot.

#: Modes taken from each candidate snapshot when choosing which Fourier modes
#: of dB to follow through the run.
MODE_CANDIDATES_PER_SNAPSHOT = 8
#: Fractions of the run at which the candidate snapshots are read.
MODE_CANDIDATE_FRACTIONS = tuple(np.unique(np.r_[np.geomspace(0.001, 0.25, 12), 0.5, 0.75, 1.0]))


def _fluctuation_plane(bx, by, bz) -> tuple[np.ndarray, tuple[str, str], tuple[float, float]]:
    """dB on the simulation plane as (3, n0, n1), with its axes and spacing in d_i."""
    bx, by, bz = (np.asarray(v, dtype=float) for v in (bx, by, bz))
    probe = SpectralAnalyzer(outdir="/tmp/psc_spectral_probe", parallel_axis="z")
    plane = probe._get_plane_slice(bx, by, bz, _spectral_plane(bx.shape))
    comps = np.stack([np.atleast_2d(plane[c]) for c in ("bx", "by", "bz")]).astype(float)
    comps -= comps.mean(axis=(1, 2), keepdims=True)
    lengths = {"x": 1.0, "y": DOMAIN_DI_Y, "z": DOMAIN_DI_Z}
    axes = plane["axes"]
    spacing = (lengths[axes[0]] / comps.shape[1], lengths[axes[1]] / comps.shape[2])
    return comps, axes, spacing


def _mode_power_map(comps: np.ndarray) -> np.ndarray:
    """Mean-square contribution of each Fourier mode, summed over components.

    Normalised with Parseval, so that summing it over the non-redundant half
    plane, doubled, gives <|dB|^2>.
    """
    n = comps.shape[1] * comps.shape[2]
    return np.sum(np.abs(np.fft.fft2(comps, axes=(1, 2))) ** 2, axis=0) / n ** 2


def mode_power(comps: np.ndarray, modes: list[tuple[int, int]],
               split: bool = False) -> np.ndarray:
    """<|dB|^2> carried by each (+k, -k) pair in ``modes`` (unshifted FFT indices).

    With ``split`` the result is (2, n_modes): the transverse part
    |dBx|^2 + |dBy|^2 and the compressive part |dBz|^2 (B0 is along z), which
    sum to the unsplit power. Their ratio is the compressibility of the mode,
    the quantity that separates the mirror branch from the ion-cyclotron
    branch (see classify_mode).
    """
    if not modes:
        return np.zeros((2, 0)) if split else np.zeros(0)
    i0, i1 = zip(*modes)
    self_conjugate = ((2 * np.asarray(i0)) % comps.shape[1] == 0) & ((2 * np.asarray(i1)) % comps.shape[2] == 0)
    weight = np.where(self_conjugate, 1.0, 2.0)
    if not split:
        return weight * _mode_power_map(comps)[list(i0), list(i1)]
    transverse = _mode_power_map(comps[:2])[list(i0), list(i1)]
    compressive = _mode_power_map(comps[2:])[list(i0), list(i1)]
    return weight * np.stack([transverse, compressive])


# --- Branch of a Fourier mode ------------------------------------------------
#
# Why: at beta_i|| = 5 and T_perp/T_par = 2 the ions are unstable to TWO
# branches whose thresholds are almost equal (Hellinger et al. 2006 fits:
# A_IC = 1.219, A_mirror = 1.226 at gamma_max = 1e-3 Omega_ci). The strongest
# Fourier mode of the v5 "mirror" runs had k_perp = 0 and a compressibility of
# ~1e-12, i.e. it was an ion-cyclotron wave, not a mirror mode. A single
# "dominant mode" gamma therefore does not measure the mirror growth rate the
# thesis compares between bi-Maxwellian and bi-kappa ions; each branch needs
# its own fit.
#
# The two branches are told apart by linear-theory polarisation, which a
# single Fourier amplitude of dB carries:
#   * mirror: non-propagating, oblique (theta_kB well above 45 deg in the
#     unstable band), and since div B = 0 forces dB to lie in the (k, B0)
#     plane, |dB_par|^2 / |dB|^2 = sin^2(theta_kB) -> compressive;
#   * ion-cyclotron (EMIC): fastest at parallel propagation, dB transverse
#     and left-hand polarised, |dB_par|^2 / |dB|^2 -> 0.
# A mode that is neither (intermediate angle or compressibility) is left
# unclassified instead of forced into a branch.

#: Minimum theta_kB and compressibility of a mirror-like mode. For a linear
#: mirror mode the compressibility is sin^2(theta_kB) >= 0.5 above 45 deg.
MIRROR_MIN_THETA_DEG = 45.0
MIRROR_MIN_COMPRESSIBILITY = 0.5
#: Maximum theta_kB and compressibility of an ion-cyclotron-like mode. Oblique
#: EMIC waves acquire a compressive part; 0.2 keeps the quasi-parallel ones.
IC_MAX_THETA_DEG = 30.0
IC_MAX_COMPRESSIBILITY = 0.2


#: Noise-settling time of a Fourier mode, in units of 1/(k v_th,i). The
#: quiet start has no fluctuations: the ion noise of a mode at wavenumber k
#: builds up over the ion transit time 1/(k v_th,i), and a mode followed from
#: t = 0 "grows" by a factor of a few during that time at a rate unrelated
#: to any instability (v6b mirror runs: gamma ~ 2-6 Omega_ci on windows
#: [0.13, 0.3-1.8] Omega_ci^-1 for weak modes). Real linear phases start from
#: the settled noise, so a mode is fitted only on t >= NOISE_SETTLING_TRANSITS
#: / (k v_th,i) (ion-driven cases; v_th,i from the larger of T_par, T_perp).
NOISE_SETTLING_TRANSITS = 2.0


def noise_settling_time(k_di: float) -> float:
    """Omega_ci t before which a mode at |k| d_i is still settling to its noise level."""
    if DRIVEN_SPECIES != "ion" or not k_di > 0:
        return 0.0
    vth_over_va = np.sqrt(BETA_I_PAR * max(1.0, TI_PERP / TI_PAR) / 2.0)
    return NOISE_SETTLING_TRANSITS / (k_di * vth_over_va)


def classify_mode(theta_kB_deg: float, compressibility: float) -> str:
    """Geometric branch of one Fourier mode of dB (see the comment above)."""
    if not (np.isfinite(theta_kB_deg) and np.isfinite(compressibility)):
        return "unclassified"
    if theta_kB_deg >= MIRROR_MIN_THETA_DEG and compressibility >= MIRROR_MIN_COMPRESSIBILITY:
        return "compressive_oblique"
    if theta_kB_deg <= IC_MAX_THETA_DEG and compressibility <= IC_MAX_COMPRESSIBILITY:
        return "transverse_parallel"
    return "unclassified"


def physical_branch(branch: str, ion_anisotropy: float) -> str:
    """Instability that a geometric branch corresponds to for the driving ions.

    Only T_perp > T_par ions drive both a compressive (mirror) and a transverse
    (ion-cyclotron) branch. For T_perp < T_par the parallel firehose is
    transverse, but the oblique firehose is Alfvenic (dB out of the plane, not
    compressive), so no physical name is attached; an isotropic control drives
    neither.
    """
    if ion_anisotropy > 1.0 + 1e-9:
        return {"compressive_oblique": "mirror",
                "transverse_parallel": "ion-cyclotron"}.get(branch, "")
    if ion_anisotropy < 1.0 - 1e-9 and branch == "transverse_parallel":
        return "parallel firehose"
    return ""


def mode_candidates(field_files: dict[int, str], kmax_di: float,
                    per_snapshot: int = MODE_CANDIDATES_PER_SNAPSHOT) -> list[dict]:
    """Fourier modes of dB worth following: the strongest at a few snapshots.

    Reading every snapshot twice to find the dominant mode first would double
    the cost of the field pass; the union of the ``per_snapshot`` strongest
    modes with |k| d_i <= ``kmax_di`` at 25/50/75/100 % of the run contains it
    (it is the largest amplitude during the late linear phase and saturation)
    while keeping the per-snapshot output to a few dozen numbers. Each +k/-k
    pair of the real field is one mode, kept in the half plane with k0 > 0.

    The strongest modes of the compressive part dB_par and of the transverse
    part dB_perp are added separately (``per_snapshot // 2`` each): when one
    branch dominates the total power, the other would otherwise never be
    followed and its growth rate could not be measured.
    """
    steps = sorted(field_files)
    if not steps:
        return []
    picks = sorted({steps[max(0, int(round(f * len(steps))) - 1)] for f in MODE_CANDIDATE_FRACTIONS})
    chosen: dict[tuple[int, int], None] = {}
    axes = spacing = shape = None
    for step in picks:
        fld = load_fields(field_files[step])
        comps, axes, spacing = _fluctuation_plane(fld["Bx"], fld["By"], fld["Bz"])
        shape = comps.shape[1:]
        k0 = 2.0 * np.pi * np.fft.fftfreq(shape[0], d=spacing[0])
        k1 = 2.0 * np.pi * np.fft.fftfreq(shape[1], d=spacing[1])
        K0, K1 = np.meshgrid(k0, k1, indexing="ij")
        allowed = ((K0 ** 2 + K1 ** 2 <= kmax_di ** 2)
                   & ((K0 > 0) | ((K0 == 0) & (K1 > 0))))
        flat = np.flatnonzero(allowed)
        for power, count in ((_mode_power_map(comps).ravel(), per_snapshot),
                             (_mode_power_map(comps[:2]).ravel(), per_snapshot // 2),
                             (_mode_power_map(comps[2:]).ravel(), per_snapshot // 2)):
            for f in flat[np.argsort(power[flat])[::-1][:count]]:
                chosen[tuple(int(v) for v in np.unravel_index(f, shape))] = None
    k0 = 2.0 * np.pi * np.fft.fftfreq(shape[0], d=spacing[0])
    k1 = 2.0 * np.pi * np.fft.fftfreq(shape[1], d=spacing[1])
    par_first = axes[0] == "z"
    modes = []
    for i0, i1 in sorted(chosen):
        k_par, k_perp = (k0[i0], k1[i1]) if par_first else (k1[i1], k0[i0])
        modes.append({
            "i0": i0, "i1": i1,
            "k_parallel_di": float(k_par), "k_perp_di": float(k_perp),
            "k_di": float(np.hypot(k_par, k_perp)),
            "theta_kB_deg": float(np.degrees(np.arctan2(abs(k_perp), abs(k_par)))),
        })
    return modes


def plot_magnetic_spectrum(spectrum: dict, step: int, outdir: Path):
    k = np.asarray(spectrum["k"], dtype=float)
    power = np.asarray(spectrum["power"], dtype=float)
    if k.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.loglog(k, power, color=ps.c("#58a6ff"), lw=2.0, label=r"$E_{B_\perp}(k)$")
    fit = spectrum["fit"]
    if fit is not None and fit.get("accepted"):
        fit_power = 10 ** (
            fit["intercept"] + fit["slope"] * np.log10(fit["k_fit"])
        )
        ax.loglog(
            fit["k_fit"],
            fit_power,
            "--",
            color=ps.c("#ff7b72"),
            lw=1.8,
            label=rf"fit: $k^{{{fit['slope']:.2f}}}$",
        )
        # Reference slope over the same range, anchored where the fit starts.
        k_ref = fit["k_fit"]
        ax.loglog(k_ref, fit_power[0] * (k_ref / k_ref[0]) ** (-5.0 / 3.0), ":",
                  color=ps.c("#f2cc60"), label=r"$k^{-5/3}$ (reference)")
    elif fit is not None:
        ax.text(0.97, 0.05, fit["reason"], transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, color=ps.MUTED_CLR)
    if fit is not None and np.isfinite(fit.get("noise_floor", np.nan)):
        ax.axhline(fit["noise_floor"], color=ps.MUTED_CLR, lw=0.8, ls="--", label="noise floor")
    ax.set_xlabel(r"$k\,[d_i^{-1}]$", color=TEXT_CLR)
    ax.set_ylabel(r"$E_{B_\perp}(k)$", color=TEXT_CLR)
    ax.set_title(
        f"Transverse magnetic spectrum - step {step} ({spectrum['plane']})",
        color=TEXT_CLR,
        fontweight="bold",
    )
    ps.plain_log_axis(ax)
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
        return float(np.sqrt(np.mean((y - model) ** 2)))

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


def _fit_density_models(x, y, sigma0):
    """Same linear-density least-squares objective for SciPy and grid solvers."""
    if curve_fit is None:
        return _grid_fit_distribution(x, y, sigma0)
    try:
        m, _ = curve_fit(maxwellian_pdf, x, y, p0=(float(np.max(y)), sigma0),
                         bounds=([0., sigma0*.05], [np.inf, sigma0*20]), maxfev=20000)
        k, _ = curve_fit(kappa_pdf_shape, x, y, p0=(float(np.max(y)), sigma0, KAPPA or 5.),
                         bounds=([0., sigma0*.05, 1.51], [np.inf, sigma0*20, KAPPA_FIT_MAX]), maxfev=30000)
        return m,k
    except (RuntimeError, ValueError, FloatingPointError):
        return _grid_fit_distribution(x,y,sigma0)


def fit_distribution(snapshot: ParticleSnapshot, species: str = "ion") -> dict:
    mask = _species_mask(snapshot, species)
    if not np.any(mask):
        return {}

    vz = snapshot.pz[mask]
    weights = snapshot.w[mask]
    centered = vz - _weighted_mean(vz, weights)
    sigma0 = math.sqrt(max(_weighted_var(centered, weights), 1e-30))
    # El dominio del histograma se fija en múltiplos de sigma0, NO en un
    # percentil de |v|. Con vmax = p99.7 una bi-Maxwelliana se recorta justo
    # en ~3*sigma0, la máscara de cola (|x| > 3*sigma0) queda vacía y
    # error_tail_maxwellian sale NaN, de modo que el contraste kappa vs
    # Maxwelliana se medía en rangos distintos para cada caso y no era
    # comparable. TAIL_SIGMA marca el inicio de la cola y FIT_SIGMA_MAX el
    # borde del ajuste; ambos son iguales para todas las distribuciones.
    vmax = FIT_SIGMA_MAX * sigma0
    if not np.isfinite(vmax) or vmax <= 0:
        return {}

    counts, edges = np.histogram(centered, bins=160, range=(-vmax, vmax),
                                 weights=weights)
    widths = np.diff(edges)
    total = float(np.sum(weights))
    hist = counts / np.maximum(total * widths, 1e-300)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Un bin con 1-2 partículas está a órdenes de magnitud de su valor
    # esperado, y el error se mide en log10: sin este filtro los bins casi
    # vacíos del extremo dominan la métrica y error_maxwellian se dispara por
    # ruido de disparo, no por desajuste. Se registra hasta dónde llega la
    # estadística útil para que el rango efectivo sea auditable.
    raw_counts, _ = np.histogram(centered, bins=edges)
    weight2, _ = np.histogram(centered, bins=edges, weights=weights ** 2)
    n_eff = np.divide(counts ** 2, weight2, out=np.zeros_like(counts), where=weight2 > 0)
    valid = np.isfinite(hist) & (raw_counts >= MIN_BIN_COUNTS) & (n_eff >= MIN_BIN_COUNTS)
    if np.count_nonzero(valid) < 12:
        return {}

    x = centers[valid]
    y = hist[valid]
    v_reliable_over_sigma = float(np.max(np.abs(x)) / sigma0)
    amp0 = float(np.nanmax(y))
    fit_solver, fallback_reason = "scipy_curve_fit", ""
    kappa_stderr = np.nan
    if curve_fit is None:
        fit_solver, fallback_reason = "grid", "scipy unavailable"
        popt_m, popt_k = _grid_fit_distribution(x, y, sigma0)
    else:
        try:
            popt_m, _ = curve_fit(
                maxwellian_pdf, x, y, p0=(amp0, sigma0),
                bounds=([0.0, sigma0 * 0.05], [np.inf, sigma0 * 20.0]),
                maxfev=20000,
            )
            popt_k, covariance_k = curve_fit(
                kappa_pdf_shape, x, y, p0=(amp0, sigma0, KAPPA or 5.0),
                bounds=([0.0, sigma0 * 0.05, 1.51], [np.inf, sigma0 * 20.0, KAPPA_FIT_MAX]),
                maxfev=30000,
            )
            kappa_stderr = float(np.sqrt(max(covariance_k[2, 2], 0.)))
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            fit_solver, fallback_reason = "grid", str(exc)
            popt_m, popt_k = _grid_fit_distribution(x, y, sigma0)

    y_m = maxwellian_pdf(x, *popt_m)
    y_k = kappa_pdf_shape(x, *popt_k)
    tail = np.abs(x) > TAIL_SIGMA * sigma0
    err_m = float(np.sqrt(np.nanmean((np.log10(y) - np.log10(y_m + 1e-300)) ** 2)))
    err_k = float(np.sqrt(np.nanmean((np.log10(y) - np.log10(y_k + 1e-300)) ** 2)))
    err_m_tail = float(np.sqrt(np.nanmean((np.log10(y[tail]) - np.log10(y_m[tail] + 1e-300)) ** 2))) if np.any(tail) else np.nan
    err_k_tail = float(np.sqrt(np.nanmean((np.log10(y[tail]) - np.log10(y_k[tail] + 1e-300)) ** 2))) if np.any(tail) else np.nan
    # La fracción suprathermal usa el mismo umbral (TAIL_SIGMA) que la máscara
    # de cola, sobre la velocidad 1D. Antes comparaba |v| 3D contra
    # 3*sqrt(var_x+var_y+var_z) = 3*sqrt(3)*sigma ~ 5.2 sigma, un corte
    # distinto del de error_tail_* y que no correspondía a su etiqueta.
    supra = float(np.sum(weights[np.abs(centered) > TAIL_SIGMA * sigma0])
                  / max(np.sum(weights), 1e-30))

    predictive = predictive_check(vz, weights, stable_rng(PROFILE_LABEL, snapshot.step, species, "vdf-heldout"), _fit_density_models)
    return {
        **predictive,
        "step": snapshot.step,
        "omega_ci_t": snapshot.time,
        "kappa_fit_stderr_conditional": kappa_stderr,
        "kappa_fit": float(popt_k[2]),
        "fit_coordinate": "u_parallel",
        "fit_frame": "global B0; window mean drift",
        "fit_objective": "unweighted density least squares",
        "fit_solver": fit_solver, "fit_fallback_reason": fallback_reason,
        "kappa_identifiability": "upper_bound_unresolved" if popt_k[2] >= 79.9 else "not_yet_model_selected",
        "n_effective": effective_sample_size(weights),
        "retained_probability": float(counts.sum() / total),
        "maxwellian_sigma": float(popt_m[1]),
        "kappa_sigma": float(popt_k[1]),
        "error_maxwellian": err_m,
        "error_kappa": err_k,
        "error_tail_maxwellian": err_m_tail,
        "error_tail_kappa": err_k_tail,
        "suprathermal_fraction": supra,
        "tail_bins": int(np.count_nonzero(tail)),
        "v_reliable_over_sigma": v_reliable_over_sigma,
        "hist_x": centers,
        "hist_y": hist,
        "fit_x": np.linspace(-vmax, vmax, 700),
        "maxwellian_params": tuple(float(v) for v in popt_m),
        "kappa_params": tuple(float(v) for v in popt_k),
    }


def particle_heat_flux(snapshot: ParticleSnapshot, species: str = "ion") -> dict:
    """Third central moments per particle, relative to B0 along z.

    A window mean removes only the window bulk flow. Multiply by local density
    and use local velocities/field directions for a spatial heat-flux density.
    """
    mask = _species_mask(snapshot, species)
    if not np.any(mask):
        return {}
    mass = abs(_weighted_mean(snapshot.m[mask], snapshot.w[mask]))
    weights = snapshot.w[mask]
    vx, vy, vz, _ = velocity_from_u(snapshot.px[mask], snapshot.py[mask], snapshot.pz[mask])
    dvx = vx - _weighted_mean(vx, weights)
    dvy = vy - _weighted_mean(vy, weights)
    dvz = vz - _weighted_mean(vz, weights)
    dv2 = dvx**2 + dvy**2 + dvz**2
    qx = 0.5 * mass * _weighted_mean(dv2 * dvx, weights)
    qy = 0.5 * mass * _weighted_mean(dv2 * dvy, weights)
    return {
        "q_parallel_particle": 0.5 * mass * _weighted_mean(dv2 * dvz, weights),
        "q_perp_particle": float(np.hypot(qx, qy)),
        "q_x_particle": qx,
        "q_y_particle": qy,
    }


def particle_energy(snapshot: ParticleSnapshot, species: str = "ion") -> dict:
    mask = _species_mask(snapshot, species)
    if not np.any(mask):
        return {}
    mass = abs(_weighted_mean(snapshot.m[mask], snapshot.w[mask]))
    weights = snapshot.w[mask]
    ux, uy, uz = snapshot.px[mask], snapshot.py[mask], snapshot.pz[mask]
    vx, vy, vz, gamma = velocity_from_u(ux, uy, uz)
    # Exact kinetic energy per particle m(gamma-1) = m u^2/(gamma+1), the
    # quantity DiagEnergies sums; the bulk part uses the mean velocity.
    kinetic = mass * _weighted_mean((ux**2 + uy**2 + uz**2) / (gamma + 1.0), weights)
    bulk = 0.5 * mass * (_weighted_mean(vx, weights) ** 2 + _weighted_mean(vy, weights) ** 2
                         + _weighted_mean(vz, weights) ** 2)
    return {"E_kin_bulk": bulk, "E_kin_thermal": kinetic - bulk}


def plot_validation(rows: list[dict], outdir: Path):
    if not rows:
        return
    first = rows[0]
    # Measured over initialised, so quantities with different units share one
    # dimensionless axis (raw T ~ 1e-2 next to beta ~ 5 made the T bars vanish).
    a0 = TI_PERP / TI_PAR
    labels = [r"$T_{\parallel i}$", r"$T_{\perp i}$", r"$A_i$", r"$R_i$", r"$\beta_{\parallel i}$"]
    ratios = np.array([first["T_parallel_i"] / TI_PAR, first["T_perp_i"] / TI_PERP,
                       first["A_i"] / a0, first["R_i"] * a0, first["beta_parallel_i"] / BETA_I_PAR])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    colors = [ps.c("#58a6ff"), ps.c("#ff7b72"), ps.c("#f2cc60"), ps.c("#d2a8ff"), ps.c("#56d364")]
    ax.bar(labels, ratios - 1.0, bottom=1.0, color=colors, alpha=0.9)
    ax.axhspan(0.98, 1.02, color=ps.MUTED_CLR, alpha=0.12, lw=0, label=r"$\pm 2\,\%$")
    ax.axhline(1.0, color=TEXT_CLR, linestyle=":", alpha=0.6)
    span = max(0.05, 1.3 * float(np.nanmax(np.abs(ratios - 1.0))))
    ax.set_ylim(1.0 - span, 1.0 + span)
    ax.set_title(f"Initial state: measured / initialised — {PROFILE_LABEL}", color=TEXT_CLR,
                 fontsize=14, fontweight="bold")
    ax.set_ylabel(r"measured / profile value ($t = 0$)", color=TEXT_CLR)
    ps.legend(ax, loc="upper right", fontsize=11)
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
    ax.plot(t, a, color=ps.c("#ff7b72"), label=r"$A_i=T_\perp/T_\parallel$", **_series_style(len(t)))
    ax.plot(t, r_inv, color=ps.c("#f2cc60"), label=r"$R_i=T_\parallel/T_\perp$", **_series_style(len(t), "s"))
    ax.axhline(1.0, color=TEXT_CLR, alpha=0.35, linestyle=":")
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel("anisotropy ratio", color=TEXT_CLR)
    ax.set_title("Ion anisotropy evolution", color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _savefig(fig, outdir / "anisotropy_vs_time.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(t, tpar, color=ps.c("#58a6ff"), label=r"$T_{\parallel i}$", **_series_style(len(t)))
    ax.plot(t, tperp, color=ps.c("#ff7b72"), label=r"$T_{\perp i}$", **_series_style(len(t)))
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel("temperature [code]", color=TEXT_CLR)
    ax.set_title("Parallel and perpendicular ion temperature", color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _savefig(fig, outdir / "temperature_parallel_perp_vs_time.png")


def plot_vdf2d(
    snapshot: ParticleSnapshot, outdir: Path, species: str = "ion", min_counts: int = 8,
) -> Path | None:
    """f(v_parallel, v_perp) as a log-density heatmap.

    The (v_parallel, v_perp) domain is a rectangle sized from independent
    percentiles of each axis, but the underlying population is closer to
    isotropic/elliptical: the rectangle's corners (large |v_par| *and* large
    v_perp at once) are almost empty, so with only a handful of raw particles
    landing there the weighted density in those bins is dominated by shot
    noise -- a speckled black/white "broken" patch instead of a smooth falloff.
    Masking bins with too few raw (unweighted) particles removes that without
    touching the physically populated core.
    """
    mask = _species_mask(snapshot, species)
    if not np.any(mask):
        return None
    vx, vy, vz, _ = velocity_from_u(snapshot.px[mask], snapshot.py[mask], snapshot.pz[mask])
    weights = snapshot.w[mask]
    vpar = (vz - _weighted_mean(vz, weights)) / VA
    vx_centered = (vx - _weighted_mean(vx, weights)) / VA
    vy_centered = (vy - _weighted_mean(vy, weights)) / VA
    vperp = np.sqrt(vx_centered**2 + vy_centered**2)
    par_abs = np.nanpercentile(np.abs(vpar), 99.7)
    perp_hi = np.nanpercentile(vperp, 99.7)
    if par_abs <= 0 or perp_hi <= 0:
        return None
    bins = (220, 150)
    hist_range = ((-par_abs, par_abs), (0.0, perp_hi))
    xedges = np.linspace(*hist_range[0], bins[0] + 1)
    yedges = np.linspace(*hist_range[1], bins[1] + 1)
    hist, metadata = cylindrical_density(vpar, vperp, weights, xedges, yedges)
    atomic_json(outdir / f"vdf_2d_{species}_step_{snapshot.step}_metadata.json", metadata)
    counts, _, _ = np.histogram2d(vpar, vperp, bins=bins, range=hist_range)
    if gaussian_filter is not None:
        counts = counts.astype(float)  # raw occupancy; no smoothing across unequal annular volumes
    hist = np.where(counts >= min_counts, hist, np.nan)

    finite_hist = hist[np.isfinite(hist) & (hist > 0)]
    if finite_hist.size == 0:
        return None
    vmin = max(np.nanpercentile(finite_hist, 1.0), np.nanmax(finite_hist) * 1e-5)
    vmax = np.nanmax(finite_hist)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(PANEL_BG)
    pcm = ax.pcolormesh(
        xedges, yedges, hist.T, cmap=cmap,
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
            colors=TEXT_CLR,
            linewidths=0.65,
            alpha=0.6,
        )
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$f(v_\parallel,v_\perp)$ [probability / $(v/v_A)^3$]", fontsize=13, color=TEXT_CLR)
    cb.ax.tick_params(which="both", direction="in", labelsize=12, colors=TEXT_CLR)
    plt.setp(plt.getp(cb.ax, "yticklabels"), color=TEXT_CLR)
    ax.axvline(0.0, color=TEXT_CLR, lw=0.8, ls=":", alpha=0.6)
    # v_perp starts at 0 in the corner where the first v_par label sits.
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, prune="lower"))
    ax.set_xlabel(r"$(v_\parallel-\langle v_\parallel\rangle)/v_A$", fontsize=14, color=TEXT_CLR)
    ax.set_ylabel(r"$v_\perp/v_A$", fontsize=14, color=TEXT_CLR)
    ax.set_title(
        rf"{species.capitalize()} VDF, step {snapshot.step}, $t\Omega_{{ci}}={snapshot.time:.2f}$",
        fontsize=14, fontweight="bold", color=TEXT_CLR,
    )
    path = outdir / f"vdf_2d_{species}_step_{snapshot.step}.png"
    _savefig(fig, path)
    return path


def plot_vdf3d(
    snapshot: ParticleSnapshot, outdir: Path, species: str = "ion", max_points: int = 35_000,
) -> Path | None:
    """3D velocity-space scatter of f(vx, vy, vz), points colored by local
    phase-space density (from a coarse 3D histogram) rather than a flat
    color, and density-weighted subsampling so the plotted cloud still reads
    as a distribution instead of a diffuse haze of equally-likely points."""
    mask = _species_mask(snapshot, species)
    if not np.any(mask):
        return None
    vx, vy, vz, _ = velocity_from_u(snapshot.px[mask], snapshot.py[mask], snapshot.pz[mask])
    weights = snapshot.w[mask]
    vx = (vx - _weighted_mean(vx, weights)) / VA
    vy = (vy - _weighted_mean(vy, weights)) / VA
    vz = (vz - _weighted_mean(vz, weights)) / VA

    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    v_lim = np.nanpercentile(speed, 99.0)
    if not np.isfinite(v_lim) or v_lim <= 0:
        return None
    keep = speed <= v_lim
    vx, vy, vz, weights = vx[keep], vy[keep], vz[keep], weights[keep]
    if vx.size < 50:
        return None

    nbins = 36
    edges = np.linspace(-v_lim, v_lim, nbins + 1)
    density3d, _ = np.histogramdd((vz, vx, vy), bins=(edges, edges, edges), weights=weights)
    if gaussian_filter is not None:
        density3d = gaussian_filter(density3d.astype(float), sigma=1.0)

    iz = np.clip(np.digitize(vz, edges) - 1, 0, nbins - 1)
    ix = np.clip(np.digitize(vx, edges) - 1, 0, nbins - 1)
    iy = np.clip(np.digitize(vy, edges) - 1, 0, nbins - 1)
    point_density = density3d[iz, ix, iy]
    valid = point_density > 0
    if np.count_nonzero(valid) < 50:
        return None
    vx, vy, vz, point_density = vx[valid], vy[valid], vz[valid], point_density[valid]

    if vx.size > max_points:
        prob = point_density / point_density.sum()
        idx = stable_rng(PROFILE_LABEL, snapshot.step, species, "vdf3d-display").choice(vx.size, size=max_points, replace=False, p=prob)
        vx, vy, vz, point_density = vx[idx], vy[idx], vz[idx], point_density[idx]

    order = np.argsort(point_density)  # draw the densest points last (on top)
    vx, vy, vz, point_density = vx[order], vy[order], vz[order], point_density[order]
    log_density = np.log10(point_density)

    fig = plt.figure(figsize=(7.6, 6.8))
    fig.patch.set_facecolor(DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(DARK_BG)
    sc = ax.scatter(
        vz, vx, vy, c=log_density, cmap="magma", s=4, alpha=0.55,
        linewidths=0, depthshade=True,
    )
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-v_lim, v_lim)
    ax.set_ylim(-v_lim, v_lim)
    ax.set_zlim(-v_lim, v_lim)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(plt.MaxNLocator(5))  # dense ticks overlap badly at this view angle
    ax.set_xlabel(r"$v_\parallel/v_A$", color=TEXT_CLR, labelpad=14)
    ax.set_ylabel(r"$v_{x,\perp}/v_A$", color=TEXT_CLR, labelpad=14)
    ax.set_zlabel(r"$v_{y,\perp}/v_A$", color=TEXT_CLR, labelpad=10)
    ax.tick_params(colors=TEXT_CLR, labelsize=10, pad=2)
    ax.view_init(elev=18, azim=-55)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANEL_BG)
        axis.pane.set_alpha(0.6)
        axis._axinfo["grid"]["color"] = GRID_CLR
        axis._axinfo["grid"]["linewidth"] = 0.5
    ax.set_title(
        rf"{species.capitalize()} 3D VDF, step {snapshot.step}, $t\Omega_{{ci}}={snapshot.time:.2f}$",
        color=TEXT_CLR, fontweight="bold", pad=14,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.1)
    cb.set_label(r"$\log_{10} f(v_x,v_y,v_z)$ [PDF]", color=TEXT_CLR)
    cb.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(plt.getp(cb.ax, "yticklabels"), color=TEXT_CLR)
    path = outdir / f"vdf_3d_{species}_step_{snapshot.step}.png"
    # mplot3d's tight-bbox calculation doesn't account for the rotated 3D axis
    # labels, so the default pad_inches clips the x-label off the bottom edge.
    _savefig(fig, path, pad_inches=0.4)
    return path


def plot_fit_metrics(rows: list[dict], outdir: Path):
    if not rows:
        return
    t = np.array([r["omega_ci_t"] for r in rows], dtype=float)
    kfit = np.array([r["kappa_fit"] for r in rows], dtype=float)
    supra = np.array([r["suprathermal_fraction"] for r in rows], dtype=float)

    # 1/kappa, so the Maxwellian limit is 0 instead of an arbitrary large
    # number; fits pinned at the upper bound are drawn open and labelled.
    at_bound = kfit >= 0.99 * KAPPA_FIT_MAX
    inv = np.where(at_bound, 0.0, 1.0 / kfit)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    # Conditional fit error propagated to 1/kappa: sigma(1/k) = sigma_k / k^2.
    err = np.array([r.get("kappa_fit_stderr_conditional", np.nan) for r in rows], dtype=float)
    inv_err = np.where(at_bound | ~np.isfinite(err), np.nan, err / kfit ** 2)
    ax.plot(t, inv, color=ps.c("#d2a8ff"), lw=1.2, alpha=0.6)
    ax.errorbar(t[~at_bound], inv[~at_bound], yerr=inv_err[~at_bound], fmt="o", capsize=3,
                color=ps.c("#d2a8ff"), label=r"fitted ($\pm 1\sigma$, conditional)")
    if at_bound.any():
        ax.plot(t[at_bound], inv[at_bound], "o", mfc="none", color=ps.c("#d2a8ff"),
                label=rf"at fit bound $\kappa={KAPPA_FIT_MAX:g}$: Maxwellian-consistent")
    if KAPPA:
        ax.axhline(1.0 / KAPPA, color=ps.c("#f2cc60"), alpha=0.8, linestyle="--",
                   label=rf"initial $\kappa_0={KAPPA:g}$")
    ax.axhline(0.0, color=ps.MUTED_CLR, lw=0.8)
    ax.set_ylim(bottom=-0.02)
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel(r"$1/\kappa_{\rm fit}$  (0 = Maxwellian)", color=TEXT_CLR)
    ax.set_title("Fitted kappa index of the window VDF", color=TEXT_CLR, fontweight="bold")
    ps.legend(ax, loc="best", fontsize=10)
    _savefig(fig, outdir / "kappa_fit_vs_time.png")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(t, supra, color=ps.c("#ff7b72"), **_series_style(len(t)))
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel(r"$P(|u_\parallel-\langle u_\parallel\rangle|>3\sigma_u)$", color=TEXT_CLR)
    ax.set_title("Suprathermal fraction vs time", color=TEXT_CLR, fontweight="bold")
    _savefig(fig, outdir / "suprathermal_fraction_vs_time.png")


def plot_distribution_fit(fit: dict, outdir: Path):
    if not fit:
        return
    x = fit["fit_x"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.step(fit["hist_x"], fit["hist_y"], where="mid", color=ps.c("#ff7b72"), lw=1.4, label="simulation")
    ax.plot(x, maxwellian_pdf(x, *fit["maxwellian_params"]), "--", color=ps.c("#58a6ff"), lw=2.0,
            label="Maxwellian fit")
    ax.plot(x, kappa_pdf_shape(x, *fit["kappa_params"]), "-", color=ps.c("#d2a8ff"), lw=2.0,
            label=(rf"Kappa fit at its bound $\kappa={KAPPA_FIT_MAX:g}$ (Maxwellian-consistent)"
                   if fit["kappa_fit"] >= 0.99 * KAPPA_FIT_MAX else rf"Kappa fit, $\kappa={fit['kappa_fit']:.2f}$"))
    ax.set_yscale("log")
    ax.set_xlabel(r"$u_\parallel-\langle u_\parallel\rangle$", color=TEXT_CLR)
    ax.set_ylabel("PDF", color=TEXT_CLR)
    ax.set_title(rf"Kappa vs Maxwellian — $t\Omega_{{ci}} = {step_to_omegaci(fit['step']):.1f}$",
                 color=TEXT_CLR, fontweight="bold")
    # Under the axes: the curves span the whole panel, tails included.
    ps.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=10, frameon=False)
    _savefig(fig, outdir / f"kappa_vs_maxwellian_step_{fit['step']}.png")


def plot_map(
    field: np.ndarray, path: Path, title: str, label: str, cmap="viridis", symmetric=False,
    smooth_sigma: float | None = None, prt_window: tuple | None = None,
):
    """``smooth_sigma`` applies a display-only Gaussian smoothing (matching the
    sigma=3.0 already used for the J_dia maps in compute_jdia): single-cell
    moment quantities like T_par/T_perp/A_i/q_par/q_perp are dominated by
    per-cell particle-counting shot noise without it -- the map otherwise
    renders as uniform salt-and-pepper noise with no visible physical
    structure. Only the plotted copy is smoothed; callers that also feed
    ``field`` into scalar statistics (mean/percentiles) or correlations pass
    the raw array there unchanged."""
    arr = np.asarray(field, dtype=float)
    if smooth_sigma is not None and gaussian_filter is not None:
        finite = np.isfinite(arr)
        if np.any(finite):
            filled = np.where(finite, arr, np.nanmedian(arr[finite]))
            arr = gaussian_filter(filled, sigma=smooth_sigma)
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
    if prt_window is not None:
        # Without this box the map does not say which part of the domain the
        # particles come from, and the VDF floats over structures that may sit
        # outside the saved region entirely.
        ext = prt_window_extent_di(prt_window[0], prt_window[1])
        z0, z1 = ext["z_di"]
        y0, y1 = ext["y_di"]
        ax.add_patch(Rectangle((z0, y0), z1 - z0, y1 - y0, fill=False,
                               edgecolor=ps.c("#00c000"), lw=2.0, ls="--",
                               label="prt window (particles for the VDFs)"))
        # Under the map, where it hides no data.
        ps.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=10, frameon=False)
    ps.spatial_axes(ax, color=TEXT_CLR)
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
    ax.fill_between(t, p10, p90, color=ps.c("#ff7b72"), alpha=0.22, label="P10-P90")
    ax.plot(t, mean_a, color=ps.c("#ff7b72"), label=r"$\langle A_i\rangle$", **_series_style(len(t)))
    ax.axhline(1.0, color=TEXT_CLR, linestyle=":", alpha=0.35)
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel(r"$A_i(x,y)$", color=TEXT_CLR)
    ax.set_title("Spatial anisotropy from moments", color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _savefig(fig, outdir / "brazil_plot_time_evolution.png")


def growth_rate(time: np.ndarray, delta_b: np.ndarray,
                amp_lo: float = 0.1, amp_hi: float = 0.9,
                t_start: float | None = None, t_end: float | None = None) -> dict:
    """Fit gamma on the exponential segment of ln(delta_b).

    Thin wrapper over growth_fit.fit_exponential_growth, the single linear-
    phase fit of the pipeline (amplitude-band window from the noise floor to
    saturation, or an explicit [t_start, t_end]). Returns {} when the series
    is too short; otherwise the keys below, including ``gamma_err`` (slope
    standard error combined with the window sensitivity) and ``fit_ok``.
    """
    t = np.asarray(time, dtype=float)
    a = np.asarray(delta_b, dtype=float)
    keep = np.isfinite(t) & (t > 0.0)
    fit = fit_exponential_growth(t[keep], a[keep], t_start=t_start, t_end=t_end,
                                 band=(amp_lo, amp_hi),
                                 min_gain=MIN_AMPLITUDE_GAIN,
                                 max_onset=None if t_start is not None else MAX_ONSET_T0)
    if fit["fit_time_range"] is None:
        return {}
    return {
        "gamma": fit["gamma"],
        "gamma_stderr": fit["gamma_stderr"],
        "gamma_window_spread": fit["gamma_window_spread"],
        "gamma_hac_stderr": fit.get("gamma_hac_stderr", np.nan),
        "gamma_hac_lags": fit.get("gamma_hac_lags", 0),
        "gamma_err": fit["gamma_err"],
        "intercept": fit["intercept"],
        "linear_phase_start": fit["linear_phase_start"],
        "linear_phase_end": fit["linear_phase_end"],
        "r_squared": fit["r_squared"],
        "amplitude_gain": fit["amplitude_gain"],
        "series_start": fit["series_start"],
        "window_source": fit["window_source"],
        "fit_ok": fit["fit_ok"],
        "fit_reject_reason": fit["fit_reject_reason"],
        "time": fit["time"],
        "ln_delta_b": fit["ln_amplitude"],
        "fit_time": fit["fit_time"],
        "fit_ln_delta_b": fit["fit_ln_amplitude"],
    }


def plot_field_time(rows: list[dict], outdir: Path):
    if not rows:
        return
    t = np.array([r["omega_ci_t"] for r in rows], dtype=float)
    rms = np.array([r.get("delta_B_vec_rms_over_B0", r["delta_B_rms_over_B0"]) for r in rows], dtype=float)
    par = np.array([r["delta_B_parallel_rms_over_B0"] for r in rows], dtype=float)
    perp = np.array([r["delta_B_perp_rms_over_B0"] for r in rows], dtype=float)
    depth = np.array([r["mirror_depth"] for r in rows], dtype=float)
    plot_mask = np.isfinite(t) & (t > 0.0)
    t, rms, par, perp, depth = (
        arr[plot_mask] for arr in (t, rms, par, perp, depth)
    )
    if len(t) == 0:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(t, rms, color=ps.c("#58a6ff"), **_series_style(len(t)))
    ax.set_yscale("log")
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel(r"$\langle|\delta\mathbf{B}|^2\rangle^{1/2}/B_0$", color=TEXT_CLR)
    ax.set_title("Magnetic fluctuation growth", color=TEXT_CLR, fontweight="bold")
    _savefig_many(fig, [outdir / "deltaB_rms_vs_time.png"])

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(t, par, color=ps.c("#ff7b72"), label=r"$\delta B_\parallel/B_0$", **_series_style(len(t)))
    ax.plot(t, perp, color=ps.c("#56d364"), label=r"$\delta B_\perp/B_0$", **_series_style(len(t), "s"))
    ax.set_yscale("log")
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel("RMS fluctuation", color=TEXT_CLR)
    ax.set_title("Parallel vs perpendicular magnetic fluctuations", color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _savefig_many(fig, [outdir / "deltaB_components_comparison.png"])

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    ax.plot(t, depth, color=ps.c("#d2a8ff"), **_series_style(len(t)))
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel(r"$1 - \min|B|/B_0$", color=TEXT_CLR)
    ax.set_title("Deepest magnetic depression (single-cell extreme)", color=TEXT_CLR, fontweight="bold")
    ax.text(0.02, 0.96, "hole populations, areas and thresholds: 07_structures",
            transform=ax.transAxes, va="top", fontsize=10, color=ps.MUTED_CLR)
    _savefig_many(fig, [outdir / "mirror_depth_vs_time.png"])

    plot_prt_window_field_time(rows, outdir)


def plot_prt_window_field_time(rows: list[dict], outdir: Path):
    """Evolución de B dentro de la ventana prt, comparada con el dominio.

    Es la serie que hace falta para leer las VDF: cada VDF sale de esta
    ventana, así que la pregunta "¿la VDF de este paso se tomó dentro de un
    hueco magnético?" se responde con estas curvas, no con las globales. Si la
    ventana y el dominio se separan, la region guardada no es representativa y
    cualquier conclusión VDF-hueco vale sólo para esa caja.
    """
    if not rows or "prt_delta_B_rms_over_B0" not in rows[0]:
        return
    t = np.array([r["omega_ci_t"] for r in rows], dtype=float)
    mask = np.isfinite(t) & (t > 0.0)

    def _col(key):
        return np.array([r.get(key, np.nan) for r in rows], dtype=float)[mask]

    t = t[mask]
    if len(t) == 0:
        return
    w_rms, g_rms = _col("prt_delta_B_rms_over_B0"), _col("delta_B_rms_over_B0")
    w_min, w_max = _col("prt_B_min_over_B0"), _col("prt_B_max_over_B0")
    w_mean = _col("prt_B_mean_over_B0")
    w_par, w_perp = (_col("prt_delta_B_parallel_rms_over_B0"),
                     _col("prt_delta_B_perp_rms_over_B0"))

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 11.0), sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        _style_axes(ax)

    axes[0].plot(t, w_rms, color=ps.c("#56d364"), label="prt window (VDF)",
                 **_series_style(len(t)))
    axes[0].plot(t, g_rms, color=ps.c("#58a6ff"), label="full domain",
                 **_series_style(len(t), "s"))
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\delta B_{\rm rms}/B_0$", color=TEXT_CLR)
    axes[0].set_title("Magnetic fluctuation: saved region vs full domain",
                      color=TEXT_CLR, fontweight="bold")

    # min/max delimitan el hueco y el pico dentro de la caja: son los dos
    # extremos que la VDF puede estar muestreando en cada snapshot.
    axes[1].fill_between(t, w_min, w_max, color=ps.c("#d2a8ff"), alpha=0.22,
                         label=r"$[\min,\max]$ range in the window")
    axes[1].plot(t, w_mean, color=ps.c("#d2a8ff"), label=r"$\langle|B|\rangle/B_0$",
                 **_series_style(len(t)))
    axes[1].axhline(1.0, color=GRID_CLR, ls=":", lw=1.0)
    axes[1].set_ylabel(r"$|B|/B_0$ in the prt window", color=TEXT_CLR)

    axes[2].plot(t, w_par, color=ps.c("#ff7b72"), label=r"$\delta B_\parallel/B_0$",
                 **_series_style(len(t)))
    axes[2].plot(t, w_perp, color=ps.c("#f2cc60"), label=r"$\delta B_\perp/B_0$",
                 **_series_style(len(t), "s"))
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    axes[2].set_ylabel("RMS in the prt window", color=TEXT_CLR)

    for ax in axes:
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
                  fontsize=10)
    fig.tight_layout()
    _savefig_many(fig, [outdir / "prt_window_B_vs_time.png"])


def plot_growth(growth: dict, outdir: Path):
    if not growth:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    series = growth.get("series_label", r"|\delta\mathbf{B}|_{\rm rms}")
    shown = ps.measured_fluctuation(growth["time"])
    ax.plot(np.asarray(growth["time"])[shown], np.asarray(growth["ln_delta_b"])[shown],
            color=ps.c("#58a6ff"), label=rf"$\ln {series}$", **_series_style(int(shown.sum())))
    err = growth.get("gamma_err", float("nan"))
    err_txt = rf"\pm{err:.2g}" if np.isfinite(err) else ""
    status = "" if growth.get("fit_ok") else " (not valid)"
    ax.plot(growth["fit_time"], growth["fit_ln_delta_b"], "--", color=ps.c("#ff7b72"),
            label=rf"$\gamma={growth['gamma']:.4g}{err_txt}\,\Omega_{{ci}}${status}")
    ax.axvspan(growth["linear_phase_start"], growth["linear_phase_end"],
               color=ps.c("#ff7b72"), alpha=0.12)
    ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
    ax.set_ylabel(rf"$\ln {series}$", color=TEXT_CLR)
    ax.set_title(growth.get("title", "Linear growth-rate fit"), color=TEXT_CLR, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    _savefig(fig, outdir / growth.get("figure_name", "growth_rate_fit.png"))


#: Smoothing of P_perp and B before differentiating, in d_i. PIC shot noise
#: dominates any gradient taken below the ion scales; the same value is the
#: default of diamagnetic_current.py so both scripts show the same current.
JDIA_SIGMA_DI = 0.5


def compute_jdia(moment_file: str, field_file: str) -> dict[str, np.ndarray]:
    """Out-of-plane diamagnetic current J_x = (B x grad P_perp)_x / B^2 per species.

    P_perp is the thermal pressure (bulk flow removed) projected on the local
    field, and the gradient is taken in code lengths (d_e), so the result is
    in code current-density units. The previous version used the raw second
    moment, B0 along z for the projection, per-cell gradients and the
    opposite sign to diamagnetic_current.py.
    """
    fld = load_fields(field_file)
    bx, by, bz = (PICDataReader.flatten_2d_slice(fld[k]).astype(float) for k in ("Bx", "By", "Bz"))
    sigma_cells = JDIA_SIGMA_DI * DI / DX_DE
    smooth = ((lambda a: gaussian_filter(a, sigma=sigma_cells, mode="wrap"))
              if gaussian_filter is not None else (lambda a: a))
    bx_s, by_s, bz_s = smooth(bx), smooth(by), smooth(bz)
    out = {}
    for suffix, mass in (("i", M_ION), ("e", M_ELEC)):
        tensor = central_pressure_tensor(load_moments(moment_file, suffix), suffix, mass)
        _, pperp, _ = field_aligned_pressures(
            tensor["Pxx"], tensor["Pyy"], tensor["Pzz"],
            tensor["Pxy"], tensor["Pyz"], tensor["Pzx"], bx, by, bz)
        pperp = np.nan_to_num(pperp, nan=float(np.nanmean(pperp)))
        out[f"J_dia_{suffix}"] = diamagnetic_current_x(smooth(pperp), by_s, bz_s, DX_DE, DX_DE)
    out["J_dia_total"] = out["J_dia_i"] + out["J_dia_e"]
    return out


def secular_heating(rows: list[dict], anisotropy_rows: list[dict] | None = None) -> dict:
    """Describe an electron-energy trend without assigning a numerical cause.

    Linearity and isotropy do not distinguish physical from numerical heating.
    A control run and convergence evidence are needed for that attribution.
    """
    t = np.array([r.get("omega_ci_t", np.nan) for r in rows], dtype=float)
    e_e = np.array([r.get("E_internal_e", np.nan) for r in rows], dtype=float)
    valid = np.isfinite(t) & np.isfinite(e_e)
    if np.count_nonzero(valid) < 4:
        return {}
    t, e_e = t[valid], e_e[valid]
    if np.ptp(t) <= 0:
        return {}

    slope, intercept = np.polyfit(t, e_e, 1)
    model = slope * t + intercept
    ss_res = float(np.sum((e_e - model) ** 2))
    ss_tot = float(np.sum((e_e - np.mean(e_e)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    a_e = np.array([r.get("A_e", np.nan) for r in (anisotropy_rows or rows)],
                   dtype=float)
    a_e_mean = float(np.nanmean(a_e)) if np.any(np.isfinite(a_e)) else float("nan")

    linear = np.isfinite(r2) and r2 > 0.9
    isotropic = np.isfinite(a_e_mean) and abs(a_e_mean - 1.0) < 0.1
    # A high R^2 does not make the trend linear: the heating of the delivered
    # runs accelerates (energy_audit.py reports its curvature and budget).
    if linear and isotropic:
        verdict = "isotropic trend, linear fit R2>0.9; physical/numerical origin not determined"
    elif linear:
        verdict = "linear fit R2>0.9; physical/numerical origin not determined"
    else:
        verdict = "linear fit inadequate; physical/numerical origin not determined"

    e_e0 = float(intercept)
    return {
        "slope_per_omegaci": float(slope),
        "intercept": e_e0,
        "r_squared": float(r2),
        "A_e_mean": a_e_mean,
        "n_points": int(t.size),
        "t_start": float(t[0]),
        "t_end": float(t[-1]),
        "fitted_change_at_end": float(slope * (t[-1] - t[0])),
        "relative_to_E_e0": float(slope * (t[-1] - t[0]) / e_e0)
        if e_e0 not in (0.0,) and np.isfinite(e_e0) else float("nan"),
        "verdict": verdict,
    }


def correlations(a_map: np.ndarray, delta_b: np.ndarray, bmag: np.ndarray,
                 jdia: np.ndarray, rho: np.ndarray, step: int,
                 b_perp: np.ndarray | None = None,
                 db_par: np.ndarray | None = None) -> dict:
    """Correlaciones espaciales entre la anisotropía y el campo local.

    `corr_A_deltaB` solía calcularse contra delta_B = |B| - B0. Como B0 es una
    constante, el coeficiente de Pearson es invariante ante ese corrimiento y
    la columna resultaba idéntica a `corr_A_Bmag` dígito a dígito en las cinco
    corridas: no medía la fluctuación, medía |B| otra vez. Se sustituye por
    medidas que sí son independientes de |B|:

      - `corr_A_dB_abs`  : amplitud de la fluctuación, ||B| - B0|. Responde a
        "¿la anisotropía es mayor donde el campo está más perturbado?",
        sin importar el signo.
      - `corr_A_dB_parallel` : componente compresiva (B_par - B0)/B0, la firma
        del modo mirror.
      - `corr_A_B_perp`  : amplitud transversal, la firma de firehose/EMIC.
    """
    def corr(x, y):
        if x is None or y is None:
            return float("nan")
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(valid) < 3:
            return float("nan")
        return float(np.corrcoef(x[valid], y[valid])[0, 1])

    return {
        "step": step,
        "omega_ci_t": step_to_omegaci(step),
        "corr_A_Bmag": corr(a_map, bmag),
        "corr_A_dB_abs": corr(a_map, np.abs(np.asarray(delta_b, dtype=float))),
        "corr_A_dB_parallel": corr(a_map, db_par),
        "corr_A_B_perp": corr(a_map, b_perp),
        "corr_A_Jdia": corr(a_map, jdia),
        "corr_A_n": corr(a_map, rho),
    }


def plot_scatter_series(series: list[tuple[float, np.ndarray, np.ndarray]], path: Path,
                        xlabel: str, ylabel: str, title: str, max_points: int = 40_000):
    """Cell-by-cell scatter of every map snapshot, coloured by time.

    Each snapshot is thinned to at most `max_points` cells with a fixed stride
    (deterministic, independent of the other snapshots).
    """
    from matplotlib import ticker
    usable = []
    for toci, x, y in series:
        x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
        idx = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
        if idx.size >= 3:
            usable.append((toci, x[idx[::max(1, idx.size // max_points)]], y[idx[::max(1, idx.size // max_points)]]))
    if not usable:
        return
    times = [t for t, _, _ in usable]
    norm = matplotlib.colors.Normalize(min(times), max(times) if max(times) > min(times) else min(times) + 1)
    cmap = plt.get_cmap(ps.CMAP_SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    fig.patch.set_facecolor(DARK_BG)
    _style_axes(ax)
    for toci, x, y in usable:
        ax.scatter(x, y, s=2, alpha=0.25, color=cmap(norm(toci)), rasterized=True)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(mappable, ax=ax, pad=0.02)
    cb.set_label(r"$t\Omega_{ci}$", color=TEXT_CLR)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.set_xlabel(xlabel, color=TEXT_CLR)
    ax.set_ylabel(ylabel, color=TEXT_CLR)
    ax.set_title(title, color=TEXT_CLR, fontweight="bold")
    _savefig(fig, path)

def _process_particle_step_worker(args):
    """Worker function for run_particles to process a single step in parallel."""
    step, filepath, max_particles, outdir, want_vdf = args
    try:
        snap = _read_particle_snapshot(filepath, max_particles)
        if snap is None:
            return step, None, [], None

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
                "R_e": elec.get("R", np.nan),
                "beta_parallel_e": 2.0 * elec.get("T_parallel", np.nan) / (B0**2 + 1e-30),
            }
            row.update(particle_heat_flux(snap, "ion"))
            row.update(particle_energy(snap, "ion"))
            for species, suffix in [("ion", "i"), ("electron", "e")]:
                row.update({f"{key}_{suffix}": value for key, value in
                            particle_kinematics_validity(snap, species).items()})

        # Both species, both a 2D reduced (v_par,v_perp) heatmap and a true 3D
        # (vx,vy,vz) scatter -- but only on the steps selected for the VDF
        # cadence; every other selected step still contributes its row above.
        vdf_paths = []
        if want_vdf:
            vdf_paths = [
                path
                for species in ("ion", "electron")
                for path in (plot_vdf2d(snap, outdir, species), plot_vdf3d(snap, outdir, species))
                if path is not None
            ]
        fit = fit_distribution(snap, "ion")

        return step, row, vdf_paths, fit
    except Exception as exc:
        print(f"\n[ERROR] Step {step} processing failed: {exc}")
        return step, None, [], None


def _field_metrics_worker(args):
    step, field_file, b0, prt_window, modes = args
    metrics = field_metrics(field_file, b0, prt_window, modes)
    return step, {
        "step": step,
        "omega_ci_t": step_to_omegaci(step),
        **{k: v for k, v in metrics.items() if np.isscalar(v)},
    }, metrics["mode_power"]


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
        # Only an accepted fit yields a citable index (spectral_analysis._fit_power_law).
        "power_law_slope": fit["slope"] if fit and fit.get("accepted") else np.nan,
        "power_law_rvalue": fit["rvalue"] if fit and fit.get("accepted") else np.nan,
        "power_law_status": "no fit" if fit is None else ("accepted" if fit["accepted"] else fit["reason"]),
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
    return step, correlations(
        maps["A"],
        PICDataReader.flatten_2d_slice(fmet["delta_B"]),
        PICDataReader.flatten_2d_slice(fmet["B_magnitude"]),
        jdia["J_dia_total"],
        maps["n"],
        step,
        b_perp=PICDataReader.flatten_2d_slice(fmet["B_perp_map"]),
        db_par=PICDataReader.flatten_2d_slice(fmet["delta_B_parallel_map"]),
    )


def _field_map_worker(args):
    step, field_file, outdir, prt_window = args
    metrics = field_metrics(field_file, B0)
    toci = step_to_omegaci(step)
    # Per-cell counting noise dominates these maps at full resolution: without
    # the display-only smoothing the mirror structures render as salt-and-
    # pepper. Only the plotted copy is smoothed; every scalar in the tables
    # comes from the raw array.
    plot_map(PICDataReader.flatten_2d_slice(metrics["delta_B_over_B0"]),
             outdir / f"deltaB_map_step_{step}.png",
             rf"$\delta B/B_0$ — $t\Omega_{{ci}} = {toci:.1f}$", r"$\delta B/B_0$",
             cmap=ps.CMAP_DIVERGING, symmetric=True, smooth_sigma=2.0,
             prt_window=prt_window)
    plot_map(PICDataReader.flatten_2d_slice(metrics["B_magnitude"]),
             outdir / f"mirror_holes_map_step_{step}.png",
             rf"$|B|$ mirror structures — $t\Omega_{{ci}} = {toci:.1f}$", r"$|B|$",
             cmap="magma", smooth_sigma=2.0, prt_window=prt_window)
    return step, True


def _thermal_map_worker(args):
    step, moment_file, field_file, species, outdir = args
    maps = moment_thermal_maps(moment_file, field_file, species)
    s = "i" if species == "ion" else "e"
    toci = step_to_omegaci(step)
    plot_map(maps["T_parallel"], outdir / f"T_parallel_{s}_map_step_{step}.png",
             rf"$T_{{\parallel {s}}}$ — $t\Omega_{{ci}} = {toci:.1f}$", rf"$T_{{\parallel {s}}}$",
             cmap=ps.CMAP_SEQUENTIAL, smooth_sigma=3.0)
    plot_map(maps["T_perp"], outdir / f"T_perp_{s}_map_step_{step}.png",
             rf"$T_{{\perp {s}}}$ — $t\Omega_{{ci}} = {toci:.1f}$", rf"$T_{{\perp {s}}}$",
             cmap=ps.CMAP_SEQUENTIAL, smooth_sigma=3.0)
    plot_map(maps["A"], outdir / f"A_{s}_map_step_{step}.png",
             rf"$A_{s}=T_{{\perp {s}}}/T_{{\parallel {s}}}$ — $t\Omega_{{ci}} = {toci:.1f}$",
             rf"$A_{s}$", cmap=ps.CMAP_DIVERGING, smooth_sigma=3.0)
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
        max_particle_steps: int | None,
        max_map_steps: int,
        selected_steps: list[int] | None,
        jobs: int = 0,
        vdf_cadence_omegaci: float = 10.0,
        growth_window: tuple[float | None, float | None] = (None, None),
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.growth_window = growth_window
        self.outdir = Path(outdir)
        self.max_particles = max_particles
        self.max_particle_steps = max_particle_steps
        self.max_map_steps = max_map_steps
        self.selected_steps = selected_steps
        self.jobs = jobs
        self.vdf_cadence_omegaci = vdf_cadence_omegaci
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
        energy_files = sorted(self.data_dir.glob("diag*.asc"))
        if energy_files:
            from energy_conservation import write_energy_analysis
            try:
                write_energy_analysis(energy_files, self.outdir)
            except ValueError as exc:
                print(f"[WARN] Global energy diagnostic unavailable: {exc}")
        print(f"Physical diagnostics written to {self.outdir}")
        if not (particle_rows or field_rows or spatial_rows):
            print("[WARN] No diagnostics were generated; check input file patterns.")

    def run_particles(self) -> list[dict]:
        if not self.particle_files:
            print("[INFO] No particle files found; skipping particle diagnostics.")
            return []

        steps = sorted(self.particle_files)
        # Every available (or --steps-selected/--max-particle-steps-capped) snapshot
        # feeds the temperature/anisotropy/energy/fit time series -- that resolution
        # is independent of how often we want a VDF frame. Only the VDF plots
        # themselves are thinned to the requested Omega_ci*t cadence; a run short
        # enough to yield one VDF frame at that cadence should not also collapse
        # its temperature-vs-time curves to a single point.
        selected = (
            [s for s in steps if s in set(self.selected_steps)] if self.selected_steps
            else _select_steps(steps, None, self.max_particle_steps)
            if self.max_particle_steps is not None else steps
        )
        vdf_steps = set(_select_steps_by_cadence(
            steps, self.selected_steps, self.vdf_cadence_omegaci, step_to_omegaci,
        ))
        print(
            f"[INFO] {len(selected)} particle snapshot(s) selected for time-series diagnostics "
            f"(of {len(steps)} available); {len(vdf_steps & set(selected))} of those get a VDF frame "
            f"at ~{self.vdf_cadence_omegaci:g} Omega_ci*t cadence."
        )
        rows = []
        fit_rows = []
        vdf_paths = []
        first_fit = None

        tasks = [
            (step, self.particle_files[step], self.max_particles, self.outdir, step in vdf_steps)
            for step in selected
        ]
        results = _run_step_tasks(
            _process_particle_step_worker, tasks, self.jobs, "Particle diagnostics"
        )
        for step, row, paths, fit in results:
            if row:
                rows.append(row)
            if paths:
                vdf_paths.extend(paths)
            if fit:
                if first_fit is None:
                    first_fit = fit
                fit_rows.append({
                    **{key: value for key, value in fit.items() if key.startswith("heldout_") or key == "mixture_ambiguity"},
                    **{key: fit[key] for key in [
                        "step", "omega_ci_t", "kappa_fit", "maxwellian_sigma",
                        "kappa_sigma", "error_maxwellian", "error_kappa",
                        "error_tail_maxwellian", "error_tail_kappa",
                        "suprathermal_fraction",
                        # Hasta dónde llegó la estadística útil: sin esto no se
                        # puede saber si dos casos compararon la misma cola.
                        "tail_bins", "v_reliable_over_sigma", "fit_coordinate", "fit_frame",
                        "fit_objective", "fit_solver", "fit_fallback_reason", "kappa_identifiability",
                        "n_effective", "retained_probability", "kappa_fit_stderr_conditional", "model_validation_status",
                    ]}
                })

        # anisotropy_table.csv is the single canonical per-step table; the
        # initial-condition check lives in validation_summary.txt.
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
            expected.append("whistler (electron-driven): expected A_e > 1 and isotropic ions")
            ok = row.get("A_e", np.nan) > 1.0 and abs(row["A_i"] - 1.0) < 0.15
        text = [
            f"Profile: {PROFILE_LABEL}",
            f"Instability: {INSTABILITY}",
            f"First available step: {row['step']}",
            f"T_parallel_i = {row['T_parallel_i']:.8g}",
            f"T_perp_i     = {row['T_perp_i']:.8g}",
            f"A_i          = {row['A_i']:.8g}",
            f"R_i          = {row['R_i']:.8g}",
            f"beta_parallel_i = {row['beta_parallel_i']:.8g}",
            f"T_parallel_e = {row.get('T_parallel_e', np.nan):.8g}",
            f"T_perp_e     = {row.get('T_perp_e', np.nan):.8g}",
            f"A_e          = {row.get('A_e', np.nan):.8g}",
            f"beta_parallel_e = {row.get('beta_parallel_e', np.nan):.8g}",
            "",
            *expected,
            f"Initial check: {'NOT_INITIAL (step 0 missing)' if row['step'] != 0 else 'PASS' if ok else 'CHECK'}",
        ]
        (self.outdir / "validation_summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")

    def run_fields(self) -> list[dict]:
        if not self.field_files:
            print("[INFO] No field files found; skipping magnetic diagnostics.")
            return []
        steps = sorted(self.field_files)
        window = self.prt_window()
        try:
            modes = mode_candidates(self.field_files, K_MAX_DI_DEFAULT)
        except Exception as exc:  # the rms fits below still run
            print(f"  [WARN] dominant-mode candidates unavailable: {exc}")
            modes = []
        mode_index = [(m["i0"], m["i1"]) for m in modes]
        tasks = [(step, self.field_files[step], B0, window, mode_index) for step in steps]
        results = _run_step_tasks(_field_metrics_worker, tasks, self.jobs, "Field diagnostics")
        rows = [row for _, row, _ in results]
        _write_csv(self.outdir / "field_fluctuation_table.csv", rows)
        plot_field_time(rows, self.outdir)
        t = np.array([r["omega_ci_t"] for r in rows])
        summary_rows = []
        fit_keys = ("gamma", "gamma_err", "gamma_stderr", "gamma_window_spread",
                    "linear_phase_start", "linear_phase_end", "r_squared",
                    "amplitude_gain", "series_start", "window_source",
                    "fit_ok", "fit_reject_reason")

        # 1. Reference: the dominant Fourier mode of dB (see mode_candidates),
        # then the strongest mode of each branch (see classify_mode).
        reference, branches = None, []
        if modes and len(results) == len(steps):
            power = np.clip(np.array([p for _, _, p in results], dtype=float), 0.0, None)
            reference, branches = self.fit_mode_growth(t, modes, power)
        if reference is not None:
            summary_rows.append({"series": "mode", "amplitude": reference["amplitude_name"],
                                 **{key: reference[key] for key in fit_keys}})
            if not reference["fit_ok"]:
                print(f"  [WARN] dominant-mode growth rate NOT valid: "
                      f"{reference['fit_reject_reason']}")
        for branch in branches:
            summary_rows.append({"series": BRANCH_SERIES[branch["branch"]],
                                 "amplitude": branch["amplitude_name"],
                                 **{key: branch[key] for key in fit_keys}})

        # 2. The domain rms of dB and of its compressive / transverse parts:
        # which one grows is part of the mode identification. They are fitted
        # on the reference linear phase, so the three numbers and the mode
        # gamma describe the same time interval; the rms includes the noise
        # of every k and comes out lower than the mode gamma.
        t_start, t_end = self.growth_window
        shared = (t_start is None and t_end is None and reference is not None
                  and reference["fit_ok"])
        if shared:
            t_start, t_end = reference["linear_phase_start"], reference["linear_phase_end"]
        for series, column, label, figure in (
            ("total", "delta_B_vec_rms", r"|\delta\mathbf{B}|_{\rm rms}", "growth_rate_fit.png"),
            ("parallel", "delta_B_parallel_rms", r"\delta B_{\parallel,\rm rms}", "growth_rate_fit_parallel.png"),
            ("perp", "delta_B_perp_rms", r"\delta B_{\perp,\rm rms}", "growth_rate_fit_perp.png"),
        ):
            growth = growth_rate(t, np.array([r[column] for r in rows]),
                                 t_start=t_start, t_end=t_end)
            if not growth:
                continue
            if shared:
                growth["window_source"] = "dominant-mode"
            growth.update(series_label=label, figure_name=figure)
            plot_growth(growth, self.outdir)
            summary_rows.append({"series": series, "amplitude": column,
                                 **{key: growth[key] for key in fit_keys}})
        _write_csv(self.outdir / "growth_rate_summary.csv", summary_rows)
        self.plot_field_maps()
        self.run_magnetic_spectra()
        return rows

    def fit_mode_growth(self, t: np.ndarray, modes: list[dict],
                        power: np.ndarray) -> tuple[dict | None, list[dict]]:
        """gamma of every followed mode: the dominant one and one per branch.

        ``power`` is (n_steps, 2, n_modes), the transverse and compressive
        <|dB|^2> of each mode (mode_power with ``split``). The dominant mode is
        the one with the largest amplitude over the run. Every mode is fitted
        on its own linear phase, classified by theta_kB and compressibility
        (classify_mode) and tabulated in mode_growth_table.csv, so a faster but
        weaker mode stays visible. Returns the dominant fit and, for each
        geometric branch that has at least one mode, the fit of its strongest
        mode: with competing mirror and ion-cyclotron instabilities the
        dominant-mode gamma is the gamma of whichever branch won, not the
        mirror growth rate.
        """
        t_start, t_end = self.growth_window
        total = power.sum(axis=1)
        amplitude = np.sqrt(total)
        anisotropy = TI_PERP / TI_PAR
        table, fits = [], []
        for j, mode in enumerate(modes):
            settled = t >= noise_settling_time(mode["k_di"])
            fit = growth_rate(t[settled], amplitude[settled, j], t_start=t_start, t_end=t_end)
            fits.append(fit)
            # Compressibility of the mode where it is measured as a linear
            # mode (the accepted fit window), otherwise power-weighted over
            # the run; saturation mixes branches through nonlinear coupling.
            window = np.ones(t.size, dtype=bool)
            source = "whole run"
            if fit and fit.get("fit_ok"):
                inside = (t >= fit["linear_phase_start"]) & (t <= fit["linear_phase_end"])
                if inside.any():
                    window, source = inside, "linear phase"
            weight = total[window, j].sum()
            compressibility = (float(power[window, 1, j].sum() / weight)
                               if weight > 0 else float("nan"))
            branch = classify_mode(mode["theta_kB_deg"], compressibility)
            table.append({
                "k_parallel_di": mode["k_parallel_di"], "k_perp_di": mode["k_perp_di"],
                "k_di": mode["k_di"], "theta_kB_deg": mode["theta_kB_deg"],
                "max_amplitude_over_B0": float(np.nanmax(amplitude[:, j]) / abs(B0)),
                "t_max_amplitude": float(t[int(np.nanargmax(amplitude[:, j]))]),
                **{key: fit.get(key, np.nan) for key in (
                    "gamma", "gamma_err", "linear_phase_start", "linear_phase_end",
                    "r_squared", "amplitude_gain", "fit_ok", "fit_reject_reason")},
                "compressibility": compressibility,
                "compressibility_window": source,
                "classification": branch,
                # A physical name only for a mode that actually grows: a
                # decaying noise mode may have mirror geometry, not a mirror
                # instability.
                "physical_branch": (physical_branch(branch, anisotropy)
                                    if fit and fit.get("fit_ok") else ""),
                "_index": j,
            })
        dominant = int(np.nanargmax(np.nanmax(amplitude, axis=0)))
        leaders = {}
        for row in table:
            j = row["_index"]
            if row["classification"] != "unclassified":
                best = leaders.get(row["classification"])
                if best is None or row["max_amplitude_over_B0"] > table[best]["max_amplitude_over_B0"]:
                    leaders[row["classification"]] = j
        fastest = max((r["gamma"] for r in table if r["fit_ok"]), default=float("nan"))
        for j, row in enumerate(table):
            row["dominant"] = int(j == dominant)
            row["branch_leader"] = int(leaders.get(row["classification"]) == j)
            row["fastest_accepted"] = int(bool(row["fit_ok"]) and row["gamma"] == fastest)
            row["candidate_snapshot_fractions"] = ";".join(map(str, MODE_CANDIDATE_FRACTIONS))
        dominant_row = table[dominant]
        # Amplitude history of the modes a paper figure may show: the dominant
        # one, each branch leader and the ten strongest. Without it the mode
        # growth can only be re-plotted from the raw fields.
        keep = sorted({dominant, *leaders.values(),
                       *np.argsort(-np.nanmax(amplitude, axis=0))[:10].tolist()})
        history = []
        for n, time in enumerate(t):
            entry = {"omega_ci_t": float(time)}
            for j in keep:
                mode = modes[j]
                tag = f"kpar{mode['k_parallel_di']:.3f}_kperp{mode['k_perp_di']:+.3f}"
                entry[f"amp_over_B0_{tag}"] = float(amplitude[n, j] / abs(B0))
                entry[f"compressive_fraction_{tag}"] = (
                    float(power[n, 1, j] / total[n, j]) if total[n, j] > 0 else float("nan"))
            history.append(entry)
        _write_csv(self.outdir / "mode_amplitude_timeseries.csv", history)
        _write_csv(self.outdir / "mode_growth_table.csv",
                   [{k: v for k, v in r.items() if k != "_index"}
                    for r in sorted(table, key=lambda r: -r["max_amplitude_over_B0"])])

        def describe(j: int) -> dict:
            row = table[j]
            return {"mode": modes[j], "gamma": row["gamma"], "gamma_err": row["gamma_err"],
                    "fit_ok": bool(row["fit_ok"]), "compressibility": row["compressibility"],
                    "physical_branch": row["physical_branch"],
                    "start": row["linear_phase_start"], "end": row["linear_phase_end"],
                    "reason": row["fit_reject_reason"]}

        atomic_json(self.outdir / "linear_phase.json", {
            "status": "PASS" if fits[dominant]["fit_ok"] else "UNVERIFIED",
            "source": "strongest-power Fourier mode", "mode": modes[dominant],
            "start": fits[dominant]["linear_phase_start"], "end": fits[dominant]["linear_phase_end"],
            "classification": dominant_row["classification"],
            "physical_branch": dominant_row["physical_branch"],
            "compressibility": dominant_row["compressibility"],
            "classification_rule": (
                f"compressive_oblique: theta_kB >= {MIRROR_MIN_THETA_DEG:g} deg and "
                f"|dB_par|^2/|dB|^2 >= {MIRROR_MIN_COMPRESSIBILITY:g}; transverse_parallel: "
                f"theta_kB <= {IC_MAX_THETA_DEG:g} deg and |dB_par|^2/|dB|^2 <= "
                f"{IC_MAX_COMPRESSIBILITY:g}"),
            "branches": {branch: describe(j) for branch, j in sorted(leaders.items())},
            "reason": fits[dominant].get("fit_reject_reason", "")})

        def labelled(j: int, prefix: str, figure: str, title: str) -> dict | None:
            fit = fits[j]
            if not fit:
                return None
            mode, row = modes[j], table[j]
            kpar, kperp = mode["k_parallel_di"], mode["k_perp_di"]
            fit = dict(fit)
            fit.update(
                amplitude_name=(f"{prefix} k_par d_i={kpar:.3f}, k_perp d_i={kperp:.3f}, "
                                f"theta={mode['theta_kB_deg']:.0f} deg, "
                                f"compressibility={row['compressibility']:.2g}"),
                series_label=(rf"|\delta\hat{{\mathbf{{B}}}}(k_\parallel d_i={kpar:.2f},"
                              rf"\,k_\perp d_i={kperp:.2f})|"),
                figure_name=figure, title=title,
            )
            plot_growth(fit, self.outdir)
            return fit

        reference = labelled(dominant, "dominant mode", "growth_rate_fit_mode.png",
                             "Linear growth-rate fit: dominant mode")
        branches = []
        for branch, j in sorted(leaders.items()):
            name = table[j]["physical_branch"] or branch.replace("_", " ")
            fit = labelled(j, f"strongest {name} mode",
                           f"growth_rate_fit_{BRANCH_SERIES[branch]}.png",
                           f"Linear growth-rate fit: {name} branch")
            if fit is not None:
                branches.append({**fit, "branch": branch})
        return reference, branches

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

    def prt_window(self) -> tuple | None:
        """Ventana de salida de partículas, resuelta una sola vez por corrida.

        Se lee del primer archivo prt disponible; si la corrida no guardó
        partículas no hay ventana que marcar y se devuelve ``None`` en vez de
        dibujar la caja del fallback, que sería una region inventada.
        """
        if getattr(self, "_prt_window", "unset") != "unset":
            return self._prt_window
        if not self.particle_files:
            self._prt_window = None
            return None
        first = self.particle_files[sorted(self.particle_files)[0]]
        lo, hi, source = prt_window_bounds(first)
        ext = prt_window_extent_di(lo, hi)
        print(f"[INFO] ventana prt ({source}): "
              f"Z=[{ext['z_di'][0]:.2f}, {ext['z_di'][1]:.2f}] d_i, "
              f"Y=[{ext['y_di'][0]:.2f}, {ext['y_di'][1]:.2f}] d_i "
              f"({100 * ext['area_fraction']:.1f} % del area)")
        self._prt_window = (lo, hi)
        return self._prt_window

    def plot_field_maps(self):
        steps = _select_steps(sorted(self.field_files), self.selected_steps, self.max_map_steps)
        window = self.prt_window()
        tasks = [(step, self.field_files[step], self.outdir, window) for step in steps]
        _run_step_tasks(_field_map_worker, tasks, self.jobs, "Field maps")

    def run_moments_and_correlations(self) -> list[dict]:
        if not self.moment_files:
            print("[INFO] No moment files found; skipping moment diagnostics.")
            return []
        rows = []
        corr_rows = []
        common = sorted(set(self.moment_files) & set(self.field_files))
        steps_for_maps = _select_steps(sorted(self.moment_files), self.selected_steps, self.max_map_steps)
        # Maps, Brazil trajectory and correlations follow the species that
        # drives the instability: for the whistler cases the ions stay
        # isotropic and an ion Brazil plot would show a single fixed point.
        species = DRIVEN_SPECIES
        s = DRIVEN_SUFFIX

        stats_tasks = [
            (step, self.moment_files[step], self.field_files.get(step), species)
            for step in sorted(self.moment_files)
        ]
        stats_results = _run_step_tasks(
            _moment_stats_worker, stats_tasks, self.jobs, "Moment statistics"
        )
        rows = [{**row, "species": species} for _, row in stats_results]

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
        for step, corr_row in corr_results:
            corr_rows.append(corr_row)

        # One scatter per field quantity with every map snapshot, coloured by
        # time (these files used to be rewritten at each step, silently
        # keeping only the last one). |B| is shown relative to B0.
        scatter = {"deltaB": [], "B": [], "Jdia": []}
        for step in [s for s in steps_for_maps if s in common]:
            maps = moment_thermal_maps(self.moment_files[step], self.field_files[step], species)
            fmet = field_metrics(self.field_files[step], B0)
            jdia = compute_jdia(self.moment_files[step], self.field_files[step])
            toci = step_to_omegaci(step)
            for key, label in (("J_dia_i", r"J_{{\rm dia},x,i}"), ("J_dia_e", r"J_{{\rm dia},x,e}"),
                               ("J_dia_total", r"J_{{\rm dia},x}")):
                plot_map(jdia[key], self.outdir / f"{key}_map_step_{step}.png",
                         rf"${label}$ — $t\Omega_{{ci}} = {toci:.1f}$",
                         rf"${label}$ [code units]", cmap=ps.CMAP_DIVERGING, symmetric=True)
            scatter["deltaB"].append((toci, maps["A"], PICDataReader.flatten_2d_slice(fmet["delta_B_over_B0"])))
            scatter["B"].append((toci, maps["A"], PICDataReader.flatten_2d_slice(fmet["B_magnitude"]) / abs(B0)))
            scatter["Jdia"].append((toci, maps["A"], jdia["J_dia_total"]))
        plot_scatter_series(scatter["deltaB"], self.outdir / f"A_{s}_vs_deltaB_scatter.png", rf"$A_{s}$",
                            r"$(|B|-B_0)/B_0$", "Anisotropy vs local field change")
        plot_scatter_series(scatter["B"], self.outdir / f"A_{s}_vs_B_scatter.png", rf"$A_{s}$",
                            r"$|B|/B_0$", "Anisotropy vs local field strength")
        plot_scatter_series(scatter["Jdia"], self.outdir / f"A_{s}_vs_Jdia_scatter.png", rf"$A_{s}$",
                            r"$J_{{\rm dia},x}$ [code units]", "Anisotropy vs diamagnetic current")

        _write_csv(self.outdir / "anisotropy_spatial_stats.csv", rows)
        _write_csv(self.outdir / "spatial_correlations.csv", corr_rows)
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

        # Con ~2400 puntos, s=40 con borde blanco solapa cada marcador con el
        # siguiente y la trayectoria se ve como una banda blanca uniforme: se
        # pierde por completo la dirección temporal, que es lo único que este
        # gráfico existe para mostrar. El tamaño y el borde se escalan con la
        # densidad de puntos.
        dense = len(beta) > 300
        ax.plot(beta, a, color=TEXT_CLR, alpha=0.25, lw=0.8, zorder=1)
        sc = ax.scatter(beta, a, c=t, cmap="plasma", zorder=2,
                        s=6 if dense else 40,
                        edgecolors="none" if dense else "white",
                        linewidths=0.0 if dense else 0.3)
        # Inicio y final explícitos: sin ellos no se sabe hacia dónde corre.
        ax.plot(beta[0], a[0], "o", mfc="none", mec=ps.c("#56d364"), mew=2.2,
                ms=13, zorder=3, label=r"start ($t=0$)")
        ax.plot(beta[-1], a[-1], "X", color=ps.c("#f85149"), ms=13, zorder=3,
                label="end")

        # Rango visible primero: así un umbral que queda fuera del recuadro no
        # aparece en la leyenda prometiendo una curva que no se ve.
        # At least 5 % of |A| as margin: a nearly constant trajectory (e.g. the
        # first snapshots of a run) would otherwise sit on the frame.
        pad = max(0.12 * (np.nanmax(a) - np.nanmin(a)), 0.05 * np.nanmax(np.abs(a)))
        y_lo = min(np.nanmin(a) - pad, 0.95)
        y_hi = max(np.nanmax(a) + pad, 1.05)

        bgrid = np.logspace(np.log10(max(np.nanmin(beta) * 0.6, 0.05)),
                            np.log10(max(np.nanmax(beta) * 1.6, 0.2)), 300)
        # Reference curves only (not kinetic thresholds of these exact VDFs):
        # the one of the declared family, drawn on the driven species' plane.
        families = ("whistler",) if DRIVEN_SPECIES == "electron" else ("mirror", "firehose")
        styles = {"mirror": ps.c("#ff7b72"), "firehose": ps.c("#58a6ff"),
                  "whistler": ps.c("#d2a8ff")}
        x_lo, x_hi = np.nanmin(beta) * 0.85, np.nanmax(beta) * 1.15
        for family in families:
            curve, label = reference_threshold(family, bgrid)
            # Drawn only inside the frame: clipped vertices would still sit
            # under the legend placed below the axes.
            inside = (curve >= y_lo) & (curve <= y_hi) & (bgrid >= x_lo) & (bgrid <= x_hi)
            visible = np.any(inside)
            ax.plot(bgrid, np.where(inside, curve, np.nan), "--", color=styles[family],
                    label=label if visible else None)
            if family == "firehose":
                ax.fill_between(bgrid, 1e-3, curve, color=styles[family], alpha=0.08,
                                where=np.isfinite(curve))
            else:
                ax.fill_between(bgrid, curve, 1e3, color=styles[family], alpha=0.08,
                                where=np.isfinite(curve))
        ax.axhline(1.0, color=TEXT_CLR, alpha=0.35, linestyle=":")

        # Escala log sólo si el rango la justifica. Estas corridas cubren un
        # factor ~2 en beta y ~1.5 en A; en log eso da ticks del tipo
        # "3x10^0, 4x10^0" que ocupan más que la información que aportan.
        def _span(values):
            lo, hi = np.nanmin(values), np.nanmax(values)
            return hi / lo if lo > 0 else np.inf

        if _span(beta) > 10:
            ax.set_xscale("log")
        if _span(a) > 10:
            ax.set_yscale("log")
        ax.set_xlim(np.nanmin(beta) * 0.85, np.nanmax(beta) * 1.15)
        ax.set_ylim(y_lo, y_hi)
        s = DRIVEN_SUFFIX
        ax.set_xlabel(rf"$\beta_{{\parallel {s}}}$", color=TEXT_CLR)
        ax.set_ylabel(rf"$A_{s}$", color=TEXT_CLR)
        ax.set_title(f"Brazil plot from moment averages ({DRIVEN_SPECIES}s)",
                     color=TEXT_CLR, fontweight="bold")
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
                  loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(r"$t\Omega_{ci}$", color=TEXT_CLR)
        cb.ax.yaxis.set_tick_params(color=TEXT_CLR)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_CLR)
        _savefig_many(fig, [self.outdir / "brazil_plot_global.png"])

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
            # This mixes window particle averages (assuming n=1) with global
            # magnetic fluctuations. It is a proxy, not a conserved total:
            # electric energy and electron bulk energy are also absent.
            e_int_i = 1.5 * (r.get("T_parallel_i", np.nan) + 2.0 * r.get("T_perp_i", np.nan)) / 3.0
            e_int_e = 1.5 * (r.get("T_parallel_e", np.nan) + 2.0 * r.get("T_perp_e", np.nan)) / 3.0
            total = e_bulk + e_int_i + e_int_e + e_b
            rows.append({
                "step": step,
                "omega_ci_t": step_to_omegaci(step),
                "E_kin_bulk": e_bulk,
                "E_kin_thermal": e_th,
                "E_internal_i": e_int_i,
                "E_internal_e": e_int_e,
                "E_B": e_b,
                "E_proxy": total,
                "is_conservation_diagnostic": False,
                "A_e": r.get("A_e", np.nan),
            })
        heating = secular_heating(rows)

        if rows and np.isfinite(rows[0]["E_proxy"]) and rows[0]["E_proxy"] != 0:
            e0 = rows[0]["E_proxy"]
            for row in rows:
                row["energy_proxy_relative_change"] = (row["E_proxy"] - e0) / e0

        _write_csv(self.outdir / "energy_table.csv", rows)
        if heating:
            _write_csv(self.outdir / "electron_energy_trend.csv", [heating])
            print(f"  secular electron heating: "
                  f"dEe/dt = {heating['slope_per_omegaci']:.3e} por Omega_ci^-1, "
                  f"R2 = {heating['r_squared']:.4f}, "
                  f"A_e medio = {heating['A_e_mean']:.3f} "
                  f"-> {heating['verdict']}")
        self.plot_energy(rows, heating)

    def plot_energy(self, rows: list[dict], heating: dict | None = None):
        if not rows:
            return
        t = np.array([r["omega_ci_t"] for r in rows])
        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        fig.patch.set_facecolor(DARK_BG)
        _style_axes(ax)
        for key, color, label in [
            ("E_kin_bulk", ps.c("#58a6ff"), "bulk"),
            ("E_internal_i", ps.c("#ff7b72"), "ion internal"),
            ("E_internal_e", ps.c("#d2a8ff"), "electron internal"),
            ("E_B", ps.c("#56d364"), "magnetic fluct."),
            ("E_proxy", ps.c("#f2cc60"), "partial energy proxy"),
        ]:
            y = np.array([r.get(key, np.nan) for r in rows], dtype=float)
            if np.any(np.isfinite(y)):
                style = _series_style(len(t))
                ax.plot(t, y, color=color, label=label, **style)
        if heating:
            ax.plot(t, heating["slope_per_omegaci"] * t + heating["intercept"],
                    ":", color=ps.c("#8b949e"), lw=2.0,
                    label=(rf"secular $e^-$ trend "
                           rf"($R^2$={heating['r_squared']:.3f})"))
        ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
        ax.set_ylabel("energy proxy [code]", color=TEXT_CLR)
        ax.set_title("Partial energy proxy", color=TEXT_CLR, fontweight="bold")
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
                  fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
                  frameon=False)
        _savefig(fig, self.outdir / "energy_partition.png")

        err = np.array([r.get("energy_proxy_relative_change", np.nan) for r in rows], dtype=float)
        if np.any(np.isfinite(err)):
            fig, ax = plt.subplots(figsize=(8.8, 5.4))
            fig.patch.set_facecolor(DARK_BG)
            _style_axes(ax)
            ax.plot(t, err, color=ps.c("#d2a8ff"), **_series_style(len(t)))
            ax.axhline(0, color=TEXT_CLR, alpha=0.35, linestyle=":")
            ax.set_xlabel(r"$t\Omega_{ci}$", color=TEXT_CLR)
            ax.set_ylabel(r"$(E(t)-E(0))/E(0)$", color=TEXT_CLR)
            ax.set_title("Relative change of partial energy proxy", color=TEXT_CLR, fontweight="bold")
            _savefig(fig, self.outdir / "energy_proxy_relative_change.png")


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
    parser.add_argument("--vdf-cadence-omegaci", type=float, default=10.0,
                        help="Physical-time spacing (Omega_ci*t) between selected particle "
                             "snapshots -- one VDF evolution frame (2D + 3D, ion + electron) "
                             "per interval, never denser than the raw particle dumps allow.")
    parser.add_argument("--max-particle-steps", type=int, default=None,
                        help="Optional hard cap on the number of particle snapshots to "
                             "process, applied on top of --vdf-cadence-omegaci (default: "
                             "no cap -- process every snapshot the cadence selects).")
    parser.add_argument("--max-map-steps", type=int, default=5,
                        help="Maximum spatial-map snapshots to render.")
    parser.add_argument("--steps", nargs="*", type=int, help="Optional explicit steps.")
    parser.add_argument("--jobs", "-j", type=int, default=0,
                        help="Number of parallel processes to use (default: 0 = use all CPUs).")
    parser.add_argument("--growth-t-start", type=float, default=None,
                        help="Explicit start of the linear phase (Omega_ci t) for the gamma fits.")
    parser.add_argument("--growth-t-end", type=float, default=None,
                        help="Explicit end of the linear phase (Omega_ci t) for the gamma fits.")
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
        vdf_cadence_omegaci=args.vdf_cadence_omegaci,
        growth_window=(args.growth_t_start, args.growth_t_end),
    ).run()


if __name__ == "__main__":
    main()
