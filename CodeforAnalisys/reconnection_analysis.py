#!/usr/bin/env python3
"""
reconnection_analysis.py — maintained reconnection diagnostics + B–kappa correlation
====================================================================================
Replaces ``legacy/reconnection_analysis.py`` (hard-coded paths, hard-coded
``dt = 0.547``, dark theme baked into the PNGs, no tabular output) with a
run-aware module in the style of the rest of the pipeline:

  1. **Outputs are discovered, not assumed.** Field, moment and particle
     series come from ``PICDataReader.discover_outputs``; the grid shape and
     the physical box are read from the snapshot coordinates, so the same
     script analyses ``psc_reconnection`` (25.6 x 51.2 d_i, mi/me = 25) and
     ``psc_reconnection_comparable`` (40 x 40 d_i, mi/me = 200) without
     editing constants. What was detected is recorded in
     ``reconnection_summary.json``.

  2. **One figure style.** Everything goes through ``plot_style`` (paper
     theme by default: white background, 300 dpi, PDF next to every PNG).

  3. **B–kappa correlation.** Per particle snapshot, the effective kappa
     index of the selected species is measured with the truncated, whitened
     estimator of ``kappa_eff`` (the same estimator used for theory and for
     the anisotropy runs). Per field snapshot, ``<|B|>`` is averaged over
     two regions: the prt output window (the volume the particles actually
     sample) and the perturbed current sheet at y = +Ly/4 (where the
     reconnection happens). The two time series are then correlated
     (Pearson on 1/kappa, plus Spearman), written as CSV + JSON and drawn
     as a shared-time-axis figure and a scatter.

Physical conventions (PSC dimensionless normalization, c = m_e = e = 1,
omega_pe = 1, lengths in d_e):

    b0 = 1 / (wpe/wce),   Omega_ci = 1 / (mass_ratio * wpe/wce),
    d_i = sqrt(mass_ratio),   dt = cfl / sqrt(1/dy^2 + 1/dz^2).

Interpretation notes, so the numbers are not over-read:

  * The prt window of ``psc_reconnection_comparable`` covers the *inflow*
    region between the two sheets (cells 0.4–0.6 of the grid), not the
    X-point. kappa_eff(t) therefore characterises the plasma feeding the
    reconnection, and the window/sheet pairing quantifies whether the
    inflow VDF responds to the field evolution *inside* the sheet. Moving
    the window is a change to the simulation case matrix, not to this
    script.
  * kappa_eff is reported as 1/kappa for correlation and plotting: the
    Maxwellian limit maps to 0 instead of a divergent value, matching
    ``kappa_evolution.py``.
  * The correlation p-values assume independent samples; consecutive
    snapshots of one run are autocorrelated, so treat them as descriptive,
    not as significance tests.

Usage::

    python reconnection_analysis.py --data-dir /path/to/run \
        --profile reconnection_comparable --outdir ../analysis_results/rec/10_reconnection

    make reconnection DATA_DIR=/path/to/run CASE=reconnection_comparable
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import plot_style as ps
ps.apply()
import matplotlib.pyplot as plt

import kappa_eff
from data_reader import PICDataReader


# ── Run profiles ─────────────────────────────────────────────────────────────
# Only the quantities that cannot be read back from the output files live
# here; grid, box and cadence always come from the data. Values must match
# the corresponding src/psc_reconnection*.cxx.
RECONNECTION_PROFILES = {
    "reconnection": {
        "label": "Double Harris reconnection (production, mi/me=25)",
        "mass_ratio": 25.0,
        "wpe_wce": 2.0,
        "cfl": 0.99,
        "kappa0": 3.0,        # setup_p.kappa in psc_reconnection.cxx
        "sheet_y_over_Ly": 0.25,   # perturbed sheet at y = +Ly/4
    },
    "reconnection_comparable": {
        "label": "Double Harris reconnection (comparable, mi/me=200)",
        "mass_ratio": 200.0,
        "wpe_wce": 2.0,
        "cfl": 0.95,
        "kappa0": 3.0,
        "sheet_y_over_Ly": 0.25,
    },
}

FIELD_COMPONENTS = ("hx_fc", "hy_fc", "hz_fc")

# Okabe-Ito series colours in the fixed pipeline order (see plot_style).
C_WINDOW = "#0072B2"   # blue      — prt window <|B|>
C_SHEET = "#D55E00"    # vermillion — perturbed-sheet <|B|>
C_KAPPA = "#009E73"    # green     — 1/kappa_eff
C_FLUX = "#CC79A7"     # purple    — reconnected-flux proxy
MUTED = "#5f5f5c"


# ── Geometry and unit resolution ─────────────────────────────────────────────

@dataclass
class RunGeometry:
    """Everything the analysis needs about one run, read from its files."""

    profile_name: str
    label: str
    mass_ratio: float
    wpe_wce: float
    kappa0: float | None
    b0: float                 # code units (= 1/wpe_wce)
    omega_ci: float
    d_i: float                # in d_e (code lengths)
    dt_code: float
    # Grid: storage order of the 2D snapshots is (nz, ny).
    ny: int
    nz: int
    y_centers: np.ndarray     # code units, length ny
    z_centers: np.ndarray     # code units, length nz
    dy: float
    dz: float
    sheet_y_code: float       # y of the perturbed sheet, code units
    provenance: dict = field(default_factory=dict)

    @property
    def Ly_code(self) -> float:
        return self.dy * self.ny

    @property
    def Lz_code(self) -> float:
        return self.dz * self.nz

    def to_di(self, x_code):
        return np.asarray(x_code, dtype=float) / self.d_i

    def step_to_wci(self, step: int) -> float:
        return step * self.dt_code * self.omega_ci

    def summary_dict(self) -> dict:
        return {
            "profile": self.profile_name,
            "label": self.label,
            "mass_ratio": self.mass_ratio,
            "wpe_wce": self.wpe_wce,
            "kappa0": self.kappa0,
            "b0_code": self.b0,
            "omega_ci_code": self.omega_ci,
            "d_i_de": self.d_i,
            "dt_code": self.dt_code,
            "grid_ny": self.ny,
            "grid_nz": self.nz,
            "Ly_di": float(self.Ly_code / self.d_i),
            "Lz_di": float(self.Lz_code / self.d_i),
            "dy_di": float(self.dy / self.d_i),
            "dz_di": float(self.dz / self.d_i),
            "sheet_y_di": float(self.sheet_y_code / self.d_i),
            "provenance": self.provenance,
        }


def _read_axis_centers(view, keys, axis: int) -> np.ndarray:
    path = PICDataReader.resolve_variable_path(keys, f"crd[{axis}]/p0/1d")
    if path is None:
        path = PICDataReader.resolve_dataset_path(
            keys, f"crd[{axis}]", f"crd[{axis}]/p0/1d")
    if path is None:
        raise KeyError(f"Coordinate crd[{axis}] not found in snapshot.")
    x = np.asarray(view.read(path), dtype=float).ravel()
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError(f"Unusable crd[{axis}] coordinate array.")
    dx = np.diff(x)
    if np.any(dx <= 0) or not np.allclose(dx, dx.mean(), rtol=2e-4, atol=1e-9):
        raise ValueError(f"Nonuniform crd[{axis}] coordinates.")
    return x


def resolve_geometry(field_file: str, profile_name: str, *,
                     mass_ratio: float | None = None,
                     wpe_wce: float | None = None,
                     cfl: float | None = None,
                     dt_code: float | None = None,
                     sheet_y_di: float | None = None) -> RunGeometry:
    """Combine the profile constants with the geometry stored in one snapshot.

    The reconnection boxes are rectangular (25.6 x 51.2 d_i in production),
    so this deliberately does not go through ``run_geometry.resolve_run``,
    which enforces the square domain of the anisotropy runs.
    """
    if profile_name not in RECONNECTION_PROFILES:
        raise ValueError(
            f"Unknown reconnection profile '{profile_name}'. "
            f"Valid: {sorted(RECONNECTION_PROFILES)}")
    prof = RECONNECTION_PROFILES[profile_name]
    mass_ratio = float(mass_ratio if mass_ratio is not None else prof["mass_ratio"])
    wpe_wce = float(wpe_wce if wpe_wce is not None else prof["wpe_wce"])
    cfl = float(cfl if cfl is not None else prof["cfl"])

    with PICDataReader.open_data_file(field_file) as view:
        keys = view.keys()
        hz = PICDataReader.resolve_dataset_path(keys, "jeh", "hz_fc/p0/3d")
        if hz is None:
            raise KeyError(f"hz_fc not found in {field_file}")
        shape = np.asarray(view.read(hz)).shape
        if len(shape) != 3 or shape[2] != 1:
            raise ValueError(
                f"Expected a 2D yz snapshot with shape (nz, ny, 1); got {shape}")
        nz, ny = int(shape[0]), int(shape[1])
        y = _read_axis_centers(view, keys, 1)
        z = _read_axis_centers(view, keys, 2)
    if y.size != ny or z.size != nz:
        raise ValueError(
            f"Coordinate lengths (ny={y.size}, nz={z.size}) do not match the "
            f"field shape (ny={ny}, nz={nz}) in {field_file}")

    dy = float(np.diff(y).mean())
    dz = float(np.diff(z).mean())
    b0 = 1.0 / wpe_wce
    omega_ci = 1.0 / (mass_ratio * wpe_wce)
    d_i = math.sqrt(mass_ratio)
    if dt_code is None:
        dt_code = cfl / math.sqrt(1.0 / dy**2 + 1.0 / dz**2)
        dt_origin = f"cfl={cfl} Courant estimate; verify runtime log"
    else:
        dt_origin = "command line"

    # The perturbed sheet sits at y = +Ly/4 in both source files; the domain
    # is y-centered, so locate it from the actual coordinate range instead of
    # assuming an origin.
    y_lo = float(y[0] - 0.5 * dy)
    Ly = dy * ny
    if sheet_y_di is None:
        sheet_y_code = y_lo + (0.5 + prof["sheet_y_over_Ly"]) * Ly
        sheet_origin = f"profile (y = +Ly/4 above the domain centre)"
    else:
        sheet_y_code = sheet_y_di * d_i
        sheet_origin = "command line"

    return RunGeometry(
        profile_name=profile_name, label=prof["label"],
        mass_ratio=mass_ratio, wpe_wce=wpe_wce, kappa0=prof["kappa0"],
        b0=b0, omega_ci=omega_ci, d_i=d_i, dt_code=float(dt_code),
        ny=ny, nz=nz, y_centers=y, z_centers=z, dy=dy, dz=dz,
        sheet_y_code=float(sheet_y_code),
        provenance={"grid": str(field_file), "dt_code": dt_origin,
                    "sheet_y": sheet_origin},
    )


# ── Field reading and region statistics ──────────────────────────────────────

def read_field_snapshot(path: str) -> dict[str, np.ndarray]:
    """Read (Bx, By, Bz) as 2D arrays of shape (nz, ny)."""
    raw = PICDataReader.read_multiple_fields_3d(
        path, "jeh", [f"{c}/p0/3d" for c in FIELD_COMPONENTS])
    out = {}
    for comp in FIELD_COMPONENTS:
        out[comp[:2]] = PICDataReader.flatten_2d_slice(raw[f"{comp}/p0/3d"])
    out["bmag"] = np.sqrt(out["hx"]**2 + out["hy"]**2 + out["hz"]**2)
    return out


@dataclass
class Region:
    """Axis-aligned analysis region in code units, with its cell slices."""

    name: str
    y_lo: float
    y_hi: float
    z_lo: float
    z_hi: float
    sl_y: slice
    sl_z: slice

    def n_cells(self) -> int:
        ny = self.sl_y.stop - self.sl_y.start
        nz = self.sl_z.stop - self.sl_z.start
        return max(ny, 0) * max(nz, 0)

    def contains(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return ((y >= self.y_lo) & (y < self.y_hi)
                & (z >= self.z_lo) & (z < self.z_hi))

    def describe(self, geom: RunGeometry) -> dict:
        return {"name": self.name,
                "y_di": [float(self.y_lo / geom.d_i), float(self.y_hi / geom.d_i)],
                "z_di": [float(self.z_lo / geom.d_i), float(self.z_hi / geom.d_i)],
                "n_cells": self.n_cells()}


def _axis_slice(centers: np.ndarray, lo: float, hi: float) -> slice:
    inside = np.nonzero((centers >= lo) & (centers < hi))[0]
    if inside.size == 0:
        return slice(0, 0)
    return slice(int(inside[0]), int(inside[-1]) + 1)


def make_box_region(geom: RunGeometry, name: str,
                    y_lo: float, y_hi: float, z_lo: float, z_hi: float) -> Region:
    return Region(name=name, y_lo=y_lo, y_hi=y_hi, z_lo=z_lo, z_hi=z_hi,
                  sl_y=_axis_slice(geom.y_centers, y_lo, y_hi),
                  sl_z=_axis_slice(geom.z_centers, z_lo, z_hi))


def window_region(geom: RunGeometry, prt_file: str) -> Region:
    """The prt output window, read from the particle file's own attributes."""
    lo, hi = PICDataReader.read_prt_window(prt_file)
    y0 = geom.y_centers[0] - 0.5 * geom.dy
    z0 = geom.z_centers[0] - 0.5 * geom.dz
    return make_box_region(
        geom, "prt_window",
        y0 + lo[1] * geom.dy, y0 + hi[1] * geom.dy,
        z0 + lo[2] * geom.dz, z0 + hi[2] * geom.dz)


def sheet_region(geom: RunGeometry, half_width_di: float) -> Region:
    """Band of +-half_width_di around the perturbed sheet, all of z."""
    hw = half_width_di * geom.d_i
    z0 = geom.z_centers[0] - 0.5 * geom.dz
    return make_box_region(
        geom, "perturbed_sheet",
        geom.sheet_y_code - hw, geom.sheet_y_code + hw,
        z0, z0 + geom.Lz_code)


def region_field_stats(bfield: dict[str, np.ndarray], region: Region,
                       b0: float) -> dict:
    """Mean/min/fluctuation of |B| over a region, normalized to b0."""
    sub = bfield["bmag"][region.sl_z, region.sl_y]
    if sub.size == 0:
        nan = float("nan")
        return {"mean_B": nan, "min_B": nan, "delta_B_rms": nan}
    mean = float(np.mean(sub))
    return {"mean_B": mean / b0,
            "min_B": float(np.min(sub)) / b0,
            "delta_B_rms": float(np.sqrt(np.mean((sub - mean) ** 2))) / b0}


def mean_b_direction(bfield: dict[str, np.ndarray], region: Region) -> np.ndarray:
    """Region-mean field direction; falls back to z when the mean cancels."""
    b = np.array([np.mean(bfield[c][region.sl_z, region.sl_y])
                  for c in ("hx", "hy", "hz")])
    norm = float(np.linalg.norm(b))
    rms = float(np.sqrt(np.mean(bfield["bmag"][region.sl_z, region.sl_y] ** 2)))
    if norm < 0.05 * max(rms, 1e-30):
        return np.array([0.0, 0.0, 1.0])
    return b / norm


def reconnected_flux_proxy(bfield: dict[str, np.ndarray],
                           geom: RunGeometry) -> float:
    """max |By|/b0 along z at the perturbed-sheet row (tearing-mode proxy).

    The legacy script measured this at y = 0, which for the double sheet is
    the midpoint *between* the sheets; here it is taken on the sheet that is
    actually perturbed and reconnecting.
    """
    iy = int(np.argmin(np.abs(geom.y_centers - geom.sheet_y_code)))
    return float(np.max(np.abs(bfield["hy"][:, iy]))) / geom.b0


# ── Particle-side kappa ──────────────────────────────────────────────────────

def field_aligned_velocities(vx, vy, vz, b_hat: np.ndarray):
    """Rotate (vx, vy, vz) into (v_par, v_perp1, v_perp2) about b_hat."""
    b_hat = np.asarray(b_hat, dtype=float)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, b_hat)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, b_hat)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(b_hat, e1)
    v_par = vx * b_hat[0] + vy * b_hat[1] + vz * b_hat[2]
    v_p1 = vx * e1[0] + vy * e1[1] + vz * e1[2]
    v_p2 = vx * e2[0] + vy * e2[1] + vz * e2[2]
    return v_par, v_p1, v_p2


def kappa_of_snapshot(prt_file: str, species: str, b_hat: np.ndarray, *,
                      region: Region | None = None,
                      max_particles: int = 2_000_000,
                      s_max: float = kappa_eff.DEFAULT_S_MAX,
                      n_boot: int = 0) -> dict:
    """kappa_eff of one particle snapshot, optionally restricted to a region.

    Selects the species by charge sign (Harris and background populations of
    the same species are analysed together: physically they are one VDF).
    Momenta are treated as velocities, valid in these non-relativistic runs.
    """
    data = PICDataReader.read_particles_with_positions(prt_file, max_particles)
    charge_sel = data["q"] > 0 if species == "ion" else data["q"] < 0
    if region is not None and "y" in data and "z" in data:
        charge_sel &= region.contains(data["y"], data["z"])
    n_sel = int(np.count_nonzero(charge_sel))
    if n_sel < 100:
        return {"kappa": float("nan"), "inv_kappa": float("nan"),
                "kappa_err": float("nan"), "K": float("nan"),
                "n_particles": n_sel, "n_eff": float(n_sel),
                "fraction_inside": float("nan")}
    v = field_aligned_velocities(
        data["px"][charge_sel], data["py"][charge_sel], data["pz"][charge_sel],
        b_hat)
    res = kappa_eff.kappa_eff_from_velocities(
        *v, data["w"][charge_sel], s_max=s_max, n_boot=n_boot)
    kappa = res["kappa"]
    res["inv_kappa"] = 0.0 if np.isinf(kappa) else (
        1.0 / kappa if np.isfinite(kappa) and kappa > 0 else float("nan"))
    res["n_particles"] = n_sel
    return res


# ── Correlation ──────────────────────────────────────────────────────────────

def correlate_series(x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> dict:
    """Pearson + Spearman between two aligned series, with a small lag scan.

    Lag ``L`` correlates x(t_i) against y(t_{i+L}); the number of usable
    pairs shrinks with |L|, so the scan is only reported when >= 8 samples
    survive at every lag.
    """
    from scipy import stats

    keep = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, float)[keep], np.asarray(y, float)[keep]
    n = int(x.size)
    out = {"n": n, "pearson_r": float("nan"), "pearson_p": float("nan"),
           "spearman_rho": float("nan"), "spearman_p": float("nan"),
           "lag_scan": None}
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return out
    r, p = stats.pearsonr(x, y)
    rho, sp = stats.spearmanr(x, y)
    out.update(pearson_r=float(r), pearson_p=float(p),
               spearman_rho=float(rho), spearman_p=float(sp))
    if n - max_lag >= 8:
        lags = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                a, b = x[:n - lag] if lag else x, y[lag:]
            else:
                a, b = x[-lag:], y[:n + lag]
            if a.size >= 3 and np.ptp(a) > 0 and np.ptp(b) > 0:
                lags[lag] = float(stats.pearsonr(a, b)[0])
        if lags:
            best = max(lags, key=lambda k: abs(lags[k]))
            out["lag_scan"] = {"r_by_lag": {str(k): v for k, v in lags.items()},
                               "best_lag": best, "best_r": lags[best]}
    return out


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_b_kappa_evolution(field_rows, kappa_rows, geom: RunGeometry,
                           outdir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.8, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.12})

    t_f = np.array([r["t_wci"] for r in field_rows])
    for key, color, label in (
            ("window_mean_B", ps.c(C_WINDOW), "prt window"),
            ("sheet_mean_B", ps.c(C_SHEET), "perturbed sheet")):
        ax1.plot(t_f, [r[key] for r in field_rows], color=color, lw=1.8,
                 label=label, zorder=3)
    ax1.set_ylabel(r"$\langle |B| \rangle / B_0$")
    ps.style_axes(ax1)
    ps.legend(ax1, loc="best", fontsize=10, title="region")

    t_k = np.array([r["t_wci"] for r in kappa_rows])
    inv = np.array([r["inv_kappa"] for r in kappa_rows])
    err = np.array([r.get("kappa_err", float("nan")) for r in kappa_rows])
    kap = np.array([r["kappa"] for r in kappa_rows])
    # sigma(1/kappa) = sigma(kappa)/kappa^2 for the bootstrap half-interval.
    inv_err = np.where(np.isfinite(err) & np.isfinite(kap) & (kap > 0),
                       err / np.maximum(kap, 1e-30) ** 2, np.nan)
    col = ps.c(C_KAPPA)
    ax2.plot(t_k, inv, color=col, lw=1.6, marker="o", ms=5.5, zorder=3,
             label=r"$1/\kappa_{\rm eff}$ (prt window)")
    band = np.isfinite(inv_err)
    if band.any():
        ax2.fill_between(t_k[band], (inv - inv_err)[band], (inv + inv_err)[band],
                         color=col, alpha=0.18, lw=0, zorder=2)
    if geom.kappa0:
        ax2.axhline(1.0 / geom.kappa0, color=MUTED, lw=0.9, ls=(0, (4, 3)),
                    zorder=1)
        ax2.text(0.995, 1.0 / geom.kappa0, rf"injected $\kappa_0={geom.kappa0:g}$",
                 transform=ax2.get_yaxis_transform(), fontsize=8, color=MUTED,
                 va="bottom", ha="right")
    ax2.axhline(0.0, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=1)
    ax2.text(0.995, 0.002, "Maxwellian limit",
             transform=ax2.get_yaxis_transform(), fontsize=8, color=MUTED,
             va="bottom", ha="right")
    ax2.set_ylabel(r"$1/\kappa_{\rm eff}$")
    ax2.set_xlabel(r"$t\,\Omega_{ci}$")
    ps.style_axes(ax2)
    ps.legend(ax2, loc="best", fontsize=10)

    out = outdir / "b_kappa_evolution.png"
    ps.save(fig, out)
    return out


def _emptiest_corner(ax, x, y):
    """Corner (in axes fractions) with the fewest data points, for the stats box."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if not keep.any():
        return {"x": 0.03, "y": 0.97, "ha": "left", "va": "top"}
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    fx = (x[keep] - x0) / (x1 - x0 or 1.0)
    fy = (y[keep] - y0) / (y1 - y0 or 1.0)
    corners = {
        ("left", "top"): np.sum((fx < 0.45) & (fy > 0.55)),
        ("right", "top"): np.sum((fx > 0.55) & (fy > 0.55)),
        ("left", "bottom"): np.sum((fx < 0.45) & (fy < 0.45)),
        ("right", "bottom"): np.sum((fx > 0.55) & (fy < 0.45)),
    }
    ha, va = min(corners, key=corners.get)
    return {"x": 0.03 if ha == "left" else 0.97,
            "y": 0.97 if va == "top" else 0.03, "ha": ha, "va": va}


def plot_b_kappa_scatter(merged, corr_window, corr_sheet, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    t = np.array([m["t_wci"] for m in merged])
    inv = np.array([m["inv_kappa"] for m in merged])
    for ax, key, corr, xlabel in (
            (axes[0], "window_mean_B", corr_window,
             r"$\langle |B| \rangle_{\rm window} / B_0$"),
            (axes[1], "sheet_mean_B", corr_sheet,
             r"$\langle |B| \rangle_{\rm sheet} / B_0$")):
        x = np.array([m[key] for m in merged])
        sc = ax.scatter(x, inv, c=t, cmap=ps.CMAP_SEQUENTIAL, s=42, zorder=3,
                        edgecolors=ps.c("#111111"), linewidths=0.4)
        ax.set_xlabel(xlabel)
        ps.style_axes(ax)
        r, p = corr["pearson_r"], corr["pearson_p"]
        rho = corr["spearman_rho"]
        pos = _emptiest_corner(ax, x, inv)
        ax.text(pos["x"], pos["y"],
                (f"$r = {r:+.2f}$ ($p = {p:.2g}$)\n"
                 rf"$\rho_s = {rho:+.2f}$,  $N = {corr['n']}$"),
                transform=ax.transAxes, fontsize=10,
                ha=pos["ha"], va=pos["va"], zorder=4,
                bbox=dict(facecolor=ps.LEGEND_BG, edgecolor=ps.GRID_CLR,
                          alpha=0.9, boxstyle="round,pad=0.35"))
    axes[0].set_ylabel(r"$1/\kappa_{\rm eff}$")
    cbar = fig.colorbar(sc, ax=axes, pad=0.015, fraction=0.035)
    cbar.set_label(r"$t\,\Omega_{ci}$")

    out = outdir / "b_kappa_scatter.png"
    ps.save(fig, out)
    return out


def _fieldline_overlay(ax, bfield, geom: RunGeometry, nlines: int = 16):
    """In-plane field lines from the flux function A_x(z, y) = int Bz dz."""
    Ax = np.cumsum(bfield["hz"], axis=0) * geom.dz
    levels = np.linspace(np.min(Ax) * 0.9, np.max(Ax) * 0.9, nlines)
    Z, Y = np.meshgrid(geom.to_di(geom.z_centers), geom.to_di(geom.y_centers))
    ax.contour(Z, Y, Ax.T, levels=levels, colors=ps.c("#111111"),
               linewidths=0.45, alpha=0.5)


def plot_overview_panel(fields: dict[int, str], geom: RunGeometry,
                        outdir: Path, n_times: int = 4) -> Path:
    steps = sorted(fields)
    picks = sorted({steps[min(int(round(i * (len(steps) - 1) / (n_times - 1))),
                              len(steps) - 1)]
                    for i in range(min(n_times, len(steps)))})
    fig, axes = plt.subplots(
        2, len(picks), figsize=(3.6 * len(picks) + 1.2, 7.4),
        sharex=True, sharey=True, squeeze=False)
    y_di = geom.to_di(geom.y_centers)
    z_di = geom.to_di(geom.z_centers)
    extent = [z_di[0], z_di[-1], y_di[0], y_di[-1]]

    ims = [None, None]
    for col, step in enumerate(picks):
        b = read_field_snapshot(fields[step])
        panels = ((0, b["hz"] / geom.b0, ps.CMAP_DIVERGING, True),
                  (1, b["bmag"] / geom.b0, ps.CMAP_SEQUENTIAL, False))
        for row, data, cmap, symmetric in panels:
            ax = axes[row][col]
            vmax = float(np.nanpercentile(np.abs(data), 99.5)) or 1.0
            vmin = -vmax if symmetric else 0.0
            ims[row] = ax.imshow(data.T, origin="lower", cmap=cmap,
                                 vmin=vmin, vmax=vmax, extent=extent,
                                 aspect="auto", rasterized=True,
                                 interpolation="nearest")
            _fieldline_overlay(ax, b, geom)
            ax.axhline(geom.sheet_y_code / geom.d_i, color=ps.c(C_SHEET),
                       lw=0.8, ls=(0, (4, 3)), alpha=0.8)
            ps.style_axes(ax)
            if row == 0:
                ax.set_title(rf"$t\,\Omega_{{ci}} = {geom.step_to_wci(step):.1f}$",
                             fontsize=12)
            if row == 1:
                ax.set_xlabel(r"$z\ [d_i]$")
        axes[0][col].tick_params(labelbottom=False)
    for row, label in ((0, r"$B_z / B_0$"), (1, r"$|B| / B_0$")):
        axes[row][0].set_ylabel(r"$y\ [d_i]$")
        cbar = fig.colorbar(ims[row], ax=list(axes[row]), pad=0.012,
                            fraction=0.03)
        cbar.set_label(label)

    out = outdir / "reconnection_overview.png"
    ps.save(fig, out)
    return out


def plot_flux_evolution(field_rows, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    t = [r["t_wci"] for r in field_rows]
    ax.plot(t, [r["flux_proxy"] for r in field_rows], color=ps.c(C_FLUX),
            lw=1.9, zorder=3)
    ax.set_xlabel(r"$t\,\Omega_{ci}$")
    ax.set_ylabel(r"$\max_z |B_y| / B_0$ at the perturbed sheet")
    ps.style_axes(ax, "Reconnected-flux proxy")
    out = outdir / "reconnected_flux.png"
    ps.save(fig, out)
    return out


# ── CSV / JSON output ────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict], columns: list[str]) -> Path:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})
    return path


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _finite_or_none(value):
    return value if isinstance(value, (int, str)) or (
        isinstance(value, float) and math.isfinite(value)) else None


# ── Driver ───────────────────────────────────────────────────────────────────

def run_analysis(args) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    discovered = PICDataReader.discover_outputs(args.data_dir)
    fields: dict[int, str] = discovered["fields"]
    particles_by_series: dict[str, dict[int, str]] = discovered["particles"]
    if not fields:
        print(f"[ERROR] No field snapshots (pfd.*) found in {args.data_dir}")
        return 1
    if len(particles_by_series) > 1:
        print(f"[ERROR] Multiple particle series found "
              f"({sorted(particles_by_series)}); keep one case per directory.")
        return 1
    prt_series = next(iter(particles_by_series.values()), {})

    geom = resolve_geometry(
        fields[min(fields)], args.profile,
        mass_ratio=args.mass_ratio, wpe_wce=args.wpe_wce, cfl=args.cfl,
        dt_code=args.dt_code, sheet_y_di=args.sheet_y_di)

    print(f"Run: {geom.label}")
    print(f"  grid {geom.nz} x {geom.ny} (z x y), box "
          f"{geom.Ly_code / geom.d_i:.1f} x {geom.Lz_code / geom.d_i:.1f} d_i, "
          f"dt = {geom.dt_code:.4f} ({geom.provenance['dt_code']})")
    print(f"  {len(fields)} field snapshots, {len(prt_series)} particle snapshots")

    sheet = sheet_region(geom, args.sheet_half_width_di)
    window = (window_region(geom, prt_series[min(prt_series)])
              if prt_series else None)
    if window is None:
        # No particles: still report <|B|> over the central inter-sheet
        # region so the field CSV keeps both columns comparable across runs.
        y0 = geom.y_centers[0] - 0.5 * geom.dy
        z0 = geom.z_centers[0] - 0.5 * geom.dz
        window = make_box_region(
            geom, "central_fallback",
            y0 + 0.4 * geom.Ly_code, y0 + 0.6 * geom.Ly_code,
            z0 + 0.4 * geom.Lz_code, z0 + 0.6 * geom.Lz_code)
        print("[WARN] No prt files: kappa(t) and the correlation are skipped; "
              "'window' <|B|> uses the central 20% of the box.")
    for region in (window, sheet):
        d = region.describe(geom)
        print(f"  region '{d['name']}': y in [{d['y_di'][0]:.2f}, "
              f"{d['y_di'][1]:.2f}] d_i, z in [{d['z_di'][0]:.2f}, "
              f"{d['z_di'][1]:.2f}] d_i ({d['n_cells']} cells)")
    if window.n_cells() == 0 or sheet.n_cells() == 0:
        print("[ERROR] An analysis region contains no cells; check "
              "--sheet-y-di / --sheet-half-width-di against the box.")
        return 1

    # ── field-side time series ────────────────────────────────────────────
    field_steps = sorted(fields)
    if args.max_field_snapshots and len(field_steps) > args.max_field_snapshots:
        stride = int(np.ceil(len(field_steps) / args.max_field_snapshots))
        kept = set(field_steps[::stride])
        kept.update(s for s in field_steps if s in prt_series)
        field_steps = sorted(kept)
        print(f"  subsampled fields to {len(field_steps)} snapshots "
              f"(stride {stride}; particle steps kept)")

    field_rows, b_dir_by_step = [], {}
    for step in field_steps:
        b = read_field_snapshot(fields[step])
        w_stats = region_field_stats(b, window, geom.b0)
        s_stats = region_field_stats(b, sheet, geom.b0)
        field_rows.append({
            "step": step, "t_wci": geom.step_to_wci(step),
            "window_mean_B": w_stats["mean_B"],
            "window_min_B": w_stats["min_B"],
            "window_delta_B_rms": w_stats["delta_B_rms"],
            "sheet_mean_B": s_stats["mean_B"],
            "sheet_min_B": s_stats["min_B"],
            "sheet_delta_B_rms": s_stats["delta_B_rms"],
            "flux_proxy": reconnected_flux_proxy(b, geom),
        })
        b_dir_by_step[step] = mean_b_direction(b, window)
    field_csv = write_csv(
        outdir / "reconnection_field_timeseries.csv", field_rows,
        ["step", "t_wci", "window_mean_B", "window_min_B",
         "window_delta_B_rms", "sheet_mean_B", "sheet_min_B",
         "sheet_delta_B_rms", "flux_proxy"])
    print(f"  wrote -> {field_csv}")

    # ── particle-side kappa(t) ────────────────────────────────────────────
    kappa_rows = []
    field_step_arr = np.array(field_steps)
    cadence = int(np.min(np.diff(field_step_arr))) if len(field_steps) > 1 else 0
    for step in sorted(prt_series):
        nearest = int(field_step_arr[np.argmin(np.abs(field_step_arr - step))])
        if cadence and abs(nearest - step) > cadence:
            print(f"[WARN] prt step {step}: nearest field snapshot is "
                  f"{nearest} (> one field cadence away); using it anyway.")
        res = kappa_of_snapshot(
            prt_series[step], args.species, b_dir_by_step[nearest],
            max_particles=args.max_particles, s_max=args.s_max,
            n_boot=args.kappa_boot)
        matched = next(r for r in field_rows if r["step"] == nearest)
        kappa_rows.append({
            "step": step, "t_wci": geom.step_to_wci(step),
            "species": args.species,
            "kappa": res["kappa"], "inv_kappa": res["inv_kappa"],
            "kappa_err": res.get("kappa_err", float("nan")),
            "K": res.get("K", float("nan")),
            "n_particles": res["n_particles"],
            "n_eff": res.get("n_eff", float("nan")),
            "fraction_inside": res.get("fraction_inside", float("nan")),
            "matched_field_step": nearest,
            "window_mean_B": matched["window_mean_B"],
            "sheet_mean_B": matched["sheet_mean_B"],
        })
        k_str = ("inf" if np.isinf(res["kappa"])
                 else f"{res['kappa']:.3f}" if np.isfinite(res["kappa"])
                 else "nan")
        print(f"  step {step:>9d}: kappa_eff = {k_str}  "
              f"(N = {res['n_particles']})")
    if kappa_rows:
        kappa_csv = write_csv(
            outdir / "reconnection_kappa_timeseries.csv", kappa_rows,
            ["step", "t_wci", "species", "kappa", "inv_kappa", "kappa_err",
             "K", "n_particles", "n_eff", "fraction_inside",
             "matched_field_step", "window_mean_B", "sheet_mean_B"])
        print(f"  wrote -> {kappa_csv}")

    # ── correlation ───────────────────────────────────────────────────────
    corr_block = None
    if len(kappa_rows) >= 3:
        inv = np.array([r["inv_kappa"] for r in kappa_rows])
        corr_window = correlate_series(
            np.array([r["window_mean_B"] for r in kappa_rows]), inv)
        corr_sheet = correlate_series(
            np.array([r["sheet_mean_B"] for r in kappa_rows]), inv)
        corr_block = {
            "quantities": "Pearson/Spearman between <|B|>/B0 and 1/kappa_eff "
                          "at the particle-output cadence",
            "caveats": [
                "p-values assume independent samples; consecutive snapshots "
                "are autocorrelated",
                "kappa_eff samples the prt window (inflow region), not the "
                "X-point",
                "correlation is not causation: both series share the "
                "instability clock",
            ],
            "window_B_vs_inv_kappa": corr_window,
            "sheet_B_vs_inv_kappa": corr_sheet,
        }
        with open(outdir / "b_kappa_correlation.json", "w") as fh:
            json.dump(corr_block, fh, indent=2, default=_json_default)
        print(f"  wrote -> {outdir / 'b_kappa_correlation.json'}")
        print(f"  window <|B|> vs 1/kappa: r = {corr_window['pearson_r']:+.3f} "
              f"(p = {corr_window['pearson_p']:.3g}, N = {corr_window['n']})")
        print(f"  sheet  <|B|> vs 1/kappa: r = {corr_sheet['pearson_r']:+.3f} "
              f"(p = {corr_sheet['pearson_p']:.3g}, N = {corr_sheet['n']})")
        print(f"  wrote -> {plot_b_kappa_evolution(field_rows, kappa_rows, geom, outdir)}")
        print(f"  wrote -> {plot_b_kappa_scatter(kappa_rows, corr_window, corr_sheet, outdir)}")
    elif kappa_rows:
        print(f"[WARN] Only {len(kappa_rows)} particle snapshot(s); a "
              "correlation needs at least 3. Time series were still written.")
        print(f"  wrote -> {plot_b_kappa_evolution(field_rows, kappa_rows, geom, outdir)}")

    # ── overview figures ──────────────────────────────────────────────────
    if not args.skip_overview:
        print(f"  wrote -> {plot_overview_panel(fields, geom, outdir, args.panel_times)}")
        print(f"  wrote -> {plot_flux_evolution(field_rows, outdir)}")

    # ── manifest ──────────────────────────────────────────────────────────
    summary = {
        "data_dir": str(discovered["data_dir"]),
        "geometry": geom.summary_dict(),
        "inputs": {
            "n_field_snapshots": len(fields),
            "field_steps": [min(fields), max(fields)],
            "n_particle_snapshots": len(prt_series),
            "particle_series": sorted(particles_by_series) or None,
            "moments_available": bool(discovered["moments"]),
        },
        "regions": {"window": window.describe(geom),
                    "sheet": sheet.describe(geom)},
        "kappa_estimator": {
            "species": args.species, "s_max": args.s_max,
            "n_boot": args.kappa_boot, "max_particles": args.max_particles,
            "note": "truncated, whitened moment estimator (kappa_eff.py); "
                    "1/kappa = 0 means Maxwellian-consistent",
        },
        "correlation": corr_block,
    }
    with open(outdir / "reconnection_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    print(f"  wrote -> {outdir / 'reconnection_summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True,
                   help="directory with one reconnection run (pfd/prt files)")
    p.add_argument("--outdir", default="../analysis_results/reconnection/10_reconnection")
    p.add_argument("--profile", default="reconnection_comparable",
                   choices=sorted(RECONNECTION_PROFILES),
                   help="which psc_reconnection*.cxx produced the data")
    p.add_argument("--species", default="ion", choices=("ion", "electron"),
                   help="species for kappa_eff (Harris + background together)")
    p.add_argument("--sheet-half-width-di", type=float, default=2.0,
                   help="half-width of the sheet region in d_i (default 2)")
    p.add_argument("--sheet-y-di", type=float, default=None,
                   help="override the perturbed-sheet y position [d_i]")
    p.add_argument("--s-max", type=float, default=kappa_eff.DEFAULT_S_MAX,
                   help="truncation radius of the kappa estimator")
    p.add_argument("--kappa-boot", type=int, default=12,
                   help="bootstrap resamples for the kappa error (0 = off)")
    p.add_argument("--max-particles", type=int, default=2_000_000,
                   help="uniform subsample cap per particle snapshot")
    p.add_argument("--max-field-snapshots", type=int, default=400,
                   help="subsample the field series beyond this many "
                        "snapshots (particle-matched steps are always kept)")
    p.add_argument("--panel-times", type=int, default=4,
                   help="columns in the overview panel")
    p.add_argument("--skip-overview", action="store_true",
                   help="only the time series and the correlation")
    # Profile overrides, for runs whose .cxx drifted from the table above.
    p.add_argument("--mass-ratio", type=float, default=None)
    p.add_argument("--wpe-wce", type=float, default=None)
    p.add_argument("--cfl", type=float, default=None)
    p.add_argument("--dt-code", type=float, default=None,
                   help="exact dt from the runtime log (overrides the CFL estimate)")
    return p


def main() -> int:
    return run_analysis(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
