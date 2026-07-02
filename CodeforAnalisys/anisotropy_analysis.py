#!/usr/bin/env python3
"""
anisotropy_analysis.py — Brazil Plot Generator
===============================================
Genera:
  1. Brazil plot acumulado con trayectoria temporal de la mediana
  2. Panel temporal: anisotropia media + beta_par medio vs Omega_ci t
  3. Brazil plots individuales para ~10 snapshots clave

Los parámetros físicos, la malla y las conversiones se toman de psc_units.py.
"""

import warnings
import argparse
import csv
import re
import os
import concurrent.futures
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path

from data_reader import PICDataReader
from psc_units import (
    B0, BETA_E_PAR, BETA_E_PERP_OVER_PAR, BETA_I_PAR,
    BETA_I_PERP_OVER_PAR, DRIVEN_SPECIES, FIELD_FILE_PATTERN, INSTABILITY,
    MASS_RATIO, MOMENT_FILE_PATTERN, M_ELEC, M_ION, PROFILE_LABEL,
    step_to_omegaci,
)

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260612)
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

# ── Parametros fisicos iniciales ──────────────────────────────────────────────
N0       = 1.0
MU0      = 1.0
ACTIVE_BETA_INITIAL = BETA_I_PAR if DRIVEN_SPECIES == "ion" else BETA_E_PAR
ACTIVE_ANISOTROPY_INITIAL = (
    BETA_I_PERP_OVER_PAR if DRIVEN_SPECIES == "ion"
    else BETA_E_PERP_OVER_PAR
)
SPECIES_SYMBOL = "i" if DRIVEN_SPECIES == "ion" else "e"
SPECIES_NAME = "ions" if DRIVEN_SPECIES == "ion" else "electrons"
OUTPUT_PREFIX = ""


def output_path(outdir: Path, stem: str, suffix: str = ".png") -> Path:
    return outdir / f"{OUTPUT_PREFIX}{stem}{suffix}"

# ── Colores ───────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
TEXT_CLR = "#e6edf3"
GRID_CLR = "#21262d"
ACCENT   = "#58a6ff"


# ── Umbrales de inestabilidad ─────────────────────────────────────────────────
def mirror_threshold(b):   return 1.0 + 1.0 / b
def firehose_threshold(b): return 1.0 - 2.0 / b
def oblique_firehose_threshold(b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0.11, 1.0 - 1.4 / (b - 0.11)**0.55, np.nan)
def ic_threshold(b):       return 1.0 + 0.43 / b**0.42
def whistler_threshold(b): return 1.0 + 0.21 / b**0.6


def instability_threshold(beta):
    """Return the active marginal-stability threshold at beta_parallel."""
    beta = np.maximum(np.asarray(beta, dtype=float), 1e-12)
    if INSTABILITY == "firehose":
        return firehose_threshold(beta)
    if INSTABILITY == "mirror":
        return mirror_threshold(beta)
    return whistler_threshold(beta)


def instability_drive(anisotropy, threshold):
    """Positive values mean that the state is on the unstable side."""
    if INSTABILITY == "firehose":
        return np.asarray(threshold) - np.asarray(anisotropy)
    return np.asarray(anisotropy) - np.asarray(threshold)


def field_aligned_pressures(
    Pxx, Pyy, Pzz, Pxy, Pyz, Pzx, Bx, By, Bz
):
    """Project a symmetric pressure tensor onto the local magnetic field."""
    B2 = Bx**2 + By**2 + Bz**2
    inv_B = 1.0 / np.sqrt(np.maximum(B2, 1e-30))
    bx, by, bz = Bx * inv_B, By * inv_B, Bz * inv_B
    p_par = (
        Pxx * bx**2 + Pyy * by**2 + Pzz * bz**2
        + 2.0 * Pxy * bx * by
        + 2.0 * Pyz * by * bz
        + 2.0 * Pzx * bz * bx
    )
    p_perp = 0.5 * (Pxx + Pyy + Pzz - p_par)
    return p_par, p_perp, B2


# ── Lectura de un snapshot ────────────────────────────────────────────────────
def process_snapshot(mom_file: str, bz_file: str, species: str = DRIVEN_SPECIES):
    """Devuelve dict{anisotropy, beta_par} o None si hay error."""
    suffix = "i" if species == "ion" else "e"
    mass = M_ION if species == "ion" else M_ELEC
    try:
        moments = PICDataReader.read_multiple_fields_3d(
            mom_file, "all_1st",
            [
                f"txx_{suffix}/p0/3d", f"tyy_{suffix}/p0/3d",
                f"tzz_{suffix}/p0/3d", f"txy_{suffix}/p0/3d",
                f"tyz_{suffix}/p0/3d", f"tzx_{suffix}/p0/3d",
                f"px_{suffix}/p0/3d",
                f"py_{suffix}/p0/3d", f"pz_{suffix}/p0/3d",
                f"rho_{suffix}/p0/3d",
            ],
        )
        b_fields = PICDataReader.read_multiple_fields_3d(
            bz_file, "jeh",
            ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"],
        )
    except Exception as exc:
        print(f"  [WARN] {exc}")
        return None

    Mxx = moments[f"txx_{suffix}/p0/3d"].ravel()
    Myy = moments[f"tyy_{suffix}/p0/3d"].ravel()
    Mzz = moments[f"tzz_{suffix}/p0/3d"].ravel()
    Mxy = moments[f"txy_{suffix}/p0/3d"].ravel()
    Myz = moments[f"tyz_{suffix}/p0/3d"].ravel()
    Mzx = moments[f"tzx_{suffix}/p0/3d"].ravel()
    px  = moments[f"px_{suffix}/p0/3d"].ravel()
    py  = moments[f"py_{suffix}/p0/3d"].ravel()
    pz  = moments[f"pz_{suffix}/p0/3d"].ravel()
    rho = moments[f"rho_{suffix}/p0/3d"].ravel()
    n = rho if species == "ion" else -rho
    Bx  = b_fields["hx_fc/p0/3d"].ravel()
    By  = b_fields["hy_fc/p0/3d"].ravel()
    Bz  = b_fields["hz_fc/p0/3d"].ravel()
    B2  = Bx**2 + By**2 + Bz**2

    safe_n = np.where(n > 0.05, n, np.nan)
    # Thermal pressure is the central second moment. Using Mii directly
    # contaminates temperature and beta when a bulk drift develops.
    Pxx = Mxx - px**2 / (safe_n * mass)
    Pyy = Myy - py**2 / (safe_n * mass)
    Pzz = Mzz - pz**2 / (safe_n * mass)
    Pxy = Mxy - px * py / (safe_n * mass)
    Pyz = Myz - py * pz / (safe_n * mass)
    Pzx = Mzx - pz * px / (safe_n * mass)
    P_par, P_perp, B2 = field_aligned_pressures(
        Pxx, Pyy, Pzz, Pxy, Pyz, Pzx, Bx, By, Bz
    )
    T_par  = P_par / safe_n
    T_perp = P_perp / safe_n
    aniso  = T_perp / (T_par + 1e-30)
    P_mag  = B2 / (2.0 * MU0)
    beta   = P_par / (P_mag + 1e-30)

    mask = (
        (P_par > 0) & (P_perp > 0) & (n > 0.05) & (B2 > 1e-10)
        & np.isfinite(beta) & np.isfinite(aniso)
        & (aniso > 0.05) & (aniso < 20.0)
        & (beta  > 0.1)  & (beta  < 2000.0)   # recorta artefactos numericos
    )
    if not np.any(mask):
        return None

    return {
        "anisotropy": aniso[mask],
        "beta_par": beta[mask],
        # Global thermodynamic state. Ratios of volume-averaged pressures are
        # the appropriate trajectory coordinates; medians remain useful for
        # showing spatial spread but are not the trajectory itself.
        "anisotropy_global": float(
            np.mean(P_perp[mask]) / np.mean(P_par[mask])
        ),
        "beta_global": float(
            2.0 * np.mean(P_par[mask]) / np.mean(B2[mask])
        ),
        "valid_cells": int(np.count_nonzero(mask)),
    }


# ── Decorar ejes Brasil ───────────────────────────────────────────────────────
def _draw_thresholds(ax, xmin, xmax, ymin, ymax):
    b = np.logspace(np.log10(xmin * 0.5), np.log10(xmax * 2), 600)

    if INSTABILITY == "whistler":
        wh = whistler_threshold(b)
        ok = (wh >= ymin * 0.7) & (wh <= ymax * 1.5)
        ax.plot(b[ok], wh[ok], "--", color="#c084fc", lw=2.2, zorder=8,
                label=r"Whistler  $1+0.21/\beta_{e\parallel}^{0.6}$")
        ax.fill_between(b, np.clip(wh, ymin, ymax * 2), ymax * 2,
                        alpha=0.08, color="#c084fc", zorder=2)
        ax.axhline(1.0, color=TEXT_CLR, alpha=0.25, lw=0.9, ls=":")
        return

    # Mirror
    m = mirror_threshold(b)
    ok = (m >= ymin * 0.7) & (m <= ymax * 1.5)
    ax.plot(b[ok], m[ok], "--", color="#ff6b6b", lw=2.2, zorder=8, alpha=0.9,
            label=r"Mirror  $1+1/\beta_\parallel$")
    ax.fill_between(b, np.clip(m, ymin, ymax * 2), ymax * 2,
                    alpha=0.08, color="#ff4444", zorder=2)

    # Firehose (fluid and oblique kinetic approximations)
    bf = b[b > 2.05]
    fh = firehose_threshold(bf)
    ok = (fh >= ymin * 0.5) & (fh <= ymax * 1.5)
    ax.plot(bf[ok], fh[ok], "--", color="#74b9ff", lw=2.2, zorder=8, alpha=0.9,
            label=r"Firehose  $1-2/\beta_\parallel$")
    ax.fill_between(bf, ymin * 0.3, np.clip(fh, ymin * 0.3, ymax),
                    alpha=0.08, color="#0984e3", zorder=2)
    bfo = b[b > 0.12]
    ofh = oblique_firehose_threshold(bfo)
    ok = np.isfinite(ofh) & (ofh >= ymin * 0.5) & (ofh <= ymax * 1.5)
    ax.plot(bfo[ok], ofh[ok], "-.", color="#c084fc", lw=1.7, zorder=8,
            alpha=0.9, label="Oblique firehose")

    # Ion-cyclotron
    ic = ic_threshold(b)
    ok = (ic >= ymin * 0.7) & (ic <= ymax * 1.5)
    ax.plot(b[ok], ic[ok], ":", color="#55efc4", lw=1.8, zorder=8, alpha=0.85,
            label=r"IC  $1+0.43/\beta_\parallel^{0.42}$")

    ax.axhline(1.0, color=TEXT_CLR, alpha=0.25, lw=0.9, ls=":")


def _style_ax(ax, title=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(which="both", colors=TEXT_CLR, direction="in", top=True, right=True, labelsize=POSTER_TICK)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_CLR)
    ax.grid(True, which="both", alpha=0.18, color=GRID_CLR, ls=":")
    if title:
        ax.set_title(title, fontsize=POSTER_TITLE, fontweight="bold", color=TEXT_CLR, pad=8)


def _robust_plot_ranges(beta, aniso):
    """Choose shared log-scale limits without hard-coding one old run."""
    beta = np.asarray(beta)
    aniso = np.asarray(aniso)
    valid = (
        np.isfinite(beta) & np.isfinite(aniso)
        & (beta > 0) & (aniso > 0)
    )
    if valid.sum() < 10:
        return 0.5, 50.0, 0.3, 5.0

    b_lo, b_hi = np.percentile(beta[valid], [0.5, 99.5])
    a_lo, a_hi = np.percentile(aniso[valid], [0.5, 99.5])
    xmin = max(0.05, b_lo / 1.25)
    xmax = max(xmin * 10.0, b_hi * 1.25)
    ymin = max(0.05, min(1.0, a_lo) / 1.2)
    ymax = max(1.5, max(1.0, a_hi) * 1.2)
    return xmin, xmax, ymin, ymax


# ── Plot 1: Brazil acumulado coloreado por tiempo ─────────────────────────────
def plot_brazil_accumulated(
    all_beta, all_aniso, snap_stats, steps, outdir: Path, b0_ref: float
):
    """Scatter coloreado por Omega_ci*t con histograma de densidad superpuesto."""
    if len(all_beta) == 0:
        print("[WARN] Sin datos para Brazil acumulado"); return

    xmin, xmax, ymin, ymax = _robust_plot_ranges(all_beta, all_aniso)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    in_range = (
        (all_beta  >= xmin) & (all_beta  <= xmax)
        & (all_aniso >= ymin) & (all_aniso <= ymax)
    )
    bv = all_beta[in_range]
    av = all_aniso[in_range]

    # Fondo: densidad 2D con bins logaritmicos
    xbins = np.logspace(np.log10(xmin), np.log10(xmax), 120)
    ybins = np.logspace(np.log10(ymin), np.log10(ymax), 80)
    H, xe, ye = np.histogram2d(bv, av, bins=[xbins, ybins])
    H = H.T
    H_masked = np.ma.masked_where(H < 1, H)
    pcm = ax.pcolormesh(xe, ye, H_masked, cmap="inferno",
                        norm=mcolors.LogNorm(vmin=1, vmax=H.max()),
                        shading="flat", zorder=3)
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("Point density [log]", fontsize=POSTER_LABEL - 1, color=TEXT_CLR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_CLR)

    _draw_thresholds(ax, xmin, xmax, ymin, ymax)

    # Condicion inicial
    beta_init = ACTIVE_BETA_INITIAL
    aniso_init = ACTIVE_ANISOTROPY_INITIAL
    ax.plot(beta_init, aniso_init, "*", color="#ffd700",
            markeredgecolor="white", markeredgewidth=0.7,
            markersize=20, zorder=12,
            label=rf"CI  ($\beta_{{{SPECIES_SYMBOL}\parallel}}={beta_init:g}$, "
                  rf"$A_{SPECIES_SYMBOL}={aniso_init:g}$)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel(rf"$\beta_{{{SPECIES_SYMBOL}\parallel}}$ — parallel pressure / magnetic pressure",
                  fontsize=POSTER_LABEL, color=TEXT_CLR, labelpad=8)
    ax.set_ylabel(rf"$T_{{{SPECIES_SYMBOL}\perp}}/T_{{{SPECIES_SYMBOL}\parallel}}$",
                  fontsize=POSTER_LABEL, color=TEXT_CLR, labelpad=8)

    if snap_stats:
        beta_med = np.array([s["beta_global"] for s in snap_stats])
        aniso_med = np.array([s["aniso_global"] for s in snap_stats])
        time_med = np.array([s["toci"] for s in snap_stats])
        ax.plot(beta_med, aniso_med, color="white", lw=2.0, alpha=0.9, zorder=10)
        ax.scatter(
            beta_med, aniso_med, c=time_med, cmap="cool", s=46,
            edgecolors="white", linewidths=0.4, zorder=11,
            label="Global state per snapshot",
        )
        ax.scatter(beta_med[0], aniso_med[0], marker="D", s=90,
                   color="#2ecc71", edgecolor="white", zorder=12, label="Measured start")
        ax.scatter(beta_med[-1], aniso_med[-1], marker="*", s=190,
                   color="#f1c40f", edgecolor="white", zorder=12, label="Measured end")
        for idx in np.unique(np.linspace(0, len(beta_med) - 1, min(6, len(beta_med)), dtype=int)):
            ax.annotate(
                rf"{time_med[idx]:.2f}",
                (beta_med[idx], aniso_med[idx]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=13, color=TEXT_CLR,
            )

    _style_ax(ax, rf"Brazil Plot — {PROFILE_LABEL}  ($m_i/m_e={int(MASS_RATIO)}$)")

    t_max_oci = step_to_omegaci(steps[-1]) if steps else 0
    ax.text(0.98, 0.02,
            f"{len(bv):,} points  |  {len(steps)} snapshots  |  "
            rf"$t_{{max}} = {t_max_oci:.1f}\,\Omega_{{ci}}^{{-1}}$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=13, color="#8b949e")

    ax.legend(fontsize=POSTER_LEGEND, framealpha=0.55,
              facecolor="#1c2128", edgecolor="#30363d", labelcolor=TEXT_CLR,
              loc="upper right")

    out = output_path(outdir, "brazil_trayectoria")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Guardado → {out}")


# ── Plot 2: Evolucion temporal  <A_i> y <beta_par> vs Omega_ci*t ──────────────
def plot_temporal_evolution(snap_data: list, outdir: Path):
    """snap_data: lista de dicts {step, toci, aniso_mean, aniso_med, beta_mean, beta_med}"""
    if not snap_data:
        return

    toci  = np.array([s["toci"]      for s in snap_data])
    a_med = np.array([s["aniso_med"] for s in snap_data])
    a_global = np.array([s["aniso_global"] for s in snap_data])
    inverse_global = 1.0 / np.maximum(a_global, 1e-30)
    a_p25 = np.array([s["aniso_p25"] for s in snap_data])
    a_p75 = np.array([s["aniso_p75"] for s in snap_data])
    b_med = np.array([s["beta_med"]  for s in snap_data])
    b_global = np.array([s["beta_global"] for s in snap_data])
    b_p25 = np.array([s["beta_p25"]  for s in snap_data])
    b_p75 = np.array([s["beta_p75"]  for s in snap_data])

    fig_a, ax1 = plt.subplots(figsize=(9.5, 6.0))
    fig_inv, ax_inv = plt.subplots(figsize=(9.5, 6.0))
    fig_b, ax2 = plt.subplots(figsize=(9.5, 6.0))
    for fig in (fig_a, fig_inv, fig_b):
        fig.patch.set_facecolor(DARK_BG)

    # ─ Panel superior: anisotropia ─
    ax1.set_facecolor(PANEL_BG)
    ax1.fill_between(toci, a_p25, a_p75, alpha=0.25, color="#ff6b6b")
    ax1.plot(toci, a_global, color="#ff6b6b", marker="o", ms=3.5, lw=2.2,
             label=r"global $\langle P_\perp\rangle/\langle P_\parallel\rangle$")
    ax1.plot(toci, a_med, color="#ffb4b4", lw=1.0, alpha=0.75,
             label="per-cell median")
    ax1.axhline(1.0, color=TEXT_CLR, alpha=0.3, lw=0.9, ls="--", label="Isotropy A=1")

    dynamic_threshold = instability_threshold(b_global)
    if INSTABILITY == "firehose":
        threshold_label = r"Firehose threshold $1-2/\beta_\parallel(t)$"
        threshold_color = "#74b9ff"
    elif INSTABILITY == "mirror":
        threshold_label = r"Mirror threshold $1+1/\beta_\parallel(t)$"
        threshold_color = "#ff9999"
    else:
        threshold_label = r"Whistler threshold $1+0.21/\beta_{e\parallel}^{0.6}$"
        threshold_color = "#c084fc"
    ax1.plot(toci, dynamic_threshold, color=threshold_color, alpha=0.9, lw=1.2,
             ls=":", label=threshold_label)

    ax1.set_ylabel(r"$T_\perp / T_\parallel$", fontsize=POSTER_LABEL, color=TEXT_CLR)
    ax1.set_xlabel(r"$t\,\Omega_{ci}$", fontsize=POSTER_LABEL, color=TEXT_CLR, labelpad=6)
    finite_a = np.concatenate([a_p25, a_p75, a_global, dynamic_threshold])
    ax1.set_ylim(max(0.05, np.nanpercentile(finite_a, 1) * 0.8),
                 np.nanpercentile(finite_a, 99) * 1.2)
    _style_ax(ax1, f"Temporal Evolution — {PROFILE_LABEL}")
    ax1.legend(fontsize=POSTER_LEGEND, framealpha=0.5,
               facecolor="#1c2128", edgecolor="#30363d", labelcolor=TEXT_CLR)

    # La razón inversa evita ambigüedad en Firehose: T_par/T_perp decrece
    # mientras A=T_perp/T_par aumenta hacia la isotropía.
    ax_inv.set_facecolor(PANEL_BG)
    ax_inv.plot(
        toci, inverse_global, color="#f9c74f", marker="o", ms=3.5, lw=2.2,
        label=r"global $\langle P_\parallel\rangle/\langle P_\perp\rangle$",
    )
    ax_inv.axhline(1.0, color=TEXT_CLR, alpha=0.3, lw=0.9, ls="--")
    ax_inv.set_ylabel(r"$T_\parallel/T_\perp$", fontsize=POSTER_LABEL, color=TEXT_CLR)
    ax_inv.set_xlabel(r"$t\,\Omega_{ci}$", fontsize=POSTER_LABEL, color=TEXT_CLR, labelpad=6)
    _style_ax(ax_inv)
    ax_inv.legend(fontsize=POSTER_LEGEND, framealpha=0.5,
                  facecolor="#1c2128", edgecolor="#30363d", labelcolor=TEXT_CLR)

    # ─ Panel inferior: beta_par ─
    ax2.set_facecolor(PANEL_BG)
    ax2.fill_between(toci, b_p25, b_p75, alpha=0.25, color="#58a6ff")
    ax2.plot(toci, b_global, color="#58a6ff", marker="o", ms=3.5, lw=2.2,
             label=r"global $2\langle P_\parallel\rangle/\langle B^2\rangle$")
    ax2.plot(toci, b_med, color="#a8d4ff", lw=1.0, alpha=0.75,
             label="per-cell median")
    ax2.set_ylabel(r"$\beta_{i\parallel}$", fontsize=POSTER_LABEL, color=TEXT_CLR)
    ax2.set_xlabel(r"$t\,\Omega_{ci}$", fontsize=POSTER_LABEL, color=TEXT_CLR, labelpad=6)
    ax2.set_yscale("log")
    _style_ax(ax2)
    ax2.legend(fontsize=POSTER_LEGEND, framealpha=0.5,
               facecolor="#1c2128", edgecolor="#30363d", labelcolor=TEXT_CLR)

    for ax in (ax1, ax_inv, ax2):
        ax.tick_params(which="both", colors=TEXT_CLR, direction="in",
                       top=True, right=True)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_CLR)
        ax.grid(True, which="both", alpha=0.18, color=GRID_CLR, ls=":")

    outputs = [
        (fig_a, output_path(outdir, "anisotropy_ratio_vs_time")),
        (fig_inv, output_path(outdir, "inverse_anisotropy_vs_time")),
        (fig_b, output_path(outdir, "beta_parallel_vs_time")),
    ]
    for fig, out in outputs:
        fig.tight_layout()
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print(f"  Guardado → {out}")


# ── Plot 3: Grid de Brazil plots por snapshot ─────────────────────────────────
def plot_brazil_grid(snap_list: list, outdir: Path, b0_ref: float, n_cols=4):
    """
    snap_list: lista de dicts {step, toci, aniso, beta_par}
    Genera un grid de N_SNAPS Brazil plots individuales.
    """
    if not snap_list:
        return

    n = len(snap_list)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4.5 * n_rows),
                             constrained_layout=True)
    fig.patch.set_facecolor(DARK_BG)
    axes_flat = np.array(axes).ravel()

    all_beta = np.concatenate([s["beta_par"] for s in snap_list])
    all_aniso = np.concatenate([s["anisotropy"] for s in snap_list])
    xmin, xmax, ymin, ymax = _robust_plot_ranges(all_beta, all_aniso)

    cmap_time = cm.plasma
    t_all = np.array([s["toci"] for s in snap_list])
    t_norm = mcolors.Normalize(vmin=t_all.min(), vmax=t_all.max())

    beta_init = ACTIVE_BETA_INITIAL
    aniso_init = ACTIVE_ANISOTROPY_INITIAL

    for i, snap in enumerate(snap_list):
        ax = axes_flat[i]
        ax.set_facecolor(PANEL_BG)

        bv = snap["beta_par"]
        av = snap["anisotropy"]
        in_r = (bv >= xmin) & (bv <= xmax) & (av >= ymin) & (av <= ymax)
        bv = bv[in_r]; av = av[in_r]

        if len(bv) > 0:
            xb = np.logspace(np.log10(xmin), np.log10(xmax), 60)
            yb = np.logspace(np.log10(ymin), np.log10(ymax), 40)
            H, xe, ye = np.histogram2d(bv, av, bins=[xb, yb])
            H = H.T
            Hm = np.ma.masked_where(H < 1, H)
            ax.pcolormesh(xe, ye, Hm, cmap="plasma",
                          norm=mcolors.LogNorm(vmin=1, vmax=max(H.max(), 2)),
                          shading="flat", zorder=3)

        _draw_thresholds(ax, xmin, xmax, ymin, ymax)
        ax.plot(beta_init, aniso_init, "*", color="#ffd700",
                markeredgecolor="white", markeredgewidth=0.6,
                markersize=14, zorder=12)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

        toci = snap["toci"]
        color = cmap_time(t_norm(toci))
        ax.set_title(rf"$t\,\Omega_{{ci}} = {toci:.2f}$",
                     fontsize=14, fontweight="bold", color=color, pad=5)

        ax.tick_params(which="both", colors=TEXT_CLR, direction="in",
                       labelsize=11, top=True, right=True)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_CLR)
        ax.grid(True, which="both", alpha=0.15, color=GRID_CLR, ls=":")

        if i % n_cols == 0:
            ax.set_ylabel(rf"$T_{{{SPECIES_SYMBOL}\perp}}/T_{{{SPECIES_SYMBOL}\parallel}}$",
                          fontsize=13, color=TEXT_CLR)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel(rf"$\beta_{{{SPECIES_SYMBOL}\parallel}}$",
                          fontsize=13, color=TEXT_CLR)

    # Ocultar ejes sobrantes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        rf"Brazil Plots per Snapshot — {PROFILE_LABEL}  ($m_i/m_e={int(MASS_RATIO)}$)",
        fontsize=POSTER_TITLE + 1, fontweight="bold", color=TEXT_CLR, y=1.01
    )

    out = output_path(outdir, "brazil_snapshots")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Guardado → {out}")


def write_summary_csv(snap_stats: list, outdir: Path):
    """Write the exact trajectory values used by the temporal plots."""
    if not snap_stats:
        return
    columns = [
        "case", "species", "step", "omega_ci_t", "anisotropy_global",
        "parallel_over_perpendicular_global", "anisotropy_median",
        "anisotropy_p25", "anisotropy_p75", "beta_parallel_global",
        "beta_parallel_median", "beta_parallel_p25", "beta_parallel_p75",
        "marginal_threshold", "instability_drive", "valid_cells",
    ]
    out = output_path(outdir, "anisotropy_evolution", ".csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for stat in snap_stats:
            writer.writerow({
                "case": PROFILE_LABEL,
                "species": DRIVEN_SPECIES,
                "step": stat["step"],
                "omega_ci_t": stat["toci"],
                "anisotropy_global": stat["aniso_global"],
                "parallel_over_perpendicular_global": 1.0 / stat["aniso_global"],
                "anisotropy_median": stat["aniso_med"],
                "anisotropy_p25": stat["aniso_p25"],
                "anisotropy_p75": stat["aniso_p75"],
                "beta_parallel_global": stat["beta_global"],
                "beta_parallel_median": stat["beta_med"],
                "beta_parallel_p25": stat["beta_p25"],
                "beta_parallel_p75": stat["beta_p75"],
                "marginal_threshold": stat["threshold"],
                "instability_drive": stat["drive"],
                "valid_cells": stat["valid_cells"],
            })
    print(f"  Guardado → {out}")


# ── Analisis principal ────────────────────────────────────────────────────────
def run_analysis(mom_pattern: str, bz_pattern: str, B0_ref: float,
                 outdir: str = "anisotropy_plots", n_grid_snaps: int = 12,
                 run_name: str = "", jobs: int = 0):
    global OUTPUT_PREFIX
    clean_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("_")
    OUTPUT_PREFIX = f"{clean_name}_" if clean_name else ""

    out = Path(outdir)
    out.mkdir(exist_ok=True, parents=True)

    mom_files = PICDataReader.find_files(mom_pattern)
    bz_files  = PICDataReader.find_files(bz_pattern)
    common    = sorted(set(mom_files) & set(bz_files))

    if not common:
        print("[ERROR] No se encontraron pares momento/campo."); return

    print(f"Snapshots encontrados: {len(common)}  (steps {common[0]}–{common[-1]})")

    # ── Acumular datos ────────────────────────────────────────────────────────
    all_beta, all_aniso = [], []
    snap_stats  = []   # para plot temporal
    grid_snaps  = []   # para grid de Brazils

    # Seleccionar ~n_grid_snaps pasos uniformes para el grid
    idx_grid = set(np.linspace(0, len(common) - 1, n_grid_snaps, dtype=int))

    num_workers = jobs if jobs > 0 else os.cpu_count() or 1
    num_workers = min(num_workers, len(common))
    print(f"Ejecutando con {num_workers} procesos en paralelo...")

    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_snapshot, mom_files[step], bz_files[step], DRIVEN_SPECIES): step
            for step in common
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            step = futures[future]
            toci = step_to_omegaci(step)
            print(f"  Procesado step {step:6d}  ({i+1}/{len(common)})  t*Oci={toci:.3f}", end="\r")
            try:
                results[step] = future.result()
            except Exception as exc:
                print(f"\n  [ERROR] Excepción procesando step {step}: {exc}")
                results[step] = None
    print()

    # Procesar secuencialmente en el orden original para mantener consistencia
    for i, step in enumerate(common):
        data = results.get(step)
        if data is None:
            continue

        bv = data["beta_par"]; av = data["anisotropy"]
        n_pts = len(bv)

        # Acumular acumulado (sub-muestreo)
        if n_pts > 0:
            sub = min(n_pts, 4000)
            idx = RNG.choice(n_pts, sub, replace=False)
            all_beta.extend(bv[idx])
            all_aniso.extend(av[idx])

        # Estadisticas por paso
        if n_pts > 5:
            threshold = float(instability_threshold(data["beta_global"]))
            snap_stats.append({
                "step":      step,
                "toci":      step_to_omegaci(step),
                "aniso_global": data["anisotropy_global"],
                "aniso_med": float(np.median(av)),
                "aniso_p25": float(np.percentile(av, 25)),
                "aniso_p75": float(np.percentile(av, 75)),
                "beta_global": data["beta_global"],
                "beta_med":  float(np.median(bv)),
                "beta_p25":  float(np.percentile(bv, 25)),
                "beta_p75":  float(np.percentile(bv, 75)),
                "threshold": threshold,
                "drive": float(
                    instability_drive(data["anisotropy_global"], threshold)
                ),
                "valid_cells": data["valid_cells"],
            })

        # Guardar snapshot completo para el grid
        if i in idx_grid and n_pts > 0:
            grid_snaps.append({
                "step": step, "toci": step_to_omegaci(step),
                "anisotropy": av, "beta_par": bv,
            })

    print(f"Puntos acumulados: {len(all_beta):,}")

    all_beta  = np.asarray(all_beta)
    all_aniso = np.asarray(all_aniso)
    print("Generando graficas...")
    plot_brazil_accumulated(all_beta, all_aniso, snap_stats, common, out, B0_ref)
    plot_temporal_evolution(snap_stats, out)
    plot_brazil_grid(grid_snaps, out, B0_ref, n_cols=4)
    write_summary_csv(snap_stats, out)
    if snap_stats:
        first, last = snap_stats[0], snap_stats[-1]
        print(
            "Evolucion global: "
            f"A_{SPECIES_SYMBOL} {first['aniso_global']:.5g} -> "
            f"{last['aniso_global']:.5g}; beta_{SPECIES_SYMBOL}|| "
            f"{first['beta_global']:.5g} -> {last['beta_global']:.5g}; "
            f"drive inestable {first['drive']:.5g} -> {last['drive']:.5g}"
        )
    print("Listo.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brazil plots para simulaciones de anisotropia PSC."
    )
    parser.add_argument("--data-dir", type=Path,
                        help="Directorio PSC; detecta pfd y pfd_moments automáticamente.")
    parser.add_argument("--moments", default=MOMENT_FILE_PATTERN,
                        help="Glob de archivos de momentos.")
    parser.add_argument("--fields",  default=FIELD_FILE_PATTERN,
                        help="Glob de archivos de campos.")
    parser.add_argument("--B0",      type=float, default=B0,
                        help="Campo de referencia B0.")
    parser.add_argument("--outdir",  default="anisotropy_plots",
                        help="Directorio de salida.")
    parser.add_argument("--nsnaps",  type=int, default=12,
                        help="Snapshots en el grid de Brazil plots.")
    parser.add_argument("--run-name", default="",
                        help="Nombre incluido en cada archivo de salida.")
    parser.add_argument("--jobs", "-j", type=int, default=0,
                        help="Número de procesos en paralelo (0 = usar todos los cores disponibles).")
    args = parser.parse_args()

    if args.data_dir:
        discovered = PICDataReader.discover_outputs(str(args.data_dir))
        mom_pattern = str(discovered["data_dir"] / "pfd_moments.*_p*.h5")
        field_pattern = str(discovered["data_dir"] / "pfd.*_p*.h5")
    else:
        mom_pattern = args.moments
        field_pattern = args.fields

    run_analysis(
        mom_pattern=mom_pattern,
        bz_pattern=field_pattern,
        B0_ref=args.B0,
        outdir=args.outdir,
        n_grid_snaps=args.nsnaps,
        run_name=args.run_name,
        jobs=args.jobs,
    )
