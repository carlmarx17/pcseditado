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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path

from data_reader import PICDataReader
from psc_units import (
    B0, FIELD_FILE_PATTERN, MASS_RATIO, MOMENT_FILE_PATTERN, M_ION,
    PROFILE_LABEL, TI_PAR, TI_PERP, step_to_omegaci,
)

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260612)

# ── Parametros fisicos iniciales ──────────────────────────────────────────────
T_PERP_I = TI_PERP
T_PAR_I  = TI_PAR
N0       = 1.0
MU0      = 1.0

# ── Colores ───────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
TEXT_CLR = "#e6edf3"
GRID_CLR = "#21262d"
ACCENT   = "#58a6ff"


# ── Umbrales de inestabilidad ─────────────────────────────────────────────────
def mirror_threshold(b):   return 1.0 + 1.0 / b
def firehose_threshold(b): return 1.0 - 1.0 / b
def ic_threshold(b):       return 1.0 + 0.43 / b**0.42


# ── Lectura de un snapshot ────────────────────────────────────────────────────
def process_snapshot(mom_file: str, bz_file: str):
    """Devuelve dict{anisotropy, beta_par} o None si hay error."""
    try:
        moments = PICDataReader.read_multiple_fields_3d(
            mom_file, "all_1st",
            [
                "txx_i/p0/3d", "tyy_i/p0/3d", "tzz_i/p0/3d",
                "px_i/p0/3d", "py_i/p0/3d", "pz_i/p0/3d",
                "rho_i/p0/3d",
            ],
        )
        b_fields = PICDataReader.read_multiple_fields_3d(
            bz_file, "jeh",
            ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"],
        )
    except Exception as exc:
        print(f"  [WARN] {exc}")
        return None

    Mxx = moments["txx_i/p0/3d"].ravel()
    Myy = moments["tyy_i/p0/3d"].ravel()
    Mzz = moments["tzz_i/p0/3d"].ravel()
    px  = moments["px_i/p0/3d"].ravel()
    py  = moments["py_i/p0/3d"].ravel()
    pz  = moments["pz_i/p0/3d"].ravel()
    n   = moments["rho_i/p0/3d"].ravel()
    Bx  = b_fields["hx_fc/p0/3d"].ravel()
    By  = b_fields["hy_fc/p0/3d"].ravel()
    Bz  = b_fields["hz_fc/p0/3d"].ravel()
    B2  = Bx**2 + By**2 + Bz**2

    safe_n = np.where(n > 0.05, n, np.nan)
    # Thermal pressure is the central second moment. Using Mii directly
    # contaminates temperature and beta when a bulk drift develops.
    Pxx = Mxx - px**2 / (safe_n * M_ION)
    Pyy = Myy - py**2 / (safe_n * M_ION)
    Pzz = Mzz - pz**2 / (safe_n * M_ION)
    T_par  = Pzz / safe_n
    T_perp = 0.5 * (Pxx + Pyy) / safe_n
    aniso  = T_perp / (T_par + 1e-30)
    P_mag  = B2 / (2.0 * MU0)
    beta   = Pzz / (P_mag + 1e-30)

    mask = (
        (Pxx > 0) & (Pyy > 0) & (Pzz > 0) & (n > 0.05) & (B2 > 1e-10)
        & np.isfinite(beta) & np.isfinite(aniso)
        & (aniso > 0.05) & (aniso < 20.0)
        & (beta  > 0.1)  & (beta  < 2000.0)   # recorta artefactos numericos
    )
    return {"anisotropy": aniso[mask], "beta_par": beta[mask]}


# ── Decorar ejes Brasil ───────────────────────────────────────────────────────
def _draw_thresholds(ax, xmin, xmax, ymin, ymax):
    b = np.logspace(np.log10(xmin * 0.5), np.log10(xmax * 2), 600)

    # Mirror
    m = mirror_threshold(b)
    ok = (m >= ymin * 0.7) & (m <= ymax * 1.5)
    ax.plot(b[ok], m[ok], "--", color="#ff6b6b", lw=2.2, zorder=8, alpha=0.9,
            label=r"Mirror  $1+1/\beta_\parallel$")
    ax.fill_between(b, np.clip(m, ymin, ymax * 2), ymax * 2,
                    alpha=0.08, color="#ff4444", zorder=2)

    # Firehose (beta > 1)
    bf = b[b > 1.05]
    fh = firehose_threshold(bf)
    ok = (fh >= ymin * 0.5) & (fh <= ymax * 1.5)
    ax.plot(bf[ok], fh[ok], "--", color="#74b9ff", lw=2.2, zorder=8, alpha=0.9,
            label=r"Firehose  $1-1/\beta_\parallel$")
    ax.fill_between(bf, ymin * 0.3, np.clip(fh, ymin * 0.3, ymax),
                    alpha=0.08, color="#0984e3", zorder=2)

    # Ion-cyclotron
    ic = ic_threshold(b)
    ok = (ic >= ymin * 0.7) & (ic <= ymax * 1.5)
    ax.plot(b[ok], ic[ok], ":", color="#55efc4", lw=1.8, zorder=8, alpha=0.85,
            label=r"IC  $1+0.43/\beta_\parallel^{0.42}$")

    ax.axhline(1.0, color=TEXT_CLR, alpha=0.25, lw=0.9, ls=":")


def _style_ax(ax, title=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(which="both", colors=TEXT_CLR, direction="in", top=True, right=True, labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_CLR)
    ax.grid(True, which="both", alpha=0.18, color=GRID_CLR, ls=":")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_CLR, pad=8)


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
    cbar.set_label("Densidad de puntos [log]", fontsize=11, color=TEXT_CLR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_CLR)

    _draw_thresholds(ax, xmin, xmax, ymin, ymax)

    # Condicion inicial
    beta_init  = (N0 * T_PAR_I) / (b0_ref**2 / 2.0)
    aniso_init = T_PERP_I / T_PAR_I
    ax.plot(beta_init, aniso_init, "*", color="#ffd700",
            markeredgecolor="white", markeredgewidth=0.7,
            markersize=20, zorder=12,
            label=rf"CI  ($\beta_{{i\parallel}}={beta_init:.0f}$, $A_i={aniso_init:.1f}$)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$\beta_{i\parallel}$ — presión paralela / presión magnética",
                  fontsize=12, color=TEXT_CLR, labelpad=8)
    ax.set_ylabel(r"$T_\perp / T_\parallel$", fontsize=13, color=TEXT_CLR, labelpad=8)

    if snap_stats:
        beta_med = np.array([s["beta_med"] for s in snap_stats])
        aniso_med = np.array([s["aniso_med"] for s in snap_stats])
        time_med = np.array([s["toci"] for s in snap_stats])
        ax.plot(beta_med, aniso_med, color="white", lw=1.2, alpha=0.75, zorder=10)
        ax.scatter(
            beta_med, aniso_med, c=time_med, cmap="cool", s=28,
            edgecolors="white", linewidths=0.25, zorder=11,
            label="Mediana espacial por snapshot",
        )

    _style_ax(ax, rf"Brazil Plot — {PROFILE_LABEL}  ($m_i/m_e={int(MASS_RATIO)}$)")

    t_max_oci = step_to_omegaci(steps[-1]) if steps else 0
    ax.text(0.98, 0.02,
            f"{len(bv):,} puntos  |  {len(steps)} snapshots  |  "
            rf"$t_{{max}} = {t_max_oci:.1f}\,\Omega_{{ci}}^{{-1}}$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#8b949e")

    ax.legend(fontsize=10, framealpha=0.55,
              facecolor="#1c2128", edgecolor="#30363d", labelcolor=TEXT_CLR,
              loc="upper right")

    out = outdir / "brazil_acumulado.png"
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
    a_p25 = np.array([s["aniso_p25"] for s in snap_data])
    a_p75 = np.array([s["aniso_p75"] for s in snap_data])
    b_med = np.array([s["beta_med"]  for s in snap_data])
    b_p25 = np.array([s["beta_p25"]  for s in snap_data])
    b_p75 = np.array([s["beta_p75"]  for s in snap_data])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                   gridspec_kw={"hspace": 0.08})
    fig.patch.set_facecolor(DARK_BG)

    # ─ Panel superior: anisotropia ─
    ax1.set_facecolor(PANEL_BG)
    ax1.fill_between(toci, a_p25, a_p75, alpha=0.25, color="#ff6b6b")
    ax1.plot(toci, a_med, color="#ff6b6b", lw=2.0, label=r"mediana $T_\perp/T_\parallel$")
    ax1.axhline(1.0, color=TEXT_CLR, alpha=0.3, lw=0.9, ls="--", label="Isotropía A=1")

    mirror_dynamic = mirror_threshold(np.maximum(b_med, 1e-12))
    ax1.plot(toci, mirror_dynamic, color="#ff9999", alpha=0.8, lw=1.2,
             ls=":", label=r"Umbral Mirror $1+1/\beta_\parallel(t)$")

    ax1.set_ylabel(r"$T_\perp / T_\parallel$", fontsize=12, color=TEXT_CLR)
    finite_a = np.concatenate([a_p25, a_p75, mirror_dynamic])
    ax1.set_ylim(max(0.05, np.nanpercentile(finite_a, 1) * 0.8),
                 np.nanpercentile(finite_a, 99) * 1.2)
    _style_ax(ax1, f"Evolución Temporal — {PROFILE_LABEL}")
    ax1.legend(fontsize=9.5, framealpha=0.5,
               facecolor="#1c2128", edgecolor="#30363d", labelcolor=TEXT_CLR)

    # ─ Panel inferior: beta_par ─
    ax2.set_facecolor(PANEL_BG)
    ax2.fill_between(toci, b_p25, b_p75, alpha=0.25, color="#58a6ff")
    ax2.plot(toci, b_med, color="#58a6ff", lw=2.0, label=r"mediana $\beta_{i\parallel}$")
    ax2.set_ylabel(r"$\beta_{i\parallel}$", fontsize=12, color=TEXT_CLR)
    ax2.set_xlabel(r"$t\,\Omega_{ci}$", fontsize=12, color=TEXT_CLR, labelpad=6)
    ax2.set_yscale("log")
    _style_ax(ax2)
    ax2.legend(fontsize=9.5, framealpha=0.5,
               facecolor="#1c2128", edgecolor="#30363d", labelcolor=TEXT_CLR)

    for ax in (ax1, ax2):
        ax.tick_params(which="both", colors=TEXT_CLR, direction="in",
                       top=True, right=True)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_CLR)
        ax.grid(True, which="both", alpha=0.18, color=GRID_CLR, ls=":")

    out = outdir / "evolucion_temporal.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
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

    beta_init  = (N0 * T_PAR_I) / (b0_ref**2 / 2.0)
    aniso_init = T_PERP_I / T_PAR_I

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
                     fontsize=10, fontweight="bold", color=color, pad=5)

        ax.tick_params(which="both", colors=TEXT_CLR, direction="in",
                       labelsize=7.5, top=True, right=True)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_CLR)
        ax.grid(True, which="both", alpha=0.15, color=GRID_CLR, ls=":")

        if i % n_cols == 0:
            ax.set_ylabel(r"$T_\perp/T_\parallel$", fontsize=9, color=TEXT_CLR)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel(r"$\beta_{i\parallel}$", fontsize=9, color=TEXT_CLR)

    # Ocultar ejes sobrantes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        rf"Brazil Plots por Snapshot — {PROFILE_LABEL}  ($m_i/m_e={int(MASS_RATIO)}$)",
        fontsize=14, fontweight="bold", color=TEXT_CLR, y=1.01
    )

    out = outdir / "brazil_grid_snapshots.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Guardado → {out}")


# ── Analisis principal ────────────────────────────────────────────────────────
def run_analysis(mom_pattern: str, bz_pattern: str, B0_ref: float,
                 outdir: str = "anisotropy_plots", n_grid_snaps: int = 12):

    out = Path(outdir)
    out.mkdir(exist_ok=True)

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

    for i, step in enumerate(common):
        toci = step_to_omegaci(step)
        print(f"  step {step:6d}  ({i+1}/{len(common)})  "
              f"t*Oci={toci:.3f}", end="\r")

        data = process_snapshot(mom_files[step], bz_files[step])
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
            snap_stats.append({
                "step":      step,
                "toci":      toci,
                "aniso_med": float(np.median(av)),
                "aniso_p25": float(np.percentile(av, 25)),
                "aniso_p75": float(np.percentile(av, 75)),
                "beta_med":  float(np.median(bv)),
                "beta_p25":  float(np.percentile(bv, 25)),
                "beta_p75":  float(np.percentile(bv, 75)),
            })

        # Guardar snapshot completo para el grid
        if i in idx_grid and n_pts > 0:
            grid_snaps.append({
                "step": step, "toci": toci,
                "anisotropy": av, "beta_par": bv,
            })

    print()  # nueva linea tras \r
    print(f"Puntos acumulados: {len(all_beta):,}")

    all_beta  = np.asarray(all_beta)
    all_aniso = np.asarray(all_aniso)
    print("Generando graficas...")
    plot_brazil_accumulated(all_beta, all_aniso, snap_stats, common, out, B0_ref)
    plot_temporal_evolution(snap_stats, out)
    plot_brazil_grid(grid_snaps, out, B0_ref, n_cols=4)
    print("Listo.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brazil plots para simulaciones de anisotropia PSC."
    )
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
    args = parser.parse_args()

    run_analysis(
        mom_pattern=args.moments,
        bz_pattern=args.fields,
        B0_ref=args.B0,
        outdir=args.outdir,
        n_grid_snaps=args.nsnaps,
    )
