#!/usr/bin/env python3
"""
vdf_spatial.py — VDF resuelta en el espacio dentro de la ventana prt
====================================================================
El resto de la pipeline promedia la VDF sobre toda la ventana de partículas,
de modo que un hueco magnético y el plasma de fondo se mezclan en el mismo
histograma. Aquí se usan las posiciones que PSC sí escribe en los archivos
prt (`hdf5_prt`: x, y, z, px, py, pz, q, m, w) y que
`PICDataReader.read_particles_snapshot` descarta.

Dos diagnósticos:

  1. **Mapa de anisotropía por macro-celda.** La ventana prt se divide en
     `--macrocells` x `--macrocells` bloques; en cada uno se calculan
     T_par, T_perp y A a partir de las partículas. Es independiente de
     `moment_thermal_maps`, que los obtiene de los momentos de la grilla, y
     sirve para cruzarlos.

  2. **VDF condicionada al |B| local.** Las partículas se clasifican por el
     |B| de su celda en `hole` (percentil bajo), `ambient` y `peak`
     (percentil alto), y se comparan f(v_par) y f(v_perp) de cada población.
     Esta es la pregunta de los magnetic holes: si la anisotropía se regula
     localmente, la VDF dentro del hueco no es la del fondo.

Convención de ejes (verificada contra `rho_i`, corr = 0.81 vs 0.01 para la
transpuesta): los arrays de campo/momentos llegan como (Nz, Ny), eje 0 = Z,
eje 1 = Y. Las posiciones de las partículas vienen en unidades de código
(d_e) con el origen en el centro del dominio, así que
`celda = (coord + DOMAIN_DE/2) / dx_code`.

La anisotropía se mide respecto al **campo local** b = B/|B|, no respecto a
z global: dentro de un hueco el campo se dobla y proyectar sobre z mezcla
las componentes. Se reportan ambas para poder comparar.

Uso típico:
    python vdf_spatial.py --data-dir ../corridas_locales/mi_prueba --outdir salida
    python vdf_spatial.py --data-dir /ruta/corrida --steps 600000 900000
    python vdf_spatial.py --data-dir /ruta/corrida --species electron --macrocells 12
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import plot_style as ps

ps.apply()

from data_reader import PICDataReader
from psc_units import (
    B0, DOMAIN_DE, DOMAIN_DI_Y, DOMAIN_DI_Z, DI, N_GRID_Y, N_GRID_Z,
    PROFILE_LABEL, step_to_omegaci,
)

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
})

DPI = 200
DX_CODE = DOMAIN_DE / N_GRID_Y          # tamaño de celda en unidades de código


# ── Lectura ──────────────────────────────────────────────────────────────────

def load_particles(path: str, species: str, max_particles: int) -> dict:
    """Partículas de una especie con su celda (iy, iz) ya calculada."""
    data = PICDataReader.read_particles_with_positions(path, max_particles)
    mask = data["q"] > 0 if species == "ion" else data["q"] < 0
    if not np.any(mask):
        raise ValueError(f"No hay partículas de especie '{species}' en {path}")

    out = {k: data[k][mask] for k in ("y", "z", "px", "py", "pz", "w", "m")}
    out["iy"] = np.floor((out["y"] + DOMAIN_DE / 2.0) / DX_CODE).astype(int)
    out["iz"] = np.floor((out["z"] + DOMAIN_DE / 2.0) / DX_CODE).astype(int)
    np.clip(out["iy"], 0, N_GRID_Y - 1, out=out["iy"])
    np.clip(out["iz"], 0, N_GRID_Z - 1, out=out["iz"])
    out["mass"] = float(np.abs(np.average(out["m"], weights=out["w"])))
    return out


def load_b_field(field_file: str) -> dict:
    """(Bx, By, Bz) como arrays 2D (Nz, Ny) más |B| y la fluctuación."""
    fields = PICDataReader.read_multiple_fields_3d(
        field_file, "jeh", ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"])
    bx = PICDataReader.flatten_2d_slice(fields["hx_fc/p0/3d"]).astype(float)
    by = PICDataReader.flatten_2d_slice(fields["hy_fc/p0/3d"]).astype(float)
    bz = PICDataReader.flatten_2d_slice(fields["hz_fc/p0/3d"]).astype(float)
    bmag = np.sqrt(bx**2 + by**2 + bz**2)
    return {"bx": bx, "by": by, "bz": bz, "bmag": bmag,
            "delta_b_over_b0": (bmag - B0) / B0}


# ── Marco local y momentos ───────────────────────────────────────────────────

def local_frame_velocities(part: dict, bfield: dict) -> dict:
    """Proyecta las velocidades sobre el campo *local* de cada partícula.

    Devuelve v_par y las dos componentes perpendiculares (v_perp1, v_perp2)
    en una base ortonormal ligada a b = B/|B|, además de la versión global
    (z como eje paralelo) para comparar.
    """
    iz, iy = part["iz"], part["iy"]
    bx = bfield["bx"][iz, iy]
    by = bfield["by"][iz, iy]
    bz = bfield["bz"][iz, iy]
    bmag = np.sqrt(bx**2 + by**2 + bz**2)
    bmag = np.where(bmag > 1e-30, bmag, 1e-30)
    ux, uy, uz = bx / bmag, by / bmag, bz / bmag

    # Referencia para construir la base perpendicular; se cambia de eje donde
    # b es casi paralelo a z para no degenerar el producto cruz.
    near_z = np.abs(uz) > 0.9
    rx = np.where(near_z, 1.0, 0.0)
    ry = np.zeros_like(ux)
    rz = np.where(near_z, 0.0, 1.0)

    e1x, e1y, e1z = ry * uz - rz * uy, rz * ux - rx * uz, rx * uy - ry * ux
    norm = np.sqrt(e1x**2 + e1y**2 + e1z**2)
    norm = np.where(norm > 1e-30, norm, 1e-30)
    e1x, e1y, e1z = e1x / norm, e1y / norm, e1z / norm
    e2x, e2y, e2z = uy * e1z - uz * e1y, uz * e1x - ux * e1z, ux * e1y - uy * e1x

    vx, vy, vz = part["px"], part["py"], part["pz"]
    return {
        "v_par": vx * ux + vy * uy + vz * uz,
        "v_perp1": vx * e1x + vy * e1y + vz * e1z,
        "v_perp2": vx * e2x + vy * e2y + vz * e2z,
        "v_par_global": vz,
        "v_perp1_global": vx,
        "v_perp2_global": vy,
        "bmag_local": bmag,
    }


def _wvar(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or np.sum(weights) <= 0:
        return float("nan")
    mean = np.average(values, weights=weights)
    return float(np.average((values - mean) ** 2, weights=weights))


def anisotropy_noise_floor(count: int) -> float:
    """Error relativo esperado de A por ruido de muestreo con N partículas.

    Para un estimador de varianza con N muestras independientes,
    Var(s^2)/sigma^4 ~ 2/N. T_par usa una componente (2/N) y T_perp promedia
    dos (1/N), de modo que sigma_A / A ~ sqrt(3/N). Sin este piso, la escala
    de color de un mapa de A convierte ruido PIC en estructura aparente.
    """
    if count <= 0:
        return float("nan")
    return float(np.sqrt(3.0 / count))


def anisotropy_from(v_par, v_p1, v_p2, weights, mass) -> dict:
    """T_par, T_perp y A de un subconjunto de partículas."""
    tpar = mass * _wvar(v_par, weights)
    tperp = 0.5 * mass * (_wvar(v_p1, weights) + _wvar(v_p2, weights))
    if not np.isfinite(tpar) or tpar <= 0:
        return {"T_parallel": tpar, "T_perp": tperp, "A": float("nan"),
                "count": int(v_par.size)}
    return {"T_parallel": float(tpar), "T_perp": float(tperp),
            "A": float(tperp / tpar), "count": int(v_par.size)}


# ── Diagnóstico 1: mapa por macro-celda ──────────────────────────────────────

def macrocell_map(part: dict, vel: dict, lo, hi, nblocks: int) -> dict:
    """Anisotropía por bloque espacial dentro de la ventana prt."""
    y0, y1 = int(lo[1]), int(hi[1])
    z0, z1 = int(lo[2]), int(hi[2])
    edges_y = np.linspace(y0, y1, nblocks + 1)
    edges_z = np.linspace(z0, z1, nblocks + 1)

    by = np.clip(np.digitize(part["iy"], edges_y) - 1, 0, nblocks - 1)
    bz = np.clip(np.digitize(part["iz"], edges_z) - 1, 0, nblocks - 1)
    inside = ((part["iy"] >= y0) & (part["iy"] < y1) &
              (part["iz"] >= z0) & (part["iz"] < z1))

    a_map = np.full((nblocks, nblocks), np.nan)      # [bloque_z, bloque_y]
    tpar_map = np.full((nblocks, nblocks), np.nan)
    tperp_map = np.full((nblocks, nblocks), np.nan)
    count_map = np.zeros((nblocks, nblocks), dtype=int)

    flat = bz * nblocks + by
    order = np.argsort(flat[inside], kind="stable")
    idx_inside = np.flatnonzero(inside)[order]
    keys = flat[idx_inside]
    bounds = np.searchsorted(keys, np.arange(nblocks * nblocks + 1))

    for cell in range(nblocks * nblocks):
        sel = idx_inside[bounds[cell]:bounds[cell + 1]]
        if sel.size < 200:                      # ruido PIC: bloque insuficiente
            continue
        stats = anisotropy_from(vel["v_par"][sel], vel["v_perp1"][sel],
                                vel["v_perp2"][sel], part["w"][sel],
                                part["mass"])
        jz, jy = divmod(cell, nblocks)
        a_map[jz, jy] = stats["A"]
        tpar_map[jz, jy] = stats["T_parallel"]
        tperp_map[jz, jy] = stats["T_perp"]
        count_map[jz, jy] = stats["count"]

    return {"A": a_map, "T_parallel": tpar_map, "T_perp": tperp_map,
            "count": count_map, "edges_y": edges_y, "edges_z": edges_z,
            "nblocks": nblocks}


# ── Diagnóstico 2: VDF condicionada al |B| local ─────────────────────────────

def condition_on_field(part: dict, vel: dict, lo, hi, percentile: float) -> dict:
    """Separa hole / ambient / peak por el |B| local de cada partícula."""
    y0, y1 = int(lo[1]), int(hi[1])
    z0, z1 = int(lo[2]), int(hi[2])
    inside = ((part["iy"] >= y0) & (part["iy"] < y1) &
              (part["iz"] >= z0) & (part["iz"] < z1))
    b_local = vel["bmag_local"]
    b_in = b_local[inside]
    if b_in.size == 0:
        return {}

    b_lo = np.percentile(b_in, percentile)
    b_hi = np.percentile(b_in, 100.0 - percentile)

    groups = {
        "hole": inside & (b_local <= b_lo),
        "ambient": inside & (b_local > b_lo) & (b_local < b_hi),
        "peak": inside & (b_local >= b_hi),
        "all": inside,
    }

    out = {"thresholds": {"b_lo_over_B0": float(b_lo / B0),
                          "b_hi_over_B0": float(b_hi / B0),
                          "percentile": percentile}}
    for name, sel in groups.items():
        idx = np.flatnonzero(sel)
        if idx.size < 200:
            continue
        stats = anisotropy_from(vel["v_par"][idx], vel["v_perp1"][idx],
                                vel["v_perp2"][idx], part["w"][idx],
                                part["mass"])
        glob = anisotropy_from(vel["v_par_global"][idx],
                               vel["v_perp1_global"][idx],
                               vel["v_perp2_global"][idx],
                               part["w"][idx], part["mass"])
        stats["A_global_z"] = glob["A"]
        stats["b_mean_over_B0"] = float(np.mean(b_local[idx]) / B0)
        stats["idx"] = idx
        out[name] = stats
    return out


def vdf_profiles(part: dict, vel: dict, groups: dict, nbins: int = 120) -> dict:
    """f(v_par) y f(v_perp) normalizadas, en una malla común a los 3 grupos."""
    present = [g for g in ("hole", "ambient", "peak") if g in groups]
    if not present:
        return {}
    all_idx = np.concatenate([groups[g]["idx"] for g in present])
    vpar_all = vel["v_par"][all_idx]
    vperp_all = np.sqrt(vel["v_perp1"][all_idx]**2 + vel["v_perp2"][all_idx]**2)
    vpar_max = float(np.percentile(np.abs(vpar_all), 99.5))
    vperp_max = float(np.percentile(vperp_all, 99.5))
    if not (vpar_max > 0 and vperp_max > 0):
        return {}

    par_edges = np.linspace(-vpar_max, vpar_max, nbins + 1)
    perp_edges = np.linspace(0.0, vperp_max, nbins + 1)
    profiles = {"v_par_centers": 0.5 * (par_edges[:-1] + par_edges[1:]),
                "v_perp_centers": 0.5 * (perp_edges[:-1] + perp_edges[1:])}

    for name in present:
        idx = groups[name]["idx"]
        w = part["w"][idx]
        vperp = np.sqrt(vel["v_perp1"][idx]**2 + vel["v_perp2"][idx]**2)
        h_par, _ = np.histogram(vel["v_par"][idx], bins=par_edges,
                                weights=w, density=True)
        h_perp, _ = np.histogram(vperp, bins=perp_edges, weights=w)
        # f(v_perp) por unidad de área en el plano perpendicular
        area = np.pi * (perp_edges[1:]**2 - perp_edges[:-1]**2)
        h_perp = h_perp / (np.sum(w) * area)
        profiles[f"{name}_par"] = h_par
        profiles[f"{name}_perp"] = h_perp
    return profiles


# ── Figuras ──────────────────────────────────────────────────────────────────

def plot_overview(bfield, mac, window, groups, profiles, step, outdir, prefix):
    """Foto de la región guardada + anisotropía por macro-celda + VDFs."""
    toci = step_to_omegaci(step)
    fig = plt.figure(figsize=(18.5, 12.0))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.45)

    z0, z1 = window["z_di"]
    y0, y1 = window["y_di"]

    # (a) dominio completo en fluctuación de campo, con la ventana prt marcada
    ax = fig.add_subplot(gs[0, 0])
    db = bfield["delta_b_over_b0"]
    lim = float(np.percentile(np.abs(db), 99))
    im = ax.imshow(db.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   aspect="equal", extent=[0, DOMAIN_DI_Z, 0, DOMAIN_DI_Y])
    ax.add_patch(Rectangle((z0, y0), z1 - z0, y1 - y0, fill=False,
                           edgecolor=ps.c("#00ff00"), lw=2.4, ls="--"))
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\delta |B| / B_0$", labelpad=2)
    ax.set_xlabel(r"$Z\ [d_i]$  ($\parallel B_0$)")
    ax.set_ylabel(r"$Y\ [d_i]$")
    ax.set_title("Full domain\n(green = prt window)")

    # (b) zoom a la ventana prt, misma cantidad
    ax = fig.add_subplot(gs[0, 1])
    iz0, iz1 = window["cells_z"]
    iy0, iy1 = window["cells_y"]
    db_win = db[iz0:iz1, iy0:iy1]
    lim_w = float(np.percentile(np.abs(db_win), 99))
    im = ax.imshow(db_win.T, origin="lower", cmap="RdBu_r",
                   vmin=-lim_w, vmax=lim_w, aspect="equal",
                   extent=[z0, z1, y0, y1])
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\delta |B| / B_0$", labelpad=2)
    for e in mac["edges_z"]:
        ax.axvline(e * DOMAIN_DI_Z / N_GRID_Z, color="k", lw=0.4, alpha=0.35)
    for e in mac["edges_y"]:
        ax.axhline(e * DOMAIN_DI_Y / N_GRID_Y, color="k", lw=0.4, alpha=0.35)
    ax.set_xlabel(r"$Z\ [d_i]$")
    ax.set_ylabel(r"$Y\ [d_i]$")
    ax.set_title("prt window: what is saved\n(grid = macro-cells)")

    # (c) anisotropía por macro-celda, desde partículas.
    # La escala se fija en múltiplos del ruido de muestreo, de forma que un
    # mapa plano se vea plano en vez de amplificar ruido PIC.
    ax = fig.add_subplot(gs[0, 2])
    a = mac["A"]
    if np.any(np.isfinite(a)):
        amid = float(np.nanmedian(a))
        typical_n = int(np.median(mac["count"][mac["count"] > 0])) \
            if np.any(mac["count"] > 0) else 0
        sigma = amid * anisotropy_noise_floor(typical_n) if typical_n else np.nan
        observed = float(np.nanstd(a))
        span = 3.0 * sigma if np.isfinite(sigma) and sigma > 0 else \
            (float(np.nanstd(a)) or 0.1)
        im = ax.imshow(a.T, origin="lower", cmap="viridis", aspect="equal",
                       extent=[z0, z1, y0, y1],
                       vmin=amid - span, vmax=amid + span)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(r"$A = T_\perp / T_\parallel$", labelpad=2)
        ratio = observed / sigma if np.isfinite(sigma) and sigma > 0 else np.nan
        ax.set_title(r"$A$ per macro-cell ($\hat{b}$ frame)"
                     "\n" + rf"scale $\pm 3\sigma_{{\rm noise}}$, "
                     rf"spread/noise = {ratio:.1f}", fontsize=14)
    else:
        ax.set_title(r"$A$ per macro-cell ($\hat{b}$ frame)", fontsize=14)
    ax.set_xlabel(r"$Z\ [d_i]$")
    ax.set_ylabel(r"$Y\ [d_i]$")

    # (d) f(v_par) por población
    ax = fig.add_subplot(gs[1, 0])
    colors = {"hole": ps.c("#1f77b4"), "ambient": ps.c("#7f7f7f"), "peak": ps.c("#d62728")}
    labels = {"hole": "hole (low $|B|$)", "ambient": "ambient",
              "peak": "peak (high $|B|$)"}
    if profiles:
        for name, color in colors.items():
            key = f"{name}_par"
            if key in profiles:
                ax.semilogy(profiles["v_par_centers"], profiles[key],
                            color=color, lw=1.8, label=labels[name])
    ax.set_xlabel(r"$v_\parallel$ [code units]")
    ax.set_ylabel(r"$f(v_\parallel)$")
    ax.set_title(r"Parallel VDF conditioned on local $|B|$")
    ax.legend(framealpha=0.9)

    # (e) f(v_perp) por población
    ax = fig.add_subplot(gs[1, 1])
    if profiles:
        for name, color in colors.items():
            key = f"{name}_perp"
            if key in profiles:
                ax.semilogy(profiles["v_perp_centers"], profiles[key],
                            color=color, lw=1.8, label=labels[name])
    ax.set_xlabel(r"$v_\perp$ [code units]")
    ax.set_ylabel(r"$f(v_\perp)$")
    ax.set_title(r"Perpendicular VDF conditioned on local $|B|$")
    ax.legend(framealpha=0.9)

    # (f) resumen numérico
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    lines = [f"{PROFILE_LABEL}", f"step {step}   " +
             rf"$t \approx {toci:.1f}\,\Omega_{{ci}}^{{-1}}$", ""]
    thr = groups.get("thresholds", {})
    if thr:
        lines.append(rf"cut: percentile {thr['percentile']:.0f} / "
                     rf"{100 - thr['percentile']:.0f}")
        lines.append(rf"$|B|_{{\rm hole}} \leq {thr['b_lo_over_B0']:.3f}\,B_0$")
        lines.append(rf"$|B|_{{\rm peak}} \geq {thr['b_hi_over_B0']:.3f}\,B_0$")
        lines.append("")
    for name in ("hole", "ambient", "peak", "all"):
        g = groups.get(name)
        if not g:
            continue
        err = g["A"] * anisotropy_noise_floor(g["count"])
        lines.append(
            rf"{labels.get(name, name):<18s} A={g['A']:.4f}$\pm${err:.4f}   "
            rf"$A_z$={g['A_global_z']:.4f}   N={g['count']:,}")

    # La pregunta de los magnetic holes: A dentro del hueco vs en el pico,
    # medido contra el ruido de muestreo de ambas poblaciones.
    hole, peak = groups.get("hole"), groups.get("peak")
    if hole and peak:
        diff = hole["A"] - peak["A"]
        err = np.sqrt((hole["A"] * anisotropy_noise_floor(hole["count"]))**2 +
                      (peak["A"] * anisotropy_noise_floor(peak["count"]))**2)
        nsig = diff / err if err > 0 else np.nan
        verdict = ("SIGNIFICANT" if abs(nsig) >= 3 else "consistent with noise")
        lines += ["", rf"$A_{{\rm hole}} - A_{{\rm peak}}$ = {diff:+.4f} "
                      rf"$\pm$ {err:.4f}",
                  rf"   = {nsig:+.1f}$\sigma$  ->  {verdict}"]
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=12, family="monospace")

    fig.suptitle(
        rf"Spatially resolved VDF — {PROFILE_LABEL}, step {step}, "
        rf"$t \approx {toci:.1f}\,\Omega_{{ci}}^{{-1}}$", y=0.98)
    out = outdir / f"{prefix}vdf_spatial_step{step:09d}.png"
    ps.save(fig, out)
    plt.close(fig)
    return out


# ── Salidas tabulares ────────────────────────────────────────────────────────

def write_macrocell_csv(mac, window, step, outdir, prefix) -> Path:
    out = outdir / f"{prefix}vdf_macrocells_step{step:09d}.csv"
    dy = DOMAIN_DI_Y / N_GRID_Y
    dz = DOMAIN_DI_Z / N_GRID_Z
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "omega_ci_t", "block_z", "block_y", "z_center_di",
                    "y_center_di", "T_parallel", "T_perp", "A", "count"])
        n = mac["nblocks"]
        for jz in range(n):
            zc = 0.5 * (mac["edges_z"][jz] + mac["edges_z"][jz + 1]) * dz
            for jy in range(n):
                yc = 0.5 * (mac["edges_y"][jy] + mac["edges_y"][jy + 1]) * dy
                w.writerow([step, f"{step_to_omegaci(step):.6f}", jz, jy,
                            f"{zc:.4f}", f"{yc:.4f}",
                            f"{mac['T_parallel'][jz, jy]:.8g}",
                            f"{mac['T_perp'][jz, jy]:.8g}",
                            f"{mac['A'][jz, jy]:.6g}",
                            int(mac["count"][jz, jy])])
    return out


def summary_rows(groups: dict, mac: dict, step: int) -> list[dict]:
    thr = groups.get("thresholds", {})
    hole, peak = groups.get("hole"), groups.get("peak")
    if hole and peak:
        diff = hole["A"] - peak["A"]
        diff_err = np.sqrt(
            (hole["A"] * anisotropy_noise_floor(hole["count"]))**2 +
            (peak["A"] * anisotropy_noise_floor(peak["count"]))**2)
        diff_sigma = diff / diff_err if diff_err > 0 else float("nan")
    else:
        diff = diff_err = diff_sigma = float("nan")

    finite = mac["count"][mac["count"] > 0]
    typical_n = int(np.median(finite)) if finite.size else 0
    mac_noise = (float(np.nanmedian(mac["A"])) *
                 anisotropy_noise_floor(typical_n)) if typical_n else float("nan")

    rows = []
    for name in ("hole", "ambient", "peak", "all"):
        g = groups.get(name)
        if not g:
            continue
        rows.append({
            "step": step,
            "omega_ci_t": step_to_omegaci(step),
            "population": name,
            "b_mean_over_B0": g["b_mean_over_B0"],
            "T_parallel": g["T_parallel"],
            "T_perp": g["T_perp"],
            "A_local_b": g["A"],
            "A_local_b_error": g["A"] * anisotropy_noise_floor(g["count"]),
            "A_global_z": g["A_global_z"],
            "count": g["count"],
            "b_lo_over_B0": thr.get("b_lo_over_B0", float("nan")),
            "b_hi_over_B0": thr.get("b_hi_over_B0", float("nan")),
            "A_hole_minus_peak": diff,
            "A_hole_minus_peak_error": diff_err,
            "A_hole_minus_peak_sigma": diff_sigma,
            "A_macrocell_median": float(np.nanmedian(mac["A"])),
            "A_macrocell_std": float(np.nanstd(mac["A"])),
            "A_macrocell_noise": mac_noise,
            "A_macrocell_p10": float(np.nanpercentile(mac["A"], 10)),
            "A_macrocell_p90": float(np.nanpercentile(mac["A"], 90)),
        })
    return rows


# ── Driver ───────────────────────────────────────────────────────────────────

def describe_window(lo, hi) -> dict:
    ny, nz = int(hi[1] - lo[1]), int(hi[2] - lo[2])
    dy = DOMAIN_DI_Y / N_GRID_Y
    dz = DOMAIN_DI_Z / N_GRID_Z
    return {
        "cells_y": [int(lo[1]), int(hi[1])],
        "cells_z": [int(lo[2]), int(hi[2])],
        "n_cells_y": ny, "n_cells_z": nz,
        "y_di": [lo[1] * dy, hi[1] * dy],
        "z_di": [lo[2] * dz, hi[2] * dz],
        "size_di": [ny * dy, nz * dz],
        "fraction_of_area": (ny / N_GRID_Y) * (nz / N_GRID_Z),
    }


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outputs = PICDataReader.discover_outputs(args.data_dir)
    fields = outputs["fields"]
    particles = outputs["particles"]
    if not fields or not particles:
        print(f"ERROR: faltan pfd.* o prt_*.* en {args.data_dir}")
        return 1
    series = next(iter(particles.values()))

    prt_steps = sorted(series)
    if args.steps:
        prt_steps = [s for s in args.steps if s in series]
        missing = sorted(set(args.steps) - set(series))
        if missing:
            print(f"[WARN] steps sin archivo prt: {missing}")
    elif args.max_snapshots and len(prt_steps) > args.max_snapshots:
        pick = np.linspace(0, len(prt_steps) - 1, args.max_snapshots)
        prt_steps = [prt_steps[i] for i in dict.fromkeys(pick.round().astype(int))]
    if not prt_steps:
        print("ERROR: ningún step seleccionado.")
        return 1

    field_steps = np.array(sorted(fields))
    lo, hi = PICDataReader.read_prt_window(series[prt_steps[0]])
    window = describe_window(lo, hi)

    print(f"Perfil:          {PROFILE_LABEL}")
    print(f"Ventana prt:     celdas Y [{window['cells_y'][0]}, {window['cells_y'][1]}), "
          f"Z [{window['cells_z'][0]}, {window['cells_z'][1]})")
    print(f"                 {window['size_di'][0]:.2f} x {window['size_di'][1]:.2f} d_i "
          f"({100 * window['fraction_of_area']:.1f} % del área)")
    print(f"Macro-celdas:    {args.macrocells} x {args.macrocells}")
    print(f"Steps:           {len(prt_steps)}")

    all_rows: list[dict] = []
    for step in prt_steps:
        # el snapshot de campos más cercano al de partículas
        near = int(field_steps[np.argmin(np.abs(field_steps - step))])
        if abs(near - step) > args.max_step_mismatch:
            print(f"[WARN] step {step}: campo más cercano en {near}, "
                  f"desfase {abs(near - step)} > {args.max_step_mismatch}; se omite.")
            continue

        part = load_particles(series[step], args.species, args.max_particles)
        bfield = load_b_field(fields[near])
        vel = local_frame_velocities(part, bfield)
        mac = macrocell_map(part, vel, lo, hi, args.macrocells)
        groups = condition_on_field(part, vel, lo, hi, args.percentile)
        profiles = vdf_profiles(part, vel, groups)

        png = plot_overview(bfield, mac, window, groups, profiles, step,
                            outdir, args.prefix)
        csv_path = write_macrocell_csv(mac, window, step, outdir, args.prefix)
        all_rows.extend(summary_rows(groups, mac, step))

        hole = groups.get("hole", {})
        peak = groups.get("peak", {})
        if hole and peak and all_rows:
            sigma = all_rows[-1]["A_hole_minus_peak_sigma"]
            print(f"  step {step:>9} (campo {near}): "
                  f"A_hueco={hole['A']:.4f}  A_pico={peak['A']:.4f}  "
                  f"dif={sigma:+.1f}sigma  -> {png.name}")
        else:
            print(f"  step {step:>9} (campo {near}): {png.name}  |  {csv_path.name}")

    if all_rows:
        summary = outdir / f"{args.prefix}vdf_hole_vs_peak_summary.csv"
        with open(summary, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Resumen: {summary}")

    meta = outdir / f"{args.prefix}vdf_spatial_metadata.json"
    with open(meta, "w") as fh:
        json.dump({"profile": PROFILE_LABEL, "species": args.species,
                   "prt_window": window, "macrocells": args.macrocells,
                   "percentile": args.percentile,
                   "steps": [int(s) for s in prt_steps]}, fh, indent=2)
    return 0


def parse_args():
    p = argparse.ArgumentParser(
        description="VDF resuelta en el espacio dentro de la ventana prt.")
    p.add_argument("--data-dir", default="../build/src")
    p.add_argument("--outdir", default="vdf_spatial_plots")
    p.add_argument("--prefix", default="")
    p.add_argument("--species", choices=["ion", "electron"], default="ion")
    p.add_argument("--steps", nargs="*", type=int)
    p.add_argument("--max-snapshots", type=int, default=6)
    p.add_argument("--macrocells", type=int, default=8,
                   help="número de bloques por eje dentro de la ventana prt")
    p.add_argument("--percentile", type=float, default=15.0,
                   help="percentil de |B| que define hueco y pico")
    p.add_argument("--max-particles", type=int, default=4_000_000)
    p.add_argument("--max-step-mismatch", type=int, default=3000,
                   help="desfase máximo permitido entre snapshot prt y de campos")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
