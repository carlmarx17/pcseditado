#!/usr/bin/env python3
"""
structures_analysis.py — saturated magnetic structures and their pressure balance
=================================================================================
Nonlinear mirror (and oblique-firehose) saturation produces compressive
structures: magnetic holes or peaks elongated along B0, anticorrelated with
the density and close to total-pressure balance. This script measures those
properties per field/moment snapshot, so the saturation figure of the thesis
rests on numbers rather than on a visual impression of a map:

* **|B| statistics.** delta|B|/B0 after a Gaussian smoothing of ``--sigma-di``
  (PIC noise removal only; structures are several d_i). Its skewness is the
  standard discriminator of saturated mirror structures: negative for
  magnetic holes (dips), positive for peaks (Soucek, Lucek & Dandouras 2008,
  JGR 113, A04203; Genot et al. 2009, Ann. Geophys. 27, 601).
* **Structure catalogue.** Connected regions (periodic in both directions)
  with delta|B|/B0 below -theta (holes) or above +theta (peaks), with
  theta = max(``--threshold-sigma`` x std, ``--min-amplitude``). For each:
  area, depth or height, extent along B0 and across it (2x the RMS
  extent, in d_i), and the angle of the major axis to B0.
* **Density correlation.** Pearson r(delta n_i/n_i, delta|B|/B0) and the
  regression slope; mirror structures are anti-correlated.
* **Total-pressure balance.** P_T = B^2/2 + P_perp,i + P_perp,e with the
  thermal pressures projected on the local field. The ratio
  std(P_T)/std(B^2/2) is ~0 for pressure-balanced (static, mirror-like)
  structures and ~1 when nothing compensates the magnetic pressure.
* **Lifetime.** The interval during which the hole (or peak) area fraction
  exceeds half its maximum.

Outputs (``--outdir``, normally ``07_structures/``):
  structures_table.csv, structures_summary.json,
  structures_catalog_<step>.csv, structures_vs_time.png,
  pressure_balance_vs_time.png, structures_map_<step>.png
"""

from __future__ import annotations

import argparse
import csv
from analysis_contract import strict_dumps, magnetic_cell_centres_yz
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import plot_style as ps
from structure_tracking import StructureTracker
from data_reader import PICDataReader
from plasma_physics import central_pressure_tensor, field_aligned_pressures
from psc_units import (
    B0, DI, DOMAIN_DI, DX_DE, M_ELEC, M_ION, PROFILE_LABEL, step_to_omegaci,
)

ps.apply()

MOMENT_NAMES = ["rho", "txx", "tyy", "tzz", "txy", "tyz", "tzx", "px", "py", "pz"]


def load_snapshot(field_file: str, moment_file: str | None) -> dict:
    data = PICDataReader.read_multiple_fields_3d(
        field_file, "jeh", ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"])
    snap = {c: PICDataReader.flatten_2d_slice(data[f"h{c}_fc/p0/3d"]).astype(float) for c in "xyz"}
    snap["x"], snap["y"], snap["z"] = magnetic_cell_centres_yz(snap["x"], snap["y"], snap["z"])
    if moment_file is not None:
        for suffix, mass in (("i", M_ION), ("e", M_ELEC)):
            raw = PICDataReader.read_multiple_fields_3d(
                moment_file, "all_1st", [f"{n}_{suffix}/p0/3d" for n in MOMENT_NAMES])
            mom = {k.split("/")[0]: PICDataReader.flatten_2d_slice(v).astype(float)
                   for k, v in raw.items()}
            tensor = central_pressure_tensor(mom, suffix, mass)
            _, pperp, _ = field_aligned_pressures(
                tensor["Pxx"], tensor["Pyy"], tensor["Pzz"], tensor["Pxy"],
                tensor["Pyz"], tensor["Pzx"], snap["x"], snap["y"], snap["z"])
            snap[f"n_{suffix}"] = tensor["n"]
            snap[f"pperp_{suffix}"] = pperp
    return snap


def periodic_label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components of a boolean map that is periodic in both axes."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return labels, 0
    parent = np.arange(count + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a_edge, b_edge in ((labels[0, :], labels[-1, :]), (labels[:, 0], labels[:, -1])):
        for a, b in zip(a_edge, b_edge):
            if a and b:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(count + 1)])
    unique = np.unique(roots[1:])
    remap = np.zeros(count + 1, dtype=int)
    remap[1:] = np.searchsorted(unique, roots[1:]) + 1
    return remap[labels], int(unique.size)


def structure_catalog(db: np.ndarray, labels: np.ndarray, count: int, kind: str,
                      cell_di: float) -> list[dict]:
    """Geometry of each labelled structure, with minimum-image coordinates."""
    nz, ny = db.shape
    rows = []
    for lab in range(1, count + 1):
        iz, iy = np.nonzero(labels == lab)
        if iz.size == 0:
            continue
        # Unwrap around the first pixel so a structure crossing the periodic
        # boundary is not split into two distant halves.
        dz = (iz - iz[0] + nz // 2) % nz - nz // 2
        dy = (iy - iy[0] + ny // 2) % ny - ny // 2
        amp = db[iz, iy]
        wts = np.abs(amp)
        wts = wts / wts.sum() if wts.sum() > 0 else np.full(amp.size, 1.0 / amp.size)
        mz, my = np.sum(wts * dz), np.sum(wts * dy)
        czz = np.sum(wts * (dz - mz) ** 2)
        cyy = np.sum(wts * (dy - my) ** 2)
        czy = np.sum(wts * (dz - mz) * (dy - my))
        evals, evecs = np.linalg.eigh(np.array([[czz, czy], [czy, cyy]]))
        major = evecs[:, 1]
        angle = float(np.degrees(np.arctan2(abs(major[1]), abs(major[0]))))
        rows.append({
            "kind": kind, "label": lab, "cells": int(iz.size),
            "area_di2": float(iz.size * cell_di ** 2),
            "amplitude": float(np.min(amp) if kind == "hole" else np.max(amp)),
            "z_center_di": float(((iz[0] + mz) % nz) * cell_di),
            "y_center_di": float(((iy[0] + my) % ny) * cell_di),
            "L_parallel_di": float(2.0 * np.sqrt(czz) * cell_di),
            "L_perp_di": float(2.0 * np.sqrt(cyy) * cell_di),
            "elongation": float(np.sqrt(evals[1] / evals[0])) if evals[0] > 0 else float("inf"),
            "angle_to_B0_deg": angle,
        })
    return rows


def analyse(snap: dict, sigma_cells: float, threshold_sigma: float,
            min_amplitude: float, min_cells: int) -> tuple[dict, list[dict], dict]:
    smooth = lambda a: ndimage.gaussian_filter(a, sigma_cells, mode="wrap")
    bmag = np.sqrt(snap["x"] ** 2 + snap["y"] ** 2 + snap["z"] ** 2)
    db_b0 = smooth(bmag) / abs(B0) - 1.0
    db = db_b0 - db_b0.mean()  # morphology about the instantaneous spatial background
    std = float(np.std(db))
    theta = max(threshold_sigma * std, min_amplitude)
    cell_di = DX_DE / DI
    catalog = []
    maps = {"db": db, "db_b0": db_b0, "theta": theta}
    row = {"mean_B_change_over_B0": float(db_b0.mean()),
           "delta_B_parallel_std_over_B0": float(np.std(smooth(snap["z"])) / abs(B0)),
           "structure_reference": "instantaneous domain mean |B|", "delta_B_mag_std": std, "threshold": theta,
           "skewness_B": float(np.mean((db - db.mean()) ** 3) / std ** 3) if std > 0 else float("nan")}
    for kind, mask in (("hole", db < -theta), ("peak", db > theta)):
        labels, count = periodic_label(mask)
        cat = [c for c in structure_catalog(db, labels, count, kind, cell_di) if c["cells"] >= min_cells]
        catalog.extend(cat)
        kept = [c["label"] for c in cat]
        maps[f"{kind}_mask"] = np.isin(labels, kept)
        maps[f"{kind}_labels"] = np.where(np.isin(labels, kept), labels, 0)
        row[f"n_{kind}s"] = len(cat)
        row[f"{kind}_area_fraction"] = float(sum(c["cells"] for c in cat) / db.size)
        for key in ("amplitude", "L_parallel_di", "L_perp_di", "angle_to_B0_deg"):
            vals = np.array([c[key] for c in cat], dtype=float)
            wts = np.array([c["cells"] for c in cat], dtype=float)
            row[f"{kind}_{key}_mean"] = float(np.sum(vals * wts) / np.sum(wts)) if cat else float("nan")
    if "n_i" in snap:
        n_i = smooth(snap["n_i"])
        dn = n_i / np.mean(n_i) - 1.0
        dbb = db - db.mean()
        if np.std(dn) > 0 and np.std(dbb) > 0:
            row["corr_n_B"] = float(np.corrcoef(dn.ravel(), dbb.ravel())[0, 1])
            row["slope_dn_dB"] = float(np.sum(dn * dbb) / np.sum(dbb * dbb))
        pmag_old = 0.5 * smooth(bmag) ** 2
        pmag = smooth(0.5 * bmag ** 2)
        row["magnetic_pressure_smoothing_difference_rms"] = float(np.sqrt(np.mean((pmag - pmag_old) ** 2)))
        pt = pmag + smooth(snap["pperp_i"]) + smooth(snap["pperp_e"])
        row["pressure_balance_ratio"] = (float(np.std(pt) / np.std(pmag))
                                         if np.std(pmag) > 0 else float("nan"))
        maps["dn"] = dn
    return row, catalog, maps


def lifetime(rows: list[dict], key: str) -> dict:
    t = np.array([r["omega_ci_t"] for r in rows], dtype=float)
    f = np.array([r.get(key, np.nan) for r in rows], dtype=float)
    if not np.any(np.isfinite(f)) or np.nanmax(f) <= 0:
        return {"t_start": None, "t_end": None, "duration": None, "max_fraction": None}
    half = 0.5 * np.nanmax(f)
    above = np.flatnonzero(f >= half)
    return {"t_start": float(t[above[0]]), "t_end": float(t[above[-1]]),
            "duration": float(t[above[-1]] - t[above[0]]), "max_fraction": float(np.nanmax(f))}


def plot_time(rows: list[dict], outdir: Path):
    t = np.array([r["omega_ci_t"] for r in rows])
    col = lambda k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    hole, peak = ps.c("#58a6ff"), ps.c("#ff7b72")
    axes[0, 0].plot(t, col("hole_area_fraction"), "-", color=hole, label="holes")
    axes[0, 0].plot(t, col("peak_area_fraction"), "-", color=peak, label="peaks")
    axes[0, 0].set_ylabel("area fraction")
    axes[0, 1].plot(t, -col("hole_amplitude_mean"), "-", color=hole, label=r"hole depth $-\delta|B|/B_0$")
    axes[0, 1].plot(t, col("peak_amplitude_mean"), "-", color=peak, label=r"peak height $\delta|B|/B_0$")
    axes[0, 1].plot(t, col("delta_B_mag_std"), ":", color=ps.MUTED_CLR, label=r"std $\delta|B|/B_0$")
    axes[0, 1].set_ylabel("amplitude")
    axes[1, 0].plot(t, col("skewness_B"), "-", color=ps.c("#d2a8ff"))
    axes[1, 0].axhline(0.0, color=ps.MUTED_CLR, lw=0.8, ls=":")
    axes[1, 0].set_ylabel(r"skewness of $|B|$")
    axes[1, 0].set_title("< 0: holes dominate, > 0: peaks dominate", fontsize=10)
    for kind, color in (("hole", hole), ("peak", peak)):
        axes[1, 1].plot(t, col(f"{kind}_L_parallel_di_mean"), "-", color=color,
                        label=rf"{kind}s $L_\parallel$")
        axes[1, 1].plot(t, col(f"{kind}_L_perp_di_mean"), "--", color=color,
                        label=rf"{kind}s $L_\perp$")
    axes[1, 1].set_ylabel(r"extent $[d_i]$")
    for ax in axes.ravel():
        ps.style_axes(ax)
        ax.ticklabel_format(axis="y", useOffset=False)
        if ax.get_legend_handles_labels()[0]:
            ps.legend(ax, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel(r"$t\,\Omega_{ci}$")
    fig.suptitle(f"Magnetic structures — {PROFILE_LABEL}", fontsize=14)
    fig.tight_layout()
    ps.save(fig, outdir / "structures_vs_time.png")

    if "corr_n_B" in rows[0]:
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.plot(t, col("corr_n_B"), "-", color=ps.c("#58a6ff"),
                label=r"$r(\delta n_i/n_i,\ \delta|B|/B_0)$")
        ax.plot(t, col("pressure_balance_ratio"), "-", color=ps.c("#ff7b72"),
                label=r"$\mathrm{std}(P_T)/\mathrm{std}(B^2/2)$")
        ax.axhline(0.0, color=ps.MUTED_CLR, lw=0.8, ls=":")
        ax.axhline(-1.0, color=ps.MUTED_CLR, lw=0.6, ls=":")
        ax.set_xlabel(r"$t\,\Omega_{ci}$")
        ax.set_title(r"Density–field correlation and total-pressure balance "
                     r"($P_T = B^2/2 + P_{\perp i} + P_{\perp e}$)", fontsize=12)
        ps.style_axes(ax)
        ps.legend(ax, fontsize=10)
        ps.save(fig, outdir / "pressure_balance_vs_time.png")


def plot_map(maps: dict, step: int, outdir: Path):
    ncols = 2 if "dn" in maps else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 5.4), squeeze=False)
    extent = [0, DOMAIN_DI, 0, DOMAIN_DI]
    lim = float(np.max(np.abs(maps["db"]))) or 1.0
    ax = axes[0, 0]
    # Maps are stored (Nz, Ny); transposed, z (along B0) runs horizontally as
    # in every other map of the pipeline (plot_style.spatial_axes).
    im = ax.imshow(maps["db"].T, origin="lower", extent=extent, cmap=ps.CMAP_DIVERGING,
                   vmin=-lim, vmax=lim)
    for key, color in (("hole_mask", "#0072B2"), ("peak_mask", "#D55E00")):
        ax.contour(maps[key].astype(float).T, levels=[0.5], colors=[color], linewidths=0.8,
                   extent=extent, origin="lower")
    fig.colorbar(im, ax=ax, pad=0.02).set_label(r"$\delta|B|/B_0$ (smoothed)")
    ax.set_title(rf"$\delta|B|/B_0$, $\pm\theta={maps['theta']:.3g}$ contours")
    if "dn" in maps:
        ax = axes[0, 1]
        lim_n = float(np.max(np.abs(maps["dn"]))) or 1.0
        im = ax.imshow(maps["dn"].T, origin="lower", extent=extent, cmap=ps.CMAP_DIVERGING,
                       vmin=-lim_n, vmax=lim_n)
        fig.colorbar(im, ax=ax, pad=0.02).set_label(r"$\delta n_i/n_i$")
        ax.set_title(r"$\delta n_i/n_i$")
    for ax in axes[0]:
        ps.spatial_axes(ax)
    fig.suptitle(rf"{PROFILE_LABEL} — $t\Omega_{{ci}} = {step_to_omegaci(step):.1f}$", fontsize=13)
    fig.tight_layout()
    ps.save(fig, outdir / f"structures_map_{step}.png")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Saturated magnetic structures and pressure balance.")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--fields", default=None)
    p.add_argument("--moments", default=None)
    p.add_argument("--outdir", default="structures")
    p.add_argument("--sigma-di", type=float, default=0.25,
                   help="Gaussian smoothing before thresholding, in d_i (default 0.25)")
    p.add_argument("--threshold-sigma", type=float, default=1.0,
                   help="structure threshold in units of the std of delta|B|/B0 (default 1)")
    p.add_argument("--min-amplitude", type=float, default=0.02,
                   help="absolute floor of the threshold in delta|B|/B0 (default 0.02)")
    p.add_argument("--min-cells", type=int, default=9, help="discard smaller structures")
    p.add_argument("--every", type=int, default=1, help="use every N-th snapshot")
    p.add_argument("--map-steps", type=int, nargs="*", default=None,
                   help="steps with a map (default: largest std of delta|B| and the last)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    fields = PICDataReader.find_files(args.fields or str(data_dir / "pfd.*_p*.h5"))
    moments = PICDataReader.find_files(args.moments or str(data_dir / "pfd_moments.*_p*.h5"))
    steps = sorted(fields)[:: max(args.every, 1)]
    if not steps:
        print("[ERROR] no field snapshots")
        return 1
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sigma_cells = args.sigma_di * DI / DX_DE
    rows = []
    # Without explicit --map-steps, map the snapshot with the strongest
    # structures (largest std of delta|B|) and the last one; only those two
    # map sets are kept in memory.
    best = last = None
    tracker = StructureTracker()
    for step in steps:
        snap = load_snapshot(fields[step], moments.get(step))
        row, catalog, maps = analyse(snap, sigma_cells, args.threshold_sigma,
                                     args.min_amplitude, args.min_cells)
        rows.append({"step": step, "omega_ci_t": step_to_omegaci(step), **row})
        tracker.update(step_to_omegaci(step), maps)
        if args.map_steps is not None:
            if step in args.map_steps:
                plot_map(maps, step, outdir)
                write_csv(outdir / f"structures_catalog_{step}.csv", catalog)
            continue
        if best is None or row["delta_B_mag_std"] > best[1]:
            best = (step, row["delta_B_mag_std"], maps, catalog)
        last = (step, row["delta_B_mag_std"], maps, catalog)
    write_csv(outdir / "structures_table.csv", rows)
    write_csv(outdir / "structure_tracks.csv", tracker.rows)
    write_csv(outdir / "structure_lifetimes.csv", tracker.summary())
    sensitivity = []
    for selected in sorted({steps[0], steps[-1]}):
        snap = load_snapshot(fields[selected], moments.get(selected))
        for smooth_factor in (.5, 1., 2.):
            for threshold_factor in (.5, 1., 2.):
                measured, _, _ = analyse(snap, sigma_cells*smooth_factor,
                                         args.threshold_sigma*threshold_factor, args.min_amplitude, args.min_cells)
                sensitivity.append({"step": selected, "sigma_di": args.sigma_di*smooth_factor,
                                    "threshold_sigma": args.threshold_sigma*threshold_factor, **measured})
    write_csv(outdir / "structure_sensitivity.csv", sensitivity)
    if args.map_steps is None:
        for step, _, maps, catalog in {best[0]: best, last[0]: last}.values():
            plot_map(maps, step, outdir)
            write_csv(outdir / f"structures_catalog_{step}.csv", catalog)
    plot_time(rows, outdir)
    summary = {"holes": lifetime(rows, "hole_area_fraction"),
               "peaks": lifetime(rows, "peak_area_fraction"),
               "sigma_di": args.sigma_di, "threshold_sigma": args.threshold_sigma,
               "min_amplitude": args.min_amplitude,
               "structure_reference": "instantaneous spatial mean |B|",
               "duration_definition": "population activity interval above half maximum filling fraction; not individual lifetime",
               "scientific_status": "UNVERIFIED",
               "reason": "Morphology does not independently identify an instability branch",
               "final_skewness_B": rows[-1]["skewness_B"],
               "final_pressure_balance_ratio": rows[-1].get("pressure_balance_ratio"),
               "final_corr_n_B": rows[-1].get("corr_n_B")}
    (outdir / "structures_summary.json").write_text(strict_dumps(summary, indent=2))
    print(strict_dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
