#!/usr/bin/env python3
"""
estimator_consistency.py — are the anisotropy estimators measuring the same thing?
=================================================================================
The pipeline quotes the anisotropy A = T_perp/T_par from four estimators
that differ in region, frame, drift subtraction and averaging:

  prt_global   particles of the prt window, B0 = z as the parallel axis, the
               window-mean drift removed (physical_diagnostics.py tables).
  prt_local    the same particles split into blocks, each with its own
               drift and the direction of its local B; A is the ratio of
               the window-summed pressures.
  mom_window   PSC moment maps (bulk flow removed per cell, projected on the
               local B per cell) restricted to the prt window; reported as
               the ratio of mean pressures and as the mean of the per-cell A.
  mom_domain   the same over the full domain, with the cell filter of
               anisotropy_analysis.py (Brazil plot); the fraction of cells
               that filter rejects is reported.

They coincide for a uniform, drift-free plasma and separate as soon as the
structures develop; comparing them as if they were the same estimator is
one of the open points of the physics audit (section 4). This script puts
all four on one time axis, for both species, and tabulates the differences.

Outputs (``--outdir``, normally ``09_physical_diagnostics/``):
  estimator_consistency.csv, estimator_consistency.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps
from analysis_contract import sample_rng, effective_sample_size, magnetic_cell_centres_yz
from data_reader import PICDataReader
from plasma_physics import central_pressure_tensor, field_aligned_pressures, velocity_from_u
from psc_units import (
    DOMAIN_DE, DRIVEN_SPECIES, M_ELEC, M_ION, N_GRID_Y, N_GRID_Z,
    PARTICLE_FILE_PATTERN, PROFILE_LABEL, step_to_omegaci,
)

ps.apply()

DX_CODE = DOMAIN_DE / N_GRID_Y
MOMENT_NAMES = ["rho", "txx", "tyy", "tzz", "txy", "tyz", "tzx", "px", "py", "pz"]
SPECIES = {"ion": ("i", M_ION, 1.0), "electron": ("e", M_ELEC, -1.0)}


def particle_estimators(path: str, species: str, bmap: dict, lo, hi,
                        nblocks: int, max_particles: int, rng) -> dict:
    data = PICDataReader.read_particles_with_positions(path, max_particles, rng=sample_rng(path, species, "estimators"))
    sel = data["q"] > 0 if species == "ion" else data["q"] < 0
    if not np.any(sel):
        return {}
    w = data["w"][sel]
    mass = float(abs(np.average(data["m"][sel], weights=w)))
    ux, uy, uz = data["px"][sel], data["py"][sel], data["pz"][sel]
    vx, vy, vz, _ = velocity_from_u(ux, uy, uz)
    iy = np.clip(np.floor((data["y"][sel] + DOMAIN_DE / 2) / DX_CODE).astype(int), 0, N_GRID_Y - 1)
    iz = np.clip(np.floor((data["z"][sel] + DOMAIN_DE / 2) / DX_CODE).astype(int), 0, N_GRID_Z - 1)

    def pressures(idx, b):
        """Pressure-like sums (weight x per-particle central <u v>) along b."""
        ww = w[idx]
        W = ww.sum()
        if W <= 0 or idx.size < 10:
            return 0.0, 0.0, 0.0
        u = np.stack([ux[idx], uy[idx], uz[idx]])
        v = np.stack([vx[idx], vy[idx], vz[idx]])
        mu = (u * ww).sum(axis=1) / W
        mv = (v * ww).sum(axis=1) / W
        tensor = np.einsum("ai,bi,i->ab", u, v, ww) / W - np.outer(mu, mv)
        tensor = 0.5 * (tensor + tensor.T)
        tpar = float(b @ tensor @ b)
        tperp = 0.5 * (float(np.trace(tensor)) - tpar)
        return mass * W * tpar, mass * W * tperp, W

    zhat = np.array([0.0, 0.0, 1.0])
    ppar, pperp, _ = pressures(np.arange(w.size), zhat)
    out = {"A_prt_global": pperp / ppar if ppar > 0 else np.nan,
           "n_effective": effective_sample_size(w), "n_particles": int(w.size),
           "particle_window_fraction": float(np.prod(np.asarray(hi)-lo)/(N_GRID_Y*N_GRID_Z))}

    zedges = np.linspace(int(lo[2]), int(hi[2]), nblocks + 1).astype(int)
    yedges = np.linspace(int(lo[1]), int(hi[1]), nblocks + 1).astype(int)
    bz_idx = np.searchsorted(zedges, iz, side="right") - 1
    by_idx = np.searchsorted(yedges, iy, side="right") - 1
    block = np.where((bz_idx >= 0) & (bz_idx < nblocks) & (by_idx >= 0) & (by_idx < nblocks),
                     bz_idx * nblocks + by_idx, -1)
    order = np.argsort(block, kind="stable")
    bounds = np.searchsorted(block[order], np.arange(nblocks * nblocks + 1))
    spar = sperp = 0.0
    for jz in range(nblocks):
        for jy in range(nblocks):
            cell = jz * nblocks + jy
            idx = order[bounds[cell]:bounds[cell + 1]]
            zs, ys = slice(zedges[jz], zedges[jz + 1]), slice(yedges[jy], yedges[jy + 1])
            b = np.array([bmap[c][zs, ys].mean() for c in ("x", "y", "z")])
            b = b / np.linalg.norm(b)
            p1, p2, _ = pressures(idx, b)
            spar += p1
            sperp += p2
    out["A_prt_local"] = sperp / spar if spar > 0 else np.nan
    return out


def moment_estimators(moment_file: str, bmap: dict, species: str, lo, hi) -> dict:
    suffix, mass, _ = SPECIES[species]
    raw = PICDataReader.read_multiple_fields_3d(
        moment_file, "all_1st", [f"{n}_{suffix}/p0/3d" for n in MOMENT_NAMES])
    mom = {k.split("/")[0]: PICDataReader.flatten_2d_slice(v).astype(float) for k, v in raw.items()}
    t = central_pressure_tensor(mom, suffix, mass)
    ppar, pperp, b2 = field_aligned_pressures(t["Pxx"], t["Pyy"], t["Pzz"], t["Pxy"], t["Pyz"],
                                              t["Pzx"], bmap["x"], bmap["y"], bmap["z"])
    n = t["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        a_cell = pperp / ppar
        beta = 2.0 * ppar / b2
    win = (slice(int(lo[2]), int(hi[2])), slice(int(lo[1]), int(hi[1])))
    # Same cell filter as anisotropy_analysis.process_snapshot.
    mask = ((ppar > 0) & (pperp > 0) & (n > 0.05) & (b2 > 1e-10) & np.isfinite(beta)
            & np.isfinite(a_cell) & (a_cell > 0.05) & (a_cell < 20.0) & (beta > 0.1) & (beta < 2000.0))
    finite_w = np.isfinite(a_cell[win])
    return {
        "A_mom_window_ratio": float(np.nanmean(pperp[win]) / np.nanmean(ppar[win])),
        "A_mom_window_cellmean": float(np.nanmean(a_cell[win][finite_w])) if finite_w.any() else np.nan,
        "A_mom_domain_ratio": float(np.mean(pperp[mask]) / np.mean(ppar[mask])) if mask.any() else np.nan,
        "A_mom_domain_cellmean": float(np.mean(a_cell[mask])) if mask.any() else np.nan,
        "filtered_cell_fraction": float(1.0 - mask.mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Compare the anisotropy estimators of the pipeline.")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--outdir", default="physical_diagnostics")
    p.add_argument("--macrocells", type=int, default=8)
    p.add_argument("--max-particles", type=int, default=2_000_000)
    p.add_argument("--species", nargs="+", default=["ion", "electron"], choices=list(SPECIES))
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    prt = PICDataReader.find_files(str(data_dir / PARTICLE_FILE_PATTERN))
    fields = PICDataReader.find_files(str(data_dir / "pfd.*_p*.h5"))
    moments = PICDataReader.find_files(str(data_dir / "pfd_moments.*_p*.h5"))
    steps = sorted(set(prt) & set(fields) & set(moments))
    if not steps:
        print("[MISSING] no step has particles, fields and moments together; "
              "the estimators cannot be compared on the same snapshot.")
        return 0
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260928)
    rows = []
    for step in steps:
        raw = PICDataReader.read_multiple_fields_3d(
            fields[step], "jeh", ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"])
        bmap = {c: PICDataReader.flatten_2d_slice(raw[f"h{c}_fc/p0/3d"]).astype(float) for c in "xyz"}
        bmap["x"], bmap["y"], bmap["z"] = magnetic_cell_centres_yz(bmap["x"], bmap["y"], bmap["z"])
        lo, hi = PICDataReader.read_prt_window(prt[step])
        for species in args.species:
            row = {"step": step, "omega_ci_t": step_to_omegaci(step), "species": species}
            row.update(particle_estimators(prt[step], species, bmap, lo, hi, args.macrocells,
                                           args.max_particles, rng))
            row.update(moment_estimators(moments[step], bmap, species, lo, hi))
            ref = row.get("A_mom_window_ratio", np.nan)
            for key in ("A_prt_global", "A_prt_local", "A_mom_window_cellmean",
                        "A_mom_domain_ratio", "A_mom_domain_cellmean"):
                if key in row and np.isfinite(ref) and ref != 0:
                    row[f"{key}_rel_diff"] = float(row[key] / ref - 1.0)
            rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (outdir / "estimator_consistency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.species, outdir)
    print(f"Estimator comparison written to {outdir / 'estimator_consistency.csv'}")
    return 0


def plot(rows: list[dict], species_list: list[str], outdir: Path):
    styles = [("A_prt_global", "o-", "#58a6ff", "particles, window, $B_0$ frame"),
              ("A_prt_local", "s-", "#56d364", "particles, window, local frame"),
              ("A_mom_window_ratio", "-", "#ff7b72", r"moments, window, $\langle P_\perp\rangle/\langle P_\parallel\rangle$"),
              ("A_mom_window_cellmean", "--", "#ff7b72", r"moments, window, $\langle A\rangle$"),
              ("A_mom_domain_ratio", "-", "#d2a8ff", r"moments, domain, $\langle P_\perp\rangle/\langle P_\parallel\rangle$")]
    order = sorted(species_list, key=lambda s: s != DRIVEN_SPECIES)
    height = 3.6 * len(order) + 1.2
    fig, axes = plt.subplots(len(order), 1, figsize=(8.8, height), sharex=True, squeeze=False)
    handles = {}
    for ax, species in zip(axes[:, 0], order):
        sr = [r for r in rows if r["species"] == species]
        t = np.array([r["omega_ci_t"] for r in sr])
        for key, fmt, color, label in styles:
            y = np.array([r.get(key, np.nan) for r in sr], dtype=float)
            if np.any(np.isfinite(y)):
                line, = ax.plot(t, y, fmt, ms=2.5 if len(t) > 60 else 3.5, lw=1.3,
                                color=ps.c(color), label=label)
                handles.setdefault(label, line)
        s = SPECIES[species][0]
        ax.set_ylabel(rf"$A_{s} = T_{{\perp {s}}}/T_{{\parallel {s}}}$")
        ax.set_title(f"{species}s" + (" (driven species)" if species == DRIVEN_SPECIES else ""),
                     fontsize=12)
        ps.style_axes(ax)
    axes[-1, 0].set_xlabel(r"$t\,\Omega_{ci}$")
    fig.suptitle(f"Anisotropy estimators — {PROFILE_LABEL}", fontsize=13)
    # One legend for both panels, below them: inside, it covers the curves.
    if handles:
        fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=2,
                   fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.72 / height, 1, 1))
    ps.save(fig, outdir / "estimator_consistency.png")


if __name__ == "__main__":
    raise SystemExit(main())
