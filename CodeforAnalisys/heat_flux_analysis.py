#!/usr/bin/env python3
"""
heat_flux_analysis.py — particle heat flux (third central moment) in the prt window
===================================================================================
The heat flux of species s is the third central moment of its VDF,

    q = (m/2) ∫ |v - U|^2 (v - U) f d^3v ,      q_par = q · b ,  q_perp = |q - q_par b| .

PSC does not deposit third moments, so it can only be measured from the
particles. The previous version of this script plotted P_par * U_par, a
convective enthalpy term that is not q (it vanishes identically for a
skewed VDF at rest and is non-zero for a drifting Maxwellian that carries no
heat flux). This version measures q itself:

* **Local frame.** The prt window is split into ``--macrocells`` x
  ``--macrocells`` blocks; in each block the bulk velocity U and the field
  direction b (block mean of the local B at the particles' cells) are those
  of the block, so a spatially varying drift or a tilted field does not leak
  into q.
* **Normalisation.** Per block, q is divided by the free-streaming scale
  q0 = (3/2) n T v_T with v_T = sqrt(2T/m) and T = (T_par + 2 T_perp)/3;
  both are per-particle moments, so the density cancels and q/q0 is
  dimensionless and comparable between species and cases.
* **Truncation.** For a bi-Kappa with kappa = 3 the sixth moment of the
  ideal distribution diverges: the sampling variance of the third moment is
  infinite and the untruncated estimate is dominated by the few fastest
  particles. Every moment is therefore also computed over |v - U| <= s_max
  sqrt(T/m) for each ``--s-max`` (default 4, 6, 8); the untruncated value is
  reported for reference only. Quote a truncated value and state s_max.
* **Uncertainty.** The particles are split into ``--subsamples`` disjoint
  random groups; the standard error of the window mean across the groups is
  the sampling error reported with each value.

Positions are in code units (d_e) with the origin at the domain centre
(``Grid_t::Domain`` corner = -L/2 in psc_anisotropy_case.hxx). PSC writes
u = gamma v; velocities are converted before any moment is taken.

Outputs (``--outdir``, normally ``06_heat_flux/``):
  heat_flux_table.csv        one row per (step, species)
  heat_flux_blocks_<step>.csv per-block values at the mapped steps
  heat_flux_vs_time.png      <q_par/q0> and <|q_par|/q0> with error bands
  heat_flux_truncation.png   sensitivity of <|q_par|/q0> to s_max
  heat_flux_map_<s>_<step>.png  q_par/q0 per block over the window

Usage:
    python heat_flux_analysis.py --data-dir RUN --outdir out/06_heat_flux
    python heat_flux_analysis.py --data-dir RUN --species electron --macrocells 10
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
from plasma_physics import velocity_from_u
from psc_units import (
    B0, DI, DOMAIN_DE, KAPPA, N_GRID_Y, N_GRID_Z, PARTICLE_FILE_PATTERN,
    PROFILE_LABEL, step_to_omegaci,
)

ps.apply()

DX_CODE = DOMAIN_DE / N_GRID_Y
DEFAULT_S_MAX = (4.0, 6.0, 8.0)

#: Sampling floor of <|q_par|>/q0. For an isotropic Maxwellian (sigma^2 = T/m)
#: the estimator taken about the *sample* mean is, to first order,
#: q_par ~ (m/2)[<|v|^2 v_par> - 5 sigma^2 <v_par>], whose per-particle
#: variance is (m/2)^2 (35 - 2*5*5 + 25) sigma^6 = (m/2)^2 10 sigma^6
#: (E[|v|^4 v_par^2] = 35, E[|v|^2 v_par^2] = 5 sigma^4). With q0 = 1.5 T v_T
#: = 3/sqrt(2) m sigma^3, the mean of |q_par|/q0 over N particles of a VDF
#: with *no* heat flux is sqrt(2/pi) sqrt(10)/(3 sqrt(2)) / sqrt(N) ~ 0.59/sqrt(N)
#: (checked by Monte Carlo in test_new_diagnostics.py). <|q_par|> is only a
#: measured heat flux where it clearly exceeds this floor (a lower bound for
#: Kappa distributions, whose sampling variance is larger).
ABS_Q_FLOOR_COEFF = float(np.sqrt(2.0 / np.pi) * np.sqrt(10.0) / (3.0 * np.sqrt(2.0)))
SPECIES_COLOR = {"ion": ps.c("#ff7b72"), "electron": ps.c("#58a6ff")}
SPECIES_SYMBOL = {"ion": "i", "electron": "e"}


# ── Reading ──────────────────────────────────────────────────────────────────

def load_species(path: str, species: str, max_particles: int, rng) -> dict | None:
    data = PICDataReader.read_particles_with_positions(path, max_particles, rng=rng)
    mask = data["q"] > 0 if species == "ion" else data["q"] < 0
    if not np.any(mask):
        return None
    out = {k: data[k][mask] for k in ("y", "z", "px", "py", "pz", "w", "m")}
    out["iy"] = np.clip(np.floor((out["y"] + DOMAIN_DE / 2.0) / DX_CODE).astype(int), 0, N_GRID_Y - 1)
    out["iz"] = np.clip(np.floor((out["z"] + DOMAIN_DE / 2.0) / DX_CODE).astype(int), 0, N_GRID_Z - 1)
    out["mass"] = float(abs(np.average(out["m"], weights=out["w"])))
    out["vx"], out["vy"], out["vz"], _ = velocity_from_u(out["px"], out["py"], out["pz"])
    return out


def load_b(field_file: str | None, shape: tuple[int, int]) -> tuple[dict, str]:
    """(Bx, By, Bz) as (Nz, Ny) maps; uniform B0 z-hat if no field snapshot."""
    if field_file is None:
        return ({"bx": np.zeros(shape), "by": np.zeros(shape), "bz": np.full(shape, B0)},
                "uniform B0 (no field snapshot at this step)")
    data = PICDataReader.read_multiple_fields_3d(
        field_file, "jeh", ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"])
    flat = {k: PICDataReader.flatten_2d_slice(v).astype(float) for k, v in data.items()}
    bx, by, bz = magnetic_cell_centres_yz(flat["hx_fc/p0/3d"], flat["hy_fc/p0/3d"], flat["hz_fc/p0/3d"])
    return ({"bx": bx, "by": by, "bz": bz}, "local cell-centred B")


# ── Moments ──────────────────────────────────────────────────────────────────

def heat_flux_moments(vx, vy, vz, w, mass: float, b: np.ndarray,
                      s_max: float | None = None) -> dict:
    """q_par/q0, q_perp/q0, T_par, T_perp of one population in the frame of b.

    The bulk velocity and temperature are those of the population itself.
    With ``s_max`` the population is first cut to |v-U| <= s_max sqrt(T/m),
    U and T being those of the full population (the truncation radius must
    not depend on the moments it truncates); the moments are then central
    moments of the kept particles about their own mean.
    """
    nan = {"q_par_over_q0": np.nan, "q_perp_over_q0": np.nan,
           "T_par": np.nan, "T_perp": np.nan, "count": 0,
           "q_par_per_particle": np.nan, "q0_per_particle": np.nan,
           "q_par_null_se": np.nan, "abs_q_matched_floor": np.nan, "n_effective": 0.0}
    wsum = float(np.sum(w))
    if vx.size < 10 or wsum <= 0:
        return nan
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    ux, uy, uz = (float(np.sum(w * v) / wsum) for v in (vx, vy, vz))
    dvx, dvy, dvz = vx - ux, vy - uy, vz - uz
    dv2 = dvx * dvx + dvy * dvy + dvz * dvz
    if s_max is not None:
        t_full = mass * float(np.sum(w * dv2) / wsum) / 3.0
        keep = dv2 <= (s_max ** 2) * t_full / mass
        dvx, dvy, dvz, w = dvx[keep], dvy[keep], dvz[keep], w[keep]
        wsum = float(np.sum(w))
        if dvx.size < 10 or wsum <= 0:
            return nan
        # Central moments of the truncated population about its own mean:
        # the momentum of the removed tail must not reappear as a spurious
        # third moment of a symmetric core.
        dvx, dvy, dvz = (d - float(np.sum(w * d) / wsum) for d in (dvx, dvy, dvz))
        dv2 = dvx * dvx + dvy * dvy + dvz * dvz
    dpar = dvx * b[0] + dvy * b[1] + dvz * b[2]
    t_par = mass * float(np.sum(w * dpar * dpar) / wsum)
    t_perp = 0.5 * mass * float(np.sum(w * (dv2 - dpar * dpar)) / wsum)
    temp = (t_par + 2.0 * t_perp) / 3.0
    if not temp > 0:
        return nan
    qx, qy, qz = (0.5 * mass * float(np.sum(w * dv2 * d) / wsum) for d in (dvx, dvy, dvz))
    q_par = qx * b[0] + qy * b[1] + qz * b[2]
    q_perp = float(np.sqrt(max(qx * qx + qy * qy + qz * qz - q_par * q_par, 0.0)))
    q0 = 1.5 * temp * np.sqrt(2.0 * temp / mass)
    # Influence function of the sample-centred third moment under a symmetric null.
    # This accounts for estimating U, unlike the raw sixth moment alone.
    v = np.stack([dvx, dvy, dvz])
    covariance = (v * w) @ v.T / wsum
    influence = .5 * mass * (dv2 * dpar - np.trace(covariance) * dpar - 2 * (b @ covariance @ v)) / q0
    null_se = float(np.sqrt(np.sum((w * influence) ** 2)) / wsum)
    return {"q_par_over_q0": q_par / q0, "q_perp_over_q0": q_perp / q0,
            "q_par_per_particle": float(q_par), "q0_per_particle": float(q0),
            "q_par_null_se": null_se, "abs_q_matched_floor": float(np.sqrt(2 / np.pi) * null_se),
            "n_effective": effective_sample_size(w),
            "T_par": t_par, "T_perp": t_perp, "count": int(dvx.size)}


def block_heat_flux(part: dict, bmap: dict, lo, hi, nblocks: int,
                    s_max: float | None, groups: np.ndarray | None = None,
                    group: int | None = None) -> list[dict]:
    """Per-block moments over the prt window (optionally one subsample group)."""
    iz0, iz1, iy0, iy1 = int(lo[2]), int(hi[2]), int(lo[1]), int(hi[1])
    zedges = np.linspace(iz0, iz1, nblocks + 1).astype(int)
    yedges = np.linspace(iy0, iy1, nblocks + 1).astype(int)
    sel_all = np.ones(part["iz"].size, dtype=bool) if group is None else groups == group
    bz_idx = np.searchsorted(zedges, part["iz"], side="right") - 1
    by_idx = np.searchsorted(yedges, part["iy"], side="right") - 1
    inside = sel_all & (bz_idx >= 0) & (bz_idx < nblocks) & (by_idx >= 0) & (by_idx < nblocks)
    # One sort instead of nblocks^2 boolean masks over every particle.
    idx = np.flatnonzero(inside)
    block_id = bz_idx[idx] * nblocks + by_idx[idx]
    order = np.argsort(block_id, kind="stable")
    idx, block_id = idx[order], block_id[order]
    bounds = np.searchsorted(block_id, np.arange(nblocks * nblocks + 1))
    rows = []
    for jz in range(nblocks):
        for jy in range(nblocks):
            cell = jz * nblocks + jy
            sel = idx[bounds[cell]:bounds[cell + 1]]
            zs, ys = slice(zedges[jz], zedges[jz + 1]), slice(yedges[jy], yedges[jy + 1])
            b = np.array([np.mean(bmap["bx"][zs, ys]), np.mean(bmap["by"][zs, ys]),
                          np.mean(bmap["bz"][zs, ys])])
            mom = heat_flux_moments(part["vx"][sel], part["vy"][sel], part["vz"][sel],
                                    part["w"][sel], part["mass"], b, s_max)
            rows.append({"jz": jz, "jy": jy, "weight": float(np.sum(part["w"][sel])),
                         "B_over_B0": float(np.linalg.norm(b) / B0),
                         "volume_code": float((zedges[jz+1]-zedges[jz])*(yedges[jy+1]-yedges[jy])*DX_CODE**2), **mom})
    return rows


def window_mean(rows: list[dict], key: str, absolute: bool = False) -> float:
    vals = np.array([r[key] for r in rows], dtype=float)
    wts = np.array([r["weight"] for r in rows], dtype=float)
    ok = np.isfinite(vals) & (wts > 0)
    if not np.any(ok):
        return float("nan")
    vals = np.abs(vals[ok]) if absolute else vals[ok]
    return float(np.sum(wts[ok] * vals) / np.sum(wts[ok]))


def integrated_flux_ratio(blocks):
    """Integrated q / integrated q0; particle weights incorporate density and volume."""
    valid = [r for r in blocks if np.isfinite(r.get('q_par_per_particle', np.nan))
             and np.isfinite(r.get('q0_per_particle', np.nan)) and r['weight'] > 0]
    den = sum(r['weight'] * r['q0_per_particle'] for r in valid)
    return sum(r['weight'] * r['q_par_per_particle'] for r in valid) / den if den > 0 else float('nan')


# ── Driver ───────────────────────────────────────────────────────────────────

def analyse_step(step: int, prt_file: str, field_file: str | None, species: str,
                 args, rng) -> tuple[dict, list[dict]] | None:
    rng = sample_rng(prt_file, species, "heatflux")
    part = load_species(prt_file, species, args.max_particles, rng)
    if part is None:
        return None
    lo, hi = PICDataReader.read_prt_window(prt_file)
    bmap, frame = load_b(field_file, (N_GRID_Z, N_GRID_Y))
    blocks = block_heat_flux(part, bmap, lo, hi, args.macrocells, None)
    row = {
        "step": step, "omega_ci_t": step_to_omegaci(step), "species": species,
        "frame": frame, "n_particles": int(part["vx"].size),
        "n_effective": effective_sample_size(part["w"]),
        "averaging": "particle-weighted mean of local q/q0; not volume transport",
        "sampling": "sha256 run-step-species-purpose",
        "q_par_over_q0": window_mean(blocks, "q_par_over_q0"),
        "abs_q_par_over_q0": window_mean(blocks, "q_par_over_q0", absolute=True),
        "q_perp_over_q0": window_mean(blocks, "q_perp_over_q0"),
        "abs_q_par_noise_floor": window_mean(
            [{**b, "floor": ABS_Q_FLOOR_COEFF / np.sqrt(b["count"]) if b["count"] else np.nan}
             for b in blocks], "floor"),
    }
    for s_max in args.s_max:
        tb = block_heat_flux(part, bmap, lo, hi, args.macrocells, s_max)
        row[f"q_par_over_q0_smax{s_max:g}"] = window_mean(tb, "q_par_over_q0")
        row[f"abs_q_par_over_q0_smax{s_max:g}"] = window_mean(tb, "q_par_over_q0", absolute=True)
        row[f"q_perp_over_q0_smax{s_max:g}"] = window_mean(tb, "q_perp_over_q0")
    # Sampling error from disjoint subsamples, at the reference truncation.
    ref = args.s_max[len(args.s_max) // 2] if args.s_max else None
    groups = rng.integers(0, args.subsamples, part["vx"].size)
    sub_mean, sub_abs = [], []
    for g in range(args.subsamples):
        sb = block_heat_flux(part, bmap, lo, hi, args.macrocells, ref, groups, g)
        sub_mean.append(window_mean(sb, "q_par_over_q0"))
        sub_abs.append(window_mean(sb, "q_par_over_q0", absolute=True))
    for key, values in (("q_par_over_q0", sub_mean), ("abs_q_par_over_q0", sub_abs)):
        vals = np.array(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        row[f"{key}_err"] = (float(np.std(vals, ddof=1) / np.sqrt(vals.size))
                             if vals.size > 1 else float("nan"))
    reference_blocks = block_heat_flux(part, bmap, lo, hi, args.macrocells, ref)
    row["q_integrated_over_q0_integrated"] = integrated_flux_ratio(reference_blocks)
    row["abs_q_matched_noise_floor"] = window_mean(reference_blocks, "abs_q_matched_floor")
    row["null_model"] = "symmetric observed distribution, central-moment influence, asymptotic normal; truncated finite-moment estimate"
    row["null_status"] = "WARN_asymptotic_requires_calibration" if ref is not None else "UNVERIFIED_untruncated_tail_variance"
    row["error_reference_s_max"] = ref if ref is not None else float("nan")
    return row, blocks


def plot_time(rows: list[dict], outdir: Path, s_ref: float | None):
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.4), sharex=True)
    suffix = f"_smax{s_ref:g}" if s_ref is not None else ""
    for species in ("ion", "electron"):
        sr = [r for r in rows if r["species"] == species]
        if not sr:
            continue
        t = np.array([r["omega_ci_t"] for r in sr])
        s = SPECIES_SYMBOL[species]
        color = SPECIES_COLOR[species]
        for ax, key, label in (
            (axes[0], "abs_q_par_over_q0", rf"$\langle|q_{{\parallel {s}}}|\rangle/q_{{0{s}}}$"),
            (axes[1], "q_par_over_q0", rf"$\langle q_{{\parallel {s}}}\rangle/q_{{0{s}}}$"),
        ):
            y = np.array([r.get(key + suffix, r[key]) for r in sr], dtype=float)
            err = np.array([r.get(f"{key}_err", np.nan) for r in sr], dtype=float)
            ax.plot(t, y, "o-", ms=3.5, color=color, label=label)
            if np.any(np.isfinite(err)):
                ax.fill_between(t, y - err, y + err, color=color, alpha=0.2, lw=0)
            if key == "abs_q_par_over_q0":
                floor = np.array([r.get("abs_q_matched_noise_floor", np.nan) for r in sr], dtype=float)
                ax.plot(t, floor, "--", lw=1.2, color=color, alpha=0.8,
                        label=rf"matched symmetric-null floor (asymptotic), {s}")
    trunc = f", $|v-U|\\leq{s_ref:g}\\,(T/m)^{{1/2}}$" if s_ref is not None else ""
    axes[0].set_ylabel(r"$\langle|q_\parallel|\rangle/q_0$")
    axes[1].set_ylabel(r"$\langle q_\parallel\rangle/q_0$")
    axes[1].axhline(0.0, color=ps.MUTED_CLR, lw=0.8, ls=":")
    axes[1].set_xlabel(r"$t\,\Omega_{ci}$")
    axes[0].set_title(f"Heat flux in the prt window — {PROFILE_LABEL}{trunc}", fontsize=13)
    for ax in axes:
        ps.style_axes(ax)
        # Beside the panel: inside, it covers the uncertainty bands.
        ps.legend(ax, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    ps.save(fig, outdir / "heat_flux_vs_time.png")


def plot_truncation(rows: list[dict], s_values: list[float], outdir: Path):
    if not s_values or not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for species in ("ion", "electron"):
        sr = [r for r in rows if r["species"] == species]
        if not sr:
            continue
        last = sr[-1]
        y = [last[f"abs_q_par_over_q0_smax{s:g}"] for s in s_values]
        ax.plot(s_values, y, "o-", color=SPECIES_COLOR[species],
                label=rf"{species}, $t\Omega_{{ci}}={last['omega_ci_t']:.1f}$")
        ax.axhline(last["abs_q_par_over_q0"], color=SPECIES_COLOR[species], ls=":", lw=1.0)
    ax.set_xlabel(r"truncation radius $s_{\max}$ [$(T/m)^{1/2}$]")
    ax.set_ylabel(r"$\langle|q_\parallel|\rangle/q_0$")
    kappa = "bi-Maxwellian" if KAPPA is None else rf"bi-Kappa $\kappa={KAPPA:g}$"
    ax.set_title(f"Truncation sensitivity ({kappa}); dotted: untruncated", fontsize=12)
    ps.style_axes(ax)
    ps.legend(ax, fontsize=10)
    ps.save(fig, outdir / "heat_flux_truncation.png")


def plot_block_map(blocks: list[dict], lo, hi, nblocks: int, species: str,
                   step: int, outdir: Path):
    grid = np.full((nblocks, nblocks), np.nan)
    for r in blocks:
        grid[r["jz"], r["jy"]] = r["q_par_over_q0"]
    if not np.any(np.isfinite(grid)):
        return
    dx_di = DX_CODE / DI
    # grid is (block_z, block_y); transposed, z (along B0) runs horizontally.
    extent = [lo[2] * dx_di, hi[2] * dx_di, lo[1] * dx_di, hi[1] * dx_di]
    lim = float(np.nanmax(np.abs(grid))) or 1.0
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(grid.T, origin="lower", extent=extent, cmap=ps.CMAP_DIVERGING,
                   vmin=-lim, vmax=lim, aspect="equal")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    s = SPECIES_SYMBOL[species]
    cb.set_label(rf"$q_{{\parallel {s}}}/q_{{0{s}}}$")
    ps.spatial_axes(ax)
    ax.set_title(rf"$q_{{\parallel {s}}}/q_0$ per block — $t\Omega_{{ci}}={step_to_omegaci(step):.1f}$",
                 fontsize=12)
    ps.save(fig, outdir / f"heat_flux_map_{s}_{step}.png")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".")
    p.add_argument("--particles", default=None, help="prt glob (default: profile basename in --data-dir)")
    p.add_argument("--fields", default=None, help="pfd glob (default: --data-dir/pfd.*_p*.h5)")
    p.add_argument("--outdir", default="heat_flux")
    p.add_argument("--species", nargs="+", default=["ion", "electron"], choices=["ion", "electron"])
    p.add_argument("--macrocells", type=int, default=8,
                   help="blocks per side of the prt window (local frame, default 8)")
    p.add_argument("--s-max", type=float, nargs="*", default=list(DEFAULT_S_MAX),
                   help="truncation radii in units of (T/m)^1/2 (default 4 6 8)")
    p.add_argument("--subsamples", type=int, default=8,
                   help="disjoint random groups for the sampling error (default 8)")
    p.add_argument("--max-particles", type=int, default=2_000_000)
    p.add_argument("--steps", type=int, nargs="*", default=None)
    p.add_argument("--map-steps", type=int, default=3,
                   help="number of evenly spaced steps with a per-block map")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    prt = PICDataReader.find_files(args.particles or str(data_dir / PARTICLE_FILE_PATTERN))
    fields = PICDataReader.find_files(args.fields or str(data_dir / "pfd.*_p*.h5"))
    if not prt:
        print("[ERROR] no particle files: the heat flux needs the prt output.")
        return 1
    steps = sorted(prt) if not args.steps else [s for s in sorted(prt) if s in set(args.steps)]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260928)
    map_steps = set(steps[:: max(1, len(steps) // max(args.map_steps, 1))][: args.map_steps])
    rows = []
    for step in steps:
        for species in args.species:
            result = analyse_step(step, prt[step], fields.get(step), species, args, rng)
            if result is None:
                continue
            row, blocks = result
            rows.append(row)
            print(f"  step {step} {species}: <|q_par|>/q0 = {row['abs_q_par_over_q0']:.3g} "
                  f"({row['frame']})")
            if step in map_steps:
                lo, hi = PICDataReader.read_prt_window(prt[step])
                plot_block_map(blocks, lo, hi, args.macrocells, species, step, outdir)
                write_csv(outdir / f"heat_flux_blocks_{SPECIES_SYMBOL[species]}_{step}.csv", blocks)
    write_csv(outdir / "heat_flux_table.csv", rows)
    s_ref = args.s_max[len(args.s_max) // 2] if args.s_max else None
    plot_time(rows, outdir, s_ref)
    plot_truncation(rows, list(args.s_max), outdir)
    print(f"Heat-flux diagnostics written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
