#!/usr/bin/env python3
"""
plot_prt.py — Particle diagnostic suite for PSC PIC simulations.

Reads HDF5 particle files and generates publication-quality diagnostics of
the driven species (ions for mirror/firehose, electrons for whistler):
  1. kappa_comparison_{parallel,perpendicular}.png -- f(v) at t = 0 and at the
     last snapshot against the Maxwellian and the maximum-likelihood kappa,
     both with the variance measured at that time, and the ratio to the
     Maxwellian (kappa_mle_summary.csv)
  2. goodness_of_fit.png -- F_PIC - F_model with the KS 95 % band
     (goodness_of_fit_metrics.csv: KS and Anderson-Darling)
  3. distribution_change_{ions,electrons}.png -- log10 f(v,t)/f(v,0)
  4. vdf_1d_evolution.png -- f(v_par), f(|v_perp|) at selected times
     (vdf_tail_fractions.csv)
  5. particle_energy_partition.png (the heat flux is heat_flux_analysis.py)

2D/3D VDF snapshots and the T_perp/T_par ("Brazil") evolution are no longer
generated here -- physical_diagnostics.py (09_physical_diagnostics/) and
anisotropy_analysis.py (01_anisotropy/) already cover both, denser and
without this module's corner-noise-in-low-count-bins artifact.

Usage:
    python plot_prt.py [path_to_prt_file | directory | glob_pattern]

Defaults to  ../build/src/prt.000000000.h5  if no argument is given.
"""

import sys
import os
import glob
import re
import gc
import csv
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
try:
    from scipy.special import gamma as gamma_func
    from scipy import stats as scipy_stats
except ImportError:
    gamma_func = None
    scipy_stats = None

from plasma_physics import (
    kappa_marginal_cdf, kappa_marginal_pdf, kappa_mle, kappa_speed_pdf_2d, velocity_from_u,
)
from psc_units import (
    B0, DRIVEN_SPECIES, KAPPA, TI_PAR, TI_PERP, VA_OVER_C, ZI, BETA_I_PAR as _BETA_I_PAR_SIM,
    BETA_I_PERP_OVER_PAR as _TI_RATIO_SIM, M_ION, PROFILE_LABEL, step_to_omegaci,
)

import plot_style as ps

# Shared theme (paper: 300 dpi, PDF copy, content check at every save).
ps.apply()

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = "prt_plots"
MAX_EVOLUTION_FILES = 12
MAX_PARTICLES = 500_000  # submuestreo global si hay más partículas
RNG_SEED = 20260612
OUTPUT_PREFIX = ""

STEP_RE = re.compile(r"\.(\d+)(?:_p\d+)?\.h5$")


def _style_paper_axes(ax):
    ax.set_facecolor("white")
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=14)
    ax.minorticks_on()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _save_paper_figure(fig, path: str):
    ps.save(fig, path)
    print(f"  Saved: {path}")


def output_file(outdir: str, filename: str) -> str:
    return os.path.join(outdir, f"{OUTPUT_PREFIX}{filename}")


def _sample_indices(n_total: int, max_particles: int):
    """Return a deterministic uniform subsample for reproducible figures."""
    if n_total <= max_particles:
        return slice(None)
    rng = np.random.default_rng(RNG_SEED)
    return np.sort(rng.choice(n_total, max_particles, replace=False))

# ── Simulation parameters (from psc_units — single source of truth) ──────────
Zi: float = ZI
VA_OVER_C_: float = VA_OVER_C
BETA_I_PAR: float = _BETA_I_PAR_SIM
TI_PERP_OVER_TI_PAR: float = _TI_RATIO_SIM
BETA_NORM: float = 1.0




# ══════════════════════════════════════════════════════════════════════════════
#  I/O Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_particles(filepath: str, verbose: bool = True,
                   max_particles: int = MAX_PARTICLES) -> np.ndarray:
    """Load particle data from a PSC prt.*.h5 file.

    Only reads columns needed for analysis (q, m, px, py, pz) to save RAM.
    Subsamples uniformly if total particles exceeds max_particles.
    """
    if verbose:
        print(f"Loading particles from: {filepath}")
    with h5py.File(filepath, "r") as f:
        dset = f["particles"]["p0"]["1d"]
        n_total = dset["q"].shape[0]
        if n_total > max_particles:
            idx = _sample_indices(n_total, max_particles)
            if verbose:
                print(f"  Subsampling {max_particles:,} / {n_total:,} particles")
        else:
            idx = slice(None)
        # Read only needed fields — avoids loading x,y,z positions
        q  = dset["q"][idx].astype(np.float32)
        m  = dset["m"][idx].astype(np.float32)
        px = dset["px"][idx].astype(np.float32)
        py = dset["py"][idx].astype(np.float32)
        pz = dset["pz"][idx].astype(np.float32)

    # Pack into a structured array (same interface as before)
    dtype = np.dtype([("q", np.float32), ("m", np.float32),
                      ("px", np.float32), ("py", np.float32), ("pz", np.float32)])
    data = np.empty(len(q), dtype=dtype)
    data["q"]  = q;  data["m"]  = m
    data["px"] = px; data["py"] = py; data["pz"] = pz
    if verbose:
        print(f"  Loaded {len(data):,} particles ({data.nbytes / 1e6:.1f} MB)")
    return data


def load_particle_phase_space(filepath: str,
                              max_particles: int = MAX_PARTICLES) -> dict:
    """Load only momentum fields needed for distribution analysis (memory-light)."""
    with h5py.File(filepath, "r") as f:
        dset = f["particles"]["p0"]["1d"]
        n_total = dset["q"].shape[0]
        idx = _sample_indices(n_total, max_particles)
        q  = dset["q"][idx].astype(np.float32)
        px = dset["px"][idx].astype(np.float32)
        py = dset["py"][idx].astype(np.float32)
        pz = dset["pz"][idx].astype(np.float32)
        w = (dset["w"][idx].astype(np.float64) if "w" in (dset.dtype.names or ())
             else np.ones(q.shape[0]))

    ion_mask  = q > 0
    elec_mask = q < 0
    result = {
        "ions_px":        px[ion_mask],
        "ions_py":        py[ion_mask],
        "ions_pz":        pz[ion_mask],
        "ions_perp":      np.sqrt(px[ion_mask]**2 + py[ion_mask]**2),
        "ions_w":         w[ion_mask],
        "electrons_px":   px[elec_mask],
        "electrons_py":   py[elec_mask],
        "electrons_pz":   pz[elec_mask],
        "electrons_perp": np.sqrt(px[elec_mask]**2 + py[elec_mask]**2),
        "electrons_w":    w[elec_mask],
    }
    del q, px, py, pz, w, ion_mask, elec_mask
    gc.collect()
    return result


def extract_step(filepath: str) -> int:
    """Extract the integer step from a PSC particle filename."""
    match = STEP_RE.search(os.path.basename(filepath))
    if not match:
        raise ValueError(f"Could not extract step from filename: {filepath}")
    return int(match.group(1))


def resolve_particle_files(input_path: str) -> list[str]:
    """Resolve a file, directory, or glob pattern into an ordered file list."""
    if os.path.isdir(input_path):
        candidates = sorted(glob.glob(os.path.join(input_path, "prt*.h5")))
    elif any(ch in input_path for ch in "*?[]"):
        candidates = sorted(glob.glob(input_path))
    else:
        candidates = [input_path]

    files = [p for p in candidates if os.path.isfile(p)]
    if not files:
        raise FileNotFoundError(f"No particle files matched: {input_path}")

    return sorted(files, key=extract_step)


def sample_filepaths(
    filepaths: list[str], max_files: int = MAX_EVOLUTION_FILES
) -> list[str]:
    """Uniformly sub-sample filepaths for temporal scans."""
    if len(filepaths) <= max_files:
        return filepaths
    indices = np.unique(np.linspace(0, len(filepaths) - 1, max_files, dtype=int))
    return [filepaths[i] for i in indices]


# ══════════════════════════════════════════════════════════════════════════════
#  Species helpers
# ══════════════════════════════════════════════════════════════════════════════

def separate_species(data: np.ndarray, verbose: bool = True):
    """Separate particles by charge sign into ions and electrons."""
    ions = data[data["q"] > 0]
    electrons = data[data["q"] < 0]
    if verbose:
        print(f"  Ions: {len(ions):,},  Electrons: {len(electrons):,}")
    return ions, electrons


# ══════════════════════════════════════════════════════════════════════════════
#  Theoretical distributions
# ══════════════════════════════════════════════════════════════════════════════

def kappa_1d(v: np.ndarray, kappa: float, v_th: float) -> np.ndarray:
    """1-D kappa marginal with standard deviation v_th (plasma_physics.kappa_marginal_pdf)."""
    return kappa_marginal_pdf(v, v_th, kappa)


def maxwellian_1d(v: np.ndarray, v_th: float) -> np.ndarray:
    """1-D Maxwellian with standard deviation v_th."""
    return kappa_marginal_pdf(v, v_th, None)


#: Beyond this a fitted kappa is reported as "Maxwellian-consistent": the tail
#: of a kappa > 50 differs from a Gaussian only far beyond the sampled range.
KAPPA_DISPLAY_MAX = 50.0
#: Bins with fewer raw particles are not drawn: their density is shot noise.
MIN_BIN_COUNTS = 10
#: Colours of the data and of the three reference curves (Okabe-Ito).
C_DATA, C_MAXW, C_KFIT, C_K0 = "#D55E00", "#0072B2", "#009E73", "#555555"
DRIVEN = "ions" if DRIVEN_SPECIES == "ion" else "electrons"
SPECIES_TITLE = {"ions": "Ion", "electrons": "Electron"}


def _velocities(phase: dict, species: str):
    """v_z (along B0), v_x, v_y in v_A, each centred on its window-mean flow, and weights."""
    vx, vy, vz, _ = velocity_from_u(phase[f"{species}_px"], phase[f"{species}_py"],
                                    phase[f"{species}_pz"])
    w = np.asarray(phase[f"{species}_w"], dtype=float)
    centred = [(a - np.average(a, weights=w)) / VA_OVER_C_ for a in (vz, vx, vy)]
    return (*centred, w)


def _component(snapshot: tuple, which: str):
    """(values, weights) of the parallel component, or of both perpendicular ones pooled.

    v_x and v_y are two samples of the same gyrotropic perpendicular marginal,
    so pooling them doubles the statistics of the perpendicular curve.
    """
    vz, vx, vy, w = snapshot
    if which == "parallel":
        return vz, w
    return np.concatenate([vx, vy]), np.concatenate([w, w])


def _density(values, weights, edges, min_counts: int = MIN_BIN_COUNTS):
    """Normalised density per unit velocity and its shot-noise error; sparse bins NaN."""
    counts, _ = np.histogram(values, bins=edges)
    wsum, _ = np.histogram(values, bins=edges, weights=weights)
    w2sum, _ = np.histogram(values, bins=edges, weights=weights ** 2)
    norm = float(np.sum(weights)) * np.diff(edges)
    ok = counts >= min_counts
    return np.where(ok, wsum / norm, np.nan), np.where(ok, np.sqrt(w2sum) / norm, np.nan)


def _kappa_label(est: dict) -> str:
    k, lo, hi = est["kappa"], est["kappa_lo"], est["kappa_hi"]
    if not np.isfinite(k) or k > KAPPA_DISPLAY_MAX:
        bound = lo if np.isfinite(lo) and lo < KAPPA_DISPLAY_MAX else KAPPA_DISPLAY_MAX
        return rf"$\kappa$ fit: Maxwellian-consistent ($\kappa>{bound:.0f}$)"
    up = rf"^{{+{hi - k:.2f}}}" if np.isfinite(hi) else r"^{+\infty}"
    return rf"$\kappa$ fit $= {k:.2f}{up}_{{-{k - lo:.2f}}}$"


def _kappa_is_resolved(est: dict) -> bool:
    return bool(np.isfinite(est["kappa"]) and est["kappa"] <= KAPPA_DISPLAY_MAX)


def _load_snapshot(path: str, species: str):
    phase = load_particle_phase_space(path)
    snap = _velocities(phase, species)
    del phase
    gc.collect()
    return step_to_omegaci(extract_step(path)), snap


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 2: Kappa vs. Maxwellian comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_kappa_comparison(first_path: str, last_path: str, outdir: str, species: str = DRIVEN):
    """Driven-species velocity distribution at t = 0 and at the last snapshot.

    Why the curves are built this way: a model is only comparable with data of
    the same variance, so the Maxwellian and every kappa curve use the variance
    *measured at that time* (the temperature grows and the anisotropy relaxes
    during the run). The kappa curve is the maximum-likelihood kappa of the
    same sample (plasma_physics.kappa_mle, variance fixed), with its profile-
    likelihood interval; the initial kappa_0 is drawn dotted as a reference.
    At t = 0 the fit must return kappa_0 (a check of the loading); later it
    measures the tail. The lower panels divide by the Maxwellian, where a tail
    is a rise at large |v| instead of a slight change of slope on a log axis.
    Parallel is v along the global B0 (prt window, bulk flow removed); the two
    perpendicular components are pooled.
    """
    snaps = [_load_snapshot(path, species) for path in (first_path, last_path)]
    rows = []
    for comp, xlabel in (("parallel", r"$v_\parallel/v_A$"),
                         ("perpendicular", r"$v_{\perp,j}/v_A$  ($j = x, y$ pooled)")):
        samples = [(t, *_component(snap, comp)) for t, snap in snaps]
        sigmas = [float(np.sqrt(np.average(v ** 2, weights=w))) for _, v, w in samples]
        edges = np.linspace(-6.0 * max(sigmas), 6.0 * max(sigmas), 97)
        centres = 0.5 * (edges[:-1] + edges[1:])
        vfine = np.linspace(edges[0], edges[-1], 700)
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=True, sharey="row",
                                 gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.07,
                                              "wspace": 0.07})
        f_min, x_extent = np.inf, 0.0
        for col, (t, values, weights) in enumerate(samples):
            est = kappa_mle(values, weights)
            sigma = est["sigma"]
            f, err = _density(values, weights, edges)
            if np.any(f > 0):
                f_min = min(f_min, float(np.nanmin(f[f > 0])))
            f_m = kappa_marginal_pdf(centres, sigma, None)
            top, bottom = axes[0, col], axes[1, col]
            top.errorbar(centres, f, yerr=err, fmt="o", ms=3.2, color=C_DATA, ecolor=C_DATA,
                         elinewidth=0.8, capsize=0, zorder=3, label="PIC")
            top.plot(vfine, kappa_marginal_pdf(vfine, sigma, None), "--", color=C_MAXW, lw=1.8,
                     label=r"Maxwellian (measured $\sigma$)")
            populated = centres[np.isfinite(f)]
            if populated.size:
                x_extent = max(x_extent, float(np.max(np.abs(populated))))
            bottom.errorbar(centres, f / f_m, yerr=err / f_m, fmt="o", ms=3.2, color=C_DATA,
                            ecolor=C_DATA, elinewidth=0.8, capsize=0, zorder=3)
            # A Maxwellian-consistent sample has no kappa curve to draw (the
            # column title says so); the Maxwellian already is the model.
            if _kappa_is_resolved(est):
                fit_curve = kappa_marginal_pdf(vfine, sigma, est["kappa"])
                top.plot(vfine, fit_curve, "-", color=C_KFIT, lw=1.9,
                         label=r"$\kappa$ maximum-likelihood fit (measured $\sigma$)")
                bottom.plot(vfine, fit_curve / kappa_marginal_pdf(vfine, sigma, None), "-",
                            color=C_KFIT, lw=1.9)
            if KAPPA is not None:
                k0 = kappa_marginal_pdf(vfine, sigma, KAPPA)
                top.plot(vfine, k0, ":", color=C_K0, lw=1.6,
                         label=rf"$\kappa_0 = {KAPPA:g}$ (measured $\sigma$)")
                bottom.plot(vfine, k0 / kappa_marginal_pdf(vfine, sigma, None), ":",
                            color=C_K0, lw=1.6)
            bottom.axhline(1.0, color=ps.MUTED_CLR, lw=0.8)
            top.set_yscale("log")
            bottom.set_yscale("log")
            top.set_title(rf"$t\,\Omega_{{ci}} = {t:.1f}$" + "\n" + _kappa_label(est), fontsize=12.5)
            bottom.set_xlabel(xlabel, fontsize=13)
            rows.append({"component": comp, "omega_ci_t": t, "sigma_over_vA": sigma,
                         "kappa_mle": est["kappa"], "kappa_lo": est["kappa_lo"],
                         "kappa_hi": est["kappa_hi"], "n_eff": est["n_eff"],
                         "kappa_initial": KAPPA if KAPPA is not None else "inf"})
        # The lowest measured density sets the axis: a Maxwellian drawn to 6
        # sigma would otherwise push it decades below the data.
        if np.isfinite(f_min):
            axes[0, 0].set_ylim(0.4 * f_min, None)
        axes[1, 0].set_ylim(0.1, 30.0)
        if x_extent > 0:
            axes[1, 0].set_xlim(-1.08 * x_extent, 1.08 * x_extent)
        axes[0, 0].set_ylabel(r"$f(v)\,v_A$", fontsize=13)
        axes[1, 0].set_ylabel("PIC / Maxwellian", fontsize=13)
        handles = {}
        for ax in axes[0]:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(label, handle)
        # Data first (matplotlib lists error-bar containers after the lines).
        handles = {"PIC": handles.pop("PIC"), **handles} if "PIC" in handles else handles
        fig.legend(list(handles.values()), list(handles), loc="upper center",
                   bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False, fontsize=10.5)
        fig.suptitle(f"{SPECIES_TITLE[species]} {comp} velocity distribution — {PROFILE_LABEL}",
                     fontsize=15, fontweight="bold", y=1.04)
        fig.text(0.5, 0.995, "models carry the variance measured at each time; "
                 "prt window, global $B_0$ frame, bulk flow removed",
                 ha="center", va="bottom", fontsize=10, color=ps.MUTED_CLR)
        _save_paper_figure(fig, output_file(outdir, f"kappa_comparison_{comp}.png"))
    _write_rows(output_file(outdir, "kappa_mle_summary.csv"), rows)


def _write_rows(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 3: Goodness-of-Fit (A-D & K-S)
# ══════════════════════════════════════════════════════════════════════════════

def _ad_p_value(ad_stat_norm: float) -> float:
    """Approximate p-value for the Anderson-Darling statistic (Stephens 1974)."""
    a = ad_stat_norm
    if a >= 0.6:
        p = np.exp(1.2937 - 5.709 * a + 0.0186 * a**2)
    elif a >= 0.34:
        p = np.exp(0.9177 - 4.279 * a - 1.38 * a**2)
    elif a >= 0.2:
        p = 1 - np.exp(-8.318 + 42.796 * a - 59.938 * a**2)
    else:
        p = 1 - np.exp(-13.436 + 101.14 * a - 223.73 * a**2)
    return float(np.clip(p, 0.0, 1.0))


def _anderson_darling(cdf_sorted: np.ndarray) -> float:
    n = cdf_sorted.size
    i = np.arange(1, n + 1)
    lo = np.clip(cdf_sorted, 1e-12, 1 - 1e-12)
    hi = np.clip(cdf_sorted[::-1], 1e-12, 1 - 1e-12)
    return float(-n - np.mean((2 * i - 1) * (np.log(lo) + np.log(1 - hi))))


def plot_goodness_of_fit(last_path: str, outdir: str, species: str = DRIVEN, nsample: int = 20_000):
    """Empirical CDF minus model CDF at the last snapshot, with the KS 95 % band.

    Why the difference and not the CDFs: the three CDFs lie on top of each
    other at plotting resolution, while F_emp - F_model shows where (core or
    tail) a model fails and by how much, against the band 1.36/sqrt(n) inside
    which a correct model stays with 95 % probability. The models carry the
    measured variance; kappa is the maximum-likelihood fit (and kappa_0 when
    the run was loaded with one). With 1e5-1e6 particles any small systematic
    difference is 'significant', so D_KS is the quantity to compare, not p.
    """
    t, snap = _load_snapshot(last_path, species)
    rng = np.random.default_rng(RNG_SEED)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True, gridspec_kw={"wspace": 0.06})
    rows = []
    for ax, comp in zip(axes, ("parallel", "perpendicular")):
        values, weights = _component(snap, comp)
        est = kappa_mle(values, weights)
        sigma = est["sigma"]
        pick = rng.choice(values.size, size=min(nsample, values.size), replace=False,
                          p=weights / weights.sum())
        sample = np.sort(values[pick])
        n = sample.size
        ecdf_hi = np.arange(1, n + 1) / n
        x = sample / sigma
        models = [("Maxwellian", None, C_MAXW, "--")]
        if _kappa_is_resolved(est):
            models.append((rf"$\kappa$ fit $= {est['kappa']:.2f}$", est["kappa"], C_KFIT, "-"))
        if KAPPA is not None:
            models.append((rf"$\kappa_0 = {KAPPA:g}$", KAPPA, C_K0, ":"))
        row = {"component": comp, "omega_ci_t": t, "n": n, "sigma_over_vA": sigma,
               "kappa_mle": est["kappa"]}
        for name, kappa, colour, style in models:
            cdf = kappa_marginal_cdf(sample, sigma, kappa)
            diff = ecdf_hi - cdf
            d_ks = float(max(np.max(np.abs(diff)), np.max(np.abs(ecdf_hi - 1.0 / n - cdf))))
            ad = _anderson_darling(cdf)
            ax.plot(x, diff, style, color=colour, lw=1.6, label=rf"{name}: $D_{{KS}}={d_ks:.3f}$")
            key = "maxwellian" if kappa is None else ("kappa_fit" if kappa == est["kappa"] else "kappa_initial")
            row.update({f"ks_stat_{key}": d_ks,
                        f"ks_p_{key}": float(scipy_stats.kstwobign.sf(np.sqrt(n) * d_ks)),
                        f"ad_stat_{key}": ad,
                        f"ad_p_{key}": _ad_p_value(ad * (1 + 4 / n - 25 / n ** 2))})
        band = 1.36 / np.sqrt(n)
        ax.axhspan(-band, band, color=ps.MUTED_CLR, alpha=0.15, lw=0,
                   label=r"KS 95 % band, $1.36/\sqrt{n}$")
        ax.axhline(0.0, color=ps.MUTED_CLR, lw=0.8)
        ax.set_xlim(-5, 5)
        ax.set_xlabel(r"$v_\parallel/\sigma$" if comp == "parallel" else r"$v_{\perp,j}/\sigma$",
                      fontsize=13)
        ax.set_title(f"{comp} ($n = {n:,}$)".replace(",", r"\,"), fontsize=13)
        ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
                  frameon=False)
        rows.append(row)
    axes[0].set_ylabel(r"$F_{\rm PIC}(v) - F_{\rm model}(v)$", fontsize=13)
    fig.suptitle(rf"{SPECIES_TITLE[species]} VDF goodness of fit — $t\,\Omega_{{ci}} = {t:.1f}$, "
                 f"{PROFILE_LABEL}", fontsize=14, fontweight="bold", y=1.03)
    _save_paper_figure(fig, output_file(outdir, "goodness_of_fit.png"))
    _write_rows(output_file(outdir, "goodness_of_fit_metrics.csv"), rows)


# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════

def print_summary(ions, electrons, step: int):
    """Print summary statistics of the particle data."""
    print("\n" + "=" * 60)
    print(f"PARTICLE DATA SUMMARY (step {step}, t*Omega_ci={step_to_omegaci(step):.3f})")
    print("=" * 60)

    for name, sp in [("IONS", ions), ("ELECTRONS", electrons)]:
        m = np.abs(sp["m"][0])
        T_perp = 0.5 * m * (np.var(sp["px"]) + np.var(sp["py"]))
        T_par = m * np.var(sp["pz"])
        print(f"\n  {name}:")
        print(f"    Count: {len(sp):,}")
        print(f"    Mass (m): {sp['m'][0]:.4f}")
        print(f"    Charge (q): {sp['q'][0]:.4f}")
        print(f"    T_perp = {T_perp:.5f}, T_par = {T_par:.5f}")
        print(f"    T_perp/T_par = {T_perp / T_par:.4f}")

    print("\n" + "=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 6: Change of the distribution relative to t = 0
# ══════════════════════════════════════════════════════════════════════════════

#: Ratios below / above one decade are drawn at the colour-scale ends.
RATIO_DECADES = 1.0
#: A bin enters the ratio only with this many raw particles at both times.
RATIO_MIN_COUNTS = 25


def plot_distribution_change(filepaths: list[str], outdir: str, bins_par: int = 64,
                             bins_perp: int = 48):
    """log10 [f(v, t) / f(v, 0)] of ions and electrons, parallel and perpendicular.

    Why a ratio: f(v, t) itself spans five decades and hardly changes on a log
    colour scale, so the evolution (anisotropy relaxation, heating, tail
    erosion or growth) was invisible. Dividing by the initial distribution
    shows where phase-space density was lost (blue) or gained (red): a
    relaxing T_perp > T_par plasma moves density from large |v_perp| to large
    |v_par|; heating broadens both; an eroding tail turns the edges blue.
    Bins with fewer than RATIO_MIN_COUNTS particles at either time are masked.
    """
    print("\nBuilding distribution-change maps...")
    paths = sample_filepaths(filepaths)
    times = np.array([step_to_omegaci(extract_step(p)) for p in paths])
    snaps = []
    for path in paths:
        phase = load_particle_phase_space(path)
        snaps.append({sp: _velocities(phase, sp) for sp in ("ions", "electrons")})
        del phase; gc.collect()
    if len(times) == 1:
        t_edges = np.array([times[0] - 0.5, times[0] + 0.5])
    else:
        dt = np.diff(times)
        t_edges = np.concatenate([[times[0] - 0.5 * dt[0]], 0.5 * (times[:-1] + times[1:]),
                                  [times[-1] + 0.5 * dt[-1]]])
    cmap = plt.get_cmap(ps.CMAP_DIVERGING).copy()
    cmap.set_bad(ps.PANEL_BG)
    for species in ("ions", "electrons"):
        comps = []
        for which in ("parallel", "perp"):
            values = []
            for snap in snaps:
                vz, vx, vy, w = snap[species]
                values.append((vz if which == "parallel" else np.hypot(vx, vy), w))
            spread = max(float(np.sqrt(np.average(v ** 2, weights=w))) for v, w in values)
            edges = (np.linspace(-4.5 * spread, 4.5 * spread, bins_par + 1) if which == "parallel"
                     else np.linspace(0.0, 3.5 * spread, bins_perp + 1))
            dens, counts = [], []
            for v, w in values:
                c, _ = np.histogram(v, bins=edges)
                h, _ = np.histogram(v, bins=edges, weights=w)
                dens.append(h / (w.sum() * np.diff(edges)))
                counts.append(c)
            dens, counts = np.array(dens), np.array(counts)
            ok = (counts >= RATIO_MIN_COUNTS) & (counts[0] >= RATIO_MIN_COUNTS)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(ok, np.log10(dens / dens[0]), np.nan)
            comps.append((edges, ratio))
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), gridspec_kw={"wspace": 0.18})
        for ax, (edges, ratio), label in zip(
                axes, comps, (r"$v_\parallel/v_A$  ($\parallel B_0$)", r"$|v_\perp|/v_A$")):
            mesh = ax.pcolormesh(t_edges, edges, ratio.T, cmap=cmap, shading="flat",
                                 vmin=-RATIO_DECADES, vmax=RATIO_DECADES, rasterized=True)
            ax.set_xlabel(r"$t\,\Omega_{ci}$", fontsize=13)
            ax.set_ylabel(label, fontsize=13)
            rows_ok = np.any(np.isfinite(ratio), axis=0)
            if rows_ok.any():
                lo_edge, hi_edge = edges[:-1][rows_ok].min(), edges[1:][rows_ok].max()
                ax.set_ylim(lo_edge, hi_edge)
        cbar = fig.colorbar(mesh, ax=axes, pad=0.015, fraction=0.03)
        cbar.set_label(r"$\log_{10}\,[f(v,t)/f(v,0)]$", fontsize=12)
        fig.suptitle(f"{SPECIES_TITLE[species]} distribution relative to $t = 0$ — {PROFILE_LABEL}",
                     fontsize=14, fontweight="bold")
        _save_paper_figure(fig, output_file(outdir, f"distribution_change_{species}.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 9: 1-D VDF evolution with suprathermal-tail quantification
# ══════════════════════════════════════════════════════════════════════════════

def plot_1d_vdf_evolution(filepaths: list[str], outdir: str, n_times: int = 5, nbins: int = 90,
                          species: str = DRIVEN):
    """f(v_par) and f(|v_perp|) of the driven species at selected times.

    Each curve is normalised by the total particle weight of its snapshot and
    omits bins with fewer than MIN_BIN_COUNTS particles (shot noise). The
    references carry the t = 0 variances: the Maxwellian (Gaussian in v_par,
    Rayleigh in |v_perp|) and, for a run loaded with kappa_0, the kappa of the
    same variances (plasma_physics.kappa_marginal_pdf / kappa_speed_pdf_2d),
    so the departure of each later curve from the initial state is visible
    against the right reference. Tail fractions beyond 3 sigma(0) go to
    vdf_tail_fractions.csv.
    """
    print("\nBuilding 1-D VDF evolution...")
    paths = sample_filepaths(filepaths, max_files=n_times)
    loaded = [_load_snapshot(p, species) for p in paths]
    vz0, vx0, vy0, w0 = loaded[0][1]
    s_par = float(np.sqrt(np.average(vz0 ** 2, weights=w0)))
    s_perp = float(np.sqrt(0.5 * np.average(vx0 ** 2 + vy0 ** 2, weights=w0)))
    vpar_max = 6.0 * max(float(np.sqrt(np.average(s[1][0] ** 2, weights=s[1][3]))) for s in loaded)
    vperp_max = 4.5 * max(float(np.sqrt(0.5 * np.average(s[1][1] ** 2 + s[1][2] ** 2, weights=s[1][3])))
                          for s in loaded)
    e_par = np.linspace(-vpar_max, vpar_max, nbins + 1)
    e_perp = np.linspace(0.0, vperp_max, nbins // 2 + 1)
    c_par = 0.5 * (e_par[:-1] + e_par[1:])
    c_perp = 0.5 * (e_perp[:-1] + e_perp[1:])
    cmap = plt.get_cmap(ps.CMAP_SEQUENTIAL)
    colours = [cmap(0.88 * i / max(len(loaded) - 1, 1)) for i in range(len(loaded))]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), gridspec_kw={"wspace": 0.2})
    tails = []
    for (t, (vz, vx, vy, w)), colour in zip(loaded, colours):
        vperp = np.hypot(vx, vy)
        label = rf"$t\,\Omega_{{ci}} = {t:.1f}$"
        axes[0].plot(c_par, _density(vz, w, e_par)[0], color=colour, lw=1.7, label=label)
        axes[1].plot(c_perp, _density(vperp, w, e_perp)[0], color=colour, lw=1.7)
        tails.append({"omega_ci_t": t,
                      "fraction_abs_v_par_gt_3sigma_par0": float(np.average(np.abs(vz) > 3 * s_par, weights=w)),
                      "fraction_v_perp_gt_3sigma_perp0": float(np.average(vperp > 3 * s_perp, weights=w))})
    v1 = np.linspace(-vpar_max, vpar_max, 600)
    v2 = np.linspace(0.0, vperp_max, 400)
    axes[0].plot(v1, kappa_marginal_pdf(v1, s_par, None), "k--", lw=1.3, label=r"Maxwellian, $t = 0$ $\sigma$")
    axes[1].plot(v2, kappa_speed_pdf_2d(v2, s_perp, None), "k--", lw=1.3)
    if KAPPA is not None:
        axes[0].plot(v1, kappa_marginal_pdf(v1, s_par, KAPPA), "k:", lw=1.5,
                     label=rf"$\kappa_0 = {KAPPA:g}$, $t = 0$ $\sigma$")
        axes[1].plot(v2, kappa_speed_pdf_2d(v2, s_perp, KAPPA), "k:", lw=1.5)
    f_all = np.concatenate([np.asarray(l.get_ydata(), float) for ax in axes for l in ax.lines[:len(loaded)]])
    f_floor = 0.4 * np.nanmin(f_all[f_all > 0]) if np.any(f_all > 0) else 1e-5
    # x range: where some curve has data (sparse bins are NaN), not the binning range.
    for ax, centres in ((axes[0], c_par), (axes[1], c_perp)):
        filled = np.any([np.isfinite(np.asarray(l.get_ydata(), float)) for l in ax.lines[:len(loaded)]], axis=0)
        if filled.any():
            reach = float(np.max(np.abs(centres[filled])))
            ax.set_xlim(*((-1.06 * reach, 1.06 * reach) if ax is axes[0] else (0.0, 1.06 * reach)))
    for ax, xlabel, ylabel, title in (
            (axes[0], r"$v_\parallel/v_A$", r"$f(v_\parallel)\,v_A$", "parallel"),
            (axes[1], r"$|v_\perp|/v_A$", r"$f(|v_\perp|)\,v_A$", "perpendicular speed")):
        ax.set_yscale("log")
        ax.set_ylim(f_floor, None)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=10)
    fig.suptitle(f"{SPECIES_TITLE[species]} velocity distribution — {PROFILE_LABEL}",
                 fontsize=14, fontweight="bold", y=1.03)
    _save_paper_figure(fig, output_file(outdir, "vdf_1d_evolution.png"))
    _write_rows(output_file(outdir, "vdf_tail_fractions.csv"), tails)



# ══════════════════════════════════════════════════════════════════════════════
#  Plot 10: Particle energy partition (bulk and thermal)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_particle_energies(filepath: str) -> dict:
    """Return kinetic and thermal energies from a particle snapshot.

    PSC stores u = gamma*v. For this non-relativistic run:
      E_bulk = 0.5*m*|<u>|^2
      E_th   = 0.5*m*<|u-<u>|^2>
    Values are per macroparticle, so species with the same macro-weight can be
    compared and summed without introducing sample-size-dependent factors.
    """
    with h5py.File(filepath, "r") as f:
        dset  = f["particles"]["p0"]["1d"]
        n_tot = dset["q"].shape[0]
        idx = _sample_indices(n_tot, MAX_PARTICLES)
        q  = dset["q"][idx].astype(np.float64)
        m  = dset["m"][idx].astype(np.float64)
        px = dset["px"][idx].astype(np.float64)
        py = dset["py"][idx].astype(np.float64)
        pz = dset["pz"][idx].astype(np.float64)

    ion_mask  = q > 0
    elec_mask = q < 0

    result = {}
    for name, mask in [("ion", ion_mask), ("elec", elec_mask)]:
        mi     = float(np.abs(m[mask][0])) if mask.any() else 1.0
        vx     = px[mask]
        vy     = py[mask]
        vz     = pz[mask]
        bulk_v2 = np.mean(vx)**2 + np.mean(vy)**2 + np.mean(vz)**2
        rand_v2 = np.mean((vx - np.mean(vx))**2 +
                           (vy - np.mean(vy))**2 +
                           (vz - np.mean(vz))**2)
        t_perp = 0.5 * mi * (np.var(vx) + np.var(vy))
        t_par  = mi * np.var(vz)
        result[f"{name}_kinetic_bulk"]    = 0.5 * mi * bulk_v2
        result[f"{name}_thermal_energy"]  = 0.5 * mi * rand_v2
        result[f"{name}_t_perp"] = t_perp
        result[f"{name}_t_par"]  = t_par

    return result


def plot_energy_partition(filepaths: list[str], outdir: str):
    """Bulk and thermal energy of ions and electrons in the prt window, relative to t = 0.

    Per macroparticle (ions and electrons have the same count), normalised to
    the initial total particle energy E_0. The magnetic energy is not repeated
    here: 09_physical_diagnostics and the energy audit give it from every field
    snapshot and from DiagEnergies.
    """
    print("\nBuilding energy partition plot...")
    paths = sample_filepaths(filepaths, max_files=MAX_EVOLUTION_FILES)
    times = np.array([step_to_omegaci(extract_step(p)) for p in paths])
    series = {key: [] for key in ("ion_kinetic_bulk", "ion_thermal_energy",
                                  "elec_kinetic_bulk", "elec_thermal_energy")}
    for path in paths:
        energies = _compute_particle_energies(path)
        for key in series:
            series[key].append(energies[key])
        gc.collect()
    series = {key: np.asarray(value, dtype=float) for key, value in series.items()}
    e0 = sum(value[0] for value in series.values())
    e0 = e0 if e0 > 1e-30 else 1.0

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    _style_paper_axes(ax)
    for key, colour, marker, label in (
            ("ion_thermal_energy", "#E69F00", "s", "ion thermal"),
            ("elec_thermal_energy", "#0072B2", "^", "electron thermal"),
            ("ion_kinetic_bulk", "#009E73", "d", "ion bulk"),
            ("elec_kinetic_bulk", "#56B4E9", "x", "electron bulk")):
        ax.plot(times, series[key] / e0, marker + "-", color=colour, lw=1.8, ms=5, label=label)
    ax.set_ylabel(r"$E/E_0$", fontsize=14)
    ax.set_xlabel(r"$t\,\Omega_{ci}$", fontsize=14)
    ax.set_title(f"Particle energy in the prt window — {PROFILE_LABEL}", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    _save_paper_figure(fig, output_file(outdir, "particle_energy_partition.png"))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global OUTPUT_PREFIX
    parser = argparse.ArgumentParser(
        description=f"Particle diagnostics for {PROFILE_LABEL}."
    )
    parser.add_argument(
        "input", nargs="?",
        default=os.path.join(os.path.dirname(__file__), "..", "build", "src"),
        help="Particle file, data directory, or glob pattern.",
    )
    parser.add_argument(
        "--outdir",
        help="Output directory (default: <data-dir>/prt_plots).",
    )
    parser.add_argument(
        "--run-name", default="",
        help="Name prefixed to every generated file.",
    )
    args = parser.parse_args()
    input_path = args.input
    clean_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name).strip("_")
    OUTPUT_PREFIX = f"{clean_name}_" if clean_name else ""

    try:
        filepaths = resolve_particle_files(input_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Usage: python plot_prt.py [path_to_prt_file.h5 | directory | glob]")
        sys.exit(1)

    filepath = filepaths[-1]
    outdir = (
        os.path.abspath(args.outdir)
        if args.outdir
        else os.path.join(os.path.dirname(os.path.abspath(filepath)), OUTPUT_DIR)
    )
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}\n")

    if len(filepaths) > 1:
        print(f"Resolved {len(filepaths)} particle files "
              f"({os.path.basename(filepaths[0])} ... {os.path.basename(filepath)}).")
        plot_distribution_change(filepaths, outdir);   gc.collect()
        plot_1d_vdf_evolution(filepaths, outdir);      gc.collect()
        plot_energy_partition(filepaths, outdir);      gc.collect()

    data = load_particles(filepath)
    ions, electrons = separate_species(data)
    del data; gc.collect()
    print_summary(ions, electrons, extract_step(filepath))
    del ions, electrons; gc.collect()

    print("\nGenerating distribution-model plots...")
    plot_kappa_comparison(filepaths[0], filepath, outdir); gc.collect()
    plot_goodness_of_fit(filepath, outdir);               gc.collect()

    print(f"\nAll plots saved to: {outdir}")


if __name__ == "__main__":
    main()
