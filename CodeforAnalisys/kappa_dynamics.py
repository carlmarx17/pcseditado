#!/usr/bin/env python3
"""
kappa_dynamics.py — the ion kappa index in time, and what the magnetic field does to it
=====================================================================================
The suprathermal index of the driven ions is not constant during the runs
(v6b, local-field frame: kappa 3 -> 4.8 and 5 -> 8.9 by t Omega_ci = 158). This
module asks whether the magnetic field organises that change, in the two ways
it can:

1. **In space: the local field strength.** Adiabatic motion maps f along the
   field conserving mu and energy (Liouville). A bi-kappa stays a bi-kappa
   with the *same* kappa for passing particles, so without trapping or
   scattering kappa does not depend on the local b = |B|/B_ref
   (liouville_kappa.py gives the trapped-domain closures). The b-binned index
   of vdf_spatial.py is regressed on ln b at every snapshot, S = d(1/kappa)/d ln b,
   and the hole - peak difference of the |B|-percentile populations is
   followed in time. The time an ion needs to cross the particle window at
   the thermal speed is compared with the relaxation time: when crossing is
   much faster, spatial structure in kappa is erased and only a global change
   can survive.

2. **In time: the energy of the fluctuations.** Resonant wave-particle
   scattering drives the tail towards a Maxwellian at a rate that follows the
   fluctuation energy W(t) = <|dB|^2>/B0^2. A process that does not see the
   waves (the numerical relaxation of a PIC plasma, for one) acts at a constant
   rate from t = 0. With the suprathermal excess relaxing exponentially,

       d ln(1/kappa)/dt = -(nu_0 + c W(t))
       =>  ln[(1/kappa)/(1/kappa_0)] = -nu_0 t - c F(t),   F(t) = int_0^t W dt'

   (t in Omega_ci^-1), fitted per run by weighted least squares; windowed
   rates -d ln(1/kappa)/dt against the window-mean W show the same relation
   point by point. An isotropic control with the same kappa (no drive, no
   waves) measures nu_0 on its own: pass it with --control, or it is found
   next to the run (<instability>_<distribution>_isotropic).

The index is the truncated, whitened kurtosis estimator of kappa_eff.py in
the local-field frame, as 1/kappa: 0 is the Maxwellian. Newer products carry
it continued through the Maxwellian (negative = flatter than Maxwellian, e.g.
a resonant plateau) with errors on both sides; older ones give 1/kappa = 0
without error for a Maxwellian-consistent sample.

Inputs are analysis products (no raw data), per run root:
  03_particles/vdf_kappa_series.csv       index at every particle snapshot
                                          (vdf_hole_vs_peak_summary.csv: the
                                          figure snapshots only, older runs)
  03_particles/vdf_kappa_b_series.csv     index per b bin at every snapshot
                                          (vdf_b_profile_step*.csv, older runs)
  09_physical_diagnostics/field_fluctuation_table.csv   dB(t)
  09_physical_diagnostics/growth_rate_summary.csv       linear phase

Outputs (--outdir):
  kappa_field_evolution.png  1/kappa(t), the model-free tail content P(|dv_par| > 3 sigma)
                             and the fluctuation energy at the same times
  kappa_relaxation.png       the two-term relaxation model and the rate against W
  kappa_vs_local_field.png   1/kappa(b) at several times, S(t), hole - peak(t)
  kappa_shape_evolution.png  f(v_par) over the Gaussian of the same variance in
                             time and at four instants, with the cyclotron
                             resonance: tail (wings above 1) against flattening
                             (shoulders above 1, core and wings below)
  kappa_shape_metrics.csv    core / shoulder / tail probability over the Gaussian
  kappa_dynamics_timeseries.csv, kappa_relaxation_rates.csv,
  kappa_local_field.csv, kappa_relaxation_fit.json

Usage:
  python kappa_dynamics.py RUN_MAXW RUN_K5 RUN_K3 --outdir OUT
  python kappa_dynamics.py RUN_K3 --control ../v6c/mirror_bikappa3_isotropic --outdir OUT
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

os.environ.pop("PSC_ANALYSIS_DATA_DIR", None)
import plot_style as ps  # noqa: E402

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter  # noqa: E402

import psc_units  # noqa: E402
from growth_fit import reference_growth_row  # noqa: E402

#: Okabe-Ito in the order of paper_figures.py: Maxwellian, kappa 5, kappa 3.
SERIES = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
#: kappa values labelled on the right axis of the 1/kappa panels.
KAPPA_TICKS = (3, 4, 5, 7, 10, 20)
#: A b bin enters the local-field regression only with this many particles.
MIN_BIN_COUNT = 5000
#: Runs whose initial 1/kappa is below this have no suprathermal excess to relax.
MIN_EXCESS = 0.02
#: Number of windows for the rate-vs-W panel.
RATE_WINDOWS = 10
#: At most this many snapshots are drawn as 1/kappa(b) profiles.
PROFILE_SNAPSHOTS = 6
#: Fraction of a Gaussian beyond 3 standard deviations, 2 [1 - Phi(3)].
GAUSS_TAIL_3SIGMA = 0.0026997960632601866


def _rows(path: Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── Reading ──────────────────────────────────────────────────────────────────

def index_of(row: dict) -> tuple[float, float]:
    """(1/kappa, error) of a product row: the signed index when the pipeline wrote it."""
    inv = _f(row.get("inv_kappa_signed"))
    if np.isfinite(inv):
        err = _f(row.get("inv_kappa_signed_error", row.get("inv_kappa_signed_err")))
        return inv, err
    kappa = _f(row.get("kappa_eff"))
    if np.isposinf(kappa):
        return 0.0, float("nan")
    if not np.isfinite(kappa) or kappa <= 0:
        return float("nan"), float("nan")
    err_k = _f(row.get("kappa_eff_error", row.get("kappa_err")))
    return 1.0 / kappa, (err_k / kappa ** 2 if np.isfinite(err_k) else float("nan"))


def run_label(profile: dict, name: str) -> str:
    kappa = profile.get("kappa")
    label = "bi-Maxwellian" if kappa is None else rf"bi-$\kappa$, $\kappa_0={kappa:g}$"
    return label + (" (isotropic control)" if "isotropic" in name else "")


def load_field(root: Path) -> dict | None:
    """W(t) = <|dB|^2>/B0^2, its compressive part <dB_z^2>/B0^2, and F = int W dt."""
    path = root / "09_physical_diagnostics" / "field_fluctuation_table.csv"
    if not path.exists():
        return None
    rows = _rows(path)
    t = np.array([_f(r.get("omega_ci_t")) for r in rows])
    vec = np.array([_f(r.get("delta_B_vec_rms_over_B0")) for r in rows])
    par = np.array([_f(r.get("delta_B_parallel_rms_over_B0")) for r in rows])
    ok = np.isfinite(t) & np.isfinite(vec)
    order = np.argsort(t[ok])
    t, w, w_par = t[ok][order], vec[ok][order] ** 2, par[ok][order] ** 2
    if t.size < 2:
        return None
    fluence = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
    return {"t": t, "W": w, "W_par": w_par, "F": fluence}


def linear_phase_end(root: Path) -> float:
    path = root / "09_physical_diagnostics" / "growth_rate_summary.csv"
    if not path.exists():
        return float("nan")
    row = reference_growth_row(_rows(path)) or {}
    return _f(row.get("linear_phase_end"))


def load_b_profiles(root: Path, step_time: dict) -> list[dict]:
    """Rows (t, b, count, 1/kappa, error) of the index per b bin, every snapshot available."""
    part = root / "03_particles"
    dense = part / "vdf_kappa_b_series.csv"
    out = []
    if dense.exists():
        for r in _rows(dense):
            inv, err = index_of(r)
            out.append({"t": _f(r["omega_ci_t"]), "b": _f(r.get("b_mean")),
                        "count": _f(r.get("count"), 0.0), "inv": inv, "err": err})
        return out
    for path in sorted(part.glob("vdf_b_profile_step*.csv")):
        step = int(path.stem.split("step")[-1])
        if step not in step_time:
            continue
        for r in _rows(path):
            inv, err = index_of(r)
            out.append({"t": step_time[step], "b": _f(r.get("b_mean")),
                        "count": _f(r.get("count"), 0.0), "inv": inv, "err": err})
    return out


def load_tail(root: Path, series_rows: list[dict]) -> dict | None:
    """Fraction of ions beyond 3 sigma in v_par, the model-free tail content.

    A kappa tail raises it above the Gaussian 0.27 %; a flattened core (a
    plateau from resonant diffusion) lowers it. From the local-field series
    when written, else from fit_metrics.csv (v_par along the global B0).
    """
    local = [(_f(r["omega_ci_t"]), _f(r.get("tail_fraction_par_3sigma"))) for r in series_rows
             if r.get("population") == "all" and np.isfinite(_f(r.get("tail_fraction_par_3sigma")))]
    if local:
        a = np.array(sorted(local))
        return {"t": a[:, 0], "frac": a[:, 1], "frame": "local field"}
    path = root / "09_physical_diagnostics" / "fit_metrics.csv"
    if not path.exists():
        return None
    rows = [(_f(r.get("omega_ci_t")), _f(r.get("suprathermal_fraction"))) for r in _rows(path)]
    a = np.array(sorted(x for x in rows if np.isfinite(x[0]) and np.isfinite(x[1])))
    return {"t": a[:, 0], "frac": a[:, 1], "frame": r"global $B_0$"} if a.size else None


def load_shape(root: Path) -> dict | None:
    """Standardised histograms of v_par and v_perp at every snapshot (vdf_spatial.py)."""
    path = root / "03_particles" / "vdf_shape_series.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def resonant_speed(root: Path, profile: dict) -> float:
    """|v_res|/v_A = (1 - |omega|/Omega_ci)/(|k_par| d_i) of the dominant measured mode.

    The n = 1 cyclotron resonance of an ion-cyclotron wave (0 < omega < Omega_ci),
    from the frequency the dispersion analysis resolved; nan for another
    branch, an electron-driven profile or an unresolved frequency.
    """
    if profile.get("driven_species", "ion") != "ion":
        return float("nan")
    for path in sorted((root / "04_spectra").glob("dispersion_modes*.json")):
        report = json.loads(path.read_text())
        mode = report.get("dominant") or report.get("strongest_spatial_peak") or {}
        k, w = abs(_f(mode.get("k_parallel_d_i"))), abs(_f(mode.get("omega_over_omega_ci")))
        if mode.get("frequency_resolved") and k > 0 and 0 < w < 1:
            return float((1.0 - w) / k)
    return float("nan")


def load_run(root) -> dict | None:
    """Everything this module uses from one run root; None without a kappa product."""
    root = Path(root)
    part = root / "03_particles"
    source = next((p for p in (part / "vdf_kappa_series.csv",
                               part / "vdf_hole_vs_peak_summary.csv") if p.exists()), None)
    if source is None:
        return None
    profile = psc_units._PROFILES.get(root.name, {})
    pops: dict[str, list] = {}
    source_rows = _rows(source)
    for r in source_rows:
        inv, err = index_of(r)
        pops.setdefault(r["population"], []).append(
            (_f(r["omega_ci_t"]), inv, err, _f(r.get("A_local_b")), _f(r.get("step"))))
    series = {}
    for name, items in pops.items():
        a = np.array(sorted(items), dtype=float)
        series[name] = {"t": a[:, 0], "inv": a[:, 1], "err": a[:, 2], "A": a[:, 3],
                        "step": a[:, 4]}
    if "all" not in series:
        return None
    step_time = {int(s): t for s, t in zip(series["all"]["step"], series["all"]["t"])
                 if np.isfinite(s)}
    meta_path = part / "vdf_spatial_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    window = meta.get("prt_window", {}).get("size_di")
    return {
        "name": root.name, "root": root, "profile": profile,
        "kappa0": profile.get("kappa"), "control": "isotropic" in root.name,
        "label": run_label(profile, root.name), "source": source.name,
        "dense": source.name == "vdf_kappa_series.csv",
        "pop": series, "field": load_field(root), "t_lin_end": linear_phase_end(root),
        "b_profiles": load_b_profiles(root, step_time),
        "tail": load_tail(root, source_rows),
        # quiet-start noise build-up of the box fundamental (psc_units.noise_settling_time)
        "t_settle": (psc_units.NOISE_SETTLING_TRANSITS * profile["domain_di"] / (2.0 * np.pi)
                     / np.sqrt(profile["beta_i_par"] * max(1.0, profile["Ti_perp_over_Ti_par"]) / 2.0)
                     if profile.get("driven_species", "ion") == "ion" and profile.get("domain_di") else 0.0),
        "shape": load_shape(root),
        "v_res": resonant_speed(root, profile),
        "window_di": float(min(window)) if window else float("nan"),
        # thermal speed along B of the driven ions, in v_A (profile)
        "vth_par": float(np.sqrt(profile.get("beta_i_par", np.nan) / 2.0)),
    }


def find_control(run: dict, explicit: list[dict]) -> dict | None:
    """The isotropic control with the same kappa: given with --control, or a sibling root."""
    same = lambda other: (other is not None and other["control"]
                          and other["kappa0"] == run["kappa0"]
                          and other["profile"].get("instability") == run["profile"].get("instability"))
    for other in explicit:
        if same(other):
            return other
    for sibling in sorted(Path(run["root"]).parent.iterdir()):
        if sibling.is_dir() and "isotropic" in sibling.name and sibling != run["root"]:
            candidate = load_run(sibling)
            if same(candidate):
                return candidate
    return None


# ── Analysis ─────────────────────────────────────────────────────────────────

def _weighted_fit(x: np.ndarray, y: np.ndarray, err: np.ndarray, intercept: bool = False) -> dict:
    """Linear least squares y = X beta with 1/err^2 weights; errors inflated by sqrt(chi2/dof) > 1."""
    x = np.atleast_2d(np.asarray(x, dtype=float).T).T if np.ndim(x) == 1 else np.asarray(x, float)
    if intercept:
        x = np.column_stack([np.ones(len(y)), x])
    ok = np.all(np.isfinite(x), axis=1) & np.isfinite(y) & np.isfinite(err) & (err > 0)
    x, y, err = x[ok], y[ok], err[ok]
    npar = x.shape[1]
    nan = {"beta": np.full(npar, np.nan), "err": np.full(npar, np.nan), "r2": float("nan"),
           "chi2_dof": float("nan"), "n": int(ok.sum())}
    if y.size <= npar:
        return nan
    w = 1.0 / err
    a, b = x * w[:, None], y * w
    beta, *_ = np.linalg.lstsq(a, b, rcond=None)
    cov = np.linalg.pinv(a.T @ a)
    res = y - x @ beta
    chi2_dof = float(np.sum((res * w) ** 2) / (y.size - npar))
    cov *= max(chi2_dof, 1.0)
    # Centred R^2 in every case, so fits with and without intercept compare.
    total = np.sum((y - np.average(y, weights=w ** 2)) ** 2)
    r2 = float(1.0 - np.sum(res ** 2) / total) if total > 0 else float("nan")
    return {"beta": beta, "err": np.sqrt(np.diag(cov)), "r2": r2, "chi2_dof": chi2_dof,
            "n": int(y.size)}


def fluence_at(run: dict, t: np.ndarray) -> np.ndarray:
    field = run["field"]
    if field is None:
        return np.full(np.shape(t), np.nan)
    return np.interp(t, field["t"], field["F"])


def relaxation(run: dict) -> dict | None:
    """ln[(1/kappa)/(1/kappa_0)] = -nu_0 t - c F(t) for a run with a suprathermal excess."""
    s = run["pop"]["all"]
    t, inv, err = s["t"], s["inv"], s["err"]
    if t.size < 3 or not (np.isfinite(inv[0]) and inv[0] > MIN_EXCESS):
        return None
    keep = (t > t[0]) & (inv > 0) & np.isfinite(err)
    y = np.log(inv[keep] / inv[0])
    ey = np.hypot(err[keep] / inv[keep], err[0] / inv[0] if np.isfinite(err[0]) else 0.0)
    tt = t[keep] - t[0]
    fl = fluence_at(run, t[keep]) - fluence_at(run, np.array([t[0]]))[0]
    out = {"t": t[keep], "y": y, "y_err": ey, "F": fl, "inv0": float(inv[0]), "t0": float(t[0])}
    base = _weighted_fit(-tt, y, ey)
    out["baseline_only"] = {"nu0": float(base["beta"][0]), "nu0_err": float(base["err"][0]),
                            "r2": base["r2"], "chi2_dof": base["chi2_dof"]}
    if run["control"] or not np.any(np.isfinite(fl)):
        out["nu0"], out["nu0_err"] = out["baseline_only"]["nu0"], out["baseline_only"]["nu0_err"]
        out["c"], out["c_err"] = float("nan"), float("nan")
        out["r2"], out["chi2_dof"] = base["r2"], base["chi2_dof"]
        return out
    wave = _weighted_fit(-fl, y, ey)
    out["wave_only"] = {"c": float(wave["beta"][0]), "c_err": float(wave["err"][0]),
                        "r2": wave["r2"], "chi2_dof": wave["chi2_dof"]}
    both = _weighted_fit(np.column_stack([-tt, -fl]), y, ey)
    out["nu0"], out["c"] = (float(v) for v in both["beta"])
    out["nu0_err"], out["c_err"] = (float(v) for v in both["err"])
    out["r2"], out["chi2_dof"] = both["r2"], both["chi2_dof"]
    return out


def joint_fit(runs: list[dict], relax: dict) -> dict | None:
    """One wave coefficient c shared by every driven run, a background nu_0 per run.

    If the wave-driven erosion of the tail is one process, a single c must fit
    all the kappa_0; chi2/dof of the joint fit against the separate fits says
    whether it does.
    """
    driven = [r for r in runs if not r["control"] and relax.get(r["name"])
              and np.isfinite(relax[r["name"]].get("c", np.nan))]
    if len(driven) < 2:
        return None
    blocks, ys, errs = [], [], []
    for i, run in enumerate(driven):
        fit = relax[run["name"]]
        cols = np.zeros((fit["t"].size, len(driven) + 1))
        cols[:, i] = -(fit["t"] - fit["t0"])
        cols[:, -1] = -fit["F"]
        blocks.append(cols)
        ys.append(fit["y"])
        errs.append(fit["y_err"])
    result = _weighted_fit(np.vstack(blocks), np.concatenate(ys), np.concatenate(errs))
    separate = [relax[r["name"]]["chi2_dof"] for r in driven]
    return {"runs": [r["name"] for r in driven],
            "c": float(result["beta"][-1]), "c_err": float(result["err"][-1]),
            "nu0": {r["name"]: float(v) for r, v in zip(driven, result["beta"][:-1])},
            "nu0_err": {r["name"]: float(v) for r, v in zip(driven, result["err"][:-1])},
            "chi2_dof": result["chi2_dof"], "chi2_dof_separate": separate}


def windowed_rates(run: dict, n_windows: int = RATE_WINDOWS) -> list[dict]:
    """-d ln(1/kappa)/dt in consecutive windows, with the window-mean W."""
    s = run["pop"]["all"]
    t, inv, err = s["t"], s["inv"], s["err"]
    ok = (inv > 0) & np.isfinite(err) & (err > 0)
    t, inv, err = t[ok], inv[ok], err[ok]
    if t.size < 3:
        return []
    edges = np.unique(np.linspace(0, t.size - 1, min(n_windows, t.size - 1) + 1).round().astype(int))
    rows = []
    for i, j in zip(edges[:-1], edges[1:]):
        span = t[j] - t[i]
        if j - i == 1:     # two snapshots: the difference quotient
            rate = -(np.log(inv[j]) - np.log(inv[i])) / span
            rate_err = np.hypot(err[i] / inv[i], err[j] / inv[j]) / span
        else:
            sel = slice(i, j + 1)
            fit = _weighted_fit(t[sel], np.log(inv[sel]), err[sel] / inv[sel], intercept=True)
            rate, rate_err = -fit["beta"][1], fit["err"][1]
        f0, f1 = fluence_at(run, np.array([t[i], t[j]]))
        rows.append({"run": run["name"], "t_start": float(t[i]), "t_end": float(t[j]),
                     "rate": float(rate), "rate_err": float(rate_err),
                     "W_mean": float((f1 - f0) / span) if span > 0 else float("nan"),
                     "n_snapshots": int(j - i + 1)})
    return rows


def local_field_slopes(run: dict) -> list[dict]:
    """S = d(1/kappa)/d ln b per snapshot, and the hole - peak difference."""
    by_t: dict[float, list[dict]] = {}
    for r in run["b_profiles"]:
        if (r["count"] >= MIN_BIN_COUNT and np.isfinite(r["inv"]) and np.isfinite(r["err"])
                and r["err"] > 0 and r["b"] > 0):
            by_t.setdefault(round(r["t"], 6), []).append(r)
    hole, peak = run["pop"].get("hole"), run["pop"].get("peak")
    rows = []
    for t in sorted(by_t):
        bins = by_t[t]
        row = {"run": run["name"], "omega_ci_t": t, "n_bins": len(bins),
               "b_min": min(b["b"] for b in bins), "b_max": max(b["b"] for b in bins),
               "slope": float("nan"), "slope_err": float("nan"),
               "hole_minus_peak": float("nan"), "hole_minus_peak_err": float("nan")}
        if len(bins) >= 3:
            fit = _weighted_fit(np.log([b["b"] for b in bins]), np.array([b["inv"] for b in bins]),
                                np.array([b["err"] for b in bins]), intercept=True)
            row["slope"], row["slope_err"] = float(fit["beta"][1]), float(fit["err"][1])
        if hole is not None and peak is not None:
            ih, ip = np.argmin(np.abs(hole["t"] - t)), np.argmin(np.abs(peak["t"] - t))
            if abs(hole["t"][ih] - t) < 1e-3 and abs(peak["t"][ip] - t) < 1e-3:
                row["hole_minus_peak"] = float(hole["inv"][ih] - peak["inv"][ip])
                row["hole_minus_peak_err"] = float(np.hypot(hole["err"][ih], peak["err"][ip]))
        rows.append(row)
    return rows


# ── Figures ──────────────────────────────────────────────────────────────────

def _kappa_axis(ax) -> None:
    """kappa labels on the right of a 1/kappa axis (limits must be final).

    A secondary axis on the same scale, not a twin: it draws labels only and
    is not a second panel of data.
    """
    lo, hi = ax.get_ylim()
    same = (lambda x: x, lambda x: x)
    right = ax.secondary_yaxis("right", functions=same)
    ticks = [(1.0 / k, f"{k:g}") for k in KAPPA_TICKS if lo <= 1.0 / k <= hi]
    if lo <= 0.0 <= hi:
        ticks.append((0.0, r"$\infty$"))
    right.set_yticks([v for v, _ in ticks], labels=[s for _, s in ticks])
    right.set_ylabel(r"$\kappa$")
    right.tick_params(which="minor", right=False)


def _draw_index(ax, run: dict, color: str, population: str = "all", **style) -> None:
    s = run["pop"].get(population)
    if s is None:
        return
    t, inv, err = s["t"], s["inv"], s["err"]
    ls = "--" if run["control"] else "-"
    if run["dense"]:
        ax.plot(t, inv, ls, color=color, lw=1.6, label=run["label"], **style)
        band = np.isfinite(err)
        ax.fill_between(t[band], (inv - err)[band], (inv + err)[band], color=color, alpha=0.2, lw=0)
    else:
        ax.errorbar(t, inv, yerr=np.where(np.isfinite(err), err, 0.0), fmt="o" + ls, ms=5,
                    lw=1.6, capsize=2.5, color=color, label=run["label"],
                    mfc="white" if run["control"] else color, **style)


def _gauss_ratio_of_kappa(kappa) -> float:
    """Fraction beyond 3 sigma of the kappa marginal, over the Gaussian one."""
    from plasma_physics import kappa_marginal_cdf
    return float(2.0 * (1.0 - kappa_marginal_cdf(3.0, 1.0, kappa)) / GAUSS_TAIL_3SIGMA)


def plot_field_evolution(runs: list[dict], colors: dict, path: Path) -> Path:
    with_tail = any(r.get("tail") for r in runs)
    heights = [1.35, 0.95, 1.0] if with_tail else [1.35, 1.0]
    fig, axes = plt.subplots(len(heights), 1, figsize=(8.2, 7.6 + 2.6 * with_tail), sharex=True,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.16 if with_tail else 0.08})
    top, bottom = axes[0], axes[-1]
    middle = axes[1] if with_tail else None
    frames = set()
    for run in runs:
        col = colors[run["name"]]
        _draw_index(top, run, col)
        if run["kappa0"] and not run["control"]:
            top.axhline(1.0 / run["kappa0"], color=col, lw=0.9, ls=":", alpha=0.9)
        if np.isfinite(run["t_lin_end"]) and not run["control"]:
            for ax in axes:
                ax.axvline(run["t_lin_end"], color=col, lw=0.9, ls=(0, (5, 2, 1, 2)), alpha=0.8)
        tail = run.get("tail")
        if middle is not None and tail is not None:
            frames.add(tail["frame"])
            middle.plot(tail["t"], tail["frac"] / GAUSS_TAIL_3SIGMA, "--" if run["control"] else "-",
                        color=col, lw=1.4)
            if run["kappa0"] and not run["control"]:
                middle.axhline(_gauss_ratio_of_kappa(run["kappa0"]), color=col, lw=0.9, ls=":")
        field = run["field"]
        if field is not None and not run["control"]:
            shown = ps.measured_fluctuation(field["t"], run["t_settle"]) & (field["W"] > 0)
            bottom.plot(field["t"][shown], field["W"][shown], "-", color=col, lw=1.6)
            par = shown & (field["W_par"] > 0)
            bottom.plot(field["t"][par], field["W_par"][par], ":", color=col, lw=1.4)
    top.axhline(0.0, color=ps.MUTED_CLR, lw=0.8, ls="--")
    top.set_ylabel(r"$1/\kappa_{\rm eff}$  (0 = Maxwellian)")
    top.set_title("Ion suprathermal index (local-field frame), tail content and magnetic "
                  "fluctuation energy" if with_tail else
                  "Ion suprathermal index (local-field frame) and magnetic fluctuation energy",
                  fontsize=12)
    lo, hi = top.get_ylim()
    top.set_ylim(min(lo, -0.01), hi)
    _kappa_axis(top)
    if middle is not None:
        middle.axhline(1.0, color=ps.MUTED_CLR, lw=0.8, ls="--")
        middle.set_yscale("log")
        middle.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        middle.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        middle.yaxis.set_minor_formatter(NullFormatter())
        middle.set_ylabel("ions beyond " r"$3\sigma_\parallel$" "\n(Gaussian = 1)")
        # Above the panel, not inside: the curves can reach any height.
        middle.set_title("model-free tail content (" + ", ".join(sorted(frames)) +
                         r" frame): above 1 a suprathermal tail, below 1 a flattened core",
                         fontsize=8.8, color=ps.MUTED_CLR, loc="left", pad=3)
    bottom.set_yscale("log")
    ps.plain_log_axis(bottom, "y")
    bottom.set_ylabel(r"$\langle|\delta\mathbf{B}|^2\rangle/B_0^2$")
    bottom.set_xlabel(r"$t\,\Omega_{ci}$")
    handles, labels = top.get_legend_handles_labels()
    extra = [Line2D([], [], color=ps.MUTED_CLR, ls=":", lw=0.9, label=r"loaded $1/\kappa_0$"),
             Line2D([], [], color=ps.MUTED_CLR, ls=(0, (5, 2, 1, 2)), lw=0.9, label="end of linear phase"),
             Line2D([], [], color=ps.MUTED_CLR, ls="-", lw=1.6, label=r"total $|\delta\mathbf{B}|^2$"),
             Line2D([], [], color=ps.MUTED_CLR, ls=":", lw=1.4,
                    label=r"compressive $\delta B_\parallel^2$")]
    fig.legend(handles + extra, labels + [h.get_label() for h in extra], loc="upper center",
               bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, fontsize=9.5)
    ps.save(fig, path)
    return path


def plot_relaxation(runs: list[dict], colors: dict, fits: dict, rates: dict, path: Path,
                    joint: dict | None = None) -> Path | None:
    shown = [r for r in runs if fits.get(r["name"])]
    if not shown:
        return None
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.9), gridspec_kw={"wspace": 0.28})
    for run in shown:
        col, fit = colors[run["name"]], fits[run["name"]]
        ls = "--" if run["control"] else "-"
        left.errorbar(fit["t"], fit["y"], yerr=fit["y_err"], fmt="o", ms=3.5 if run["dense"] else 5,
                      color=col, mfc="white" if run["control"] else col, capsize=0, lw=1.0,
                      label=run["label"])
        tt = np.linspace(fit["t0"], fit["t"].max(), 200)
        base = -fit["nu0"] * (tt - fit["t0"])
        if np.isfinite(fit.get("c", np.nan)):
            model = base - fit["c"] * (fluence_at(run, tt) - fluence_at(run, np.array([fit["t0"]]))[0])
            left.plot(tt, model, ls, color=col, lw=1.6)
            left.plot(tt, base, ":", color=col, lw=1.3)
        else:
            left.plot(tt, base, ls, color=col, lw=1.4)
        # Rates in units of 1e-3 Omega_ci, set here rather than as a folded
        # axis offset (which would re-run the layout).
        rr = [r for r in rates.get(run["name"], []) if np.isfinite(r["rate"]) and r["W_mean"] > 0]
        if rr:
            right.errorbar([r["W_mean"] for r in rr], [1e3 * r["rate"] for r in rr],
                           yerr=[1e3 * r["rate_err"] for r in rr], fmt="o", ms=5, capsize=2.5,
                           color=col, mfc="white" if run["control"] else col, lw=1.0)
            if np.isfinite(fit.get("c", np.nan)):
                w = np.logspace(np.log10(min(r["W_mean"] for r in rr)) - 0.2,
                                np.log10(max(r["W_mean"] for r in rr)) + 0.1, 100)
                right.plot(w, 1e3 * (fit["nu0"] + fit["c"] * w), ls, color=col, lw=1.4)
            else:
                right.axhline(1e3 * fit["nu0"], color=col, lw=1.2, ls=ls)
    left.axhline(0.0, color=ps.MUTED_CLR, lw=0.8)
    left.set_xlabel(r"$t\,\Omega_{ci}$")
    left.set_ylabel(r"$\ln[(1/\kappa)/(1/\kappa_0)]$")
    left.set_title(r"(a) $\ln[(1/\kappa)/(1/\kappa_0)] = -\nu_0 t - c\,\int_0^t W\,dt'$", fontsize=12)
    right.set_xscale("log")
    right.set_xlabel(r"window mean $W = \langle|\delta\mathbf{B}|^2\rangle/B_0^2$")
    right.set_ylabel(r"$-d\ln(1/\kappa)/dt$  $[10^{-3}\,\Omega_{ci}]$")
    right.set_title(r"(b) relaxation rate against the fluctuation energy: $\nu_0 + c\,W$", fontsize=12)
    right.axhline(0.0, color=ps.MUTED_CLR, lw=0.8)
    # Decades only: W spans ~2 decades and plain labels of every minor tick collide.
    right.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    right.xaxis.set_major_formatter(LogFormatterMathtext())
    right.xaxis.set_minor_formatter(NullFormatter())
    handles, labels = left.get_legend_handles_labels()
    extra = [Line2D([], [], color=ps.MUTED_CLR, lw=1.6, label=r"fit $-\nu_0 t - cF$"),
             Line2D([], [], color=ps.MUTED_CLR, lw=1.3, ls=":", label=r"$-\nu_0 t$ part")]
    fig.legend(handles + extra, labels + [h.get_label() for h in extra], loc="upper center",
               bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False, fontsize=9.5)
    notes = []
    for run in shown:
        fit = fits[run["name"]]
        text = rf"{run['label']}: $\nu_0 = ({1e3 * fit['nu0']:.2f}\pm{1e3 * fit['nu0_err']:.2f})\times10^{{-3}}$"
        if np.isfinite(fit.get("c", np.nan)):
            text += rf", $c = {fit['c']:.3f}\pm{fit['c_err']:.3f}$"
        notes.append(text)
    if joint:
        notes.append(rf"one $c$ for all $\kappa_0$: $c = {joint['c']:.3f}\pm{joint['c_err']:.3f}$ "
                     rf"($\chi^2/\nu = {joint['chi2_dof']:.1f}$)")
    # Lower left of (a): the curves start at 0 and fall to the right.
    left.text(0.02, 0.03, "\n".join(notes), transform=left.transAxes, ha="left", va="bottom",
              fontsize=8.8, color=ps.TEXT_CLR)
    ps.save(fig, path)
    return path


def plot_local_field(runs: list[dict], colors: dict, slopes: dict, relax: dict, path: Path) -> Path | None:
    profiled = [r for r in runs if any(np.isfinite(b["inv"]) and b["count"] >= MIN_BIN_COUNT
                                       for b in r["b_profiles"])
                and (r["kappa0"] or any(np.isfinite(b["err"]) for b in r["b_profiles"]))]
    if not profiled:
        return None
    ncol = max(len(profiled), 2)
    fig = plt.figure(figsize=(4.3 * ncol + 0.8, 8.4))
    # The GridSpec below places every panel; a placeholder engine keeps
    # plot_style from running tight_layout over it when it folds an axis
    # offset into a label (colour-bar axes are not tight_layout compatible).
    fig.set_layout_engine("none")
    grid = GridSpec(2, 2 * ncol, figure=fig, height_ratios=[1.1, 1.0], hspace=0.45, wspace=0.9,
                    top=0.85, right=0.9)
    top_axes = []
    cmap = plt.get_cmap("viridis")
    t_all = [b["t"] for r in profiled for b in r["b_profiles"]]
    t_max = max(t_all) if t_all else 1.0
    for i, run in enumerate(profiled):
        ax = fig.add_subplot(grid[0, 2 * i:2 * i + 2])
        top_axes.append(ax)
        times = sorted({round(b["t"], 6) for b in run["b_profiles"]})
        pick = [times[k] for k in np.unique(np.linspace(0, len(times) - 1,
                                                        min(PROFILE_SNAPSHOTS, len(times))).round().astype(int))]
        for t in pick:
            bins = sorted((b for b in run["b_profiles"] if round(b["t"], 6) == t
                           and b["count"] >= MIN_BIN_COUNT and np.isfinite(b["inv"])),
                          key=lambda b: b["b"])
            if not bins:
                continue
            ax.errorbar([b["b"] for b in bins], [b["inv"] for b in bins],
                        yerr=[b["err"] if np.isfinite(b["err"]) else 0.0 for b in bins],
                        fmt="o-", ms=4, lw=1.2, capsize=2, color=cmap(t / t_max))
        ax.set_xlabel(r"$b = |\mathbf{B}|/B_{\rm ref}$ (local)")
        ax.set_ylabel(r"$1/\kappa$")
        ax.set_title(run["label"], fontsize=11.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, t_max))
    fig.colorbar(sm, ax=top_axes, pad=0.02, fraction=0.03).set_label(r"$t\,\Omega_{ci}$")

    ax_s = fig.add_subplot(grid[1, :ncol])
    ax_h = fig.add_subplot(grid[1, ncol:])
    for run in profiled:
        col = colors[run["name"]]
        rows = slopes.get(run["name"], [])
        good = [r for r in rows if np.isfinite(r["slope"])]
        if good:
            ax_s.errorbar([r["omega_ci_t"] for r in good], [r["slope"] for r in good],
                          yerr=[r["slope_err"] for r in good], fmt="o", ms=4, capsize=2,
                          color=col, label=run["label"])
        hp = [r for r in rows if np.isfinite(r["hole_minus_peak"])]
        if hp:
            # In units of 1e-3, set here: an axis offset folded into the label
            # later would trigger a tight_layout that ignores this GridSpec.
            ax_h.errorbar([r["omega_ci_t"] for r in hp], [1e3 * r["hole_minus_peak"] for r in hp],
                          yerr=[1e3 * r["hole_minus_peak_err"] if np.isfinite(r["hole_minus_peak_err"])
                                else 0.0 for r in hp], fmt="s", ms=4, capsize=2, color=col,
                          label=run["label"])
    for ax, title, ylabel in (
            (ax_s, r"(a) local-field slope $S = d(1/\kappa)/d\ln b$" "\n"
                   r"adiabatic (Liouville) mapping of passing ions: $S = 0$", r"$S$"),
            (ax_h, r"(b) $|B|$ hole $-$ peak (15 % tails)",
             r"$10^3\,(1/\kappa_{\rm hole} - 1/\kappa_{\rm peak})$")):
        ax.axhline(0.0, color=ps.MUTED_CLR, lw=0.9, ls="--")
        ax.set_xlabel(r"$t\,\Omega_{ci}$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11.5)
    handles, labels = ax_s.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=len(handles),
                   frameon=False, fontsize=9.5)
    note = mixing_note(profiled, relax)
    fig.suptitle("Does the local field strength organise the index?", fontsize=13, y=0.975)
    if note:
        fig.text(0.5, 0.94, note, ha="center", va="top", fontsize=9.5, color=ps.MUTED_CLR)
    ps.save(fig, path)
    return path


def mixing_note(runs: list[dict], relax: dict) -> str:
    """Window crossing time at the thermal speed against the relaxation time of 1/kappa."""
    crossing, taus = float("nan"), []
    for run in runs:
        fit = relax.get(run["name"])
        if not fit or run["control"] or not fit["y"].size:
            continue
        if np.isfinite(run["window_di"]) and run["vth_par"] > 0:
            crossing = run["window_di"] / run["vth_par"]
            window = run["window_di"]
        y_end = fit["y"][-1]
        if y_end < 0:
            kappa0 = rf"$\kappa_0={run['kappa0']:g}$" if run["kappa0"] else run["label"]
            taus.append(rf"{kappa0}: $\approx{-(fit['t'][-1] - fit['t0']) / y_end:.0f}$")
    if not np.isfinite(crossing) or not taus:
        return ""
    return (rf"ion crossing of the {window:.1f} $d_i$ particle window at $v_{{th,\parallel}}$: "
            rf"$\approx{crossing:.1f}\,\Omega_{{ci}}^{{-1}}$;  e-folding time of $1/\kappa$ "
            rf"[$\Omega_{{ci}}^{{-1}}$] " + ", ".join(taus))


def _gauss_in_bins(edges: np.ndarray) -> np.ndarray:
    """Standard normal density averaged over each bin."""
    from scipy.stats import norm
    return np.diff(norm.cdf(edges)) / np.diff(edges)


def shape_metrics(run: dict) -> list[dict]:
    """Probability in the core, the shoulders and the tails of u, over the Gaussian one.

    A kappa tail: core > 1, shoulders < 1, tails > 1. A flattened (plateau-
    like) distribution: core < 1, shoulders > 1, tails < 1.
    """
    shape = run.get("shape")
    if shape is None:
        return []
    edges = shape["edges"]
    centres, width = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    gauss = _gauss_in_bins(edges)
    a = np.abs(centres)
    bands = {"core_ratio": a < 0.5, "shoulder_ratio": (a > 1.0) & (a < 2.0), "tail_ratio": a > 3.0}
    rows = []
    for k, t in enumerate(shape["t"]):
        d = shape["par_density"][k]
        row = {"run": run["name"], "omega_ci_t": float(t),
               "sigma_par_vA": float(shape["par_sigma_vA"][k]),
               "u_res": float(run["v_res"] / shape["par_sigma_vA"][k]) if np.isfinite(run["v_res"]) else float("nan")}
        for key, band in bands.items():
            row[key] = float(np.sum((d * width)[band]) / np.sum((gauss * width)[band]))
        rows.append(row)
    return rows


def _shape_times(run: dict, t: np.ndarray) -> list[float]:
    """t = 0, end of the linear phase, 30 Omega_ci^-1 into saturation, last snapshot."""
    t_end = run["t_lin_end"]
    wanted = ([t[0], t_end, t_end + 30.0, t[-1]] if np.isfinite(t_end)
              else list(np.quantile(t, [0.0, 0.33, 0.66, 1.0])))
    picks = []
    for w in wanted:
        k = int(np.argmin(np.abs(t - w)))
        if k not in picks:
            picks.append(k)
    return picks


def _flat_top(u: np.ndarray, p: float = 3.0) -> np.ndarray:
    """Unit-variance generalised normal of exponent p > 2: flatter than a Gaussian."""
    from scipy.special import gamma as gamma_fn
    a = np.sqrt(gamma_fn(1.0 / p) / gamma_fn(3.0 / p))
    return p / (2.0 * a * gamma_fn(1.0 / p)) * np.exp(-np.abs(u / a) ** p)


def plot_shape_evolution(runs: list[dict], colors: dict, path: Path) -> Path | None:
    """f(v_par) over the Gaussian of the same variance, in time and at four instants.

    u = (v_par - <v_par>)/sigma_par(t): heating and anisotropy relaxation are
    divided out, only the shape is left. A suprathermal tail is red at |u| > 3;
    a flattening by resonant diffusion is red shoulders near the resonant
    velocity with blue core and wings.
    """
    shaped = [r for r in runs if r.get("shape") is not None]
    shaped = ([r for r in shaped if not r["control"]] + [r for r in shaped if r["control"]])[:5]
    if not shaped:
        return None
    from plasma_physics import kappa_marginal_pdf
    n = len(shaped)
    fig = plt.figure(figsize=(3.9 * (n + 1), 8.4))
    fig.set_layout_engine("none")          # the GridSpec places every panel
    grid = GridSpec(2, n + 1, figure=fig, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.34,
                    top=0.84, bottom=0.2, left=0.06, right=0.98)
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad=ps.PANEL_BG)
    norm = plt.Normalize(-0.5, 0.5)
    tcolors = plt.get_cmap("viridis")
    mesh = None
    for i, run in enumerate(shaped):
        sh = run["shape"]
        edges, t = sh["edges"], sh["t"]
        centres = 0.5 * (edges[1:] + edges[:-1])
        gauss = _gauss_in_bins(edges)
        ratio = sh["par_density"] / gauss[None, :]
        counts = sh["par_counts"]
        # Map: change of shape since the first snapshot (same standardisation),
        # so a kappa run's own tail does not saturate the colours; the absolute
        # comparison with the Gaussian is the row below.
        change = sh["par_density"] / np.where(sh["par_density"][0] > 0, sh["par_density"][0], np.nan)[None, :]
        good = (counts >= 30) & (counts[0] >= 30)[None, :] & (change > 0)
        logr = np.log10(np.where(good, change, np.nan))
        mid = 0.5 * (t[1:] + t[:-1]) if t.size > 1 else np.array([])
        t_edges = (np.concatenate([[t[0] - (mid[0] - t[0])], mid, [t[-1] + (t[-1] - mid[-1])]])
                   if t.size > 1 else np.array([t[0] - 0.5, t[0] + 0.5]))
        ax = fig.add_subplot(grid[0, i])
        mesh = ax.pcolormesh(t_edges, edges, logr.T, cmap=cmap, norm=norm, shading="flat",
                             rasterized=True)
        if np.isfinite(run["v_res"]):
            u_res = run["v_res"] / sh["par_sigma_vA"]
            for sign in (1.0, -1.0):
                ax.plot(t, sign * u_res, "--", color="0.1", lw=1.1)
        if np.isfinite(run["t_lin_end"]):
            ax.axvline(run["t_lin_end"], color="0.1", lw=0.9, ls=(0, (5, 2, 1, 2)))
        ax.set_ylim(-5, 5)
        ax.set_xlabel(r"$t\,\Omega_{ci}$")
        if i == 0:
            ax.set_ylabel(r"$u = \delta v_\parallel/\sigma_\parallel(t)$")
        ax.set_title(run["label"], fontsize=11)

        bx = fig.add_subplot(grid[1, i])
        picks = _shape_times(run, t)
        for j, k in enumerate(picks):
            ok = counts[k] >= 30
            r = ratio[k][ok]
            err = r / np.sqrt(np.maximum(counts[k][ok], 1))
            col = tcolors(j / max(len(picks) - 1, 1))
            bx.plot(centres[ok], r, color=col, lw=1.4, label=rf"$t\Omega_{{ci}}={t[k]:.0f}$")
            bx.fill_between(centres[ok], r - err, r + err, color=col, alpha=0.25, lw=0)
        if run["kappa0"]:
            uu = np.linspace(-5, 5, 400)
            bx.plot(uu, kappa_marginal_pdf(uu, 1.0, run["kappa0"]) / _gauss_in_bins(np.linspace(-5, 5, 401)),
                    ":", color=ps.MUTED_CLR, lw=1.3, label=rf"loaded $\kappa_0={run['kappa0']:g}$")
        if np.isfinite(run["v_res"]) and len(picks) > 2:
            u_sat = run["v_res"] / sh["par_sigma_vA"][picks[2]]
            for sign in (1.0, -1.0):
                bx.axvline(sign * u_sat, color="0.1", lw=0.9, ls="--")
        bx.axhline(1.0, color=ps.MUTED_CLR, lw=0.8)
        bx.set_yscale("log")
        bx.set_ylim(0.2, 8.0)
        bx.set_xlim(-5, 5)
        bx.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        bx.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        bx.yaxis.set_minor_formatter(NullFormatter())
        bx.set_xlabel(r"$u$")
        if i == 0:
            bx.set_ylabel(r"$f(u)\,/\,$Gaussian")
        bx.legend(fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.3), frameon=False, ncol=2)

    cell = fig.add_subplot(grid[0, n])
    cell.axis("off")
    cax = cell.inset_axes([0.08, 0.12, 0.08, 0.76])
    fig.colorbar(mesh, cax=cax).set_label(r"$\log_{10}[f(u,t)/f(u,0)]$" "\nred: gained, blue: lost",
                                          fontsize=10)
    rx = fig.add_subplot(grid[1, n])
    uu = np.linspace(-5, 5, 400)
    g = _gauss_in_bins(np.linspace(-5, 5, 401))
    rx.plot(uu, kappa_marginal_pdf(uu, 1.0, 3.0) / g, color=ps.c("#D55E00"), lw=1.6,
            label=r"$\kappa=3$: suprathermal tail")
    rx.plot(uu, _flat_top(uu) / g, color=ps.c("#0072B2"), lw=1.6, label="flattened (plateau-like)")
    rx.axhline(1.0, color=ps.MUTED_CLR, lw=0.8, label="Gaussian")
    rx.set_yscale("log")
    rx.set_ylim(0.2, 8.0)
    rx.set_xlim(-5, 5)
    rx.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    rx.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    rx.yaxis.set_minor_formatter(NullFormatter())
    rx.set_xlabel(r"$u$")
    rx.set_title("reference shapes, same variance", fontsize=10.5)
    rx.legend(fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.3), frameon=False)
    fig.suptitle(r"Shape of the ion $f(v_\parallel)$: change since $t=0$ (top) and against a Gaussian "
                 "of the same variance (bottom)", fontsize=13, y=0.975)
    fig.text(0.5, 0.925, r"local-field frame, $u=(v_\parallel-\langle v_\parallel\rangle)/\sigma_\parallel(t)$; "
             r"dashed: $n=1$ cyclotron resonance $\pm|v_{\rm res}|/\sigma_\parallel$ of the measured mode; "
             "dash-dot: end of the linear phase; bins with fewer than 30 ions left blank",
             ha="center", fontsize=9, color=ps.MUTED_CLR)
    ps.save(fig, path)
    return path


# ── Tables ───────────────────────────────────────────────────────────────────

def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def timeseries_rows(run: dict) -> list[dict]:
    s = run["pop"]["all"]
    field = run["field"]
    rows = []
    for k, t in enumerate(s["t"]):
        w = float(np.interp(t, field["t"], field["W"])) if field else float("nan")
        rows.append({"run": run["name"], "control": run["control"], "omega_ci_t": float(t),
                     "inv_kappa": float(s["inv"][k]), "inv_kappa_err": float(s["err"][k]),
                     "kappa": float(1.0 / s["inv"][k]) if s["inv"][k] > 0 else float("inf"),
                     "A_local": float(s["A"][k]), "W": w,
                     "F": float(fluence_at(run, np.array([t]))[0]),
                     "tail_fraction_3sigma": (float(np.interp(t, run["tail"]["t"], run["tail"]["frac"]))
                                              if run.get("tail") else float("nan")),
                     "source": run["source"]})
    return rows


def _json_ready(value):
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items() if k not in ("t", "y", "y_err", "F")}
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


# ── Driver ───────────────────────────────────────────────────────────────────

def analyse(roots: list, controls: list, outdir: Path) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs = [r for r in (load_run(p) for p in roots) if r is not None]
    explicit = [r for r in (load_run(p) for p in controls) if r is not None]
    if not runs:
        raise SystemExit("[ERROR] no run with a kappa product (03_particles/vdf_*.csv)")
    paired = []
    for run in runs:
        if run["control"]:
            continue
        control = find_control(run, explicit)
        if control is not None and all(control["name"] != r["name"] for r in runs + paired):
            paired.append(control)
    everything = runs + paired
    colors = {}
    for i, run in enumerate(r for r in runs if not r["control"]):
        colors[run["name"]] = ps.c(SERIES[i % len(SERIES)])
    for run in everything:
        if run["name"] not in colors:
            twin = next((r for r in runs if not r["control"] and r["kappa0"] == run["kappa0"]), None)
            colors[run["name"]] = colors[twin["name"]] if twin else ps.c(SERIES[-1])

    relax = {r["name"]: relaxation(r) for r in everything}
    rates = {r["name"]: windowed_rates(r) for r in everything if relax.get(r["name"])}
    slopes = {r["name"]: local_field_slopes(r) for r in everything}
    joint = joint_fit(everything, relax)

    plot_field_evolution(everything, colors, outdir / "kappa_field_evolution.png")
    plot_relaxation(everything, colors, relax, rates, outdir / "kappa_relaxation.png", joint)
    plot_local_field([r for r in everything if not r["control"]] or everything, colors, slopes,
                     relax, outdir / "kappa_vs_local_field.png")
    plot_shape_evolution(everything, colors, outdir / "kappa_shape_evolution.png")
    _write(outdir / "kappa_shape_metrics.csv", [row for r in everything for row in shape_metrics(r)])

    _write(outdir / "kappa_dynamics_timeseries.csv", [row for r in everything for row in timeseries_rows(r)])
    _write(outdir / "kappa_relaxation_rates.csv", [row for rows in rates.values() for row in rows])
    _write(outdir / "kappa_local_field.csv", [row for rows in slopes.values() for row in rows])
    summary = {
        "model": "ln[(1/kappa)/(1/kappa_0)] = -nu_0 t - c F(t), F = int_0^t <|dB|^2>/B0^2 dt (t in 1/Omega_ci)",
        "index": "truncated whitened kurtosis kappa_eff (kappa_eff.py), local-field frame, prt window",
        "runs": {r["name"]: {"label": r["label"], "control": r["control"], "source": r["source"],
                             "kappa0": r["kappa0"],
                             "relaxation": _json_ready(relax[r["name"]]) if relax.get(r["name"]) else None}
                 for r in everything},
        "joint_fit_common_c": _json_ready(joint) if joint else None,
    }
    (outdir / "kappa_relaxation_fit.json").write_text(json.dumps(summary, indent=2))
    for r in everything:
        fit = relax.get(r["name"])
        if fit:
            c = f", c = {fit['c']:.4f} +- {fit['c_err']:.4f}" if np.isfinite(fit.get("c", np.nan)) else ""
            print(f"  {r['name']}: nu0 = {fit['nu0']:.3e} +- {fit['nu0_err']:.1e}{c} (R2 = {fit['r2']:.4f})")
    if joint:
        print(f"  common c over {len(joint['runs'])} runs: {joint['c']:.4f} +- {joint['c_err']:.4f} "
              f"(chi2/dof {joint['chi2_dof']:.2f}; separate {', '.join(f'{v:.2f}' for v in joint['chi2_dof_separate'])})")
    return {"runs": everything, "relaxation": relax, "rates": rates, "slopes": slopes, "joint": joint}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run analysis roots (Maxwellian first)")
    parser.add_argument("--control", action="append", default=[], type=Path,
                        help="analysis root of an isotropic control (matched by kappa); "
                             "siblings named *_isotropic are found automatically")
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    analyse(args.runs, args.control, args.outdir)
    print(f"kappa dynamics written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
