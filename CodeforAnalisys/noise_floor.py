#!/usr/bin/env python3
"""
noise_floor.py — magnetic noise floor and null test of growth fits from a stable control
========================================================================================
A PIC run has magnetic fluctuations without any instability: the particle
noise. In the moderate mirror series that noise is not a constant floor. The
electrons heat numerically, and in the isotropic controls (stable by
construction) the domain rms of dB grows at 0.009 Omega_ci, about half the
logarithmic rate of T_e. Two things follow, and this script measures both from
products the pipeline already wrote (no snapshots):

1. **Noise floor of the rms fluctuation.** <|dB|^2> of an anisotropic run is
   the wave plus that noise. The control gives the noise on the same time
   axis; as in ``energy_audit.scale_control`` it is multiplied by a factor
   fitted on the quiet interval (after the quiet-start build-up, before the
   fitted linear phase) and held afterwards:

       noise(t) = s <|dB|^2>_ctrl(t),     excess(t) = <|dB|^2>_run(t) - noise(t)

   for the vector fluctuation and for its components along and across B0. The
   noise fraction noise/run says from when the rms series shows the
   instability at all (it stays above one half until t Omega_ci of about 50 in
   the moderate mirror runs), and whether the compressive component ever does.
   A growth rate fitted to an rms series over a window where the noise
   dominates is the growth of the noise, whatever its R^2.

2. **Null test of the growth fits.** The control has no unstable mode, so
   every exponential fit its Fourier modes admit is a false positive. For each
   accepted fit of the run (amplitude gain G over a window of duration W) the
   largest fitted gain exp(slope W) that any tracked mode of the control
   reaches over any window of the same duration is the null level. A fit is
   above the null when G is at least NULL_MARGIN times that level, at the null
   when the control reproduces it, marginal in between. The margin covers what
   the control cannot match: its box is smaller, so its modes sit at other k,
   and it tracks a dozen modes rather than all of them.

Neither part assigns a mechanism or corrects the dynamics. They say which
fluctuation measurements a stable plasma with the same numerics would also
produce.

Usage:
    python noise_floor.py RUN_DIR [RUN_DIR ...] --outdir OUT \\
        [--control RUN_NAME=CONTROL_NAME ...]

The runs include the controls; pairing is that of energy_audit.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import energy_audit
from analysis_contract import align_time, atomic_json
from growth_fit import reference_growth_row
from psc_units import NOISE_SETTLING_TRANSITS

#: rms series of growth_rate_summary.csv -> column of field_fluctuation_table.csv.
COMPONENTS = {"total": "delta_B_vec_rms_over_B0",
              "parallel": "delta_B_parallel_rms_over_B0",
              "perp": "delta_B_perp_rms_over_B0"}
#: Above this noise fraction of <|dB|^2> the rms series is the noise.
NOISE_DOMINATED = 0.5
#: The excess is drawn where it exceeds this many times the scatter of the
#: subtraction in the quiet interval; below it the difference is the residual
#: of two nearly equal numbers, a spiky trace decades under the noise.
EXCESS_SIGMA = 3.0
#: A fit is above the null when its gain is this many times the control's largest.
NULL_MARGIN = 2.0
PHYS = "09_physical_diagnostics"
ACCEPTED = ("1", "True", "true")


def settling_time(audit: dict) -> float:
    """Omega_ci t of the quiet-start noise build-up of the box fundamental.

    psc_units.noise_settling_time for the run's own manifest instead of the
    active profile: a cross-run script has one profile per run.
    """
    setup = audit.get("setup") or {}
    try:
        if audit.get("driven_species", "ion") != "ion":
            return 0.0
        vth_over_va = np.sqrt(setup["beta_i_parallel"] * max(1.0, setup["A_i"]) / 2.0)
        return float(NOISE_SETTLING_TRANSITS * setup["domain_di"] / (2.0 * np.pi * vth_over_va))
    except (KeyError, TypeError, ValueError):
        return 0.0


def _table(path: Path) -> dict | None:
    """Numeric columns of a CSV as arrays; None when the file is missing or empty."""
    rows = energy_audit._read_csv(path)
    if not rows:
        return None
    out = {}
    for key in rows[0]:
        column = energy_audit._col(rows, key)
        if np.any(np.isfinite(column)):
            out[key] = column
    return out if "omega_ci_t" in out else None


def subtract_noise(t, rms, control_t, control_rms, settle: float, quiet_end: float | None) -> dict | None:
    """Noise and excess of one rms series, from the control's on the run's times."""
    t, rms = np.asarray(t, dtype=float), np.asarray(rms, dtype=float)
    try:
        control, _ = align_time(control_t, np.asarray(control_rms, dtype=float) ** 2, t)
    except ValueError:
        return None
    power = rms ** 2
    ok = np.isfinite(control) & np.isfinite(power) & (power > 0) & (t > 0) & (t >= settle)
    if ok.sum() < 2:
        return None
    quiet = np.flatnonzero(ok & (t <= quiet_end)) if quiet_end is not None else np.zeros(0, dtype=int)
    calibrated = quiet.size >= energy_audit.MIN_CALIBRATION_SAMPLES
    scale = energy_audit._scale(power, control, quiet) if calibrated else 1.0
    noise = scale * control
    tt, fraction, excess = t[ok], noise[ok] / power[ok], power[ok] - noise[ok]
    below = fraction < NOISE_DOMINATED
    above = np.flatnonzero(~below)
    if not below[-1]:
        dominates_from = None
    else:
        dominates_from = float(tt[above[-1] + 1]) if above.size else float(tt[0])
    return {
        "scale": float(scale), "calibrated": bool(calibrated),
        # How well the scaled control reproduces the run where there is no wave.
        "quiet_scatter": (float(np.std((power[quiet] - noise[quiet]) / noise[quiet])) if calibrated else None),
        "t_end_omegaci": float(tt[-1]),
        "noise_fraction_end": float(fraction[-1]),
        "min_noise_fraction": float(np.min(fraction)),
        "signal_dominates_from_omegaci": dominates_from,
        "excess_over_B0sq_end": float(excess[-1]),
        "_series": {"omega_ci_t": tt, "power": power[ok], "noise": noise[ok],
                    "excess": excess, "noise_fraction": fraction},
    }


def null_gain(t, amplitudes: list, duration: float) -> dict | None:
    """Largest fitted gain exp(slope * duration) of noise modes over a window of that duration."""
    t = np.asarray(t, dtype=float)
    if t.size < 3 or not np.isfinite(duration) or duration <= 0:
        return None
    dt = float(np.median(np.diff(t)))
    n = int(round(duration / dt)) + 1
    if n < 3 or n > t.size:
        return None
    x = (np.arange(n) - 0.5 * (n - 1)) * dt
    sxx = float(np.sum(x * x))
    best, windows = -np.inf, 0
    for amplitude in amplitudes:
        a = np.asarray(amplitude, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(np.isfinite(a) & (a > 0), np.log(a), np.nan)
        view = np.lib.stride_tricks.sliding_window_view(y, n)
        valid = np.isfinite(view).all(axis=1)
        if valid.any():
            best = max(best, float(np.max(view[valid] @ x) / sxx))
            windows += int(valid.sum())
    if not windows:
        return None
    return {"gain": float(np.exp(best * duration)), "windows": windows}


def _verdict(gain: float, null: dict | None) -> tuple[str, float | None]:
    if null is None or not np.isfinite(gain):
        return "untested", None
    ratio = gain / null["gain"]
    return ("above_null" if ratio >= NULL_MARGIN else "at_null" if ratio <= 1.0 else "marginal"), float(ratio)


def null_test(fits: list[dict], control_modes: dict, settle: float) -> list[dict]:
    """Every accepted fit against the largest gain the stable control reaches in as long a window."""
    use = control_modes["omega_ci_t"] >= settle
    t = control_modes["omega_ci_t"][use]
    amplitudes = [column[use] for key, column in control_modes.items() if key.startswith("amp_over_B0")]
    out, cache = [], {}
    for fit in fits:
        try:
            start, end = float(fit["linear_phase_start"]), float(fit["linear_phase_end"])
            gain, gamma = float(fit["amplitude_gain"]), float(fit["gamma"])
        except (KeyError, TypeError, ValueError):
            continue
        duration = end - start
        key = round(duration, 6)
        if key not in cache:
            cache[key] = null_gain(t, amplitudes, duration)
        verdict, ratio = _verdict(gain, cache[key])
        out.append({"gamma": gamma, "linear_phase_start": start, "linear_phase_end": end,
                    "duration": duration, "amplitude_gain": gain,
                    "null_gain": cache[key]["gain"] if cache[key] else None,
                    "null_windows": cache[key]["windows"] if cache[key] else 0,
                    "gain_over_null": ratio, "verdict": verdict,
                    **{k: fit[k] for k in ("series", "k_parallel_di", "k_perp_di", "physical_branch",
                                           "classification") if k in fit}})
    return out


def audit_pair(run: dict, control: dict) -> dict | None:
    """Noise floor and null test of one run against its control (energy_audit records)."""
    root, control_root = Path(run["root"]) / PHYS, Path(control["root"]) / PHYS
    fluct, control_fluct = _table(root / "field_fluctuation_table.csv"), _table(control_root / "field_fluctuation_table.csv")
    control_modes = _table(control_root / "mode_amplitude_timeseries.csv")
    if fluct is None or control_fluct is None:
        return None
    settle = max(settling_time(run), settling_time(control))
    lp = run.get("linear_phase") or {}
    out = {"run": run["run"], "control": control["run"], "kappa": run.get("kappa"),
           "settle_omegaci": settle, "quiet_end_omegaci": lp.get("start"), "components": {}}
    for name, column in COMPONENTS.items():
        if column in fluct and column in control_fluct:
            part = subtract_noise(fluct["omega_ci_t"], fluct[column], control_fluct["omega_ci_t"],
                                  control_fluct[column], settle, lp.get("start"))
            if part is not None:
                out["components"][name] = part
    if not out["components"]:
        return None

    # rms-series fits: the growth of the noise wherever the noise dominates the window.
    summary = energy_audit._read_csv(root / "growth_rate_summary.csv")
    out["rms_fits"] = []
    for row in summary:
        series = row.get("series")
        part = out["components"].get(series)
        if part is None or str(row.get("fit_ok")) not in ACCEPTED:
            continue
        s = part["_series"]
        start, end = float(row["linear_phase_start"]), float(row["linear_phase_end"])
        inside = (s["omega_ci_t"] >= start) & (s["omega_ci_t"] <= end)
        if not inside.any():
            continue
        mean = float(np.mean(s["noise_fraction"][inside]))
        out["rms_fits"].append({"series": series, "gamma": float(row["gamma"]),
                                "linear_phase_start": start, "linear_phase_end": end,
                                "mean_noise_fraction": mean, "noise_dominated": bool(mean > NOISE_DOMINATED)})
    total = out["components"].get("total") or next(iter(out["components"].values()))
    never = sorted(name for name, part in out["components"].items()
                   if part["signal_dominates_from_omegaci"] is None)
    noisy = [f for f in out["rms_fits"] if f["noise_dominated"]]
    onset = total["signal_dominates_from_omegaci"]
    reason = (f"Control {control['run']} (x{total['scale']:.3f}): the instability exceeds the noise in "
              f"<|dB|^2> from t Omega_ci = {onset:.3g}" if onset is not None else
              f"Control {control['run']} (x{total['scale']:.3f}): <|dB|^2> never exceeds twice the noise")
    if never:
        reason += "; " + ", ".join(never) + " rms never above the noise"
    if noisy:
        reason += "; accepted rms fits over a noise-dominated window: " + ", ".join(
            f"{f['series']} (gamma {f['gamma']:.3g}, noise {100 * f['mean_noise_fraction']:.0f}%)" for f in noisy)
    out.update(noise_status="WARN" if never or noisy else "PASS", noise_reason=reason)

    # Null test: the reference fit decides the status, the mode table is listed.
    reference = reference_growth_row(summary)
    out["null"] = {"reference": None, "mode_fits": []}
    if control_modes is None:
        out.update(null_status="UNVERIFIED", null_reason="Control without mode_amplitude_timeseries.csv: no null test")
        return out
    if reference and str(reference.get("fit_ok")) in ACCEPTED:
        tested = null_test([reference], control_modes, settle)
        out["null"]["reference"] = tested[0] if tested else None
    accepted = [r for r in energy_audit._read_csv(root / "mode_growth_table.csv") if str(r.get("fit_ok")) in ACCEPTED]
    out["null"]["mode_fits"] = null_test(accepted, control_modes, settle)
    counts = {v: sum(1 for f in out["null"]["mode_fits"] if f["verdict"] == v)
              for v in ("above_null", "marginal", "at_null", "untested")}
    out["null"]["counts"] = counts
    ref = out["null"]["reference"]
    listed = (f"; of {len(out['null']['mode_fits'])} accepted mode fits {counts['above_null']} are above the null, "
              f"{counts['at_null'] + counts['marginal']} are reproduced by the control or marginal"
              if out["null"]["mode_fits"] else "")
    if ref is None:
        out.update(null_status="UNVERIFIED", null_reason="No accepted reference growth fit to test" + listed)
    elif ref["verdict"] == "untested":
        out.update(null_status="UNVERIFIED", null_reason=(
            f"The control is shorter than the fit window ({ref['duration']:.3g} Omega_ci^-1)" + listed))
    else:
        status = {"above_null": "PASS", "marginal": "UNVERIFIED", "at_null": "FAIL"}[ref["verdict"]]
        out.update(null_status=status, null_reason=(
            f"Reference fit: gain x{ref['amplitude_gain']:.3g} over {ref['duration']:.3g} Omega_ci^-1 against "
            f"x{ref['null_gain']:.3g} for the largest of {ref['null_windows']} windows of the stable control "
            f"({ref['gain_over_null']:.3g} times the null)" + listed))
    return out


def audit_paired(audits: list[dict]) -> list[dict]:
    """Noise-floor record of every run that energy_audit.pair_controls gave a control."""
    by_name = {a["run"]: a for a in audits}
    results = []
    for a in audits:
        name = (a.get("baseline") or {}).get("control")
        if name in by_name:
            result = audit_pair(a, by_name[name])
            if result is not None:
                results.append(result)
    return results


# ── Output ──────────────────────────────────────────────────────────────────

def plot(results: list[dict], path: Path) -> bool:
    drawn = [r for r in results if "total" in r["components"]]
    if not drawn:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import plot_style as ps
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
    labels = energy_audit._labels(drawn)
    for r in drawn:
        color = energy_audit._color(r.get("kappa"))
        s = r["components"]["total"]["_series"]
        t = s["omega_ci_t"]
        axes[0].plot(t, s["power"], "-", color=color, lw=1.4, alpha=0.55)
        axes[0].plot(t, s["noise"], "--", color=color, lw=1.4, alpha=0.55)
        scatter = r["components"]["total"]["quiet_scatter"] or 0.0
        resolved = np.where(s["excess"] > EXCESS_SIGMA * scatter * s["noise"], s["excess"], np.nan)
        axes[0].plot(t, resolved, "-", color=color, lw=2.4, label=labels[id(r)])
        axes[1].plot(t, s["noise_fraction"], "-", color=color, lw=2.0, label=labels[id(r)])
        if "parallel" in r["components"]:
            p = r["components"]["parallel"]["_series"]
            axes[1].plot(p["omega_ci_t"], p["noise_fraction"], ":", color=color, lw=2.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\langle|\delta\mathbf{B}|^2\rangle/B_0^2$")
    handles, _ = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color=ps.MUTED_CLR, lw=1.4, alpha=0.55, label="run"),
                Line2D([], [], color=ps.MUTED_CLR, lw=1.4, ls="--", alpha=0.55, label="noise (scaled control)"),
                Line2D([], [], color=ps.MUTED_CLR, lw=2.4,
                       label=rf"run minus noise (above {EXCESS_SIGMA:g}$\sigma$ of the subtraction)")]
    settle = max(r["settle_omegaci"] for r in drawn)
    if settle > 0:
        # As a legend entry: a free-standing note collides with the legend below the panel.
        handles.append(Line2D([], [], ls="none", label=rf"quiet-start build-up ($t\Omega_{{ci}} < {settle:.1f}$) not shown"))
    ps.legend(axes[0], handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=2, fontsize=9, frameon=False)
    axes[1].axhline(NOISE_DOMINATED, color=ps.MUTED_CLR, ls="--", lw=1.0)
    axes[1].set_ylim(0.0, 1.15)
    # Two lines: with the legend below the panel the one-line label is clipped.
    axes[1].set_ylabel("Noise fraction of the\nrms fluctuation energy")
    handles, _ = axes[1].get_legend_handles_labels()
    handles += [Line2D([], [], color=ps.MUTED_CLR, lw=2.0, label=r"$|\delta\mathbf{B}|^2$"),
                Line2D([], [], color=ps.MUTED_CLR, lw=2.0, ls=":", label=r"$\delta B_\parallel^2$ (compressive)"),
                Line2D([], [], color=ps.MUTED_CLR, lw=1.0, ls="--", label="noise = signal")]
    ps.legend(axes[1], handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=2, fontsize=9, frameon=False)
    for ax, title in zip(axes, ("(a) magnetic fluctuation energy", "(b) how much of it is noise")):
        ps.style_axes(ax, title)
        ax.set_xlabel(r"$t\Omega_{ci}$")
    ps.save(fig, path)
    return True


def write_outputs(results: list[dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    atomic_json(outdir / "noise_floor.json", {
        "runs": energy_audit.strip_private(results),
        "criteria": {"noise_dominated_fraction": NOISE_DOMINATED, "null_margin": NULL_MARGIN,
                     "calibration": "factor fitted between the quiet-start build-up and the fitted linear phase"},
        "scope": "which fluctuation measurements a stable plasma with the same numerics also produces; "
                 "no mechanism assigned, dynamics not corrected"})
    with (outdir / "noise_floor.csv").open("w", newline="") as handle:
        fields = ["run", "control", "noise_status", "null_status", "component", "scale", "quiet_scatter",
                  "signal_dominates_from_omegaci", "min_noise_fraction", "noise_fraction_end", "t_end_omegaci",
                  "reference_gain", "reference_null_gain", "reference_gain_over_null",
                  "noise_reason", "null_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            ref = (r.get("null") or {}).get("reference") or {}
            for name, part in r["components"].items():
                writer.writerow({**r, **part, "component": name,
                                 "reference_gain": ref.get("amplitude_gain"),
                                 "reference_null_gain": ref.get("null_gain"),
                                 "reference_gain_over_null": ref.get("gain_over_null")})
    with (outdir / "noise_floor_fits.csv").open("w", newline="") as handle:
        fields = ["run", "kind", "series", "k_parallel_di", "k_perp_di", "physical_branch", "gamma",
                  "linear_phase_start", "linear_phase_end", "amplitude_gain", "null_gain", "gain_over_null",
                  "verdict", "mean_noise_fraction"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            for fit in r.get("rms_fits", []):
                writer.writerow({**fit, "run": r["run"], "kind": "rms series",
                                 "verdict": "noise_dominated" if fit["noise_dominated"] else "signal_dominated"})
            for fit in (r.get("null") or {}).get("mode_fits", []):
                writer.writerow({**fit, "run": r["run"], "kind": "Fourier mode"})
    with (outdir / "noise_floor_series.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "component", "omega_ci_t", "dB2_over_B0sq", "noise_over_B0sq",
                         "excess_over_B0sq", "noise_fraction"])
        for r in results:
            for name, part in r["components"].items():
                s = part["_series"]
                for row in zip(s["omega_ci_t"], s["power"], s["noise"], s["excess"], s["noise_fraction"]):
                    writer.writerow([r["run"], name, *(f"{v:.8g}" for v in row)])
    plot(results, outdir / "noise_floor.png")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="Result directories of single runs, controls included, "
                                           "optionally LABEL=DIR")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--control", action="append", default=[], metavar="RUN_NAME=CONTROL_NAME",
                   help="Pair a run with a control explicitly, as in energy_audit.py")
    a = p.parse_args()
    try:
        explicit = energy_audit.parse_controls(a.control)
    except ValueError as exc:
        p.error(str(exc))
    audits = []
    for spec in a.runs:
        label, sep, path = spec.partition("=")
        if not sep:
            label, path = None, spec
        if not Path(path).is_dir():
            p.error(f"Not a results directory: {path}")
        audits.append(energy_audit.audit_run(path, label))
    try:
        energy_audit.pair_controls(audits, explicit)
    except ValueError as exc:
        p.error(str(exc))
    results = audit_paired(audits)
    if not results:
        print("No run has both a control and the fluctuation tables; nothing written")
        return 1
    write_outputs(results, a.outdir)
    for r in results:
        print(f"{r['run']}:")
        print(f"  noise floor: {r['noise_status']} -- {r['noise_reason']}")
        print(f"  null test:   {r['null_status']} -- {r['null_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
