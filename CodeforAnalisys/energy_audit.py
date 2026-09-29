#!/usr/bin/env python3
"""
energy_audit.py — cross-run audit of the energy budget from existing products
=============================================================================
Reads only what the pipeline already wrote (no snapshots), so it runs on a
laptop against a delivered results tree:

  <run>/*_analysis_manifest.json                        parameters, dx, B0
  <run>/09_physical_diagnostics/global_energy_table.csv   DiagEnergies (if any)
  <run>/09_physical_diagnostics/energy_table.csv          prt-window moments
  <run>/09_physical_diagnostics/growth_rate_summary.csv   linear-phase window

Per run it answers four questions, each with its own status:

1. **Diagnostic mapping.** At t = 0 the DiagEnergies columns must reproduce
   the profile's energy partition, E_s/E_B = beta_s,par (1/2 + A_s): this
   checks species order, the 1/2 of the field energy and the volume, without
   knowing the volume. Tolerance: 1 %, or 2.5 theta_e if larger, i.e. at least
   twice the leading relativistic correction 5 theta_e / 4 of m<gamma-1>
   against 3T/2 (theta_e = T_e/m_e c^2); shot noise of ~10^8 particles is far
   below it.
2. **Closure.** A periodic box without sources conserves energy, so a change
   of the total is numerical. It is compared with the physical signal, the
   energy released by the species that drives the instability. The budget
   FAILS when the error reaches half of that signal: a diagnostic whose error
   is as large as what it has to resolve cannot support an energy-transfer
   statement. Smaller errors stay UNVERIFIED; a PASS needs convergence runs.
3. **Window proxy.** Without DiagEnergies the prt-window moments are the only
   evidence. The window is ~4 % of a statistically homogeneous domain, so its
   energy may fluctuate by transport but cannot grow systematically: the
   proxy FAILS when the electrons gain more than twice what the ions and the
   magnetic fluctuations release there, and that gain exceeds 1 % of the
   initial thermal energy.
4. **Resolution and early-time error.** dx/lambda_De(t) from the measured
   T_e(t), beta_e(t), and T_e/T_e0 at the end of the fitted linear phase, so an
   early-time result carries its own error statement.

Across runs that share every parameter except the velocity distribution, the
spread of the electron heating tests whether it is common-mode (numerical
setup) or depends on the distribution being compared.

**Isotropic controls.** A run with A_i = A_e = 1 and the resolution, particles
per cell and electrons of an anisotropic run (``psc_mirror_*_isotropic``, in a
smaller box) is stable, so everything its electrons gain is the numerical
heating of that setup; it is compared per unit volume. Each anisotropic run is
paired with such a control, preferably of the same distribution and ion thermal
energy, and the control's energy changes are subtracted on a common time axis
(no extrapolation). This assumes the numerical heating is additive, i.e. not
changed by the instability itself; the pairing records how closely the control
matches. The baseline-corrected closure PASSES when what remains of the total
energy change is at most a tenth of the energy the ions release, FAILS at half
of it, and is UNVERIFIED in between; the raw budget keeps its own status.

No mechanism is assigned here. dx/lambda_De >> 1 with first-order particle
shapes makes grid heating a candidate; only controlled reruns (dx, ppc, shape
order) can confirm it.

Usage:
    python energy_audit.py RUN_DIR [RUN_DIR ...] --outdir OUT \\
        [--control RUN_NAME=CONTROL_NAME ...]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis_contract import align_time, atomic_json
from growth_fit import reference_growth_row

#: Closure FAILS when |Delta E_total| >= this fraction of the driver's release.
CLOSURE_FAIL_FRACTION = 0.5
#: Baseline-corrected closure PASSES when the residual is at most this fraction.
CLOSURE_PASS_FRACTION = 0.1
#: Window proxy FAILS when the electron gain exceeds this multiple of the release.
WINDOW_GAIN_FACTOR = 2.0
#: ... and the gain exceeds this fraction of the initial thermal energy.
WINDOW_MIN_GAIN = 0.01
#: Runs whose electron heating (T_e/T_e0 - 1) differs by less than this are common-mode.
COMMON_MODE_SPREAD = 0.1
#: Below this T_e/T_e0 - 1 there is no heating to apportion between runs.
NEGLIGIBLE_HEATING = 0.05
#: Window and global electron heating factors agreeing within this are domain-wide.
WINDOW_GLOBAL_AGREEMENT = 0.1
#: Parameters that must match for two runs to differ only in the distribution.
SETUP_KEYS = ("mass_ratio", "B0", "beta_i_parallel", "A_i", "beta_e_parallel", "A_e",
              "domain_di", "grid", "nicell_from_profile", "dx_de")
#: Parameters a control must share with the run it corrects: resolution (dx,
#: hence dt), particles per cell and electrons. The box may differ: the heating
#: is local, and a smaller control box is compared per unit volume.
CONTROL_KEYS = ("mass_ratio", "B0", "beta_e_parallel", "A_e", "nicell_from_profile", "dx_de")
ENERGY_COLUMNS = ("E_E", "E_B", "E_e", "E_i", "E_total")


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _col(rows: list[dict], key: str) -> np.ndarray:
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key, "nan")))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out, dtype=float)


def _interp(t: np.ndarray, y: np.ndarray, at: float | None) -> float | None:
    ok = np.isfinite(t) & np.isfinite(y)
    if at is None or not np.isfinite(at) or ok.sum() < 2 or not (t[ok][0] <= at <= t[ok][-1]):
        return None
    return float(np.interp(at, t[ok], y[ok]))


def _manifest(root: Path) -> dict:
    found = sorted(root.glob("*_analysis_manifest.json"))
    return json.loads(found[0].read_text()) if len(found) == 1 else {}


def _linear_phase(phys: Path) -> dict:
    row = reference_growth_row(_read_csv(phys / "growth_rate_summary.csv"))
    if not row or str(row.get("fit_ok")) not in ("1", "True", "true"):
        return {}
    return {"series": row.get("series") or "total",
            "start": float(row["linear_phase_start"]), "end": float(row["linear_phase_end"])}


def expected_partition(p: dict) -> dict:
    """E_s/E_B at t = 0 for n = 1: n T_par/2 + n T_perp over B0^2/2."""
    return {"E_i_over_E_B": p["beta_i_parallel"] * (0.5 + p["A_i"]),
            "E_e_over_E_B": p["beta_e_parallel"] * (0.5 + p["A_e"])}


def is_isotropic_control(audit: dict) -> bool:
    setup = audit.get("setup") or {}
    return setup.get("A_i") == 1.0 and setup.get("A_e") == 1.0


def _closure_status(error_over_release: float) -> str:
    if error_over_release >= CLOSURE_FAIL_FRACTION:
        return "FAIL"
    return "PASS" if error_over_release <= CLOSURE_PASS_FRACTION else "UNVERIFIED"


def audit_global(rows: list[dict], p: dict, driven: str) -> dict:
    t = _col(rows, "omega_ci_t")
    e = {k: _col(rows, k) for k in ENERGY_COLUMNS}
    if t.size < 2 or not np.all(np.isfinite([e[k][0] for k in e])) or e["E_B"][0] <= 0:
        return {"status": "UNVERIFIED", "reason": "DiagEnergies table unreadable"}
    theta_e = p["beta_e_parallel"] * p["B0"] ** 2 / 2.0
    tol = max(0.01, 2.5 * theta_e)
    want = expected_partition(p)
    got = {"E_i_over_E_B": e["E_i"][0] / e["E_B"][0], "E_e_over_E_B": e["E_e"][0] / e["E_B"][0]}
    dev = {k: got[k] / want[k] - 1.0 for k in want}
    mapping_ok = all(abs(v) <= tol for v in dev.values())
    d = {k: v - v[0] for k, v in e.items()}
    key = "E_i" if driven == "ion" else "E_e"
    release = -float(d[key][-1])
    error = float(d["E_total"][-1])
    error_over_release = abs(error) / release if release > 0 else float("inf")
    other = "E_e" if driven == "ion" else "E_i"
    fields_release = -float(d["E_B"][-1] + d["E_E"][-1])
    result = {
        "mapping": {"status": "PASS" if mapping_ok else "FAIL",
                    "measured": got, "expected": want, "relative_deviation": dev, "tolerance": tol,
                    "checks": "species order, 1/2 field-energy factor, volume (ratios are volume-free)"},
        "t_end_omegaci": float(t[-1]),
        "total_relative_change_end": error / float(e["E_total"][0]),
        "driver_release": release,
        "fields_release": fields_release,
        "non_driver_gain": float(d[other][-1]),
        "error_over_driver_release": error_over_release,
        "electron_heating_factor": float(e["E_e"][-1] / e["E_e"][0]),
        "_series": {"omega_ci_t": t, **e},
    }
    if not mapping_ok:
        result.update(status="FAIL", reason="DiagEnergies columns do not reproduce the initial energy partition")
    elif error_over_release >= CLOSURE_FAIL_FRACTION:
        result.update(status="FAIL", reason=(
            f"Total energy changed by {100 * result['total_relative_change_end']:+.1f}% "
            f"= {error_over_release:.2g}x the energy released by the {driven}s; "
            "the budget is dominated by non-conservation"))
    else:
        result.update(status="UNVERIFIED", reason=(
            f"Non-conservation {error_over_release:.2g}x the {driven} release; "
            "acceptance needs convergence evidence"))
    return result


def audit_window(rows: list[dict], p: dict) -> dict:
    t = _col(rows, "omega_ci_t")
    ee, ei = _col(rows, "E_internal_e"), _col(rows, "E_internal_i")
    bulk, eb, ae = _col(rows, "E_kin_bulk"), _col(rows, "E_B"), _col(rows, "A_e")
    ok = np.isfinite(t) & np.isfinite(ee) & np.isfinite(ei)
    if ok.sum() < 2 or ee[ok][0] <= 0:
        return {"status": "UNVERIFIED", "reason": "prt-window energy table missing or unreadable"}
    t, ee, ei, bulk, eb, ae = (x[ok] for x in (t, ee, ei, bulk, eb, ae))
    te_ratio = ee / ee[0]
    t_e = ee / 1.5                                   # n = 1: E_int = 3T/2
    dx_over_lde = p["dx_de"] / np.sqrt(t_e)          # lambda_De = sqrt(T_e / n e^2)
    gain = float(ee[-1] - ee[0])
    release = float(-(ei[-1] - ei[0]) - np.nan_to_num(bulk[-1] - bulk[0]) - np.nan_to_num(eb[-1] - eb[0]))
    thermal0 = float(ee[0] + ei[0])
    fails = gain > WINDOW_MIN_GAIN * thermal0 and gain > WINDOW_GAIN_FACTOR * max(release, 0.0)
    ratio = gain / release if release > 0 else float("inf")
    # Shape of the heating: e-folding rate of T_e and whether dT_e/dt grows.
    # A finite-grid instability slows down as lambda_De approaches dx; heating
    # that accelerates while dx/lambda_De falls is not that textbook signature.
    log_rate = float(np.polyfit(t, np.log(te_ratio), 1)[0]) if np.ptp(t) > 0 else float("nan")
    mid = _interp(t, te_ratio, 0.5 * (t[0] + t[-1]))
    acceleration = ((te_ratio[-1] - mid) / (mid - te_ratio[0])
                    if mid is not None and mid != te_ratio[0] else float("nan"))
    return {
        "status": "FAIL" if fails else "UNVERIFIED",
        "reason": (f"prt window: electrons gained {ratio:.2g}x what ions and magnetic fluctuations released "
                   f"(T_e x{te_ratio[-1]:.2f}); not explained by transport in a homogeneous box"
                   if fails else "Window budget within transport fluctuations; not a conservation test"),
        "electron_heating_factor": float(te_ratio[-1]),
        "electron_gain_over_release": ratio,
        "T_e0_measured": float(t_e[0]),
        "T_e0_profile": p["beta_e_parallel"] * p["B0"] ** 2 / 2.0,
        "log_heating_rate_per_omegaci": log_rate,
        "second_half_over_first_half_heating": acceleration,
        "A_e_final": float(ae[-1]) if np.isfinite(ae[-1]) else None,
        "A_e_mean": float(np.nanmean(ae)) if np.any(np.isfinite(ae)) else None,
        "dx_over_lambda_De_initial": float(dx_over_lde[0]),
        "dx_over_lambda_De_final": float(dx_over_lde[-1]),
        "beta_e_final": float(p["beta_e_parallel"] * te_ratio[-1]),
        "_series": {"omega_ci_t": t, "T_e_over_T_e0": te_ratio, "dx_over_lambda_De": dx_over_lde,
                    "beta_e": p["beta_e_parallel"] * te_ratio, "E_internal_e": ee, "E_internal_i": ei},
    }


def audit_run(root, name: str | None = None) -> dict:
    root = Path(root)
    phys = root / "09_physical_diagnostics"
    manifest = _manifest(root)
    p = manifest.get("physics", {})
    base = {"run": name or root.name, "root": str(root.resolve()), "case": manifest.get("case"),
            "kappa": p.get("kappa")}
    need = ("beta_i_parallel", "A_i", "beta_e_parallel", "A_e", "B0", "dx_de")
    if not all(isinstance(p.get(k), (int, float)) for k in need):
        return {**base, "status": "UNVERIFIED", "reason": "Manifest without the physical parameters of the run"}
    driven = manifest.get("driven_species", "ion")
    glob_rows = _read_csv(phys / "global_energy_table.csv")
    result = {**base, "driven_species": driven,
              "ion_thermal_energy": p["beta_i_parallel"] * (0.5 + p["A_i"]),
              "global": audit_global(glob_rows, p, driven) if glob_rows else None,
              "window": audit_window(_read_csv(phys / "energy_table.csv"), p),
              "linear_phase": _linear_phase(phys),
              "setup": {k: p.get(k) for k in SETUP_KEYS}}
    result["role"] = "isotropic_control" if is_isotropic_control(result) else "run"
    w, g, lp = result["window"], result["global"], result["linear_phase"]
    series = w.get("_series")
    if series is not None and lp:
        t = series["omega_ci_t"]
        result["early_time"] = {
            "growth_series": lp["series"], "linear_phase_omegaci": [lp["start"], lp["end"]],
            "T_e_over_T_e0_at_start": _interp(t, series["T_e_over_T_e0"], lp["start"]),
            "T_e_over_T_e0_at_end": _interp(t, series["T_e_over_T_e0"], lp["end"]),
            "beta_e_at_end": _interp(t, series["beta_e"], lp["end"]),
            "dx_over_lambda_De_at_end": _interp(t, series["dx_over_lambda_De"], lp["end"])}
    if g and "electron_heating_factor" in g and "electron_heating_factor" in w:
        agree = abs(w["electron_heating_factor"] / g["electron_heating_factor"] - 1.0)
        result["window_vs_global"] = {"relative_difference": agree,
                                      "domain_wide": bool(agree <= WINDOW_GLOBAL_AGREEMENT)}
    evidence = g if g else w
    result["status"] = "FAIL" if (g and g["status"] == "FAIL") or w["status"] == "FAIL" else evidence["status"]
    result["reason"] = (g["reason"] if g and g["status"] == "FAIL" else w["reason"]) if result["status"] == "FAIL" \
        else ("No global DiagEnergies; " if not g else "") + evidence["reason"]
    if result["role"] == "isotropic_control":
        result["reason"] = "Isotropic control, its heating is the numerical baseline: " + result["reason"]
    return result


def group_runs(audits: list[dict]) -> list[dict]:
    """Runs sharing every setup parameter but the distribution: is the heating common-mode?"""
    groups: dict[str, list[dict]] = {}
    for a in audits:
        factor = (a.get("window") or {}).get("electron_heating_factor")
        if a.get("setup") and factor is not None:
            groups.setdefault(json.dumps(a["setup"], sort_keys=True), []).append(a)
    out = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        f = np.array([m["window"]["electron_heating_factor"] for m in members])
        heating = f - 1.0
        base = {"runs": [m["run"] for m in members], "kappa": [m["kappa"] for m in members],
                "setup": json.loads(key), "electron_heating_factors": f.tolist()}
        if np.max(np.abs(heating)) < NEGLIGIBLE_HEATING:
            out.append({**base, "relative_spread": None, "common_mode": None,
                        "interpretation": f"No significant electron heating (T_e/T_e0 at most x{f.max():.2f})"})
            continue
        spread = float(np.ptp(heating) / np.mean(np.abs(heating)))
        common = spread <= COMMON_MODE_SPREAD
        out.append({**base, "relative_spread": spread, "common_mode": bool(common),
                    "interpretation": (
                        f"Electron heating x{f.min():.2f}-x{f.max():.2f} differs by {100 * spread:.1f}% between "
                        "distributions: it belongs to the shared numerical setup, not to the distribution "
                        "being compared" if common else
                        f"Electron heating differs by {100 * spread:.1f}% between distributions")})
    return out


# ── Isotropic controls ──────────────────────────────────────────────────────

def match_control(run: dict, audits: list[dict], explicit: str | None = None) -> tuple[dict | None, dict]:
    """The control whose numerics and electrons are those of `run`, best match first."""
    if explicit is not None:
        found = [a for a in audits if a["run"] == explicit]
        if not found:
            raise ValueError(f"Control {explicit!r} is not among the audited runs")
        candidates = found
    else:
        candidates = [a for a in audits if a is not run and a.get("role") == "isotropic_control"]
    setup = run.get("setup") or {}
    usable = [a for a in candidates
              if all((a.get("setup") or {}).get(k) == setup.get(k) for k in CONTROL_KEYS)]
    if not usable:
        return None, {}

    def closeness(a):
        return (a.get("kappa") == run.get("kappa"),
                -abs(a["ion_thermal_energy"] / run["ion_thermal_energy"] - 1.0))

    best = max(usable, key=closeness)
    return best, {"same_distribution": best.get("kappa") == run.get("kappa"),
                  "ion_thermal_energy_ratio": best["ion_thermal_energy"] / run["ion_thermal_energy"],
                  "shared": list(CONTROL_KEYS), "explicit": explicit is not None}


def subtract_control(run: dict, control: dict, match: dict) -> dict:
    """Energy changes of `run` minus those of its isotropic control, on the run's times."""
    out = {"control": control["run"], "match": match}
    ws, wc = (run.get("window") or {}).get("_series"), (control.get("window") or {}).get("_series")
    if ws is not None and wc is not None:
        t = ws["omega_ci_t"]
        ctrl, cover = align_time(wc["omega_ci_t"], wc["E_internal_e"], t)
        ok = np.isfinite(ctrl)
        if ok.sum() >= 2:
            ee0 = ws["E_internal_e"][0]
            excess = (ws["E_internal_e"] - ee0) - (ctrl - ctrl[ok][0])
            last = np.flatnonzero(ok)[-1]
            out["window"] = {
                "coverage": cover, "t_end_omegaci": float(t[last]),
                "T_e_factor_run": float(ws["T_e_over_T_e0"][last]),
                "T_e_factor_control": float(ctrl[last] / ctrl[ok][0]),
                "T_e_factor_baseline_corrected": float(1.0 + excess[last] / ee0),
                "_series": {"omega_ci_t": t[ok], "run": ws["T_e_over_T_e0"][ok],
                            "control": ctrl[ok] / ctrl[ok][0], "corrected": 1.0 + excess[ok] / ee0}}
            lp = run.get("linear_phase") or {}
            if lp:
                out["window"]["T_e_factor_baseline_corrected_at_linear_end"] = _interp(
                    t[ok], 1.0 + excess[ok] / ee0, lp["end"])
    gs, gc = (run.get("global") or {}).get("_series"), (control.get("global") or {}).get("_series")
    if gs is None or gc is None:
        out.update(status="UNVERIFIED", reason=(
            f"Control {control['run']}: no DiagEnergies in both runs, so no baseline-corrected closure"))
        return out
    t = gs["omega_ci_t"]
    # Per unit volume: B0 is uniform, so E_B(0) is proportional to the volume
    # and rescales the control's changes to the run's box.
    volume_ratio = float(gs["E_B"][0] / gc["E_B"][0])
    out["volume_ratio_run_over_control"] = volume_ratio
    d = {}
    for k in ENERGY_COLUMNS:
        ctrl, cover = align_time(gc["omega_ci_t"], gc[k], t)
        d[k] = ((gs[k] - gs[k][0]) - volume_ratio * (ctrl - ctrl[0]) if np.isfinite(ctrl[0])
                else np.full_like(t, np.nan))
    ok = np.isfinite(d["E_total"])
    if ok.sum() < 2:
        out.update(status="UNVERIFIED", reason=f"Control {control['run']} does not overlap the run in time")
        return out
    last = np.flatnonzero(ok)[-1]
    release = -float(d["E_i"][last]) if run.get("driven_species", "ion") == "ion" else -float(d["E_e"][last])
    error = float(d["E_total"][last])
    e0 = float(gs["E_total"][0])
    if release <= 1e-6 * abs(e0):
        # The control lost as much driver energy as the run: after the
        # subtraction there is no instability signal left to close.
        out.update(status="UNVERIFIED", reason=(
            f"Baseline-corrected with {control['run']}: no net ion energy release left "
            "after subtracting the control, nothing to close"))
        return out
    ratio = abs(error) / release
    status = _closure_status(ratio)
    out["global"] = {
        "coverage": cover, "t_end_omegaci": float(t[last]),
        "driver_release_corrected": release, "residual_total_change": error,
        "residual_over_driver_release": ratio,
        "electron_excess_gain": float(d["E_e"][last]),
        "raw_total_relative_change": float((gs["E_total"][last] - gs["E_total"][0]) / e0),
        "corrected_total_relative_change": error / e0,
        "_series": {"omega_ci_t": t[ok], **{k: d[k][ok] / e0 for k in ENERGY_COLUMNS},
                    "raw_total": (gs["E_total"][ok] - gs["E_total"][0]) / e0}}
    caveat = "" if match["same_distribution"] else "; control of another distribution"
    out.update(status=status, reason=(
        f"Baseline-corrected with {control['run']}: residual total change {ratio:.2g}x the ion release "
        f"(raw {100 * out['global']['raw_total_relative_change']:+.1f}%, corrected "
        f"{100 * out['global']['corrected_total_relative_change']:+.2f}% of E_total){caveat}"))
    return out


def pair_controls(audits: list[dict], explicit: dict[str, str] | None = None) -> None:
    """Attach `baseline` (control subtraction) to every anisotropic run that has a control."""
    explicit = explicit or {}
    unknown = set(explicit) - {a["run"] for a in audits}
    if unknown:
        raise ValueError(f"--control names runs that were not audited: {sorted(unknown)}")
    for a in audits:
        if a.get("role") != "run" or not a.get("setup"):
            continue
        control, match = match_control(a, audits, explicit.get(a["run"]))
        if control is not None:
            a["baseline"] = subtract_control(a, control, match)


# ── Output ──────────────────────────────────────────────────────────────────

def _color(kappa):
    import plot_style as ps
    return ps.c({None: "#58a6ff", 5.0: "#ff7b72", 3.0: "#56d364"}.get(kappa, "#d2a8ff"))


def _label(a: dict) -> str:
    name = "bi-Maxwellian" if a.get("kappa") is None else f"bi-kappa {a['kappa']:g}"
    return name + (" (isotropic control)" if a.get("role") == "isotropic_control" else "")


def _labels(audits: list[dict]) -> dict[int, str]:
    """Legend label per audit; runs of one distribution are told apart by name."""
    base = {id(a): _label(a) for a in audits}
    counts = {}
    for text in base.values():
        counts[text] = counts.get(text, 0) + 1
    return {k: (f"{v} [{a['run']}]" if counts[v] > 1 else v)
            for a in audits for k, v in [(id(a), base[id(a)])]}


def plot(audits: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import plot_style as ps
    ps.apply()
    with_global = [a for a in audits if (a.get("global") or {}).get("_series") and a.get("role") == "run"]
    ncol = 3 if with_global else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.6), constrained_layout=True)
    names = _labels(audits)
    styles = ["-", "--", "-.", ":"]
    seen: dict[str, int] = {}
    for a in audits:
        s = (a.get("window") or {}).get("_series")
        if s is None:
            continue
        color = _color(a.get("kappa"))
        # Runs of one distribution (resolution variants) share the colour and
        # differ in line style; controls are dashed.
        n = seen.get(_label(a), 0); seen[_label(a)] = n + 1
        style = "--" if a.get("role") == "isotropic_control" else styles[n % len(styles)]
        axes[0].plot(s["omega_ci_t"], s["T_e_over_T_e0"], style, color=color, label=names[id(a)])
        axes[1].plot(s["omega_ci_t"], s["dx_over_lambda_De"], style, color=color, label=names[id(a)])
        end = (a.get("early_time") or {}).get("linear_phase_omegaci", [None, None])[1]
        value = (a.get("early_time") or {}).get("T_e_over_T_e0_at_end")
        if end is not None and value is not None and a.get("role") == "run":
            axes[0].plot([end], [value], "o", ms=6, mfc="none", color=color)
    axes[0].set_ylabel(r"$T_e/T_{e0}$ (prt window)")
    axes[1].axhline(1.0, color=ps.MUTED_CLR, ls="--", lw=1.0)
    axes[1].text(0.97, 1.0, r"$\lambda_{De}=\Delta x$", color=ps.MUTED_CLR, fontsize=11,
                 ha="right", va="bottom", transform=axes[1].get_yaxis_transform())
    axes[1].set_ylabel(r"$\Delta x/\lambda_{De}$")
    axes[1].set_ylim(bottom=0)
    handles, _ = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], ls="none", marker="o", mfc="none", color=ps.MUTED_CLR,
                          label="end of fitted linear phase"))
    ps.legend(axes[0], handles=handles, loc="best", fontsize=9)
    for ax, title in zip(axes, ("(a) electron heating", "(b) Debye-length resolution")):
        ps.style_axes(ax, title)
        ax.set_xlabel(r"$t\Omega_{ci}$")
    if with_global:
        ax = axes[2]
        a = with_global[0]
        s = a["global"]["_series"]
        e0 = s["E_total"][0]
        for key, color, name in (("E_e", "#d2a8ff", "electrons"), ("E_i", "#58a6ff", "ions"),
                                 ("E_B", "#56d364", "magnetic"), ("E_total", "#111111", "total")):
            ax.plot(s["omega_ci_t"], (s[key] - s[key][0]) / e0, color=ps.c(color), label=name,
                    lw=2.4 if key == "E_total" else 1.8)
        ax.axhline(0.0, color=ps.MUTED_CLR, lw=0.8)
        ax.set_ylabel(r"$\Delta E_s/E_{\rm total}(0)$ (DiagEnergies)")
        ax.set_xlabel(r"$t\Omega_{ci}$")
        ps.style_axes(ax, f"(c) global budget, {_label(a)}")
        ps.legend(ax, loc="upper left", fontsize=10)
    ps.save(fig, path)


def plot_baseline(audits: list[dict], path: Path) -> bool:
    """Run, control and baseline-corrected heating; corrected global budget."""
    pairs = [a for a in audits if (a.get("baseline") or {}).get("window")]
    if not pairs:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import plot_style as ps
    ps.apply()
    with_global = [a for a in pairs if (a["baseline"].get("global") or {}).get("_series")]
    ncol = 2 if with_global else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.8 * ncol, 4.6), constrained_layout=True, squeeze=False)
    axes = axes[0]
    for a in pairs:
        s = a["baseline"]["window"]["_series"]
        color = _color(a.get("kappa"))
        axes[0].plot(s["omega_ci_t"], s["run"], "-", color=color, lw=1.4, alpha=0.55)
        axes[0].plot(s["omega_ci_t"], s["control"], "--", color=color, lw=1.4, alpha=0.55)
        axes[0].plot(s["omega_ci_t"], s["corrected"], "-", color=color, lw=2.4, label=_label(a))
    axes[0].axhline(1.0, color=ps.MUTED_CLR, lw=0.8)
    handles, _ = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color=ps.MUTED_CLR, lw=1.4, alpha=0.55, label="run (raw)"),
                Line2D([], [], color=ps.MUTED_CLR, lw=1.4, ls="--", alpha=0.55, label="isotropic control"),
                Line2D([], [], color=ps.MUTED_CLR, lw=2.4, label="run minus control")]
    ps.legend(axes[0], handles=handles, loc="best", fontsize=9)
    axes[0].set_ylabel(r"$T_e/T_{e0}$ (prt window)")
    axes[0].set_xlabel(r"$t\Omega_{ci}$")
    ps.style_axes(axes[0], "(a) electron heating, baseline removed")
    if with_global:
        ax = axes[1]
        for a in with_global:
            s = a["baseline"]["global"]["_series"]
            color = _color(a.get("kappa"))
            ax.plot(s["omega_ci_t"], s["raw_total"], "--", color=color, lw=1.4, alpha=0.55)
            ax.plot(s["omega_ci_t"], s["E_total"], "-", color=color, lw=2.4, label=_label(a))
        ax.axhline(0.0, color=ps.MUTED_CLR, lw=0.8)
        handles, _ = ax.get_legend_handles_labels()
        handles += [Line2D([], [], color=ps.MUTED_CLR, lw=1.4, ls="--", alpha=0.55, label="raw"),
                    Line2D([], [], color=ps.MUTED_CLR, lw=2.4, label="baseline-corrected")]
        ps.legend(ax, handles=handles, loc="upper left", fontsize=9)
        ax.set_ylabel(r"$\Delta E_{\rm total}/E_{\rm total}(0)$ (DiagEnergies)")
        ax.set_xlabel(r"$t\Omega_{ci}$")
        ps.style_axes(ax, "(b) total energy change")
    ps.save(fig, path)
    return True


def strip_private(value):
    """Drop the private plotting series (keys starting with '_') recursively."""
    if isinstance(value, dict):
        return {k: strip_private(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [strip_private(v) for v in value]
    return value


def write_outputs(audits: list[dict], groups: list[dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    atomic_json(outdir / "energy_audit.json", {
        "runs": strip_private(audits), "groups": groups,
        "criteria": {"closure_fail_fraction_of_driver_release": CLOSURE_FAIL_FRACTION,
                     "baseline_closure_pass_fraction": CLOSURE_PASS_FRACTION,
                     "window_gain_factor": WINDOW_GAIN_FACTOR, "window_min_gain": WINDOW_MIN_GAIN,
                     "common_mode_spread": COMMON_MODE_SPREAD,
                     "window_global_agreement": WINDOW_GLOBAL_AGREEMENT,
                     "control_shares": list(CONTROL_KEYS)},
        "mechanism": "not assigned. dx/lambda_De >> 1 with first-order shapes makes grid heating a "
                     "candidate, but heating that accelerates while dx/lambda_De falls is not the "
                     "textbook finite-grid-instability signature; only controlled reruns (dx, ppc, "
                     "shape order, dt) can establish the mechanism"})
    fields = ["run", "role", "kappa", "status", "global_status", "total_relative_change_end",
              "error_over_driver_release", "mapping_status", "window_status", "T_e_factor_window",
              "T_e_factor_global", "electron_gain_over_release_window", "A_e_final",
              "dx_over_lambda_De_initial", "dx_over_lambda_De_final", "beta_e_final",
              "log_heating_rate_per_omegaci", "second_half_over_first_half_heating",
              "growth_series", "linear_phase_end_omegaci", "T_e_over_T_e0_at_linear_end",
              "control", "control_same_distribution", "T_e_factor_control",
              "T_e_factor_baseline_corrected", "T_e_factor_baseline_corrected_at_linear_end",
              "baseline_status", "baseline_residual_over_driver_release",
              "baseline_corrected_total_relative_change", "reason", "baseline_reason"]
    with (outdir / "energy_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for a in audits:
            g, w, early = a.get("global") or {}, a.get("window") or {}, a.get("early_time") or {}
            b = a.get("baseline") or {}
            bw, bg = b.get("window") or {}, b.get("global") or {}
            writer.writerow({
                "run": a["run"], "role": a.get("role"), "kappa": a.get("kappa"), "status": a["status"],
                "global_status": g.get("status"), "total_relative_change_end": g.get("total_relative_change_end"),
                "error_over_driver_release": g.get("error_over_driver_release"),
                "mapping_status": (g.get("mapping") or {}).get("status"), "window_status": w.get("status"),
                "T_e_factor_window": w.get("electron_heating_factor"),
                "T_e_factor_global": g.get("electron_heating_factor"),
                "electron_gain_over_release_window": w.get("electron_gain_over_release"),
                "A_e_final": w.get("A_e_final"), "dx_over_lambda_De_initial": w.get("dx_over_lambda_De_initial"),
                "dx_over_lambda_De_final": w.get("dx_over_lambda_De_final"), "beta_e_final": w.get("beta_e_final"),
                "log_heating_rate_per_omegaci": w.get("log_heating_rate_per_omegaci"),
                "second_half_over_first_half_heating": w.get("second_half_over_first_half_heating"),
                "growth_series": early.get("growth_series"),
                "linear_phase_end_omegaci": (early.get("linear_phase_omegaci") or [None, None])[1],
                "T_e_over_T_e0_at_linear_end": early.get("T_e_over_T_e0_at_end"),
                "control": b.get("control"),
                "control_same_distribution": (b.get("match") or {}).get("same_distribution"),
                "T_e_factor_control": bw.get("T_e_factor_control"),
                "T_e_factor_baseline_corrected": bw.get("T_e_factor_baseline_corrected"),
                "T_e_factor_baseline_corrected_at_linear_end": bw.get("T_e_factor_baseline_corrected_at_linear_end"),
                "baseline_status": b.get("status"),
                "baseline_residual_over_driver_release": bg.get("residual_over_driver_release"),
                "baseline_corrected_total_relative_change": bg.get("corrected_total_relative_change"),
                "reason": a.get("reason"), "baseline_reason": b.get("reason")})
    # Runs of different length (production vs short resolution variants) are
    # compared at the last time they all reach, never at their own ends.
    ends = [float(s["omega_ci_t"][-1]) for s in
            ((a.get("window") or {}).get("_series") for a in audits) if s is not None]
    if len(ends) > 1:
        common = min(ends)
        with (outdir / "energy_audit_common_time.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run", "role", "kappa", "nicell", "dx_de", "common_omega_ci_t",
                             "T_e_over_T_e0", "dx_over_lambda_De"])
            for a in audits:
                s = (a.get("window") or {}).get("_series")
                if s is None:
                    continue
                setup = a.get("setup") or {}
                writer.writerow([a["run"], a.get("role"), a.get("kappa"), setup.get("nicell_from_profile"),
                                 setup.get("dx_de"), f"{common:.6g}",
                                 f"{_interp(s['omega_ci_t'], s['T_e_over_T_e0'], common):.6g}",
                                 f"{_interp(s['omega_ci_t'], s['dx_over_lambda_De'], common):.6g}"])
    with (outdir / "energy_audit_electron_heating.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "role", "omega_ci_t", "T_e_over_T_e0", "dx_over_lambda_De", "beta_e",
                         "T_e_over_T_e0_baseline_corrected"])
        for a in audits:
            s = (a.get("window") or {}).get("_series")
            if s is None:
                continue
            corrected = ((a.get("baseline") or {}).get("window") or {}).get("_series")
            lookup = dict(zip(corrected["omega_ci_t"], corrected["corrected"])) if corrected else {}
            for t, te, dx, be in zip(s["omega_ci_t"], s["T_e_over_T_e0"], s["dx_over_lambda_De"], s["beta_e"]):
                c = lookup.get(t)
                writer.writerow([a["run"], a.get("role"), *(f"{x:.8g}" for x in (t, te, dx, be)),
                                 f"{c:.8g}" if c is not None else ""])
    plot(audits, outdir / "energy_audit.png")
    plot_baseline(audits, outdir / "energy_audit_baseline.png")


def parse_controls(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        run, sep, control = pair.partition("=")
        if not sep or not run or not control:
            raise ValueError(f"--control expects RUN_NAME=CONTROL_NAME, got {pair!r}")
        out[run] = control
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="Result directories of single runs, controls included, "
                                           "optionally LABEL=DIR (e.g. ppc4000=.../mirror_bimaxwellian_moderate)")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--control", action="append", default=[], metavar="RUN_NAME=CONTROL_NAME",
                   help="Pair a run with a control explicitly (directory names); by default every "
                        "anisotropic run takes the closest isotropic control among the given runs")
    a = p.parse_args()
    try:
        explicit = parse_controls(a.control)
    except ValueError as exc:
        p.error(str(exc))
    audits = []
    for spec in a.runs:
        label, sep, path = spec.partition("=")
        if not sep:
            label, path = None, spec
        if not Path(path).is_dir():
            p.error(f"Not a results directory: {path}")
        audits.append(audit_run(path, label))
    try:
        pair_controls(audits, explicit)
    except ValueError as exc:
        p.error(str(exc))
    groups = group_runs(audits)
    write_outputs(audits, groups, a.outdir)
    for audit in audits:
        print(f"{audit['run']}: {audit['status']} -- {audit['reason']}")
        if audit.get("baseline"):
            print(f"  baseline: {audit['baseline']['status']} -- {audit['baseline']['reason']}")
    for group in groups:
        print(group["interpretation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
