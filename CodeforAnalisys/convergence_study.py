#!/usr/bin/env python3
"""
convergence_study.py — numerical convergence, box size and realization spread
=============================================================================
A physical conclusion is only as good as its numerical tolerance: the change
of each observable under dx, dt, particles per cell, box size and random
seed must be smaller than the effect being claimed (e.g. bi-Kappa vs
bi-Maxwellian). This script collects, for any number of analysed runs, the
observables the thesis quotes and puts them side by side:

  gamma        linear growth rate (growth_rate_summary.csv, reference row:
               the dominant Fourier mode of dB, or the vector |dB| rms in
               older summaries) with its error gamma_err
  dB_sat       saturation level: max of <|dB|^2>^1/2 / B0
  t_sat        time of that maximum
  A_final      final anisotropy of the driven species (particle table)
  heating_e    relative electron heating T_e(t_end)/T_e(0) - 1
  energy_err   max |E_total(t)-E_total(0)|/E_total(0) from DiagEnergies

Each run is ``LABEL=RESULTS_DIR`` where RESULTS_DIR is the run's analysis
root (``analysis_results/<RUN_NAME>``). Runs with the same numerical
parameters (grid, domain, ppc, dt; read from their manifests) and different
``run_tag`` are treated as realizations: their mean and standard deviation
give the statistical spread. Every run is compared with the reference
(first run or ``--reference``); the table says which parameter changed.

A convergence claim needs: base + two grid refinements, one dt and one ppc
control, a larger box, and >= 3 realizations of the central pair (see the
physics audit). This script only evaluates what exists; it does not make
the runs.

Outputs: convergence_table.csv, convergence_groups.csv, convergence.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps
from growth_fit import reference_growth_row

ps.apply()

NUMERICAL_KEYS = ("grid", "domain_di", "nicell_from_profile", "dt_code_from_profile")
PHYSICAL_KEYS = ("mass_ratio", "beta_i_parallel", "A_i", "beta_e_parallel", "A_e", "kappa", "B0")
OBSERVABLES = ("gamma", "dB_sat", "t_sat", "A_final", "heating_e", "energy_err")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_run(label: str, root: Path) -> dict:
    manifests = sorted(root.glob("*_analysis_manifest.json"))
    manifest = json.loads(manifests[0].read_text()) if manifests else {}
    physics = manifest.get("physics", {})
    phys_dir = root / "09_physical_diagnostics"
    run = {"label": label, "root": str(root), "run_tag": manifest.get("run_tag", ""),
           "driven_species": manifest.get("driven_species", "ion"),
           "conventions": physics.get("analysis_conventions_version"),
           "input_identity": manifest.get("input_identity", {}).get("sha256"),
           **{k: physics.get(k) for k in NUMERICAL_KEYS + PHYSICAL_KEYS}}

    growth = _read_csv(phys_dir / "growth_rate_summary.csv")
    total = reference_growth_row(growth)
    run["gamma_series"] = total.get("series", "total") if total else "none"
    run["gamma"] = _f(total.get("gamma")) if total else np.nan
    run["gamma_err"] = _f(total.get("gamma_err")) if total else np.nan
    run["gamma_fit_ok"] = bool(total) and str(total.get("fit_ok", "")).strip() in ("1", "True", "true")

    field = _read_csv(phys_dir / "field_fluctuation_table.csv")
    if field:
        t = np.array([_f(r.get("omega_ci_t")) for r in field])
        db = np.array([_f(r.get("delta_B_vec_rms_over_B0", r.get("delta_B_rms_over_B0"))) for r in field])
        i = int(np.nanargmax(db)) if np.any(np.isfinite(db)) else None
        run["dB_sat"] = float(db[i]) if i is not None else np.nan
        run["t_sat"] = float(t[i]) if i is not None else np.nan
    else:
        run["dB_sat"] = run["t_sat"] = np.nan

    table = _read_csv(phys_dir / "anisotropy_table.csv")
    s = "e" if run["driven_species"] == "electron" else "i"
    if table:
        run["A_final"] = _f(table[-1].get(f"A_{s}"))
        te = lambda r: (_f(r.get("T_parallel_e")) + 2.0 * _f(r.get("T_perp_e"))) / 3.0
        run["heating_e"] = te(table[-1]) / te(table[0]) - 1.0 if te(table[0]) > 0 else np.nan
    else:
        run["A_final"] = run["heating_e"] = np.nan

    energy = phys_dir / "global_energy_summary.json"
    run["energy_err"] = (_f(json.loads(energy.read_text()).get("max_abs_relative_change"))
                         if energy.exists() else np.nan)
    return run


def changed_parameters(run: dict, ref: dict) -> str:
    changed = []
    for key in NUMERICAL_KEYS + PHYSICAL_KEYS:
        a, b = run.get(key), ref.get(key)
        if a is None or b is None:
            if a != b:
                changed.append(key)
            continue
        if not np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-9):
            changed.append(key)
    if not changed and run["run_tag"] != ref["run_tag"]:
        return "realization"
    return "+".join(changed) if changed else "identical"


def group_key(run: dict) -> str:
    return json.dumps({k: run.get(k) for k in NUMERICAL_KEYS + PHYSICAL_KEYS}, sort_keys=True,
                      default=str)


def main() -> int:
    p = argparse.ArgumentParser(description="Convergence / realization comparison of analysed runs.")
    p.add_argument("runs", nargs="+", help="LABEL=RESULTS_DIR (analysis root of each run)")
    p.add_argument("--reference", default=None, help="label of the reference run (default: first)")
    p.add_argument("--outdir", default="convergence")
    args = p.parse_args()

    runs = []
    for raw in args.runs:
        label, _, path = raw.partition("=")
        if not path:
            label, path = Path(raw).name, raw
        runs.append(load_run(label, Path(path)))
    ref = next((r for r in runs if r["label"] == args.reference), runs[0])
    versions = {r["conventions"] for r in runs}
    if len(versions) > 1 or len({r['gamma_series'] for r in runs}) > 1:
        raise ValueError("Mixed conventions or growth estimators; regenerate before comparing convergence")
    identities = [r['input_identity'] for r in runs if r['input_identity']]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate input fingerprints cannot count as independent realizations")
    physical_mismatch = [r["label"] for r in runs
                         if any(str(r.get(k)) != str(ref.get(k)) for k in PHYSICAL_KEYS)]
    if physical_mismatch:
        print(f"[WARN] {physical_mismatch} differ from the reference in *physical* parameters: "
              "that is a physics comparison, not a convergence test.")

    rows = []
    for run in runs:
        row = {k: run[k] for k in ("label", "run_tag", "driven_species", *NUMERICAL_KEYS)}
        row["changed_vs_reference"] = "reference" if run is ref else changed_parameters(run, ref)
        for key in OBSERVABLES:
            row[key] = run[key]
            refv = ref[key]
            row[f"{key}_rel_diff"] = (float(run[key] / refv - 1.0)
                                      if np.isfinite(run[key]) and np.isfinite(refv) and refv != 0
                                      else np.nan)
        row["gamma_err"] = run["gamma_err"]
        row["gamma_fit_ok"] = run["gamma_fit_ok"]
        rows.append(row)

    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(group_key(run), []).append(run)
    group_rows = []
    for members in groups.values():
        grow = {"members": " ".join(m["label"] for m in members), "n_realizations": len(members),
                **{k: members[0].get(k) for k in NUMERICAL_KEYS}}
        for key in OBSERVABLES:
            vals = np.array([m[key] for m in members], dtype=float)
            vals = vals[np.isfinite(vals)]
            grow[f"{key}_mean"] = float(np.mean(vals)) if vals.size else np.nan
            grow[f"{key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
        group_rows.append(grow)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, table in (("convergence_table.csv", rows), ("convergence_groups.csv", group_rows)):
        keys = list(dict.fromkeys(k for r in table for k in r))
        with (outdir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(table)
    plot(rows, outdir)
    print(f"Convergence tables written to {outdir}")
    return 0


def plot(rows: list[dict], outdir: Path):
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    titles = {"gamma": r"$\gamma/\Omega_{ci}$", "dB_sat": r"$\max\langle|\delta B|^2\rangle^{1/2}/B_0$",
              "A_final": r"final $A$ (driven species)", "heating_e": r"$T_e(t_{\rm end})/T_e(0)-1$",
              "energy_err": r"$\max|\Delta E_{\rm tot}|/E_{\rm tot}$", "t_sat": r"$t_{\rm sat}\Omega_{ci}$"}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
    for ax, key in zip(axes.ravel(), OBSERVABLES):
        y = np.array([r[key] for r in rows], dtype=float)
        err = np.array([r["gamma_err"] if key == "gamma" else np.nan for r in rows], dtype=float)
        ax.errorbar(x, y, yerr=np.where(np.isfinite(err), err, 0.0), fmt="o",
                    color=ps.c("#58a6ff"), ecolor=ps.MUTED_CLR, capsize=3)
        if np.isfinite(y[0]):
            ax.axhline(y[0], color=ps.c("#ff7b72"), ls="--", lw=1.0)
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=9)
        ax.set_title(titles[key], fontsize=12)
        ps.style_axes(ax)
    fig.suptitle("Convergence and realization spread (dashed: reference run)", fontsize=13)
    fig.tight_layout()
    ps.save(fig, outdir / "convergence.png")


if __name__ == "__main__":
    raise SystemExit(main())
