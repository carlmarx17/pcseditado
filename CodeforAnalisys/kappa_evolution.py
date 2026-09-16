#!/usr/bin/env python3
"""
kappa_evolution.py — temporal evolution of the fitted kappa index across cases
==============================================================================
The pipeline already fits a Kappa shape to f(v_par) at every snapshot and
stores the result in `fit_metrics.csv` (written by `physical_diagnostics.py`),
and it already locates the linear phase in `growth_rate_summary.csv`. Nothing
joins the two, so the question "does the injected index survive the
instability?" has never been asked of the data.

This module answers it, for several runs at once.

Three things it does that plotting kappa_fit(t) directly does not:

  1. **Plots 1/kappa, not kappa.** The fit grid has a hard upper bound
     (kappa = 80 in the current pipeline), so a Maxwellian run does not return
     a number, it returns the ceiling. In kappa the ceiling is a meaningless
     spike; in 1/kappa it is ~0, which is exactly where a Maxwellian belongs.
     Censored samples are drawn as limits, never as data points.

  2. **Shades the linear phase** from `growth_rate_summary.csv`, so a change
     in the index can be placed before, during or after saturation instead of
     being read off a bare time axis.

  3. **Reports the model-selection ratio** R = error_maxwellian / error_kappa,
     in the tail where the two models actually differ. R < 1 means the
     Maxwellian describes the VDF better; R > 1 means the Kappa does. A run
     that starts below 1 and ends well above it has changed family, which is a
     statement no single fitted value can make.

Numerical-relaxation note: finite particle number acts like a collision
operator and drives the VDF *towards* a Maxwellian, i.e. towards larger kappa
and smaller 1/kappa. A measured *increase* of 1/kappa therefore cannot be a
particle-noise artefact -- the artefact has the opposite sign. A decrease can,
and must be controlled against a stable run.

Usage::

    python kappa_evolution.py \
        --case "mirror=../analysis_results/M_lite/09_physical_diagnostics" \
        --case "firehose=../analysis_results/F_lite/09_physical_diagnostics" \
        --outdir ../analysis_results/kappa_evolution
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import plot_style as ps
    ps.apply()
    _HAVE_PS = True
except Exception:                                    # standalone use
    _HAVE_PS = False

# Okabe-Ito, in the fixed order plot_style uses. Never cycled.
SERIES = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
MUTED = "#5f5f5c"
GRID = "#e6e3dd"

# `physical_diagnostics.kappa_pdf_shape` searches a bounded kappa grid; values
# at or above this are censored, not measured.
KAPPA_CEILING = 79.5


def _c(hex_color: str) -> str:
    return ps.c(hex_color) if _HAVE_PS else hex_color


def read_case(path: Path) -> dict:
    """fit_metrics.csv + growth_rate_summary.csv for one run."""
    rows = []
    with open(path / "fit_metrics.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            def f(key):
                try:
                    return float(r[key])
                except (KeyError, TypeError, ValueError):
                    return float("nan")
            rows.append({
                "t": f("omega_ci_t"),
                "kappa": f("kappa_fit"),
                "err_max": f("error_maxwellian"),
                "err_kap": f("error_kappa"),
                "err_max_tail": f("error_tail_maxwellian"),
                "err_kap_tail": f("error_tail_kappa"),
                "supra": f("suprathermal_fraction"),
            })
    rows.sort(key=lambda r: r["t"])

    growth = {}
    gpath = path / "growth_rate_summary.csv"
    if gpath.exists():
        with open(gpath, newline="") as fh:
            for r in csv.DictReader(fh):
                growth = {k: float(v) for k, v in r.items() if k != "fit_reject_reason" and v.strip()}
                if r.get("fit_ok", "0").lower() not in ("1", "true"):
                    growth = {}
                break
    return {"rows": rows, "growth": growth}


def _arrays(case: dict):
    rows = case["rows"]
    t = np.array([r["t"] for r in rows])
    kap = np.array([r["kappa"] for r in rows])
    inv = 1.0 / np.where(kap > 0, kap, np.nan)
    censored = kap >= KAPPA_CEILING
    # Tail errors are the discriminating ones; fall back to the full-range
    # errors when the tail was too sparse to score (NaN in the CSV).
    em = np.array([r["err_max_tail"] for r in rows])
    ek = np.array([r["err_kap_tail"] for r in rows])
    em_f = np.array([r["err_max"] for r in rows])
    ek_f = np.array([r["err_kap"] for r in rows])
    use_tail = np.isfinite(em) & np.isfinite(ek)
    ratio = np.where(use_tail, em / ek, em_f / ek_f)
    return t, inv, censored, ratio, use_tail


def plot_evolution(cases: dict, outdir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.6, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.12})

    for i, (label, case) in enumerate(cases.items()):
        col = _c(SERIES[i % len(SERIES)])
        t, inv, cens, ratio, _ = _arrays(case)
        g = case["growth"]

        # The trajectory is drawn through every sample, censored ones included,
        # because skipping them interpolates across the interval where the VDF
        # was *most* Maxwellian and hides the dip. Censored segments are dashed
        # and their markers are upper limits, never filled points.
        ax1.plot(t, inv, color=col, lw=1.8, ls=(0, (3, 2)), alpha=0.75, zorder=2)
        # Markers only: a solid chord between two uncensored samples would
        # imply the fit was resolved in between, and it was not.
        ax1.plot(t[~cens], inv[~cens], color=col, ls="none", marker="o", ms=5.5,
                 zorder=4, label=label)
        if cens.any():
            ax1.errorbar(t[cens], inv[cens], yerr=0.005, uplims=True,
                         fmt="v", ms=5, mfc="none", color=col, alpha=0.75,
                         lw=1.2, capsize=0, zorder=3)

        ax2.plot(t, ratio, color=col, lw=2.0, marker="s", ms=5, zorder=3,
                 label=label)

        t_end = g.get("linear_phase_end")
        if t_end is not None:
            for ax in (ax1, ax2):
                ax.axvline(t_end, color=col, lw=1.0, ls=(0, (1, 2)),
                           alpha=0.8, zorder=1)

    ax1.axhline(0.0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax1.text(0.995, 0.002, "Maxwellian limit", transform=
             ax1.get_yaxis_transform(), fontsize=8, color=MUTED,
             va="bottom", ha="right")
    ax1.set_ylabel(r"$1/\kappa_{\rm fit}$")
    ax1.set_ylim(-0.008, None)

    ax2.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax2.set_ylabel(r"$\varepsilon_{\rm Maxw}/\varepsilon_{\kappa}$  (tail)")
    ax2.set_xlabel(r"$t\,\Omega_{ci}$")
    ax2.set_yscale("log")

    for ax in (ax1, ax2):
        ax.grid(True, which="major", color=GRID, lw=0.5, ls=":", zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax1.legend(frameon=False, fontsize=9, loc="upper left",
               title="dotted vertical: end of linear phase",
               title_fontsize=7.5)

    ax2.text(0.012, 0.93, r"$>1$: Kappa fits better", transform=ax2.transAxes,
             fontsize=8, color=MUTED, va="top")
    ax2.text(0.012, 0.07, r"$<1$: Maxwellian fits better",
             transform=ax2.transAxes, fontsize=8, color=MUTED, va="bottom")

    out = outdir / "kappa_evolution.png"
    if _HAVE_PS:
        ps.save(fig, out)
    else:
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return out


def write_summary(cases: dict, outdir: Path) -> Path:
    out = outdir / "kappa_evolution_summary.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "t_first", "kappa_first", "t_last", "kappa_last",
                    "inv_kappa_first", "inv_kappa_last", "delta_inv_kappa",
                    "ratio_first", "ratio_last", "linear_phase_end",
                    "n_snapshots", "n_censored", "n_post_saturation"])
        for label, case in cases.items():
            t, inv, cens, ratio, _ = _arrays(case)
            if t.size == 0:
                continue
            g = case["growth"]
            t_end = g.get("linear_phase_end", float("nan"))
            n_post = int(np.sum(t > t_end)) if math.isfinite(t_end) else 0
            kap = 1.0 / np.where(inv > 0, inv, np.nan)
            w.writerow([label, f"{t[0]:.3f}", f"{kap[0]:.3f}",
                        f"{t[-1]:.3f}", f"{kap[-1]:.3f}",
                        f"{inv[0]:.5f}", f"{inv[-1]:.5f}",
                        f"{inv[-1]-inv[0]:+.5f}",
                        f"{ratio[0]:.3f}", f"{ratio[-1]:.3f}",
                        f"{t_end:.3f}", t.size, int(cens.sum()), n_post])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", action="append", default=[], metavar="LABEL=DIR",
                   help="run label and its 09_physical_diagnostics directory")
    p.add_argument("--outdir", default="kappa_evolution")
    a = p.parse_args()

    if not a.case:
        p.error("at least one --case LABEL=DIR is required")

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cases = {}
    for spec in a.case:
        if "=" not in spec:
            p.error(f"--case must be LABEL=DIR, got {spec!r}")
        label, _, d = spec.partition("=")
        path = Path(d)
        if not (path / "fit_metrics.csv").exists():
            print(f"[WARN] no fit_metrics.csv in {path}, skipping {label}")
            continue
        cases[label] = read_case(path)

    if not cases:
        print("[ERROR] no usable cases")
        return 1

    fig_path = plot_evolution(cases, outdir)
    csv_path = write_summary(cases, outdir)
    print(f"  wrote -> {fig_path}")
    print(f"  wrote -> {csv_path}")

    for label, case in cases.items():
        t, inv, cens, ratio, _ = _arrays(case)
        n_post = 0
        te = case["growth"].get("linear_phase_end")
        if te is not None:
            n_post = int(np.sum(t > te))
        if n_post < 3:
            print(f"[WARN] {label}: only {n_post} snapshot(s) after the linear "
                  f"phase; the post-saturation trend is not constrained")
        if cens.all():
            print(f"[WARN] {label}: every fit is at the grid ceiling; "
                  f"no finite kappa was resolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
