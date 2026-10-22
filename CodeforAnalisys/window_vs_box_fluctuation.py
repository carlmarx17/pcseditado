#!/usr/bin/env python3
"""
window_vs_box_fluctuation.py — is the particle window the whole box?
=====================================================================

Every result that pairs the shape of the distribution with the field pairs
two measurements made over *different volumes*: 1/kappa_eff comes from the
particles PSC dumps, which live in the central prt window (20 % of the
domain), while the field statistics are whole-box. The pairing is only
meaningful while the window is representative.

This draws the one quantity that decides it -- the magnitude fluctuation
rms(|B| - B0)/B0 -- over both volumes, and their ratio. A ratio that stays
near 1 says the window is a fair sample. A ratio that drifts says the
structures are not uniformly distributed (mirror modes trap particles in
magnetic holes, which is exactly a non-uniform distribution of |B|), and then
a kappa measured in the window cannot be set against a whole-box field
amplitude without saying so.

Reads delivered tables only (09_physical_diagnostics/field_fluctuation_table.csv).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style as ps
from run_naming import families_in, find_runs, pretty

ps.apply()

FONT = {"title": 18, "label": 16, "tick": 13, "note": 11}

#: Okabe-Ito, fixed order of the pipeline: whole box, then the prt window.
BOX_COLOUR, WINDOW_COLOUR = "#0072B2", "#D55E00"


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _column(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def series(run: Path) -> dict[str, np.ndarray]:
    """t, whole-box and prt-window magnitude fluctuation, in time order."""
    rows = _rows(run / "09_physical_diagnostics" / "field_fluctuation_table.csv")
    order = np.argsort(_column(rows, "omega_ci_t"))
    t = _column(rows, "omega_ci_t")[order]
    box = _column(rows, "delta_B_rms_over_B0")[order]
    window = _column(rows, "prt_delta_B_rms_over_B0")[order]
    return {"t": t, "box": box, "window": window}


def build(runs: list[Path], out: Path) -> Path:
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 7.6), squeeze=False,
                             sharex=True)
    fig.patch.set_facecolor(ps.FIG_BG)
    for col in range(1, n):
        axes[0][col].sharey(axes[0][0])
        axes[1][col].sharey(axes[1][0])

    hidden = 0.0
    for col, run in enumerate(runs):
        data = series(run)
        t = data["t"]
        shown = ps.measured_fluctuation(t)
        hidden = max(hidden, float(t[~shown].max()) if (~shown).any() else 0.0)
        top = axes[0][col]
        top.semilogy(t[shown], data["box"][shown], color=ps.c(BOX_COLOUR),
                     lw=2.2, label="whole box")
        top.semilogy(t[shown], data["window"][shown], color=ps.c(WINDOW_COLOUR),
                     lw=2.2, linestyle="--", label="prt window")
        top.set_title(pretty(run.name), fontsize=FONT["title"], fontweight="bold")
        top.tick_params(axis="both", labelsize=FONT["tick"])

        bottom = axes[1][col]
        # The ratio is the figure's point, so it is drawn where the two series
        # are actually measured; a ratio of noise floors is not informative.
        valid = shown & (data["box"] > 0)
        bottom.plot(t[valid], data["window"][valid] / data["box"][valid],
                    color=ps.TEXT_CLR, lw=2.0)
        bottom.axhline(1.0, color=ps.GRID_CLR, lw=1.0, linestyle=":")
        bottom.set_xlabel(r"$t\Omega_{ci}$", fontsize=FONT["label"])
        bottom.tick_params(axis="both", labelsize=FONT["tick"])
        if col:
            top.tick_params(axis="y", labelleft=False)
            bottom.tick_params(axis="y", labelleft=False)

    axes[0][0].set_ylabel(r"$\mathrm{rms}(|B|-B_0)/B_0$", fontsize=FONT["label"])
    axes[1][0].set_ylabel("window / box", fontsize=FONT["label"])
    axes[0][0].legend(fontsize=FONT["note"], loc="lower right", framealpha=0.92)
    fig.text(0.005, -0.02,
             f"quiet-start build-up ($t\\Omega_{{ci}} < {hidden:.1f}$) not shown; "
             "both rows share one scale across the columns",
             fontsize=FONT["note"], color=ps.MUTED_CLR)
    fig.tight_layout()
    ps.save(fig, out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="results root holding the run directories")
    parser.add_argument("--family", default=None,
                        help="mirror | firehose | whistler (default: the one present)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runs = find_runs(args.root, family=args.family)
    if not runs:
        raise SystemExit(f"No case runs under {args.root}")
    families = families_in(runs)
    if len(families) > 1:
        raise SystemExit(f"Mixed families {sorted(families)}; pass --family")

    out = args.out or args.root / "window_vs_box_fluctuation" / "window_vs_box_fluctuation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(build(runs, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
