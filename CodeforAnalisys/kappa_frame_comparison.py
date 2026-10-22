#!/usr/bin/env python3
"""
kappa_frame_comparison.py — how much kappa the field structure manufactures
============================================================================

This pipeline measures the shape index in the frame of the *local* field:
every macrocell is whitened by its own measured variances before the kurtosis
ratio is taken. Much of the literature instead fits one distribution to the
whole box. The two are not the same measurement, and the difference is not a
detail: a plasma that is locally Maxwellian everywhere, but whose temperature
varies from place to place, looks kappa when pooled over the box. Superposing
Maxwellians of different widths *is* a way to manufacture a suprathermal tail.

The inflation is exact and needs no particles, only the per-macrocell moments
the pipeline already writes. With the estimator's whitened speed

    s^2 = v_par^2/sigma_par^2 + v_perp^2/sigma_perp^2 ,     K = <s^4>/<s^2>^2

and global whitening by the box variances, a cell whose local widths are
a_par = sigma_par,i / sigma_par,box and a_perp likewise contributes

    <s^2>_i = a_par^2 + 2 a_perp^2
    <s^4>_i = 3 a_par^4 + 4 a_par^2 a_perp^2 + 8 a_perp^4

(chi-squared moments: <X_1^2> = 1, <X_1^4> = 3, <X_2^2> = 2, <X_2^4> = 8).
Because the box variance is the count-weighted mean of the local ones, the
denominator is exactly 3, so for locally Maxwellian cells

    K_box = [3 <a_par^4> + 4 <a_par^2 a_perp^2> + 8 <a_perp^4>] / 9 ,

against the Maxwellian 5/3. Their ratio is the **inflation factor** F: what
the structure alone does to the kurtosis ratio. F = 1 for a uniform box.

Reported per snapshot:

* ``F``: the inflation factor from the macrocell moments alone.
* ``inv_kappa_structure_only``: the index a box-frame estimator would report
  for a plasma that is locally Maxwellian everywhere -- kappa made of nothing
  but structure.
* ``inv_kappa_box_predicted``: the local measurement carried into the box
  frame, K_box = F * K_local.

Three caveats, all of which make this a **lower bound** on the real box-frame
inflation, and all of which belong in the text:

1. The macrocells are a coarse (8 x 8) sampling of the structure; variation
   inside a cell is not counted.
2. Cell-to-cell differences in bulk velocity add to the pooled fourth moment
   and are not included, because the delivered tables carry no drift column.
3. K_box = F * K_local assumes the same local shape in every cell.

The quantitative claim this supports is modest and specific: a box-frame
kappa falls while the local-frame kappa rises, so the two conventions can
disagree in *sign* on the same run, and a comparison across papers has to say
which one it used.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style as ps
from kappa_eff import (DEFAULT_S_MAX, signed_inverse_kappa, truncated_K_kappa,
                       truncated_K_maxwellian, KAPPA_MAX)
from run_naming import families_in, find_runs, pretty

ps.apply()

FONT = {"title": 18, "label": 16, "tick": 13, "note": 11}
LOCAL_COLOUR, BOX_COLOUR = "#0072B2", "#D55E00"
STEP_RE = re.compile(r"vdf_macrocells_step(\d+)\.csv$")


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def inflation_factor(t_par, t_perp, counts) -> float:
    """K_box / K_Maxwellian for locally Maxwellian cells of these moments.

    1 for a uniform box; larger when the local widths vary from cell to cell.
    """
    t_par = np.asarray(t_par, dtype=float)
    t_perp = np.asarray(t_perp, dtype=float)
    weights = np.asarray(counts, dtype=float)
    good = (np.isfinite(t_par) & np.isfinite(t_perp) & np.isfinite(weights)
            & (t_par > 0) & (t_perp > 0) & (weights > 0))
    if good.sum() < 2:
        return float("nan")
    t_par, t_perp, weights = t_par[good], t_perp[good], weights[good]
    weights = weights / weights.sum()
    # a^2 is the local variance over the count-weighted box variance, so each
    # of these means is exactly 1 and the denominator of K is exactly 3.
    a_par2 = t_par / np.sum(weights * t_par)
    a_perp2 = t_perp / np.sum(weights * t_perp)
    k_box = (3.0 * np.sum(weights * a_par2 ** 2)
             + 4.0 * np.sum(weights * a_par2 * a_perp2)
             + 8.0 * np.sum(weights * a_perp2 ** 2)) / 9.0
    return float(k_box / (5.0 / 3.0))


def local_K(inv_kappa: float, s_max: float = DEFAULT_S_MAX) -> float:
    """The truncated kurtosis ratio behind a reported signed 1/kappa."""
    if not np.isfinite(inv_kappa):
        return float("nan")
    if inv_kappa > 1.0 / KAPPA_MAX:
        return float(truncated_K_kappa(1.0 / inv_kappa, s_max))
    # The signed continuation below the Maxwellian is linear in 1/kappa.
    edge = truncated_K_kappa(KAPPA_MAX, s_max)
    slope = (edge - truncated_K_maxwellian(s_max)) * KAPPA_MAX
    return float(edge + (inv_kappa - 1.0 / KAPPA_MAX) * slope)


def series(run: Path) -> dict[str, np.ndarray]:
    """Per snapshot: t, the local 1/kappa, F, and the two box-frame indices."""
    folder = run / "03_particles"
    local_rows = [r for r in _rows(folder / "vdf_kappa_series.csv")
                  if r["population"] == "all"]
    local_by_t = {round(float(r["omega_ci_t"]), 6): float(r["inv_kappa_signed"])
                  for r in local_rows}

    t, factor, structure, predicted, local = [], [], [], [], []
    for path in sorted(folder.glob("vdf_macrocells_step*.csv")):
        rows = _rows(path)
        if not rows:
            continue
        time = float(rows[0]["omega_ci_t"])
        f = inflation_factor([r["T_parallel"] for r in rows],
                             [r["T_perp"] for r in rows],
                             [r["count"] for r in rows])
        if not np.isfinite(f):
            continue
        maxwellian = truncated_K_maxwellian(DEFAULT_S_MAX)
        t.append(time)
        factor.append(f)
        # A locally Maxwellian box, read in the box frame.
        structure.append(signed_inverse_kappa(f * maxwellian, DEFAULT_S_MAX))
        key = round(time, 6)
        nearest = min(local_by_t, key=lambda k: abs(k - key)) if local_by_t else None
        q = local_by_t.get(key, local_by_t.get(nearest, float("nan")))
        local.append(q)
        predicted.append(signed_inverse_kappa(f * local_K(q), DEFAULT_S_MAX))
    order = np.argsort(t)
    return {"t": np.array(t)[order], "inflation": np.array(factor)[order],
            "inv_kappa_local": np.array(local)[order],
            "inv_kappa_structure_only": np.array(structure)[order],
            "inv_kappa_box_predicted": np.array(predicted)[order]}


def build(runs: list[Path], out: Path) -> Path:
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 7.8), squeeze=False, sharex=True)
    fig.patch.set_facecolor(ps.FIG_BG)
    for col in range(1, n):
        axes[0][col].sharey(axes[0][0])
        axes[1][col].sharey(axes[1][0])

    for col, run in enumerate(runs):
        data = series(run)
        top = axes[0][col]
        top.plot(data["t"], data["inv_kappa_local"], color=ps.c(LOCAL_COLOUR),
                 lw=2.4, label="local-field frame (measured)")
        top.plot(data["t"], data["inv_kappa_box_predicted"], color=ps.c(BOX_COLOUR),
                 lw=2.4, linestyle="--", label="box frame (predicted)")
        top.plot(data["t"], data["inv_kappa_structure_only"], color=ps.MUTED_CLR,
                 lw=1.8, linestyle=":", label="structure only (locally Maxwellian)")
        top.axhline(0.0, color=ps.GRID_CLR, lw=0.8)
        top.set_title(pretty(run.name), fontsize=FONT["title"], fontweight="bold")
        top.tick_params(axis="both", labelsize=FONT["tick"])

        bottom = axes[1][col]
        bottom.plot(data["t"], data["inflation"], color=ps.TEXT_CLR, lw=2.0)
        bottom.axhline(1.0, color=ps.GRID_CLR, lw=1.0, linestyle=":")
        bottom.set_xlabel(r"$t\Omega_{ci}$", fontsize=FONT["label"])
        bottom.tick_params(axis="both", labelsize=FONT["tick"])
        if col:
            top.tick_params(axis="y", labelleft=False)
            bottom.tick_params(axis="y", labelleft=False)

    axes[0][0].set_ylabel(r"$1/\kappa_{\rm eff}$", fontsize=FONT["label"])
    axes[1][0].set_ylabel("inflation factor $F$", fontsize=FONT["label"])
    axes[0][0].legend(fontsize=FONT["note"], loc="center left", framealpha=0.92)
    fig.text(0.005, -0.02,
             "Lower bound: 8x8 macrocells, no cell-to-cell drift term, "
             "one local shape assumed for every cell",
             fontsize=FONT["note"], color=ps.MUTED_CLR)
    fig.tight_layout()
    ps.save(fig, out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--family", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runs = find_runs(args.root, family=args.family)
    if not runs:
        raise SystemExit(f"No case runs under {args.root}")
    if len(families_in(runs)) > 1:
        raise SystemExit("Mixed families; pass --family")

    outdir = (args.out.parent if args.out else args.root / "kappa_frame_comparison")
    outdir.mkdir(parents=True, exist_ok=True)
    out = args.out or outdir / "kappa_frame_comparison.png"

    with (outdir / "kappa_frame_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "omega_ci_t", "inflation_factor",
                         "inv_kappa_local", "inv_kappa_structure_only",
                         "inv_kappa_box_predicted"])
        for run in runs:
            data = series(run)
            for i in range(len(data["t"])):
                writer.writerow([run.name, f"{data['t'][i]:.6g}",
                                 f"{data['inflation'][i]:.6g}",
                                 f"{data['inv_kappa_local'][i]:.6g}",
                                 f"{data['inv_kappa_structure_only'][i]:.6g}",
                                 f"{data['inv_kappa_box_predicted'][i]:.6g}"])
            print(f"{run.name:34s} F: {data['inflation'][0]:.4f} -> "
                  f"{data['inflation'][-1]:.4f}   local 1/kappa "
                  f"{data['inv_kappa_local'][0]:+.4f} -> {data['inv_kappa_local'][-1]:+.4f}"
                  f"   box {data['inv_kappa_box_predicted'][0]:+.4f} -> "
                  f"{data['inv_kappa_box_predicted'][-1]:+.4f}")
    print(build(runs, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
