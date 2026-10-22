#!/usr/bin/env python3
"""
paper_story_figure.py — the one composite figure of the kappa comparison
=========================================================================

One figure, runs across the columns and the story down the rows::

                 bi-Maxwellian      kappa = 3        kappa = 5
    t0 (linear)   [dB_par map]     [dB_par map]     [dB_par map]
    t1 (saturated)[dB_par map]     [dB_par map]     [dB_par map]
    series        [dB/B0 and 1/kappa_eff against t Omega_ci]

Three choices in here are what make the three columns comparable, and each
one is a place where a composite figure usually goes wrong:

* **One colour scale for every map**, symmetric about zero and taken from the
  union of all six panels (``--clip`` percentile). Per-panel autoscaling makes
  a weak run look exactly like a strong one, which is the single most
  misleading thing a multi-run figure can do.
* **The same two wall-clock times for every column.** The runs saturate at
  different times (t_saturation = 68.6 / 62.8 / 64.4 in v6), so "the same
  phase" and "the same time" are different figures. This one shows the same
  clock and says so; ``--times`` overrides, and ``--phase-matched`` instead
  puts each column at its own t_saturation.
* **The series row skips t = 0 and the quiet-start build-up**
  (``plot_style.measured_fluctuation``), as every fluctuation-against-time
  figure in this pipeline does, with the hidden interval stated in the panel.

The maps need the raw ``pfd.<step>_p*.h5`` of each run (``--data``); the
series row reads the delivered tables only. With ``--no-maps`` the figure is
the series row alone, which is the part that can be rebuilt anywhere.
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
from data_reader import PICDataReader
from run_naming import distribution_of, find_runs, pretty
from psc_units import step_to_omegaci, DOMAIN_DI_Y, DOMAIN_DI_Z

ps.apply()

#: Large enough to stay legible at one-column width in a journal.
FONT = {"title": 19, "label": 17, "tick": 14, "note": 12}


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _column(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def short_name(root: Path) -> str:
    """'mirror_bikappa3_moderate' -> 'bikappa3'; works for every family."""
    return distribution_of(root.name)


def order_runs(roots: list[Path]) -> list[Path]:
    """Maxwellian first, then the kappa runs by decreasing tail weight."""
    from run_naming import DISTRIBUTION_ORDER
    rank = {name: i for i, name in enumerate(DISTRIBUTION_ORDER)}
    return sorted(roots, key=lambda r: (rank.get(short_name(r), len(rank)), r.name))


def parallel_fluctuation_map(data_dir: Path, step: int,
                             b0: float | None = None) -> np.ndarray:
    """(dB_par/B0)(y, z) of one snapshot, B_par along the background field.

    ``b0`` defaults to the snapshot's own <Bz>. In a periodic box with
    div B = 0 that mean is exactly the initial B0 and stays so for the whole
    run (verified to six digits on the synthetic run), which makes the map
    independent of whichever profile psc_units happens to have active --
    passing a stale B0 silently rescales every panel instead of failing.
    """
    files = PICDataReader.find_files(str(data_dir / "pfd.*.h5"))
    if step not in files:
        raise FileNotFoundError(f"no pfd snapshot for step {step} in {data_dir}")
    fields = PICDataReader.read_multiple_fields_3d(
        files[step], "jeh-", ["hz_fc/p0/3d"])
    bz = np.squeeze(fields["hz_fc/p0/3d"])
    if b0 is None:
        b0 = float(np.mean(bz))
    # B0 is along z (psc_anisotropy_case.hxx: `case HZ: return g.B0`, and
    # T[2] = Ti_par), so the compressive part of the mirror / ion-cyclotron
    # modes is dbz = Bz - B0 -- the same definition as
    # physical_diagnostics.field_metrics' delta_B_parallel_map. Reading hx
    # instead returns -1 everywhere, which the synthetic run catches at once.
    return (bz - b0) / b0


def series_panel(axis, twin, root: Path, colour: str) -> float:
    """dB/B0 (left) and 1/kappa_eff (right) against time.

    Returns the end of the hidden quiet-start interval. Both axes are shared
    across the whole row by the caller, which is what makes the columns
    comparable: on its own scale the bi-Maxwellian's 1/kappa_eff -- which is
    zero to within +-0.004, its noise floor -- draws as a large oscillation
    and invites a reading of structure that is not there.
    """
    field = _rows(root / "09_physical_diagnostics" / "field_fluctuation_table.csv")
    t_field = _column(field, "omega_ci_t")
    db = _column(field, "delta_B_vec_rms_over_B0")
    shown = ps.measured_fluctuation(t_field)
    axis.plot(t_field[shown], db[shown], color=ps.c(colour), lw=2.4,
              label=r"$\langle|\delta\mathbf{B}|^2\rangle^{1/2}/B_0$")
    axis.set_yscale("log")
    axis.tick_params(axis="both", labelsize=FONT["tick"])

    kappa_rows = [r for r in _rows(root / "03_particles" / "vdf_kappa_series.csv")
                  if r["population"] == "all"]
    kappa_rows.sort(key=lambda r: float(r["omega_ci_t"]))
    if kappa_rows:
        t_k = _column(kappa_rows, "omega_ci_t")
        q = _column(kappa_rows, "inv_kappa_signed")
        # 1/kappa is not a fluctuation: t = 0 is the initial state and stays.
        twin.plot(t_k, q, color=ps.MUTED_CLR, lw=2.0, linestyle="--",
                  label=r"$1/\kappa_{\rm eff}$")
    twin.axhline(0.0, color=ps.GRID_CLR, lw=0.8, zorder=0)
    twin.tick_params(axis="both", labelsize=FONT["tick"], labelcolor=ps.MUTED_CLR)
    return float(t_field[~shown].max()) if (~shown).any() else 0.0


def build(roots: list[Path], data_dirs: dict[str, Path], times: tuple[float, float],
          out: Path, clip: float = 99.5, b0: float | None = None,
          with_maps: bool = True) -> Path:
    roots = order_runs(roots)
    colours = ("#0072B2", "#009E73", "#D55E00")   # Maxwellian, kappa 3, kappa 5
    n = len(roots)
    n_rows = 3 if with_maps else 1
    fig, axes = plt.subplots(n_rows, n, figsize=(5.4 * n, 4.6 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor(ps.FIG_BG)

    maps: dict[tuple[int, int], np.ndarray] = {}
    if with_maps:
        for col, root in enumerate(roots):
            name = short_name(root)
            data_dir = data_dirs[name]
            steps = PICDataReader.find_files(str(data_dir / "pfd.*.h5"))
            for row, target in enumerate(times):
                step = min(steps, key=lambda s: abs(step_to_omegaci(s) - target))
                maps[(row, col)] = parallel_fluctuation_map(data_dir, step, b0)

        # One symmetric scale for all six panels: per-panel autoscaling would
        # make a weak run look like a strong one.
        pooled = np.concatenate([m.ravel() for m in maps.values()])
        limit = float(np.percentile(np.abs(pooled), clip))
        extent = (0.0, DOMAIN_DI_Y, 0.0, DOMAIN_DI_Z)
        image = None
        for (row, col), field in maps.items():
            axis = axes[row][col]
            image = axis.imshow(field, origin="lower", extent=extent,
                                cmap="RdBu_r", vmin=-limit, vmax=limit,
                                aspect="equal")
            axis.tick_params(axis="both", labelsize=FONT["tick"])
            if col == 0:
                axis.set_ylabel(f"$t\\Omega_{{ci}} = {times[row]:.0f}$\n$z/d_i$",
                                fontsize=FONT["label"])
            else:
                axis.set_yticklabels([])
            # Only the lower map row carries the shared x axis.
            if row == len(times) - 1:
                axis.set_xlabel(r"$y/d_i$", fontsize=FONT["label"])
            else:
                axis.set_xticklabels([])
        bar = fig.colorbar(image, ax=axes[:2, :].ravel().tolist(),
                           shrink=0.82, pad=0.015)
        bar.set_label(r"$\delta B_\parallel / B_0$", fontsize=FONT["label"])
        bar.ax.tick_params(labelsize=FONT["tick"])

    series_row = axes[-1]
    # One shared scale per quantity across the row. sharey has to be set up
    # before anything is drawn, and the twins are shared with each other.
    for col in range(1, n):
        series_row[col].sharey(series_row[0])
    twins = [series_row[0].twinx()]
    for col in range(1, n):
        twin = series_row[col].twinx()
        twin.sharey(twins[0])
        twins.append(twin)

    hidden = 0.0
    for col, root in enumerate(roots):
        axis = series_row[col]
        hidden = max(hidden, series_panel(axis, twins[col], root,
                                          colours[col % len(colours)]))
        axis.set_xlabel(r"$t\Omega_{ci}$", fontsize=FONT["label"])
        if col:
            axis.tick_params(axis="y", labelleft=False)
        # Only the last column keeps the right-hand numbers; the inner twins
        # would otherwise print a column of digits between the panels.
        if col == n - 1:
            twins[col].set_ylabel("$1/\\kappa_{\\rm eff}$\n(dashed)",
                                  fontsize=FONT["label"], color=ps.MUTED_CLR)
        else:
            twins[col].tick_params(axis="y", labelright=False)
    series_row[0].set_ylabel(r"$\langle|\delta\mathbf{B}|^2\rangle^{1/2}/B_0$"
                             "\n(solid)", fontsize=FONT["label"])
    # No legend at all: the two series are named by the two axes they belong
    # to, with the line style in the label. A legend inside a panel sits on
    # the curve it describes, and one centred under the row sits on the middle
    # column's x label -- plot_style's layout check rejects both.
    note = (f"quiet-start build-up ($t\\Omega_{{ci}} < {hidden:.1f}$) not shown; "
            r"$1/\kappa_{\rm eff}$ shares one scale across the row")

    for col, root in enumerate(roots):
        axes[0][col].set_title(pretty(root.name),
                               fontsize=FONT["title"], fontweight="bold")
    fig.text(0.005, -0.03, note, fontsize=FONT["note"], color=ps.MUTED_CLR)
    if not with_maps:
        fig.tight_layout()
    ps.save(fig, out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path,
                        help="analysis_results/<run> directories, one per column")
    parser.add_argument("--data", action="append", default=[], metavar="NAME=DIR",
                        help="raw pfd directory of a run, e.g. bikappa3=/cosma7/.../psc_...")
    parser.add_argument("--times", type=float, nargs=2, default=(30.0, 120.0),
                        help="the two snapshot times, in 1/Omega_ci (default 30 120)")
    parser.add_argument("--clip", type=float, default=99.5,
                        help="percentile of |dB_par| that sets the shared colour limit")
    parser.add_argument("--b0", type=float, default=None,
                        help="background field; default is each snapshot's <Bz>")
    parser.add_argument("--no-maps", action="store_true",
                        help="series row only; no raw snapshots needed")
    parser.add_argument("--out", type=Path, default=Path("paper_story_figure.png"))
    args = parser.parse_args()

    data_dirs = {}
    for item in args.data:
        name, _, directory = item.partition("=")
        data_dirs[name] = Path(directory)
    with_maps = not args.no_maps
    if with_maps:
        missing = [short_name(r) for r in args.roots if short_name(r) not in data_dirs]
        if missing:
            parser.error(f"--data missing for {', '.join(missing)} (or pass --no-maps)")

    out = build(args.roots, data_dirs, tuple(args.times), args.out,
                clip=args.clip, b0=args.b0, with_maps=with_maps)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
