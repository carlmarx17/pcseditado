#!/usr/bin/env python3
"""
field_vs_kappa.py — the magnetic field of the whole box against the measured kappa
==================================================================================
Two views of the same pair of series, from analysis products only (no snapshots),
so it runs on a laptop against a delivered results tree:

  <run>/09_physical_diagnostics/field_fluctuation_table.csv   whole-box field
  <run>/03_particles/vdf_kappa_series.csv                     kappa of the ions

1. **At chosen instants** (`field_vs_kappa_instants`): the field of every run
   against its own measured index at that time, one line per instant joining the
   runs of the series. This is the figure that says whether the field and the
   index are related *across* the series at a fixed time, which a single run
   cannot answer.
2. **Along the run** (`field_vs_kappa_trajectory`): field against index as a
   track per run, with time as the colour. The same two quantities, now showing
   the direction in which the pair moves, and where the linear phase ends.

Both panels of each figure carry the two field measures, because they answer
different questions:

* ``<|B|>/B0`` over the whole box is the compressive signature. For mirror it is
  the physical quantity (holes and peaks). For transverse modes (parallel
  firehose, EMIC, whistler) ``|B| - B0 ~ dB_perp^2 / 2 B0`` is second order, so
  the curve stays near 1 and a flat line there is not an absence of fluctuations.
* ``<|dB|^2>^(1/2)/B0`` is the amplitude whose log-slope is gamma for every
  branch, so it is the one to read for the transverse cases.

The index is the signed inverse effective kappa in the local-field frame,
measured from truncated, whitened moments in the prt window (kappa on the
opposite axis, 0 = Maxwellian shape reference). This particle window is
smaller than the whole box used for the field statistics. The nominal index the particles were loaded with is the
colour of the run, so the distance between a run's colour and its position on
the axis is how far the measured distribution moved from the loaded one. These
runs load kappa at **fixed second-moment temperature** (same temperature as the
bi-Maxwellian, not the same core), which is the convention the comparison is
made in.

A relation in these figures is a correlation in time between two measured
series, not a mechanism: both the field and the index respond to the
instability, and only the isotropic controls separate what the numerical setup
contributes (see energy_audit.py).

Usage:
    python field_vs_kappa.py RUN_DIR [RUN_DIR ...] --outdir OUT \\
        [--times 5 10 20 40] [--population all]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import plot_style as ps
from kappa_dynamics import KAPPA_TICKS, _kappa_axis, load_run

#: Okabe-Ito in the fixed order of the series: bi-Maxwellian, kappa 5, kappa 3.
KAPPA_COLOURS = {None: "#0072B2", 5.0: "#D55E00", 3.0: "#009E73"}
FALLBACK_COLOUR = "#CC79A7"

#: The two field measures, with the axis label each one needs.
FIELDS = (("B_mean", r"$\langle |B| \rangle / B_0$  (whole box)"),
          ("dB_vec", r"$\langle |\delta B|^2 \rangle^{1/2} / B_0$  (whole box)"))


def colour(kappa0) -> str:
    return ps.c(KAPPA_COLOURS.get(kappa0, FALLBACK_COLOUR))


def paired_series(run: dict, population: str = "all") -> dict | None:
    """Index and field of one run on the index's own times.

    The index comes from particle snapshots and the field from field snapshots,
    a sparser and a denser cadence: the field is interpolated onto the index
    times, never the other way round, so no index value is invented.
    """
    field, pop = run.get("field"), (run.get("pop") or {}).get(population)
    if field is None or pop is None:
        return None
    t = np.asarray(pop["t"], dtype=float)
    # Fluctuation series: no t = 0 and no quiet-start noise build-up, with the
    # settling time of this run's own profile (plot_style.measured_fluctuation).
    shown = ps.measured_fluctuation(t, run.get("t_settle"))
    inside = (t >= field["t"][0]) & (t <= field["t"][-1])      # no extrapolation
    keep = shown & inside & np.isfinite(pop["inv"])
    if keep.sum() < 2:
        return None
    t = t[keep]
    out = {"t": t, "inv": np.asarray(pop["inv"], float)[keep],
           "inv_err": np.asarray(pop["err"], float)[keep],
           "dB_vec": np.sqrt(np.interp(t, field["t"], field["W"]))}
    for key in ("B_mean", "B_std"):
        column = field.get(key)
        out[key] = (np.interp(t, field["t"], column)
                    if column is not None and np.any(np.isfinite(column))
                    else np.full(t.size, np.nan))
    return out


def local_contrast(run: dict, s: dict) -> dict | None:
    """1/kappa inside the magnetic holes and the peaks, on the field's own times.

    These are the sub-regions where |B| falls and rises, so the contrast between
    them is a statement about the field magnitude; the box average of |B| is,
    to second order, the square of the fluctuation and says nothing more.
    """
    hole, peak = (run.get("pop") or {}).get("hole"), (run.get("pop") or {}).get("peak")
    if hole is None or peak is None:
        return None
    t = s["t"]
    out = {"t": t}
    for name, pop in (("hole", hole), ("peak", peak)):
        ok = np.isfinite(pop["inv"])
        if ok.sum() < 2:
            return None
        out[name] = np.interp(t, np.asarray(pop["t"])[ok], np.asarray(pop["inv"])[ok],
                              left=np.nan, right=np.nan)
    return out


def at_instant(series: dict, time: float) -> dict | None:
    """The sample of `series` closest to `time`, or None if none is near it.

    "Near" is half the local spacing of the index series: asking for an instant
    no snapshot covers must give nothing, not the nearest unrelated snapshot.
    """
    t = series["t"]
    i = int(np.argmin(np.abs(t - time)))
    tolerance = max(np.median(np.diff(t)) if t.size > 1 else 0.0, 1e-9)
    if abs(t[i] - time) > tolerance:
        return None
    return {k: v[i] for k, v in series.items()}


def default_times(runs: list[dict], series: dict, count: int = 4) -> list[float]:
    """Instants every run reaches, including the end of the linear phase."""
    ends = [s["t"][-1] for s in series.values()]
    starts = [s["t"][0] for s in series.values()]
    if not ends:
        return []
    lo, hi = max(starts), min(ends)
    times = list(np.linspace(lo, hi, count))
    linear_ends = [r.get("t_lin_end") for r in runs]
    linear = [x for x in linear_ends if x is not None and np.isfinite(x) and lo <= x <= hi]
    if linear:
        times[1] = float(np.median(linear))
    return sorted(round(float(x), 3) for x in times)


def plot_instants(runs, series, times, path: Path) -> Path | None:
    drawn = {time: [(r, at_instant(series[r["name"]], time)) for r in runs
                    if r["name"] in series and at_instant(series[r["name"]], time)]
             for time in times}
    drawn = {t: v for t, v in drawn.items() if len(v) >= 2}
    if not drawn:
        return None
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.4))
    styles = ["-", "--", "-.", ":"]
    for ax, (key, label) in zip(axes, FIELDS):
        for n, (time, items) in enumerate(sorted(drawn.items())):
            items = sorted(items, key=lambda it: it[1]["inv"])
            x = np.array([it[1]["inv"] for it in items])
            y = np.array([it[1][key] for it in items])
            if not np.any(np.isfinite(y)):
                continue
            # The line joins the runs of the series at one instant: it is a
            # guide between measured points, not a fit.
            ax.plot(x, y, styles[n % len(styles)], color=ps.c("#888888"), lw=1.0,
                    zorder=1, label=rf"$\Omega_{{ci}} t = {time:g}$")
            for run, point in items:
                err = point["B_std"] if key == "B_mean" else None
                ax.errorbar(point["inv"], point[key],
                            xerr=point["inv_err"],
                            yerr=err if err is not None and np.isfinite(err) else None,
                            marker="o", ms=5.5, lw=0, elinewidth=0.9, capsize=2.0,
                            color=colour(run["kappa0"]), zorder=3)
        ax.set_xlabel(r"$1/\kappa$  measured, local-field frame")
        # The vertical bar of the left panel is the spread of |B| across the
        # box, not an error of the mean: said here so the axis cannot be read
        # as an uncertainty. The horizontal bar is the error of the index.
        ax.set_ylabel(label + ("\nbar: spread across the box" if key == "B_mean" else ""))
        ps.style_axes(ax)
        ps.guard_degenerate_axis(ax, "y")
        _kappa_axis(ax, "x")
    handles = [plt.Line2D([], [], marker="o", lw=0, color=colour(r["kappa0"]), label=r["label"])
               for r in runs if r["name"] in series]
    handles += [plt.Line2D([], [], color=ps.c("#888888"), ls=styles[n % len(styles)],
                           label=rf"$\Omega_{{ci}} t = {t:g}$")
                for n, t in enumerate(sorted(drawn))]
    # Explicit margins: the kappa labels sit above the panels and the legend
    # below them, so neither may share space with an axis label.
    # One row per run and one per instant would stack: four columns keep the
    # legend to two rows, and the bottom margin is sized for them.
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.34, wspace=0.30)
    fig.legend(handles=handles, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Whole-box magnetic field against the measured kappa, per instant",
                 y=0.98)
    ps.save(fig, path)
    return path


def plot_trajectory(runs, series, path: Path) -> Path | None:
    usable = [r for r in runs if r["name"] in series]
    if not usable:
        return None
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.4))
    span = [min(series[r["name"]]["t"][0] for r in usable),
            max(series[r["name"]]["t"][-1] for r in usable)]
    norm = plt.Normalize(*span)
    mappable = None
    for ax, (key, label) in zip(axes, FIELDS):
        for run in usable:
            s = series[run["name"]]
            y = s[key]
            ok = np.isfinite(y) & np.isfinite(s["inv"])
            if ok.sum() < 2:
                continue
            points = np.column_stack([s["inv"][ok], y[ok]]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap="viridis", norm=norm, linewidths=1.8,
                                zorder=2)
            lc.set_array(s["t"][ok][:-1])
            mappable = ax.add_collection(lc)
            # Where the run starts and ends, and the end of the linear phase:
            # the track alone does not say which way time runs.
            ax.plot(s["inv"][ok][0], y[ok][0], marker="o", ms=5, lw=0,
                    color=colour(run["kappa0"]), zorder=3)
            ax.plot(s["inv"][ok][-1], y[ok][-1], marker="s", ms=5, lw=0,
                    color=colour(run["kappa0"]), zorder=3)
            t_lin = run.get("t_lin_end")
            if t_lin is not None and np.isfinite(t_lin):
                xi = np.interp(t_lin, s["t"][ok], s["inv"][ok])
                yi = np.interp(t_lin, s["t"][ok], y[ok])
                ax.plot(xi, yi, marker="*", ms=10, lw=0, color=colour(run["kappa0"]),
                        zorder=4)
        ax.set_xlabel(r"$1/\kappa$  measured, local-field frame")
        ax.set_ylabel(label)
        ax.autoscale_view()
        ps.style_axes(ax)
        ps.guard_degenerate_axis(ax, "x")
        ps.guard_degenerate_axis(ax, "y")
        _kappa_axis(ax, "x")
    fig.subplots_adjust(left=0.085, right=0.86, top=0.80, bottom=0.27, wspace=0.30)
    if mappable is not None:
        # Its own axes, outside the panels: a colour bar attached to them
        # steals width after the margins are set and lands on the tick labels.
        bar = fig.colorbar(mappable, cax=fig.add_axes([0.885, 0.27, 0.016, 0.53]))
        bar.set_label(r"$\Omega_{ci} t$")
    # The track itself is coloured by time, so the run colour lives only in the
    # markers: a coloured line in the legend would name a colour not on the canvas.
    handles = [plt.Line2D([], [], marker="o", lw=0, color=colour(r["kappa0"]), label=r["label"])
               for r in usable]
    handles += [plt.Line2D([], [], marker=m, lw=0, color=ps.c("#888888"), label=lab)
                for m, lab in (("o", "start"), ("*", "end of linear phase"), ("s", "end"))]
    fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)),
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Whole-box magnetic field against the measured kappa, along each run",
                 y=0.98)
    ps.save(fig, path)
    return path


def plot_single_run(run: dict, s: dict, path: Path, local: dict | None = None) -> Path | None:
    """One run, one panel: both series against time on a shared axis.

    The field is the left axis and the index the right one, so the order of
    events is read directly instead of from the colour of a track. When the
    index was also measured inside the magnetic holes and the peaks, those two
    go on the same axis with the band between them: that contrast, not the box
    average, is where "the magnitude falls and rises" is a statement about the
    field rather than about the square of the fluctuation.
    """
    y = s["dB_vec"]
    ok = np.isfinite(y) & np.isfinite(s["inv"])
    if ok.sum() < 2:
        return None
    ps.apply()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    run_colour = colour(run["kappa0"])

    ax.plot(s["t"], y, color=run_colour, lw=2.0, zorder=3,
            label=r"$\langle |\delta B|^2 \rangle^{1/2} / B_0$  (left)")
    # The magnitude itself, which is what is asked for when the question is about
    # |B| and not about its fluctuation. In these runs it is the second-order
    # signature of the same amplitude (see plot_magnitude_vs_fluctuation), so it
    # is drawn on the same axis to make that visible rather than hidden.
    if np.any(np.isfinite(s["B_mean"])):
        ax.plot(s["t"], s["B_mean"] - 1.0, color=run_colour, lw=1.4, ls="-.", zorder=2,
                label=r"$\langle |B| \rangle / B_0 - 1$  (left)")
    ax.set_xlabel(r"$\Omega_{ci} t$")
    ax.set_ylabel(r"field, in units of $B_0$", color=run_colour)
    ax.tick_params(axis="y", labelcolor=run_colour)

    kx = ax.twinx()
    grey = ps.c("#444444")
    kx.plot(s["t"], s["inv"], color=grey, lw=1.8, ls="--", zorder=3, label=r"$1/\kappa$  all (right)")
    kx.fill_between(s["t"], s["inv"] - s["inv_err"], s["inv"] + s["inv_err"],
                    color=grey, alpha=0.18, lw=0, zorder=1)
    if local:
        hole, peak = local.get("hole"), local.get("peak")
        if hole is not None and peak is not None:
            # Between the two is the hole - peak contrast of the index.
            kx.fill_between(local["t"], hole, peak, color=ps.c("#D55E00"), alpha=0.16,
                            lw=0, zorder=1, label="hole to peak")
            kx.plot(local["t"], hole, color=ps.c("#D55E00"), lw=1.2, ls="-", zorder=2,
                    label=r"$1/\kappa$  in holes")
            kx.plot(local["t"], peak, color=ps.c("#0072B2"), lw=1.2, ls=":", zorder=2,
                    label=r"$1/\kappa$  in peaks")
    kx.set_ylabel(r"$1/\kappa$  measured, local-field frame", color=grey)
    kx.tick_params(axis="y", labelcolor=grey)

    t_lin = run.get("t_lin_end")
    if t_lin is not None and np.isfinite(t_lin):
        ax.axvline(t_lin, color=ps.c("#888888"), lw=1.1, ls=":", zorder=1)
        # At the top: the magnitude curve runs along the bottom of the panel.
        ax.annotate("end of\nlinear phase", xy=(t_lin, 0.99), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", fontsize="small",
                    color=ps.c("#888888"), ha="left", va="top")
    ps.style_axes(ax)

    # kappa labels further right, on the same scale as the 1/kappa axis.
    lo, hi = kx.get_ylim()
    same = (lambda x: x, lambda x: x)
    right = kx.secondary_yaxis(1.17, functions=same)
    ticks = [(1.0 / k, f"{k:g}") for k in KAPPA_TICKS if min(lo, hi) <= 1.0 / k <= max(lo, hi)]
    if min(lo, hi) <= 0.0 <= max(lo, hi):
        ticks.append((0.0, r"$\infty$"))
    right.set_yticks([v for v, _ in ticks], labels=[t for _, t in ticks])
    right.set_ylabel(r"$\kappa$")
    right.tick_params(which="minor", right=False)

    handles = ax.get_legend_handles_labels()[0] + kx.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + kx.get_legend_handles_labels()[1]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"{run['label']} — field and index against time", y=0.97)
    fig.subplots_adjust(left=0.095, right=0.80, top=0.89, bottom=0.30)
    ps.save(fig, path)
    return path


def plot_magnitude_vs_fluctuation(runs, series, path: Path) -> Path | None:
    """<|B|>/B0 - 1 against <|dB|^2>/2B0^2, the second-order identity.

    This is the figure that answers "the magnitude, not the fluctuation" at box
    level: for a transverse-dominated fluctuation |B| - B0 ~ |dB_perp|^2 / 2 B0,
    so the box-averaged magnitude is not an independent measurement. A slope of
    one on this plot means the magnitude carries nothing the amplitude does not;
    departures from it are the compressive part that does.
    """
    usable = [r for r in runs if np.any(np.isfinite(series[r["name"]]["B_mean"]))]
    if not usable:
        return None
    ps.apply()
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    # Both axes are a few times 1e-2 at most: scale by hand and say so in the
    # label, rather than let an automatic offset appear. plot_style folds such
    # an offset into the label and redoes the layout, which would discard the
    # margins reserved for the legend below.
    SCALE, UNIT = 1e3, r"$\times 10^{-3}$"
    lo, hi = np.inf, -np.inf
    for run in usable:
        s = series[run["name"]]
        x = SCALE * 0.5 * s["dB_vec"] ** 2
        y = SCALE * (s["B_mean"] - 1.0)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            continue
        slope = float(np.polyfit(x[ok], y[ok], 1)[0])
        ax.plot(x[ok], y[ok], lw=2.0, color=colour(run["kappa0"]),
                label=f"{run['label']}  (slope {slope:.2f})")
        lo, hi = min(lo, x[ok].min()), max(hi, x[ok].max())
    if not np.isfinite(lo):
        plt.close(fig)
        return None
    ref = np.linspace(lo, hi, 2)
    ax.plot(ref, ref, ls=":", lw=1.3, color=ps.c("#888888"),
            label=r"$|B| - B_0 = |\delta B_\perp|^2 / 2 B_0$")
    ax.set_xlabel(r"$\langle |\delta B|^2 \rangle / 2 B_0^2$  (" + UNIT + ")")
    ax.set_ylabel(r"$\langle |B| \rangle / B_0 - 1$  (" + UNIT + ")")
    ps.style_axes(ax)
    # Below the panel, never over the curves: the sign of the slope decides
    # which corner the data occupies and no corner is reliably free.
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
               frameon=False, fontsize="small", bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Box-averaged field magnitude against the fluctuation amplitude", y=0.97)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.37)
    ps.save(fig, path)
    return path


def write_table(runs, series, path: Path) -> None:
    columns = ["run", "kappa_nominal", "omega_ci_t", "inv_kappa", "inv_kappa_err",
               "kappa_measured", "B_mean_over_B0", "B_std_over_B0",
               "delta_B_vec_rms_over_B0"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for run in runs:
            s = series.get(run["name"])
            if s is None:
                continue
            for i, t in enumerate(s["t"]):
                inv = s["inv"][i]
                writer.writerow([
                    run["name"], "" if run["kappa0"] is None else f"{run['kappa0']:g}",
                    f"{t:.8g}", f"{inv:.8g}", f"{s['inv_err'][i]:.8g}",
                    f"{1.0 / inv:.8g}" if inv > 0 else "inf",
                    *(f"{s[k][i]:.8g}" if np.isfinite(s[k][i]) else ""
                      for k in ("B_mean", "B_std", "dB_vec"))])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="run analysis roots")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--times", nargs="*", type=float, default=None,
                   help="instants in Omega_ci t (default: four the runs share)")
    p.add_argument("--per-run", action="store_true",
                   help="also one figure per run: its track and its two series in time")
    p.add_argument("--population", default="all",
                   help="particle population of the index series (default: all)")
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    runs, series = [], {}
    for root in a.runs:
        run = load_run(root)
        if run is None:
            print(f"[WARN] {root}: no kappa product (03_particles); skipped")
            continue
        if run.get("control"):
            # An isotropic control has no instability to relate the field to.
            print(f"[INFO] {run['name']}: isotropic control; skipped")
            continue
        paired = paired_series(run, a.population)
        if paired is None:
            print(f"[WARN] {run['name']}: index and field do not overlap; skipped")
            continue
        runs.append(run)
        series[run["name"]] = paired
    if len(runs) < 1:
        print("ERROR: no run with both an index and a field series")
        return 1

    times = a.times if a.times else default_times(runs, series)
    written = [plot_instants(runs, series, times, a.outdir / "field_vs_kappa_instants.png"),
               plot_trajectory(runs, series, a.outdir / "field_vs_kappa_trajectory.png")]
    if a.per_run:
        written += [plot_single_run(run, series[run["name"]],
                                    a.outdir / f"field_vs_kappa_{run['name']}.png",
                                    local_contrast(run, series[run["name"]]))
                    for run in runs]
    written.append(plot_magnitude_vs_fluctuation(
        runs, series, a.outdir / "field_magnitude_vs_fluctuation.png"))
    write_table(runs, series, a.outdir / "field_vs_kappa.csv")
    if not any(np.any(np.isfinite(s["B_mean"])) for s in series.values()):
        print("[WARN] no B_mean_over_B0 column: this tree predates the whole-box "
              "mean of |B|, so only the fluctuation panel has data")
    for path in filter(None, written):
        print(f"wrote {path}")
    print(f"wrote {a.outdir / 'field_vs_kappa.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
