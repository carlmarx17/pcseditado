#!/usr/bin/env python3
"""
plot_style.py — one figure style for the whole analysis pipeline
================================================================

Before this module every script picked its own look: `physical_diagnostics.py`
and `anisotropy_analysis.py` drew on the GitHub-dark background and saved it
into the PNG, `dispersion_analysis.py` used the matplotlib default, and
`prt_region_field_cut.py` / `vdf_spatial.py` forced a white face. Dropped into
the same chapter those read as figures made by different people.

Two themes, selected with the ``PSC_FIG_THEME`` environment variable:

``paper`` (default)
    White background, dark text, print-legible line weights, 300 dpi, and a
    PDF written next to every PNG. This is what goes into the thesis and the
    journal submission -- A&A, ApJ and JGR all want a white or transparent
    background, and line plots belong in vector form.

``dark``
    The previous GitHub-dark look, kept for screen and slides.

Series colours come from the Okabe-Ito colourblind-safe palette in ``paper``:
the old ones (#58a6ff, #f2cc60, #56d364) were chosen against a near-black
background and turn washed out and nearly unreadable on white.

Usage::

    import plot_style as ps
    ps.apply()                          # once, at import time of the script
    fig.patch.set_facecolor(ps.FIG_BG)
    ax.plot(t, y, color=ps.c("#58a6ff"))
    ps.save(fig, path)                  # PNG (+ PDF in paper theme)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


THEME = os.environ.get("PSC_FIG_THEME", "paper").strip().lower()
if THEME not in ("paper", "dark"):
    print(f"[WARN] PSC_FIG_THEME='{THEME}' unknown; falling back to 'paper'.")
    THEME = "paper"

IS_PAPER = THEME == "paper"

# ── Chrome ───────────────────────────────────────────────────────────────────
FIG_BG = "#ffffff" if IS_PAPER else "#0d1117"
PANEL_BG = "#ffffff" if IS_PAPER else "#161b22"
TEXT_CLR = "#111111" if IS_PAPER else "#e6edf3"
GRID_CLR = "#b8bcc2" if IS_PAPER else "#30363d"
LEGEND_BG = "#ffffff" if IS_PAPER else "#1c2128"
MUTED_CLR = "#4d4d4d" if IS_PAPER else "#8b949e"

DPI = 300 if IS_PAPER else 180
#: Extra vector copy for the thesis; PNG alone loses line quality when the
#: figure is scaled to a column width.
EXTRA_FORMATS: tuple[str, ...] = (".pdf",) if IS_PAPER else ()

# ── Series colours ───────────────────────────────────────────────────────────
# Keys are the dark-theme hexes already spread through the scripts, so call
# sites translate with c() instead of being rewritten one by one. Unknown
# colours pass through untouched -- c() must never silently invent a colour.
_PAPER_COLORS = {
    # GitHub-dark palette -> Okabe-Ito
    "#58a6ff": "#0072B2",   # blue
    "#a8d4ff": "#56B4E9",   # light blue
    "#74b9ff": "#56B4E9",
    "#0984e3": "#0072B2",
    "#1f77b4": "#0072B2",
    "#ff7b72": "#D55E00",   # salmon -> vermillion
    "#ff6b6b": "#D55E00",
    "#ff9999": "#E69F00",
    "#ffb4b4": "#E69F00",
    "#ff4444": "#D55E00",
    "#f85149": "#D55E00",
    "#d62728": "#D55E00",
    "#f0883e": "#E69F00",
    "#56d364": "#009E73",   # green
    "#2ecc71": "#009E73",
    "#3fb950": "#009E73",
    "#55efc4": "#009E73",
    "#00ff00": "#009E73",
    "#00c000": "#009E73",   # prt window box
    "#d2a8ff": "#CC79A7",   # purple
    "#c084fc": "#CC79A7",
    "#f2cc60": "#E69F00",   # yellow -> orange (yellow is invisible on white)
    "#ffd700": "#E69F00",
    "#f9c74f": "#E69F00",
    "#f1c40f": "#E69F00",
    "#ffa657": "#E69F00",
    # Chrome referenced inline rather than through the constants above
    "#0d1117": FIG_BG,
    "#161b22": PANEL_BG,
    "#1c2128": LEGEND_BG,
    "#21262d": PANEL_BG,
    "#30363d": GRID_CLR,
    "#e6edf3": TEXT_CLR,
    "#8b949e": MUTED_CLR,
    "#7f7f7f": MUTED_CLR,
    "#111111": TEXT_CLR,
}


def c(color: str) -> str:
    """Translate a dark-theme colour to the active theme."""
    if not IS_PAPER:
        return color
    return _PAPER_COLORS.get(str(color).lower(), color)


#: Sequential colormap for power/density maps. 'turbo'/'jet' are perceptually
#: non-uniform: they invent banding where the data is smooth and collapse in
#: greyscale print.
CMAP_SEQUENTIAL = "viridis"
#: Diverging maps must stay diverging in both themes -- RdBu_r reads on white.
CMAP_DIVERGING = "RdBu_r"


def apply() -> None:
    """Install the theme into matplotlib's rcParams. Call once per script."""
    matplotlib.use("Agg")
    plt.rcParams.update({
        "figure.facecolor": FIG_BG,
        "savefig.facecolor": FIG_BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRID_CLR,
        "axes.labelcolor": TEXT_CLR,
        "text.color": TEXT_CLR,
        "xtick.color": TEXT_CLR,
        "ytick.color": TEXT_CLR,
        "grid.color": GRID_CLR,
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.0 if IS_PAPER else 0.8,
        "lines.linewidth": 1.8,
        "legend.framealpha": 0.9 if IS_PAPER else 0.55,
        "legend.facecolor": LEGEND_BG,
        "legend.edgecolor": GRID_CLR,
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "mathtext.fontset": "dejavusans",
        # Inward ticks put the labels close to the frame; with the default
        # 3.5 pt pad the x = 0 and y = 0 labels of a map touch at the corner
        # (half a 12 pt label is ~7 pt, so the pad must exceed that).
        "xtick.major.pad": 8.0,
        "ytick.major.pad": 8.0,
        "xtick.minor.pad": 8.0,
        "ytick.minor.pad": 8.0,
        # Room between a title and the top tick label of the y axis.
        "axes.titlepad": 12.0,
        # An additive offset ("+2" above a colour bar) is easy to miss and
        # collides with titles; print the values themselves.
        "axes.formatter.useoffset": False,
        # Below 1e-3 (or from 1e4) the ticks carry a power of ten, which
        # save() moves into the axis label: "0.000075" becomes "0.75 (x10^-4)".
        "axes.formatter.limits": (-3, 4),
    })


def style_axes(ax, title: str = "") -> None:
    """Shared axis chrome: inward ticks on all four sides, faint grid."""
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(which="both", colors=TEXT_CLR, direction="in",
                   top=True, right=True)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(True, which="major", alpha=0.25 if IS_PAPER else 0.18,
            color=GRID_CLR, ls=":")
    if title:
        ax.set_title(title, color=TEXT_CLR, fontweight="bold")


def plain_log_axis(ax, which: str = "both") -> None:
    """Readable labels on a log axis whose limits are already set.

    Up to ~2.5 decades the labels sit at 1-2-5 values in plain numbers;
    beyond that only the decades are labelled, as powers of ten. Minor tick
    labels are always off: matplotlib's default labels them on short ranges
    and they collide ("3x10^0", "4x10^0", ...).
    """
    from matplotlib import ticker
    for name in ("x", "y") if which == "both" else (which,):
        if getattr(ax, f"get_{name}scale")() != "log":
            continue
        axis = getattr(ax, f"{name}axis")
        lo, hi = sorted(getattr(ax, f"get_{name}lim")())
        decades = np.log10(hi / lo) if lo > 0 else np.inf
        if decades <= 2.5:
            axis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=20))
            axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            axis.set_major_locator(ticker.LogLocator(base=10, numticks=8))
            axis.set_major_formatter(ticker.LogFormatterSciNotation())
        axis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2.0, 10.0), numticks=20))
        axis.set_minor_formatter(ticker.NullFormatter())


#: Axis labels of every spatial map: z (along B0) horizontal, y vertical.
Z_LABEL = r"$z\ [d_i]$  ($\parallel B_0$)"
Y_LABEL = r"$y\ [d_i]$"


def spatial_axes(ax, parallel_note: bool = True, **label_kw) -> None:
    """One convention for all maps of the yz plane.

    Arrays are stored (Nz, Ny), so ``imshow(a.T, origin="lower",
    extent=[z0, z1, y0, y1])`` puts z, the direction of B0, on the horizontal
    axis. The aspect is equal (a structure elongated along B0 must look
    elongated) and the ticks are integers of d_i every 5 (10 for a 40 d_i box).
    """
    from matplotlib import ticker
    ax.set_aspect("equal", adjustable="box")
    for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        span = abs(lim[1] - lim[0])
        step = next((s for s in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0) if span / s <= 6), 100.0)
        axis.set_major_locator(ticker.MultipleLocator(step))
        axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xlabel(Z_LABEL if parallel_note else r"$z\ [d_i]$", **label_kw)
    ax.set_ylabel(Y_LABEL, **label_kw)


def spaced_indices(ax, x, y, candidates, min_separation_pt: float = 30.0) -> list[int]:
    """Subset of `candidates` whose points are at least `min_separation_pt` apart on the page.

    For labels written next to the points of a trajectory: where it stalls,
    labels of consecutive times would otherwise be printed on top of each other.
    """
    pix = ax.transData.transform(np.column_stack([np.asarray(x, float), np.asarray(y, float)]))
    limit = min_separation_pt * ax.figure.dpi / 72.0
    kept = []
    for i in candidates:
        if np.all(np.isfinite(pix[i])) and all(np.hypot(*(pix[i] - pix[j])) >= limit for j in kept):
            kept.append(int(i))
    return kept


def legend(ax, **kwargs):
    """Legend with the theme's colours already filled in."""
    kwargs.setdefault("facecolor", LEGEND_BG)
    kwargs.setdefault("edgecolor", GRID_CLR)
    kwargs.setdefault("labelcolor", TEXT_CLR)
    return ax.legend(**kwargs)


# ── Figure content check ─────────────────────────────────────────────────────
# A saved PNG proves nothing about its content: the v5 comparison of A(t) was
# written with its legend and axes but no visible curve, because every sample
# was isolated between NaNs and a line without markers draws nothing. Every
# save therefore records what each panel actually shows, one JSON line per
# figure in figure_qa_<script>.jsonl next to it (quality_report.py reads them).

def _finite_rows(values) -> np.ndarray:
    try:
        a = np.ma.filled(np.ma.asarray(values, dtype=float), np.nan)
    except (TypeError, ValueError):
        return np.zeros(0, dtype=bool)
    if a.ndim == 0:
        a = a.reshape(1)
    return np.isfinite(a.reshape(len(a), -1)).all(axis=1) if a.size else np.zeros(0, dtype=bool)


def _line_state(line) -> dict:
    ok = _finite_rows(line.get_xydata())
    marker = line.get_marker() not in (None, "", " ", "None", "none")
    segments = int(np.count_nonzero(ok[1:] & ok[:-1])) if ok.size > 1 else 0
    return {"points": int(ok.size), "finite": int(ok.sum()), "segments": segments,
            "visible": bool(line.get_visible() and ((marker and ok.any()) or segments > 0))}


def _collection_finite(artist) -> tuple[int, int]:
    """(entries, finite entries) of a collection or image."""
    array = artist.get_array() if hasattr(artist, "get_array") else None
    if array is not None:
        a = np.ma.filled(np.ma.asarray(array, dtype=float), np.nan)
        return int(a.size), int(np.count_nonzero(np.isfinite(a)))
    offsets = artist.get_offsets() if hasattr(artist, "get_offsets") else None
    if offsets is not None and len(offsets) and not np.allclose(offsets, 0):
        ok = _finite_rows(offsets)
        return int(ok.size), int(ok.sum())
    rows = [_finite_rows(path.vertices) for path in artist.get_paths()]
    return sum(r.size for r in rows), sum(int(r.sum()) for r in rows)


def figure_content(fig) -> dict:
    """Per-panel inventory of drawn data and the problems a reader would see."""
    panels, issues = [], []
    for index, ax in enumerate(fig.get_axes()):
        if not ax.axison or not ax.get_visible() or getattr(ax, "_colorbar", None) is not None:
            continue            # tables, text-only, hidden panels and colorbars
        name = ax.get_title() or ax.get_ylabel() or f"axes {index}"
        lines = [(l.get_label(), _line_state(l)) for l in ax.get_lines()]
        shown = sum(s["visible"] for _, s in lines)
        for label, state in lines:
            # A labelled line without any point is a legend proxy, not data.
            if not label.startswith("_") and state["points"] and not state["visible"]:
                issues.append(f"{name}: series '{label}' draws nothing "
                              f"({state['finite']} finite points, {state['segments']} segments)")
        for artist in list(ax.collections) + list(ax.images):
            entries, finite = _collection_finite(artist)
            shown += finite > 0
            if entries and not finite and artist.get_visible():
                issues.append(f"{name}: {type(artist).__name__} has no finite values")
        shown += sum(bool(np.isfinite(getattr(p, "get_height", lambda: np.nan)()))
                     for p in ax.patches)
        if shown == 0 and not ax.texts:
            issues.append(f"{name}: empty panel")
        panels.append({"panel": name, "lines": len(lines), "visible_artists": int(shown),
                       "legend": ax.get_legend() is not None})
    return {"panels": panels, "issues": issues + layout_issues(fig)}


# ── Layout check ─────────────────────────────────────────────────────────────
# Overlapping text (titles, axis and tick labels, annotations, colour bars,
# legends) and legends or annotations drawn over data are the defects a
# reviewer notices first and a successful savefig never reports.

#: Overlap below this many pixels (at the figure's own dpi) is anti-aliasing.
_OVERLAP_PX = 1.5
#: A legend/annotation hides a series when it covers more than this fraction of it.
_COVERED_FRACTION = 0.01


def _ticks_in_view(axis, view) -> list:
    lo, hi = sorted(view)
    span = hi - lo
    labels = []
    for tick in axis.get_major_ticks() + axis.get_minor_ticks():
        loc = tick.get_loc()
        if loc is None or not np.isfinite(loc) or not (lo - 1e-9 * span <= loc <= hi + 1e-9 * span):
            continue
        labels += [lab for lab in (tick.label1, tick.label2) if lab.get_visible()]
    return labels


def _text_boxes(fig, renderer) -> list:
    boxes = []

    def add(artist, what):
        if artist is None or not artist.get_visible():
            return
        if hasattr(artist, "get_text") and not artist.get_text().strip():
            return
        try:
            box = artist.get_window_extent(renderer)
        except Exception:                          # noqa: BLE001 - never fail a save
            return
        if box.width > 0 and box.height > 0 and np.all(np.isfinite(box.extents)):
            boxes.append((what, box, artist))

    for index, ax in enumerate(fig.get_axes()):
        if not ax.get_visible():
            continue
        name = ("colour bar" if getattr(ax, "_colorbar", None) is not None
                else ax.get_title() or ax.get_ylabel() or f"axes {index}")
        for title in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
            add(title, f"{name}: title")
        if ax.axison:
            add(ax.xaxis.label, f"{name}: x label")
            add(ax.yaxis.label, f"{name}: y label")
            add(ax.xaxis.get_offset_text(), f"{name}: x offset")
            add(ax.yaxis.get_offset_text(), f"{name}: y offset")
            for label in _ticks_in_view(ax.xaxis, ax.get_xlim()):
                add(label, f"{name}: x tick '{label.get_text()}'")
            for label in _ticks_in_view(ax.yaxis, ax.get_ylim()):
                add(label, f"{name}: y tick '{label.get_text()}'")
        for text in ax.texts:
            add(text, f"{name}: text '{text.get_text()[:40]}'")
        add(ax.get_legend(), f"{name}: legend")
    for text in fig.texts:
        add(text, f"figure text '{text.get_text()[:40]}'")
    if getattr(fig, "_suptitle", None) not in fig.texts:
        add(getattr(fig, "_suptitle", None), "suptitle")
    for legend in fig.legends:
        add(legend, "figure legend")
    return boxes


def _samples_in_display(ax, line) -> np.ndarray:
    """Points of a data-coordinate line in pixels, densified along its segments."""
    xy = np.asarray(line.get_xydata(), dtype=float)
    xy = xy[_finite_rows(xy)] if xy.size else xy
    if len(xy) == 0:
        return np.zeros((0, 2))
    pix = ax.transData.transform(xy)
    if len(pix) > 1 and line.get_linestyle() not in ("None", "", " ", "none"):
        t = np.linspace(0, 1, 8, endpoint=False)[:, None, None]
        pix = (pix[:-1] + t * (pix[1:] - pix[:-1])).reshape(-1, 2)
    return pix


def _covered(box, pix) -> float:
    if not len(pix):
        return 0.0
    inside = ((pix[:, 0] > box.x0) & (pix[:, 0] < box.x1) & (pix[:, 1] > box.y0) & (pix[:, 1] < box.y1))
    return float(inside.mean())


def layout_issues(fig) -> list[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = _text_boxes(fig, renderer)
    issues, crowded = [], {}
    for i, (what_a, a, artist_a) in enumerate(boxes):
        for what_b, b, artist_b in boxes[i + 1:]:
            if artist_a is artist_b:
                continue
            dx = min(a.x1, b.x1) - max(a.x0, b.x0)
            dy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if dx <= _OVERLAP_PX or dy <= _OVERLAP_PX:
                continue
            axis_a, axis_b = what_a.split(" tick ")[0], what_b.split(" tick ")[0]
            if " tick " in what_a and axis_a == axis_b:
                crowded[axis_a] = crowded.get(axis_a, 0) + 1     # same axis: one issue
            else:
                issues.append(f"overlap: {what_a} / {what_b}")
    issues += [f"crowded tick labels: {axis} ({n} overlapping pairs)" for axis, n in crowded.items()]
    for ax in fig.get_axes():
        if not ax.get_visible() or getattr(ax, "_colorbar", None) is not None:
            continue
        name = ax.get_title() or ax.get_ylabel() or "axes"
        covers = [(f"legend", ax.get_legend())] + [(f"text '{t.get_text()[:40]}'", t) for t in ax.texts]
        for what, artist in covers:
            if artist is None or not artist.get_visible() or \
                    (hasattr(artist, "get_text") and not artist.get_text().strip()):
                continue
            box = artist.get_window_extent(renderer)
            for line in ax.get_lines():
                # Reference lines (axhline/axvline) live in blended coordinates
                # and cross the whole panel by design; only data lines count.
                if not line.get_visible() or line.get_transform() != ax.transData:
                    continue
                # A label written on its own faint dotted guide is intended.
                if what != "legend" and (line.get_linestyle() == ":" or (line.get_alpha() or 1) < 0.5):
                    continue
                if _covered(box, _samples_in_display(ax, line)) > _COVERED_FRACTION:
                    label = line.get_label()
                    series = f"'{label}'" if not label.startswith("_") else "a data line"
                    issues.append(f"{name}: {what} covers {series}")
                    break
            for coll in ax.collections:
                offsets = coll.get_offsets() if hasattr(coll, "get_offsets") else None
                if offsets is None or len(offsets) < 2 or coll.get_offset_transform() != ax.transData:
                    continue
                pix = ax.transData.transform(np.asarray(offsets, float)[_finite_rows(offsets)])
                if _covered(box, pix) > _COVERED_FRACTION:
                    issues.append(f"{name}: {what} covers scattered points")
                    break
    return issues


def _record_content(fig, path: Path) -> None:
    try:
        entry = {"file": path.name, **figure_content(fig)}
    except Exception as exc:            # never lose a figure over its own check
        entry = {"file": path.name, "panels": [], "issues": [], "qa_error": repr(exc)}
    name = sys.argv[0] if sys.argv else ""
    script = Path(name).stem if name not in ("", "-", "-c") else "interactive"
    with (path.parent / f"figure_qa_{script}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _fold_colorbar_offsets(fig) -> None:
    """Move a colour bar's power of ten from the floating offset into its label.

    The "1e-5" matplotlib prints above a colour bar is easy to miss and sits
    where the panel title is; "J [code units] (x10^-5)" cannot be misread.
    """
    from matplotlib import ticker
    fig.canvas.draw()
    folded = False
    for ax in fig.get_axes():
        cb = getattr(ax, "_colorbar", None)
        axes = [ax.yaxis if cb.orientation == "vertical" else ax.xaxis] if cb is not None \
            else [ax.xaxis, ax.yaxis]
        for axis in axes:
            if not axis.get_offset_text().get_text().strip():
                continue
            # The power of ten the formatter factored out: that of the largest
            # tick in view (ScalarFormatter's own rule).
            lo, hi = sorted(axis.get_view_interval())
            locs = np.abs([v for v in axis.get_majorticklocs() if lo <= v <= hi and v != 0])
            order = int(np.floor(np.log10(locs.max()))) if locs.size else 0
            if not order:
                continue
            scale = 10.0 ** order
            formatter = ticker.FuncFormatter(lambda v, _, s=scale: f"{v / s:g}")
            label = axis.label
            factor = rf"($\times 10^{{{order}}}$)"
            text = f"{label.get_text()}  {factor}" if label.get_text() else factor
            if cb is not None:
                cb.formatter = formatter
                cb.update_ticks()
                cb.set_label(text, fontsize=label.get_fontsize(), color=label.get_color())
            else:
                axis.set_major_formatter(formatter)
                label.set_text(text)
            folded = True
    # A longer label needs the layout redone; constrained/tight engines redo
    # it at every draw, a figure without one does not.
    if folded and fig.get_layout_engine() is None:
        fig.tight_layout()


def save(fig, path, pad_inches: float = 0.1, close: bool = True) -> None:
    """Write the figure to ``path``, plus a vector copy in the paper theme."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _fold_colorbar_offsets(fig)
    except Exception:                               # noqa: BLE001 - never lose a figure
        pass
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=pad_inches,
                facecolor=fig.get_facecolor())
    _record_content(fig, path)
    for suffix in EXTRA_FORMATS:
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight",
                    pad_inches=pad_inches, facecolor=fig.get_facecolor())
    if close:
        plt.close(fig)


def save_many(fig, paths: Iterable[Path], pad_inches: float = 0.1) -> None:
    """Same figure under several names (aliases kept for older references)."""
    paths = list(paths)
    for path in paths:
        save(fig, path, pad_inches=pad_inches, close=False)
    plt.close(fig)
