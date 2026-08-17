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

import os
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt


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
    "#56d364": "#009E73",   # green
    "#2ecc71": "#009E73",
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


def legend(ax, **kwargs):
    """Legend with the theme's colours already filled in."""
    kwargs.setdefault("facecolor", LEGEND_BG)
    kwargs.setdefault("edgecolor", GRID_CLR)
    kwargs.setdefault("labelcolor", TEXT_CLR)
    return ax.legend(**kwargs)


def save(fig, path, pad_inches: float = 0.1, close: bool = True) -> None:
    """Write the figure to ``path``, plus a vector copy in the paper theme."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=pad_inches,
                facecolor=fig.get_facecolor())
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
