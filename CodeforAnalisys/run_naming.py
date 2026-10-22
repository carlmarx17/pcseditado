#!/usr/bin/env python3
"""
run_naming.py — one way to find and name the runs of a comparison
==================================================================

Three of the comparison scripts used to glob ``mirror_*`` and strip the
literal prefix ``mirror_``, so pointing them at a results root that holds the
firehose or whistler runs silently produced an empty figure (or, worse, one
built from whichever mirror runs happened to be in the same directory). The
case matrix has three instability families and the analysis has to work for
all of them.

A run directory is named ``<family>_<distribution>_<regime>``, e.g.
``mirror_bikappa3_moderate``, ``firehose_bimaxwellian_strong``. The
distribution is what distinguishes the columns of a comparison figure; the
family and the regime are what has to match across them.
"""

from __future__ import annotations

from pathlib import Path

#: Instability families of the case matrix (psc_<family>_<dist>_<regime>.cxx).
FAMILIES = ("mirror", "firehose", "whistler")

#: Regimes, stripped from the display name.
REGIMES = ("strong", "moderate", "weak", "isotropic")

#: Column order of a comparison: Maxwellian first, then by decreasing tail
#: weight. Anything unknown sorts last, alphabetically.
DISTRIBUTION_ORDER = ("bimaxwellian", "bikappa3", "bikappa5")

#: Display names of the distributions.
PRETTY = {"bimaxwellian": "bi-Maxwellian",
          "bikappa3": r"$\kappa = 3$",
          "bikappa5": r"$\kappa = 5$"}


def family_of(name: str) -> str | None:
    """'mirror_bikappa3_moderate' -> 'mirror'; None when it is not a case run."""
    for family in FAMILIES:
        if name == family or name.startswith(family + "_"):
            return family
    return None


def distribution_of(name: str) -> str:
    """The part of a run name that a comparison figure varies.

    'mirror_bikappa3_moderate' -> 'bikappa3'. Falls back to the whole name
    so an unrecognised directory still gets a stable, non-empty label.
    """
    stem = name
    family = family_of(stem)
    if family:
        stem = stem[len(family):].lstrip("_")
    for regime in REGIMES:
        if stem.endswith("_" + regime):
            stem = stem[: -len(regime) - 1]
        elif stem == regime:
            return name
    return stem or name


def regime_of(name: str) -> str | None:
    for regime in REGIMES:
        if name.endswith("_" + regime):
            return regime
    return None


def pretty(name: str) -> str:
    """Display label of a run directory."""
    return PRETTY.get(distribution_of(name), distribution_of(name))


def find_runs(root: Path, family: str | None = None,
              regime: str | None = None) -> list[Path]:
    """Case-run directories under ``root``, ordered for a comparison figure.

    With no ``family``, every family present is accepted -- but a root that
    mixes families is a mistake in a comparison figure, not something to
    silently average over, so the caller is expected to check
    ``families_in``. Ordering is by distribution (Maxwellian first), then by
    name, so the columns of a figure do not depend on the filesystem.
    """
    runs = [p for p in sorted(root.iterdir())
            if p.is_dir() and family_of(p.name)
            and (family is None or family_of(p.name) == family)
            and (regime is None or regime_of(p.name) == regime)]
    rank = {name: i for i, name in enumerate(DISTRIBUTION_ORDER)}
    return sorted(runs, key=lambda p: (rank.get(distribution_of(p.name), len(rank)),
                                       p.name))


def families_in(runs) -> set[str]:
    """The instability families present among ``runs``."""
    return {f for f in (family_of(Path(r).name) for r in runs) if f}
