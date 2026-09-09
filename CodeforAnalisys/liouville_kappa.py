#!/usr/bin/env python3
"""
liouville_kappa.py — adiabatic Liouville mapping of a bi-kappa along B(x)
=========================================================================
Kinetic theory companion to the b-binned profiles of `vdf_spatial.py`
(paper fig. 2 vs fig. 7). Everything is phrased in b = B(x)/B_ref.

The theorem
-----------
With mu = v_perp^2/(2B) and E conserved, a particle observed at field level b
with (v_par, v_perp) came from the reference field (b = 1) with

    v_perp0^2 = v_perp^2 / b ,
    v_par0^2  = v_par^2 + v_perp^2 (1 - 1/b) ,

provided it *connects* to the reference: sin^2(alpha) <= b (passing).
Liouville then gives f(b; v) = f0(v0). For a bi-kappa reference

    f0 ~ [1 + (v_par^2/theta_par^2 + v_perp^2/theta_perp^2)/kappa]^-(kappa+1)

the exponent closes algebraically: on the passing domain the distribution is
*again* a bi-kappa with the SAME kappa and theta_par, and only

    theta_perp_eff^2(b) = theta_perp^2 * b / [1 - A0 (1 - b)] ,
    A0 = theta_perp^2/theta_par^2 = T_perp0/T_par0 ,

which stays positive only while b > b_crit = 1 - 1/A0. This restricts the
full bi-Kappa extension used by the 'own' closure to a=1-b < 1/A0; it is
not a universal bound on magnetic-hole depth. The passing distribution
remains defined below b_crit. Its algebraic exponent is unchanged, but a
truncated or mixed-distribution kappa_eff estimator need not be invariant.

The three trapped-domain closures
---------------------------------
The trapped domain sin^2(alpha) > b is not populated by the mapping; a
closure must fill it:

  * ``empty``      — f = 0 (freshly formed structure; nothing scattered in).
  * ``flat``       — continuity: at fixed v_perp, f is continued from the
                     trapping boundary flat along v_par (phase mixing along
                     the bounce motion).
  * ``own``        — the trapped domain carries its own bi-kappa population
                     with index kappa_t (scattering has thermalised it),
                     same theta's as the mapped passing branch, amplitude
                     matched to the passing branch at v = 0. With
                     kappa_t = kappa0 this is the seamless filling and
                     kappa_eff(b) = kappa0 exactly.

These are alternative closures; their temporal ordering is not predicted.

All moments (n, T_par, T_perp, trapped fraction) are quadratures on a
(v_par, v_perp) grid, and kappa_eff uses the SAME truncated whitened
estimator as the simulation data (`kappa_eff.kappa_eff_from_grid`), so
theory and PIC are processed by one estimator — that symmetry is part of
the paper's method.

In space instead of in b
------------------------
`--figure-x` re-labels the same closures along the model structure
B(x)/B0 = 1 - a sech^2(x/L). It is a re-labelling and not new physics — the
moments depend on x only through b(x) — but it is what shows how much of the
interesting range of b a structure of a given depth actually samples, and it
makes the own-closure domain visible: it needs b(0) = 1 - a > b_crit, i.e.
a < a_max(A0) = 1/A0. The other closures only need b > 0. `--sweep` adds a summary of
kappa_eff at the structure centre against depth.

A result worth stating: n, A and the trapped fraction come out the same for a
bi-kappa and a bi-Maxwellian to ~0.1%, because both are set by the cone
geometry and the theta_perp_eff mapping, neither of which involves kappa.
Of the four profiles only kappa_eff separates the two reference VDFs — which
is why the measurement in `vdf_spatial.py` is built around it.

Usage:
    python liouville_kappa.py --self-test
    python liouville_kappa.py --figure --kappa0 3 --A0 2.0
    python liouville_kappa.py --figure          # defaults from psc_units
    python liouville_kappa.py --figure-x --kappa0 3 --A0 2.0 --a 0.25
    python liouville_kappa.py --sweep --kappa-list 3 5 0 --A0-list 2 --a-list 0.1 0.25 0.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import kappa_eff as ke

__all__ = [
    "b_crit", "a_max", "theta_perp_eff2", "b_profile_sech2",
    "reference_vdf", "mapped_passing_vdf", "closed_vdf",
    "closure_moments", "closure_profiles", "closure_profiles_x", "CLOSURES",
]

CLOSURES = ("empty", "flat", "own")


# ── Closed-form pieces ───────────────────────────────────────────────────────

def b_crit(A0: float) -> float:
    """Lower b limit of the full bi-Kappa extension used by the own closure.

    Equivalently a_max = 1/A0 for A0 > 1. The restricted passing branch
    and flat closure are defined for every b > 0.
    """
    return 1.0 - 1.0 / A0 if A0 > 1.0 else 0.0


def theta_perp_eff2(b, A0: float, theta_perp2: float = 1.0):
    """theta_perp_eff^2(b) of the mapped passing branch (invariant kappa)."""
    b = np.asarray(b, dtype=float)
    denom = 1.0 - A0 * (1.0 - b)
    # np.where evaluates both branches, so at b == b_crit exactly the division
    # is still performed and warns before being discarded. The guard is the
    # where(), not the division.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0.0, theta_perp2 * b / denom, np.nan)


def b_profile_sech2(x, a: float, L: float = 1.0):
    """The model magnetic structure B(x)/B0 = 1 - a sech^2(x/L)."""
    x = np.asarray(x, dtype=float)
    return 1.0 - a / np.cosh(x / L) ** 2


# ── Distributions on a (v_par, v_perp) grid ──────────────────────────────────

def _bikappa(vp2_over_thpar2, vu2_over_thperp2, kappa0):
    """Unnormalised bi-kappa (or bi-Maxwellian when kappa0 is None)."""
    arg = vp2_over_thpar2 + vu2_over_thperp2
    if kappa0 is None:
        return np.exp(-arg)
    return (1.0 + arg / kappa0) ** (-(kappa0 + 1.0))


def reference_vdf(VP, VU, kappa0, A0: float):
    """f0 at b = 1, theta_par = 1, theta_perp^2 = A0 (velocity in theta_par units)."""
    return _bikappa(VP**2, VU**2 / A0, kappa0)


def mapped_passing_vdf(VP, VU, b: float, kappa0, A0: float,
                       numeric: bool = False):
    """Liouville-mapped distribution on the passing domain (0 elsewhere).

    `numeric=True` evaluates f0(v0(v)) through the raw mapping instead of
    the closed form — used by the self-test to *prove* the closed form.
    """
    if not np.isfinite(b) or b <= 0 or not np.isfinite(A0) or A0 <= 0:
        raise ValueError("b and A0 must be positive and finite")
    v2 = VP**2 + VU**2
    with np.errstate(invalid="ignore", divide="ignore"):
        sin2a = np.where(v2 > 0, VU**2 / v2, 0.0)
    passing = sin2a <= min(b, 1.0)
    if numeric:
        vperp0_2 = VU**2 / b
        vpar0_2 = VP**2 + VU**2 * (1.0 - 1.0 / b)
        argument = np.maximum(vpar0_2, 0.0) + vperp0_2 / A0
    else:
        argument = VP**2 + VU**2 * (1.0 - A0 * (1.0 - b)) / (A0 * b)
    # Only evaluate the mapped VDF on its physical support. The extension
    # outside the passing cone may have a negative quadratic coefficient.
    f = _bikappa(np.where(passing, np.maximum(argument, 0.0), 0.0), 0.0, kappa0)
    return np.where(passing, f, 0.0)


def closed_vdf(VP, VU, b: float, kappa0, A0: float, closure: str,
               kappa_t=None):
    """Passing branch + the chosen trapped-domain closure."""
    if closure not in CLOSURES:
        raise ValueError(f"closure must be one of {CLOSURES}")
    f = mapped_passing_vdf(VP, VU, b, kappa0, A0)
    if b >= 1.0 or closure == "empty":
        return f

    v2 = VP**2 + VU**2
    with np.errstate(invalid="ignore", divide="ignore"):
        sin2a = np.where(v2 > 0, VU**2 / v2, 0.0)
    trapped = sin2a > b

    if closure == "flat":
        # At the passing boundary v_parallel0=0 and v_perp0^2=v_perp^2/b.
        f_t = _bikappa(0.0, VU**2 / (b * A0), kappa0)
    else:                                   # "own"
        th2 = float(theta_perp_eff2(b, A0, theta_perp2=A0))
        if not np.isfinite(th2):
            return np.full_like(VP, np.nan)
        kt = kappa0 if kappa_t is None else kappa_t
        f_t = _bikappa(VP**2, VU**2 / th2, kt)
        # amplitude continuity at v = 0 is automatic: both forms equal 1
        # there in these unnormalised units (same n0 * C0 prefactor).
    return np.where(trapped, f_t, f)


# ── Moments ──────────────────────────────────────────────────────────────────

def _grid(kappa0, A0: float, n: int = 1400, v_max: float | None = None):
    if v_max is None:
        # kappa tails need reach; T integrand decays like v^(-2 kappa0 + 2)
        v_max = 40.0 if kappa0 is not None else 8.0
    v_max *= max(1.0, np.sqrt(A0))
    v_par = np.linspace(-v_max, v_max, 2 * n)
    v_perp = np.linspace(0.0, v_max, n + 1)[1:]      # skip zero-Jacobian node
    VP, VU = np.meshgrid(v_par, v_perp, indexing="ij")
    return VP, VU


def closure_moments(b: float, kappa0, A0: float, closure: str,
                    kappa_t=None, s_max: float = ke.DEFAULT_S_MAX,
                    grid=None) -> dict:
    """n, T_par, T_perp, A, trapped fraction and kappa_eff of the closed f."""
    VP, VU = grid if grid is not None else _grid(kappa0, A0)
    f = closed_vdf(VP, VU, b, kappa0, A0, closure, kappa_t)
    if not np.all(np.isfinite(f)):
        return {k: float("nan") for k in
                ("n", "T_par", "T_perp", "A", "trapped_fraction",
                 "kappa_eff")} | {"b": b}
    wgt = f * VU                                     # 2 pi v_perp Jacobian
    n = float(np.sum(wgt))
    if n <= 0:
        return {k: float("nan") for k in
                ("n", "T_par", "T_perp", "A", "trapped_fraction",
                 "kappa_eff")} | {"b": b}
    t_par = float(np.sum(wgt * VP**2) / n)
    t_perp = float(0.5 * np.sum(wgt * VU**2) / n)

    v2 = VP**2 + VU**2
    with np.errstate(invalid="ignore", divide="ignore"):
        sin2a = np.where(v2 > 0, VU**2 / v2, 0.0)
    trap_frac = float(np.sum(wgt[sin2a > min(b, 1.0)]) / n)

    kap = ke.kappa_eff_from_grid(VP, VU, wgt, s_max=s_max)["kappa"]
    return {"b": b, "n": n, "T_par": t_par, "T_perp": t_perp,
            "A": t_perp / t_par, "trapped_fraction": trap_frac,
            "kappa_eff": kap}


def closure_profiles(b_values, kappa0, A0: float, closure: str,
                     kappa_t=None, s_max: float = ke.DEFAULT_S_MAX) -> dict:
    """`closure_moments` swept over b, normalised to the b = 1 reference."""
    grid = _grid(kappa0, A0)
    ref = closure_moments(1.0, kappa0, A0, closure, kappa_t, s_max, grid)
    rows = [closure_moments(float(b), kappa0, A0, closure, kappa_t, s_max,
                            grid) for b in b_values]
    out = {"b": np.asarray(b_values, dtype=float),
           "closure": closure, "kappa0": kappa0, "A0": A0,
           "b_crit": b_crit(A0)}
    for key in ("n", "T_par", "T_perp", "A", "trapped_fraction", "kappa_eff"):
        out[key] = np.array([r[key] for r in rows])
    out["n_over_n0"] = out["n"] / ref["n"]
    out["A_over_A0"] = out["A"] / ref["A"]
    return out


# ── Profiles along the model structure B(x) ──────────────────────────────────
#
# closure_profiles() is parametrised by b, which is what the PIC data is binned
# by (vdf_spatial.py) and therefore the right variable for the comparison. But
# the structure the thesis actually draws is a depression in space, and its
# depth `a` is a free parameter that b-space hides: every (a, x) with the same
# b gives the same moments, so a profile in x is a *re-labelling* of a profile
# in b, not new physics. It is still the figure the reader needs, because it is
# the one that shows how much of the interesting range of b a structure of a
# given depth actually samples.

def a_max(A0: float) -> float:
    """Depth limit of the own-closure bi-Kappa extension, not of PIC holes.

    The floor of the profile is b_min = 1 - a and the own closure needs
    b > b_crit = 1 - 1/A0, so a < 1/A0 for A0 > 1. For A0 <= 1 the only bound
    is b > 0, i.e. a < 1.
    """
    return 1.0 / A0 if A0 > 1.0 else 1.0


def closure_profiles_x(x, a: float, kappa0, A0: float, closure: str,
                       L: float = 1.0, kappa_t=None,
                       s_max: float = ke.DEFAULT_S_MAX,
                       progress=None) -> dict:
    """`closure_profiles` along B(x)/B0 = 1 - a sech^2(x/L).

    b(x) is even, so each quadrature is done once per distinct b and reused on
    the mirror point: a symmetric profile costs half its point count. Points
    below b_crit(A0) return NaN only for the own closure. Empty and flat
    closures retain their passing-cone support for b > 0.
    """
    x = np.asarray(x, dtype=float)
    b = b_profile_sech2(x, a, L)
    grid = _grid(kappa0, A0)
    bc = b_crit(A0)
    ref = closure_moments(1.0, kappa0, A0, closure, kappa_t, s_max, grid)

    keys = ("n", "T_par", "T_perp", "A", "trapped_fraction", "kappa_eff")
    cache: dict[float, dict] = {}
    rows = []
    for bi in b:
        key = round(float(bi), 12)
        if key not in cache:
            if bi > 0.0 and (closure != "own" or bi > bc):
                cache[key] = closure_moments(float(bi), kappa0, A0, closure,
                                             kappa_t, s_max, grid)
            else:
                cache[key] = {k: float("nan") for k in keys} | {"b": float(bi)}
            if progress is not None:
                progress(len(cache))
        rows.append(cache[key])

    out = {"x": x, "b": b, "a": a, "L": L, "closure": closure,
           "kappa0": kappa0, "A0": A0, "b_crit": bc, "a_max": a_max(A0)}
    for key in keys:
        out[key] = np.array([r[key] for r in rows])
    out["n_over_n0"] = out["n"] / ref["n"]
    out["A_over_A0"] = out["A"] / ref["A"]
    return out


def write_x_csv(profiles: dict[str, dict], path: Path) -> Path:
    """One row per (closure, x); the figure's numbers, reusable."""
    import csv

    cols = ("x", "b", "n_over_n0", "A", "A_over_A0", "T_par", "T_perp",
            "trapped_fraction", "kappa_eff")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["closure", "kappa0", "A0", "a", "L"] + list(cols))
        for closure, prof in profiles.items():
            kap = "inf" if prof["kappa0"] is None else f"{prof['kappa0']:g}"
            for i in range(len(prof["x"])):
                w.writerow([closure, kap, f"{prof['A0']:g}", f"{prof['a']:g}",
                            f"{prof['L']:g}"] +
                           [f"{prof[c][i]:.8g}" for c in cols])
    return path


def make_x_figure(kappa0, A0: float, a: float, kappa_t, outdir: Path,
                  s_max: float, n_x: int = 41, x_max: float = 3.0,
                  L: float = 1.0, quiet: bool = False) -> tuple[Path, Path]:
    """kappa_eff(x), n(x), A(x) and trapped fraction(x) for the three closures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()

    x = np.linspace(-x_max, x_max, n_x)
    colors = {"empty": ps.c("#1f77b4"), "own": ps.c("#9467bd"),
              "flat": ps.c("#d62728")}
    labels = {"empty": "case 1: empty",
              "own": (r"case 3: own $\kappa_t$"
                      + (rf" = {kappa_t:g}" if kappa_t is not None else "")),
              "flat": r"case 2: flat in $v_\parallel$"}

    profs, prof_max = {}, {}
    for c in CLOSURES:
        if not quiet:
            print(f"    closure {c} ...", flush=True)
        profs[c] = closure_profiles_x(x, a, kappa0, A0, c, L=L,
                                      kappa_t=kappa_t, s_max=s_max)
        prof_max[c] = closure_profiles_x(x, a, None, A0, c, L=L,
                                         kappa_t=None, s_max=s_max)

    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.5), sharex=True)
    fig.subplots_adjust(hspace=0.14, wspace=0.30)
    b_x = b_profile_sech2(x, a, L)

    # (a) the structure itself, with the depth bound marked
    ax = axes[0, 0]
    ax.plot(x, b_x, color=ps.c("#111111"), lw=2.2)
    bc = b_crit(A0)
    if bc > 0:
        ax.axhline(bc, color="0.5", lw=1.2, ls="-.",
                   label=rf"$b_{{\rm crit}} = {bc:.3f}$")
        ax.legend(framealpha=0.9, fontsize=11)
    ax.set_ylabel(r"$b = B(x)/B_0$")
    ax.set_title(rf"structure: $1 - a\,{{\rm sech}}^2(x/L)$, $a = {a:g}$")

    panels = ((axes[0, 1], "n_over_n0", r"$n / n_0$", "Density"),
              (axes[0, 2], "A", r"$A = T_\perp/T_\parallel$", "Anisotropy"),
              (axes[1, 0], "trapped_fraction", "trapped fraction",
               r"Trapped domain ($\sin^2\alpha > b$)"),
              (axes[1, 1], "kappa_eff", r"$\kappa_{\rm eff}$ (truncated moments)",
               "Spectral index"))
    for ax, key, ylabel, title in panels:
        for c in CLOSURES:
            ax.plot(x, profs[c][key], color=colors[c], lw=2.0, label=labels[c])
            if key != "kappa_eff":
                ax.plot(x, prof_max[c][key], color=colors[c], lw=1.1, ls=":",
                        alpha=0.8)
        if key == "kappa_eff" and kappa0 is not None:
            ax.axhline(kappa0, color="0.4", lw=1.0, ls="--",
                       label=rf"$\kappa_0 = {kappa0:g}$ (invariance)")
            ax.set_ylim(0.0, 3.0 * kappa0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0, 1].legend(framealpha=0.9, fontsize=10.5)
    axes[1, 1].legend(framealpha=0.9, fontsize=10.5)

    # (f) the parameters, spelled out
    ax = axes[1, 2]
    ax.axis("off")
    # n, A and the trapped fraction are set by the cone geometry plus the
    # theta_perp_eff mapping, and neither depends on kappa: the dotted curves
    # fall on top of the solid ones. The maximum separation is reported rather
    # than leaving it looking as if the dotted curves were never drawn — and
    # that coincidence is exactly the argument that kappa_eff is the only one
    # of the four observables separating a bi-kappa from a bi-Maxwellian.
    gap = max(
        float(np.nanmax(np.abs(profs[c][k] - prof_max[c][k])))
        for c in CLOSURES for k in ("n_over_n0", "A", "trapped_fraction"))
    kap_txt = "bi-Maxwellian" if kappa0 is None else rf"$\kappa_0 = {kappa0:g}$"
    lines = [kap_txt, rf"$A_0 = {A0:g}$", rf"$a = {a:g}$   "
             rf"($a_{{\max}} = {a_max(A0):.3f}$)", rf"$L = {L:g}$",
             rf"$b_{{\rm crit}} = {bc:.4f}$",
             rf"$b(0) = {1.0 - a:.4f}$", "",
             rf"$s_{{\max}} = {s_max:g}$ (kappa estimator)",
             rf"{n_x} points, $|x| \leq {x_max:g}\,L$", "",
             "solid: bi-kappa", "dotted: bi-Maxwellian",
             rf"max separation: {gap:.1e}", "",
             r"$n$, $A$ and the trapped fraction",
             r"do not depend on $\kappa$;",
             r"only $\kappa_{\rm eff}$ does."]
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, va="top",
            ha="left", fontsize=13)

    for ax in axes[1]:
        ax.set_xlabel(r"$x / L$")
    fig.suptitle(
        rf"Liouville closures along the structure — {kap_txt}, "
        rf"$A_0 = {A0:g}$, $a = {a:g}$", y=0.97)

    tag = (f"kappa{'inf' if kappa0 is None else f'{kappa0:g}'}"
           f"_A{A0:g}_a{a:g}").replace(".", "p")
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"liouville_profiles_x_{tag}.png"
    ps.save(fig, png)
    csv_path = write_x_csv(profs, outdir / f"liouville_profiles_x_{tag}.csv")
    return png, csv_path


def run_sweep(kappa_list, A0_list, a_list, kappa_t, outdir: Path, s_max: float,
              n_x: int, x_max: float, L: float) -> list[Path]:
    """One x-profile figure per (kappa0, A0, a), plus a centre-value summary.

    Require 0 <= a < 1 to keep B positive. The own closure is masked where
    its full-distribution extension is undefined; empty/flat remain available.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()

    written: list[Path] = []
    centre: dict[tuple, dict] = {}
    for kappa0 in kappa_list:
        for A0 in A0_list:
            for a in a_list:
                if not 0 <= a < 1:
                    print(f"  skip kappa0={kappa0}, A0={A0}, a={a}: "
                          "require 0 <= a < 1 for positive magnetic field")
                    continue
                print(f"  kappa0={kappa0}, A0={A0}, a={a}", flush=True)
                png, csv_path = make_x_figure(kappa0, A0, a, kappa_t, outdir,
                                              s_max, n_x=n_x, x_max=x_max, L=L,
                                              quiet=True)
                written += [png, csv_path]
                # the structure centre is where the closure shows most
                for c in CLOSURES:
                    m = closure_moments(1.0 - a, kappa0, A0, c, kappa_t, s_max)
                    centre.setdefault((kappa0, A0), {}).setdefault(c, []) \
                          .append((a, m["kappa_eff"]))

    if not centre:
        return written

    combos = sorted(centre, key=lambda t: (float("inf") if t[0] is None
                                           else t[0], t[1]))
    ncols = min(3, len(combos))
    nrows = int(np.ceil(len(combos) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.8 * nrows),
                             squeeze=False)
    colors = {"empty": ps.c("#1f77b4"), "own": ps.c("#9467bd"),
              "flat": ps.c("#d62728")}
    for ax, combo in zip(axes.ravel(), combos):
        kappa0, A0 = combo
        for c in CLOSURES:
            pts = sorted(centre[combo][c])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                    ms=5, lw=2.0, color=colors[c], label=c)
        if kappa0 is not None:
            ax.axhline(kappa0, color="0.4", lw=1.0, ls="--",
                       label=rf"$\kappa_0 = {kappa0:g}$")
            ax.set_ylim(0.0, 3.0 * kappa0)
        ax.axvline(a_max(A0), color="0.5", lw=1.0, ls="-.")
        ax.set_xlabel(r"structure depth $a$")
        ax.set_ylabel(r"$\kappa_{\rm eff}$ at $x = 0$")
        kap_txt = "bi-Maxwellian" if kappa0 is None else rf"$\kappa_0={kappa0:g}$"
        ax.set_title(rf"{kap_txt}, $A_0 = {A0:g}$")
        ax.legend(framealpha=0.9, fontsize=10.5)
    for ax in axes.ravel()[len(combos):]:
        ax.axis("off")
    fig.suptitle(r"$\kappa_{\rm eff}$ at the structure centre vs depth "
                 r"(dash-dot: $a_{\max}$)", y=0.99)
    summary = outdir / "liouville_sweep_centre_kappa.png"
    ps.save(fig, summary)
    written.append(summary)
    return written


# ── Self-test ────────────────────────────────────────────────────────────────

def self_test(verbose: bool = True) -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  {name}: {'OK' if cond else 'FAIL'}")

    kappa0, A0 = 3.0, 2.0
    VP, VU = _grid(kappa0, A0, n=700)

    # 1. The kappa-invariance theorem: raw Liouville mapping == closed form
    #    (same kappa, same theta_par, theta_perp renormalised), point-wise.
    for b in (0.6, 0.75, 0.9):
        f_num = mapped_passing_vdf(VP, VU, b, kappa0, A0, numeric=True)
        f_clo = mapped_passing_vdf(VP, VU, b, kappa0, A0)
        scale = f_num.max()
        check(f"closed form == Liouville map at b={b}",
              np.nanmax(np.abs(f_num - f_clo)) / scale < 1e-12)

    # 2. Depth bound: mapping must fail exactly below b_crit = 1 - 1/A0.
    check("theta_perp_eff^2 finite above b_crit",
          np.isfinite(theta_perp_eff2(b_crit(A0) + 0.01, A0)))
    check("theta_perp_eff^2 undefined below b_crit",
          not np.isfinite(theta_perp_eff2(b_crit(A0) - 0.01, A0)))

    # 3. b = 1 recovers the reference for every closure.
    for closure in CLOSURES:
        m = closure_moments(1.0, kappa0, A0, closure)
        check(f"b=1 reference recovered ({closure}): A -> A0",
              abs(m["A"] - A0) < 0.02 * A0)
        check(f"b=1 reference recovered ({closure}): kappa_eff -> kappa0",
              abs(m["kappa_eff"] - kappa0) < 0.1)

    # 4. Seamless filling keeps kappa invariant at every depth (the anchor).
    prof = closure_profiles(np.linspace(0.55, 0.95, 5), kappa0, A0, "own")
    check("kappa_eff(b) == kappa0 for seamless 'own' closure",
          np.nanmax(np.abs(prof["kappa_eff"] - kappa0)) < 0.1)

    # 5. Density ordering: the flat fill pins the cone interior at the
    #    boundary value, which is the *minimum* of the seamless fill there,
    #    so empty <= flat <= own (with kappa_t = kappa0).
    b_test = 0.7
    n = {c: closure_moments(b_test, kappa0, A0, c)["n"] for c in CLOSURES}
    check("n ordering empty <= flat <= own",
          n["empty"] <= n["flat"] * (1 + 1e-9) <= n["own"] * (1 + 1e-9))

    # 6. Maxwellian limit runs through the same machinery. At b = 1 there is
    #    no trapped domain and kappa_eff must be Maxwellian-consistent; at
    #    b < 1 an empty cone makes even a Maxwellian report finite kappa_eff
    #    — that closure signature IS the paper's observable, so we assert it
    #    exists rather than pretend it should not.
    m1 = closure_moments(1.0, None, A0, "empty")
    check("Maxwellian at b=1: kappa_eff Maxwellian-consistent",
          (not np.isfinite(m1["kappa_eff"])) or m1["kappa_eff"] > 20.0)
    m08 = closure_moments(0.8, None, A0, "empty")
    check("Maxwellian at b=0.8, empty cone: closure leaves a kappa_eff mark",
          np.isfinite(m08["kappa_eff"]))

    # 7. The x-profiles are a re-labelling of the b-profiles, so they must
    #    agree point-wise with closure_profiles evaluated at the same b.
    x = np.array([-1.0, 0.0, 1.0])
    a = 0.3
    px = closure_profiles_x(x, a, kappa0, A0, "flat")
    pb = closure_profiles(b_profile_sech2(x, a), kappa0, A0, "flat")
    check("x-profile == b-profile at the same b",
          np.nanmax(np.abs(px["kappa_eff"] - pb["kappa_eff"])) < 1e-9 and
          np.nanmax(np.abs(px["n_over_n0"] - pb["n_over_n0"])) < 1e-9)
    check("x-profile is even in x",
          abs(px["kappa_eff"][0] - px["kappa_eff"][2]) < 1e-12)

    # 8. Shallow structure -> the reference, at every x. This is the limit
    #    that catches a normalisation error in the b = 1 reference.
    shallow = closure_profiles_x(np.linspace(-2, 2, 5), 1e-3, kappa0, A0, "own")
    check("a -> 0: n/n0 -> 1", np.nanmax(np.abs(shallow["n_over_n0"] - 1)) < 0.01)
    check("a -> 0: A -> A0", np.nanmax(np.abs(shallow["A"] - A0)) < 0.02 * A0)

    # 9. The depth bound, as the sweep will hit it: a just under a_max is
    #    defined at the floor, a just over it is not.
    am = a_max(A0)
    check("a_max consistent with b_crit", abs((1.0 - am) - b_crit(A0)) < 1e-12)
    deep = closure_profiles_x(np.array([0.0]), am + 0.01, kappa0, A0, "own")
    check("a > a_max: own extension is undefined", not np.isfinite(deep["A"][0]))
    deep_passing = closed_vdf(VP, VU, 0.3, kappa0, A0, "empty")
    check("passing VDF exists below b_crit", np.all(np.isfinite(deep_passing)))
    return ok


# ── Figure (paper fig. 2) ────────────────────────────────────────────────────

def make_figure(kappa0, A0: float, kappa_t, outdir: Path,
                s_max: float) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()

    bc = b_crit(A0)
    b = np.linspace(max(bc + 0.02, 0.5), 1.0, 26)
    colors = {"empty": ps.c("#1f77b4"), "own": ps.c("#9467bd"),
              "flat": ps.c("#d62728")}
    labels = {"empty": "case 1: empty",
              "own": (r"case 3: own $\kappa_t$"
                      + (rf" = {kappa_t:g}" if kappa_t is not None else "")),
              "flat": r"case 2: flat in $v_\parallel$"}

    profs = {c: closure_profiles(b, kappa0, A0, c, kappa_t=kappa_t,
                                 s_max=s_max) for c in CLOSURES}
    prof_max = {c: closure_profiles(b, None, A0, c, kappa_t=None,
                                    s_max=s_max) for c in CLOSURES}

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), sharex=True)
    fig.subplots_adjust(hspace=0.12, wspace=0.28)

    panels = (("n_over_n0", r"$n / n_0$", False),
              ("A", r"$A = T_\perp/T_\parallel$", False),
              ("trapped_fraction", "trapped fraction", False),
              ("kappa_eff", r"$\kappa_{\rm eff}$ (truncated moments)", True))
    for ax, (key, ylabel, is_kappa) in zip(axes.ravel(), panels):
        for c in CLOSURES:
            ax.plot(b, profs[c][key], color=colors[c], lw=2.0,
                    label=labels[c])
            if not is_kappa:
                ax.plot(b, prof_max[c][key], color=colors[c], lw=1.1,
                        ls=":", alpha=0.8)
        if is_kappa and kappa0 is not None:
            ax.axhline(kappa0, color="0.4", lw=1.0, ls="--",
                       label=rf"$\kappa_0 = {kappa0:g}$ (invariance)")
            ax.set_ylim(0.0, 3.0 * kappa0)
        if bc > 0:
            ax.axvline(bc, color="0.5", lw=1.0, ls="-.")
        ax.set_ylabel(ylabel)
    for ax in axes[1]:
        ax.set_xlabel(r"$b = B / B_0$")
    axes[0, 0].legend(framealpha=0.9, fontsize=10)
    axes[0, 0].set_title("solid: bi-kappa   dotted: bi-Maxwellian",
                         fontsize=11)

    kap_txt = rf"$\kappa_0 = {kappa0:g}$" if kappa0 is not None \
        else "bi-Maxwellian"
    # b_crit and a_max = 1 - b_crit are different numbers and label different
    # axes; this used to print one and rename it to the other with a string
    # replace. Both are stated, against the variable each one belongs to.
    fig.suptitle(
        rf"Adiabatic Liouville closures — {kap_txt}, $A_0 = {A0:g}$, "
        rf"$b_{{\rm crit}} = {max(bc, 0):.3f}$ "
        rf"($a_{{\max}} = {a_max(A0):.3f}$, own closure only)", y=0.97)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "liouville_closures.png"
    ps.save(fig, out)
    plt.close(fig)
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Adiabatic Liouville mapping of a bi-kappa and the "
                    "three trapped-domain closures.")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--figure", action="store_true")
    p.add_argument("--kappa0", type=float, default=None,
                   help="reference kappa (omit: psc_units profile; "
                        "0 = Maxwellian)")
    p.add_argument("--A0", type=float, default=None,
                   help="reference T_perp/T_par (omit: psc_units profile)")
    p.add_argument("--kappa-t", type=float, default=None,
                   help="trapped-population kappa for the 'own' closure "
                        "(default: kappa0, the seamless case)")
    p.add_argument("--s-max", type=float, default=ke.DEFAULT_S_MAX)
    p.add_argument("--outdir", default="liouville_theory")
    p.add_argument("--figure-x", action="store_true",
                   help="profiles along B(x) = B0 [1 - a sech^2(x/L)]")
    p.add_argument("--sweep", action="store_true",
                   help="one --figure-x per (kappa, A0, a) combination")
    p.add_argument("--a", type=float, default=None,
                   help="structure depth (default: half of a_max(A0))")
    p.add_argument("--L", type=float, default=1.0,
                   help="structure width; x is reported in units of L")
    p.add_argument("--n-x", type=int, default=41,
                   help="points along x. Each distinct b costs one quadrature "
                        "(~1 s), and b(x) is even, so the cost is n_x/2 per "
                        "closure")
    p.add_argument("--x-max", type=float, default=3.0,
                   help="half-extent of the x axis, in units of L")
    p.add_argument("--kappa-list", type=float, nargs="+",
                   help="--sweep: kappa values (0 = bi-Maxwellian)")
    p.add_argument("--A0-list", type=float, nargs="+",
                   help="--sweep: T_perp/T_par values")
    p.add_argument("--a-list", type=float, nargs="+",
                   help="--sweep: structure depths")
    args = p.parse_args()

    if args.self_test:
        print("liouville_kappa self-test:")
        passed = self_test()
        print("PASSED" if passed else "FAILED")
        return 0 if passed else 1

    if args.figure or args.figure_x or args.sweep:
        kappa0, A0 = args.kappa0, args.A0
        if kappa0 is None or A0 is None:
            from psc_units import BETA_I_PERP_OVER_PAR, KAPPA
            if kappa0 is None:
                kappa0 = KAPPA
            if A0 is None:
                A0 = BETA_I_PERP_OVER_PAR
        if kappa0 is not None and kappa0 <= 0:
            kappa0 = None                            # Maxwellian
        outdir = Path(args.outdir)

        if args.figure:
            out = make_figure(kappa0, A0, args.kappa_t, outdir, args.s_max)
            print(f"Figure: {out}")

        if args.figure_x:
            a = args.a if args.a is not None else 0.5 * a_max(A0)
            if not 0 <= a < 1:
                print(f"ERROR: a = {a}; require 0 <= a < 1 for positive magnetic field.")
                return 1
            png, csv_path = make_x_figure(kappa0, A0, a, args.kappa_t, outdir,
                                          args.s_max, n_x=args.n_x,
                                          x_max=args.x_max, L=args.L)
            print(f"x profiles: {png}\n            {csv_path}")

        if args.sweep:
            kappa_list = args.kappa_list if args.kappa_list else [kappa0]
            kappa_list = [None if k is not None and k <= 0 else k
                          for k in kappa_list]
            A0_list = args.A0_list if args.A0_list else [A0]
            a_list = args.a_list if args.a_list else \
                [0.25 * a_max(A0), 0.5 * a_max(A0), 0.75 * a_max(A0)]
            written = run_sweep(kappa_list, A0_list, a_list, args.kappa_t,
                                outdir, args.s_max, args.n_x, args.x_max,
                                args.L)
            print(f"Sweep: {len(written)} files in {outdir}")
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
