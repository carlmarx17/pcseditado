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

which stays positive only while b > b_crit = 1 - 1/A0, i.e. the structure
depth is bounded by a_max = 1/A0' ... = 1 - 1/A0 for A0 > 1. Consequence
(the anchor of the paper): purely adiabatic physics cannot change kappa —
any measured kappa_eff(b) trend is a signature of how the trapped domain is
filled, or of non-adiabatic dynamics.

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

Temporal reading: empty -> own -> flat as the structure ages.

All moments (n, T_par, T_perp, trapped fraction) are quadratures on a
(v_par, v_perp) grid, and kappa_eff uses the SAME truncated whitened
estimator as the simulation data (`kappa_eff.kappa_eff_from_grid`), so
theory and PIC are processed by one estimator — that symmetry is part of
the paper's method.

Usage:
    python liouville_kappa.py --self-test
    python liouville_kappa.py --figure --kappa0 3 --A0 2.0
    python liouville_kappa.py --figure          # defaults from psc_units
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import kappa_eff as ke

__all__ = [
    "b_crit", "theta_perp_eff2", "b_profile_sech2",
    "reference_vdf", "mapped_passing_vdf", "closed_vdf",
    "closure_moments", "closure_profiles", "CLOSURES",
]

CLOSURES = ("empty", "flat", "own")


# ── Closed-form pieces ───────────────────────────────────────────────────────

def b_crit(A0: float) -> float:
    """Deepest field level the adiabatic mapping supports: b > 1 - 1/A0.

    Equivalently a_max = 1 - 1/A0 for A0 > 1; for A0 <= 1 (firehose side)
    the mapping is defined at every depth.
    """
    return 1.0 - 1.0 / A0 if A0 > 1.0 else 0.0


def theta_perp_eff2(b, A0: float, theta_perp2: float = 1.0):
    """theta_perp_eff^2(b) of the mapped passing branch (invariant kappa)."""
    b = np.asarray(b, dtype=float)
    denom = 1.0 - A0 * (1.0 - b)
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
    if numeric:
        vperp0_2 = VU**2 / b
        vpar0_2 = VP**2 + VU**2 * (1.0 - 1.0 / b)
        f = _bikappa(vpar0_2, vperp0_2 / A0, kappa0)
    else:
        th2 = theta_perp_eff2(b, A0, theta_perp2=A0)
        if not np.all(np.isfinite(th2)):
            return np.full_like(VP, np.nan)
        f = _bikappa(VP**2, VU**2 / th2, kappa0)
    v2 = VP**2 + VU**2
    with np.errstate(invalid="ignore", divide="ignore"):
        sin2a = np.where(v2 > 0, VU**2 / v2, 0.0)
    return np.where(sin2a <= min(b, 1.0), f, 0.0)


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

    th2 = float(theta_perp_eff2(b, A0, theta_perp2=A0))
    if not np.isfinite(th2):
        return np.full_like(VP, np.nan)

    if closure == "flat":
        # continue f from the trapping boundary, flat along v_par:
        # boundary at fixed v_perp sits at v_par_b^2 = v_perp^2 (1 - b)/b
        vparb2 = VU**2 * (1.0 - b) / b
        f_t = _bikappa(vparb2, VU**2 / th2, kappa0)
    else:                                   # "own"
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
    fig.suptitle(
        rf"Adiabatic Liouville closures — {kap_txt}, $A_0 = {A0:g}$, "
        rf"$a_{{\max}} = {max(bc, 0):.3f}$".replace("a_{\\max}", "b_{\\rm crit}"),
        y=0.97)
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
    args = p.parse_args()

    if args.self_test:
        print("liouville_kappa self-test:")
        passed = self_test()
        print("PASSED" if passed else "FAILED")
        return 0 if passed else 1

    if args.figure:
        kappa0, A0 = args.kappa0, args.A0
        if kappa0 is None or A0 is None:
            from psc_units import BETA_I_PERP_OVER_PAR, KAPPA
            if kappa0 is None:
                kappa0 = KAPPA
            if A0 is None:
                A0 = BETA_I_PERP_OVER_PAR
        if kappa0 is not None and kappa0 <= 0:
            kappa0 = None                            # Maxwellian
        out = make_figure(kappa0, A0, args.kappa_t, Path(args.outdir),
                          args.s_max)
        print(f"Figura: {out}")
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
