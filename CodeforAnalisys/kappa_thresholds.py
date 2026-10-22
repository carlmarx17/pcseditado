#!/usr/bin/env python3
"""
kappa_thresholds.py — instability thresholds in the bi-Kappa convention
=======================================================================

The marginal-stability condition of a temperature-anisotropy instability is
quoted as a fit to a contour of constant growth rate::

    A - 1 = S / beta_par^alpha

and the values in circulation (EMIC: S ~ 0.43, alpha ~ 0.42) are
*bi-Maxwellian* fits. Two separate things stop them from being used on a
bi-Kappa run, and they are routinely confused with each other:

1. **The beta axis.** A bi-Kappa has two temperatures in the literature. This
   code, and ``SetupParticles::createKappaMultivariate``, fix the *second
   moment*: <v^2> = T/m whatever kappa is (verified to 1e-3 by sampling, and
   measured as beta_parallel_median = 4.95-4.99 in the three v6 mirror runs
   whose nominal beta is 5). Much of the literature instead fixes the core,
   theta^2 = 2T/m, so its true second-moment temperature is

       T_true = kappa/(kappa - 3/2) * T_nominal

   a factor 2.000 at kappa = 3 and 1.429 at kappa = 5. A paper's "kappa = 3,
   beta = 5" is then this code's beta = 10, and comparing the two growth rates
   as if they were the same plasma inverts the sign of the kappa effect (see
   ``main``'s printed comparison).

   Moya et al., A&A 707, A78 (2026), arXiv:2601.13888, is the core-normalised
   case, confirmed from the text: its Eq. (4) sets theta^2 = 2 k_B T / m with
   no kappa factor, and the kinetic temperature follows as T^(kappa) = F T
   with F = kappa/(kappa - 3/2) (its Eq. 6). Its beta_par(0) = 5 is therefore
   this code's beta_par = 10 at kappa = 3. Going the other way, these runs'
   true beta_par = 5 sits at its beta_par(0) = 2.5 (kappa = 3) or 3.5
   (kappa = 5), which is where a comparison with its figures has to be read.

   That paper is **not** a like-for-like comparison for these runs in any
   case, and the difference is not the convention: it is quasi-linear theory
   (kappa evolved through a kurtosis closure, its Eqs. 15-16 and 19, with
   G(kappa) = [kappa/(kappa-3/2)][kappa/(kappa-5/2)]) for the *electron*-
   cyclotron EMEC instability driven by anisotropic bi-Kappa electrons over
   cold ions. These runs are PIC, the anisotropy and the measured kappa are
   on the *ions* (A_e = 1), and the dominant mode is the ion-cyclotron
   branch. Its suprathermal electrons gain energy from the waves and kappa
   falls; the ion tails here drain and kappa rises. Those are different
   populations under different instabilities, so neither result tests the
   other.

2. **The fit itself.** Even at matched true beta, S and alpha depend on kappa,
   because the contour is set by resonant damping and the kappa tail changes
   its shape. The Maxwellian S and alpha cannot be reused.

This module computes both, with the same solver (``linear_theory``) and the
same convention as the runs, so the thesis quotes its own thresholds instead
of borrowing incompatible ones.

Caveat worth keeping in the text: the three v6 mirror runs share
(beta_par = 5, A = 2) and therefore sit at *different distances above their
own thresholds*. "Kappa suppresses the growth rate" is a statement at matched
macroscopic state, not at matched linear drive; ``distance_above_threshold``
reports the former so the figure can say which it is.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from linear_theory import ParallelDispersion

#: mi/me and c/vA of the mirror runs (psc_units of the active profile).
MASS_RATIO = 200.0
C_OVER_VA = 176.8

#: Growth-rate contour that defines "marginal". 1e-2 is the one a PIC run can
#: actually resolve in a reasonable box; the literature's 1e-3 is available
#: with --levels but costs several times more, because every bisection step
#: near marginal drives the kappa dispersion integral to its iteration limit.
LEVELS = (1e-2,)

#: beta_parallel grid of the threshold scan.
BETAS = (1.0, 2.0, 5.0, 10.0, 20.0)

#: Distributions, in the fixed figure order: Maxwellian, kappa 5, kappa 3.
KAPPAS = (None, 5.0, 3.0)


def label_of(kappa: float | None) -> str:
    return "bi-Maxwellian" if kappa is None else f"kappa={kappa:g}"


def core_to_second_moment(kappa: float | None) -> float:
    """T_true/T_nominal for a core-normalised bi-Kappa; 1 for this code's own."""
    if kappa is None:
        return 1.0
    if kappa <= 1.5:
        raise ValueError("A finite-temperature bi-Kappa requires kappa > 1.5")
    return kappa / (kappa - 1.5)


def gamma_max(beta_par: float, anisotropy: float, kappa: float | None,
              k_lo: float = 0.05, k_hi: float = 0.85, dk: float = 0.05) -> float:
    """max_k Im(omega)/Omega_ci on the parallel 'plus' (EMIC) branch.

    Several seeds per k, keeping the largest converged growth rate. Following
    one root from the previous k alone is not enough: with a coarse grid the
    secant iteration slides onto the damped branch and stays there, and the
    whole scan then reports gamma < 0 for a plasma that is unstable (the
    bi-Maxwellian column came out NaN at every beta >= 2 that way).
    """
    solver = ParallelDispersion(beta_par, anisotropy, 1.0, 1.0,
                                MASS_RATIO, C_OVER_VA, kappa=kappa)
    best = 0.0
    dragged = None
    for k in np.arange(k_lo, k_hi, dk):
        seeds = [dragged, complex(0.3 * k, 0.05), complex(0.8 * k, 0.05)]
        local = None
        for seed in seeds:
            omega = solver.solve(k, "plus", seed)
            if np.isfinite(omega) and (local is None or np.imag(omega) > np.imag(local)):
                local = omega
        if local is not None:
            dragged = local
            best = max(best, float(np.imag(local)))
    return best


def threshold_anisotropy(beta_par: float, kappa: float | None, level: float,
                         lo: float = 1.001, hi: float = 8.0,
                         iterations: int = 12) -> float:
    """A at which gamma_max equals ``level``; NaN when ``hi`` is still stable."""
    if gamma_max(beta_par, hi, kappa) < level:
        return float("nan")
    if gamma_max(beta_par, lo, kappa) >= level:
        return float("nan")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if gamma_max(beta_par, mid, kappa) < level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_threshold(betas, thresholds) -> tuple[float, float, float]:
    """(S, alpha, r_squared) of A - 1 = S / beta^alpha, fitted in logs."""
    betas = np.asarray(betas, dtype=float)
    y = np.asarray(thresholds, dtype=float) - 1.0
    ok = np.isfinite(y) & (y > 0) & np.isfinite(betas) & (betas > 0)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    x, z = np.log(betas[ok]), np.log(y[ok])
    slope, intercept = np.polyfit(x, z, 1)
    residual = z - (slope * x + intercept)
    total = float(np.sum((z - z.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / total if total > 0 else float("nan")
    return float(math.exp(intercept)), float(-slope), r2


def distance_above_threshold(beta_par: float, anisotropy: float,
                             kappa: float | None, level: float) -> float:
    """(A - 1)/(A_threshold - 1): how far above marginal a run sits.

    The three v6 mirror runs share (beta, A) but not this number, which is why
    "kappa lowers gamma" has to be stated at matched macroscopic state.
    """
    threshold = threshold_anisotropy(beta_par, kappa, level)
    if not np.isfinite(threshold) or threshold <= 1.0:
        return float("nan")
    return (anisotropy - 1.0) / (threshold - 1.0)


def scan(levels=LEVELS, betas=BETAS, kappas=KAPPAS) -> list[dict]:
    rows = []
    for level in levels:
        for kappa in kappas:
            values = [threshold_anisotropy(b, kappa, level) for b in betas]
            S, alpha, r2 = fit_threshold(betas, values)
            for beta, value in zip(betas, values):
                rows.append({"level": level, "kappa": "inf" if kappa is None else kappa,
                             "distribution": label_of(kappa), "beta_parallel": beta,
                             "A_threshold": value, "S": S, "alpha": alpha,
                             "fit_r_squared": r2})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None,
                        help="directory for kappa_thresholds.csv")
    parser.add_argument("--beta", type=float, default=5.0,
                        help="beta_parallel of the runs to place (default 5)")
    parser.add_argument("--anisotropy", type=float, default=2.0,
                        help="A of the runs to place (default 2)")
    parser.add_argument("--levels", type=float, nargs="+", default=list(LEVELS),
                        help="growth-rate contours that define marginal")
    args = parser.parse_args()

    levels = tuple(args.levels)
    rows = scan(levels=levels)
    for level in levels:
        print(f"=== gamma_max/Omega_ci = {level:g} (EMIC, parallel, "
              f"second-moment convention)")
        header = f"{'beta_par':>9}" + "".join(f"{label_of(k):>16}" for k in KAPPAS)
        print(header)
        for beta in BETAS:
            line = f"{beta:>9.1f}"
            for kappa in KAPPAS:
                value = next(r["A_threshold"] for r in rows
                             if r["level"] == level and r["beta_parallel"] == beta
                             and r["distribution"] == label_of(kappa))
                line += f"{value:>16.4f}" if np.isfinite(value) else f"{'--':>16}"
            print(line)
        print(f"   A - 1 = S / beta^alpha")
        for kappa in KAPPAS:
            row = next(r for r in rows if r["level"] == level
                       and r["distribution"] == label_of(kappa))
            print(f"     {label_of(kappa):<16} S={row['S']:.3f}  "
                  f"alpha={row['alpha']:.3f}  R2={row['fit_r_squared']:.4f}")
        print(f"   distance above threshold at beta={args.beta:g}, A={args.anisotropy:g}:")
        for kappa in KAPPAS:
            d = distance_above_threshold(args.beta, args.anisotropy, kappa, level)
            print(f"     {label_of(kappa):<16} (A-1)/(A_thr-1) = {d:.3f}")
        print()

    print("=== convention shift: the same label read in the core-normalised convention")
    print(f"{'distribution':>16} {'T_true/T_nom':>14} {'gamma_max ours':>16} {'gamma_max theirs':>18}")
    for kappa in KAPPAS:
        factor = core_to_second_moment(kappa)
        ours = gamma_max(args.beta, args.anisotropy, kappa)
        theirs = gamma_max(args.beta * factor, args.anisotropy, kappa)
        print(f"{label_of(kappa):>16} {factor:>14.3f} {ours:>16.4f} {theirs:>18.4f}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "kappa_thresholds.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
