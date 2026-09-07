#!/usr/bin/env python3
"""
kappa_eff.py — moment-based estimator of the kappa index, truncated and whitened
================================================================================
Estimates the spectral index kappa of a (bi-)kappa velocity distribution from
weighted particle samples, using the kurtosis-type ratio

    K = <s^4> / <s^2>^2 ,        s^2 = v_par^2/sigma_par^2
                                     + v_perp1^2/sigma_perp^2
                                     + v_perp2^2/sigma_perp^2 ,

where sigma_j^2 are the *measured* per-component variances (drift subtracted).
For an untruncated isotropic kappa distribution

    K(kappa) = (5/3) (kappa - 3/2) / (kappa - 5/2)
    <=>  kappa = (5/2) (K - 1) / (K - 5/3) ,

with the Maxwellian limit K -> 5/3 as kappa -> inf.

Two physics traps this module is built around
--------------------------------------------
1. **Anisotropy aliasing.** The speed kurtosis of a *bi-Maxwellian* with
   A != 1 differs from 5/3, so a naive estimator applied to |v| reports a
   spurious finite kappa driven purely by temperature anisotropy. The fix is
   whitening: a bi-kappa in the variables s_j = v_j / theta_j is an
   *isotropic* kappa, and a bi-Maxwellian whitens to an isotropic Maxwellian.
   Whitening by the measured component variances (any fixed multiple of
   theta_j) leaves K invariant, so kappa and anisotropy decouple exactly.
2. **Divergent moments.** <v^4> exists only for kappa > 5/2, and the sample
   variance of the K estimator requires <v^8>, i.e. kappa > 9/2. For the
   kappa = 3 runs the untruncated estimator is formally undefined /
   pathologically noisy. The estimator therefore truncates at s <= s_max and
   inverts the *truncated* relation K_t(kappa, s_max), computed by quadrature.
   The truncation also mimics the finite energy range of a real instrument
   (MMS/FPI), which is part of the point: the same estimator applies to
   theory, simulation and data.

Conventions
-----------
The bi-kappa follows the PSC loader (Abdul & Mace construction):

    f(v_par, v_perp) ~ [1 + v_par^2/(kappa theta_par^2)
                          + v_perp^2/(kappa theta_perp^2)]^(-(kappa+1)) ,

with per-component variance <v_j^2> = kappa theta_j^2 / (2 kappa - 3)
(finite for kappa > 3/2) and T_j = m theta_j^2/2 * kappa/(kappa - 3/2).
Sampling rule (shared radial gamma variate — the multivariate construction,
NOT three independent 1-D kappas):

    g ~ chi^2_(2 kappa - 1),  z_j ~ N(0,1),  v_j = u_j + sqrt(kappa) theta_j z_j / sqrt(g).

All estimators accept particle weights `w` and use effective sample sizes
n_eff = (sum w)^2 / sum w^2 for error estimates.

Usage:
    import kappa_eff
    res = kappa_eff.kappa_eff_from_velocities(v_par, v_p1, v_p2, w)
    res["kappa"], res["kappa_err"], res["K"], res["n_eff"]

    python kappa_eff.py --self-test
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import integrate, optimize

__all__ = [
    "KAPPA_MIN", "KAPPA_MAX", "DEFAULT_S_MAX",
    "untruncated_K", "kappa_from_K_untruncated",
    "truncated_K_kappa", "truncated_K_maxwellian", "kappa_from_K",
    "whiten", "kappa_eff_from_velocities", "kappa_eff_from_grid",
    "sample_bikappa", "sample_bimaxwellian",
]

# Inversion domain. Below KAPPA_MIN the distribution barely has a second
# moment (needs kappa > 3/2) and the whitening variance is itself unreliable;
# above KAPPA_MAX the truncated K is Maxwellian to well below sampling noise.
KAPPA_MIN = 1.6
KAPPA_MAX = 200.0

# Default truncation radius in whitened units. For a Maxwellian the mass
# beyond s = 6 is ~1e-8 (negligible bias), while for kappa = 3 the tail
# between the core and s_max still carries the kappa signature:
# K_t(3, 6) ~ 3.0 vs K_t(inf, 6) = 5/3.
DEFAULT_S_MAX = 6.0

_TINY = 1e-300


# ── Untruncated relations (kappa > 5/2 only) ─────────────────────────────────

def untruncated_K(kappa: float) -> float:
    """K = <s^4>/<s^2>^2 of an untruncated isotropic kappa distribution."""
    kappa = float(kappa)
    if kappa <= 2.5:
        return float("inf")
    return (5.0 / 3.0) * (kappa - 1.5) / (kappa - 2.5)


def kappa_from_K_untruncated(K: float) -> float:
    """Inverse of `untruncated_K`; inf for K <= 5/3 (Maxwellian-consistent)."""
    if K <= 5.0 / 3.0:
        return float("inf")
    return 2.5 * (K - 1.0) / (K - 5.0 / 3.0)


# ── Truncated theory curve ───────────────────────────────────────────────────

def _kappa_radial_pdf(kappa: float):
    """Unnormalised radial density s^2 f(s) of the whitened isotropic kappa.

    Whitened = unit variance per component, so theta^2 = (2 kappa - 3)/kappa.
    """
    theta2 = (2.0 * kappa - 3.0) / kappa

    def pdf(s):
        return (1.0 + s * s / (kappa * theta2)) ** (-(kappa + 1.0))

    return pdf


@lru_cache(maxsize=4096)
def truncated_K_kappa(kappa: float, s_max: float) -> float:
    """K_t = <s^4>_t / <s^2>_t^2 of a whitened isotropic kappa cut at s_max.

    NOTE: the moments are taken over the *truncated* distribution but the
    whitening scale is the *untruncated* unit variance — exactly matching the
    estimator, which whitens with the full-sample variance and then cuts.
    """
    kappa, s_max = float(kappa), float(s_max)
    pdf = _kappa_radial_pdf(kappa)
    m = [integrate.quad(lambda s, n=n: s ** n * pdf(s), 0.0, s_max,
                        limit=200)[0] for n in (2, 4, 6)]
    s2 = m[1] / m[0]
    s4 = m[2] / m[0]
    return s4 / (s2 * s2)


@lru_cache(maxsize=256)
def truncated_K_maxwellian(s_max: float) -> float:
    """Maxwellian limit of `truncated_K_kappa` (unit variance per component)."""
    s_max = float(s_max)
    m = [integrate.quad(lambda s, n=n: s ** n * np.exp(-0.5 * s * s),
                        0.0, s_max, limit=200)[0] for n in (2, 4, 6)]
    return (m[2] / m[0]) / (m[1] / m[0]) ** 2


def kappa_from_K(K: float, s_max: float = DEFAULT_S_MAX) -> float:
    """Invert the truncated relation K_t(kappa; s_max) -> kappa.

    Returns inf when K is at or below the Maxwellian value (no resolvable
    suprathermal tail) and KAPPA_MIN when K exceeds the hardest tail the
    inversion domain allows (caller should treat that as a saturated bound).
    """
    if not np.isfinite(K):
        return float("nan")
    k_max_val = truncated_K_kappa(KAPPA_MAX, s_max)   # ~ Maxwellian
    k_min_val = truncated_K_kappa(KAPPA_MIN, s_max)   # hardest tail
    if K <= k_max_val:
        return float("inf")
    if K >= k_min_val:
        return KAPPA_MIN
    return float(optimize.brentq(
        lambda kap: truncated_K_kappa(kap, s_max) - K,
        KAPPA_MIN, KAPPA_MAX, xtol=1e-10, rtol=1e-12))


# ── Whitening and the estimator ──────────────────────────────────────────────

def _weighted_mean_var(values: np.ndarray, weights: np.ndarray):
    mean = np.average(values, weights=weights)
    var = np.average((values - mean) ** 2, weights=weights)
    return float(mean), float(var)


def whiten(v_par, v_p1, v_p2, weights) -> dict:
    """Drift-subtract and scale each component by its own measured std.

    v_par is scaled by its variance; the two perpendicular components share a
    common variance (gyrotropy), estimated as their average, so that residual
    perp1/perp2 sampling asymmetry is not baked into the whitening.
    Returns s2 (whitened squared speed) plus the drifts and scales used.
    """
    v_par = np.asarray(v_par, dtype=float)
    v_p1 = np.asarray(v_p1, dtype=float)
    v_p2 = np.asarray(v_p2, dtype=float)
    w = np.asarray(weights, dtype=float)

    mu_par, var_par = _weighted_mean_var(v_par, w)
    mu_p1, var_p1 = _weighted_mean_var(v_p1, w)
    mu_p2, var_p2 = _weighted_mean_var(v_p2, w)
    var_perp = 0.5 * (var_p1 + var_p2)
    if var_par <= 0 or var_perp <= 0:
        raise ValueError("Non-positive component variance; cannot whiten.")

    s2 = ((v_par - mu_par) ** 2 / var_par
          + ((v_p1 - mu_p1) ** 2 + (v_p2 - mu_p2) ** 2) / var_perp)
    return {"s2": s2, "weights": w,
            "drift": (mu_par, mu_p1, mu_p2),
            "var_par": var_par, "var_perp": var_perp}


def _K_from_s2(s2: np.ndarray, w: np.ndarray, s_max: float):
    """Truncated K and diagnostics from whitened squared speeds."""
    cut = s2 <= s_max * s_max
    w_in = w[cut]
    wsum = np.sum(w_in)
    if wsum <= 0:
        return float("nan"), 0.0, 0.0
    s2_in = s2[cut]
    m2 = np.average(s2_in, weights=w_in)
    m4 = np.average(s2_in * s2_in, weights=w_in)
    n_eff = wsum * wsum / np.sum(w_in * w_in)
    frac_in = wsum / np.sum(w)
    return float(m4 / (m2 * m2)), float(n_eff), float(frac_in)


def kappa_eff_from_velocities(v_par, v_p1, v_p2, weights=None, *,
                              s_max: float = DEFAULT_S_MAX,
                              n_boot: int = 0,
                              rng: np.random.Generator | None = None) -> dict:
    """Truncated, whitened kappa_eff from weighted particle velocities.

    Velocities must be in a field-aligned frame (v_par, v_perp1, v_perp2);
    drifts are removed internally. With n_boot > 0 a weighted bootstrap
    (re-whitening each resample, so scale uncertainty is propagated) provides
    the 16–84 percentile interval `kappa_lo/kappa_hi` and `kappa_err`.
    """
    v_par = np.asarray(v_par, dtype=float)
    n = v_par.size
    w = (np.ones(n) if weights is None
         else np.asarray(weights, dtype=float))
    if n < 100:
        return {"kappa": float("nan"), "K": float("nan"), "n_eff": float(n),
                "kappa_err": float("nan"), "kappa_lo": float("nan"),
                "kappa_hi": float("nan"), "s_max": s_max,
                "fraction_inside": float("nan")}

    wh = whiten(v_par, v_p1, v_p2, w)
    K, n_eff, frac_in = _K_from_s2(wh["s2"], w, s_max)
    kappa = kappa_from_K(K, s_max)

    out = {"kappa": kappa, "K": K, "n_eff": n_eff, "s_max": s_max,
           "fraction_inside": frac_in,
           "var_par": wh["var_par"], "var_perp": wh["var_perp"],
           "kappa_err": float("nan"),
           "kappa_lo": float("nan"), "kappa_hi": float("nan")}

    if n_boot > 0:
        rng = rng or np.random.default_rng(0)
        p = w / np.sum(w)
        boots = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True, p=p)
            try:
                wb = whiten(v_par[idx], np.asarray(v_p1)[idx],
                            np.asarray(v_p2)[idx], np.ones(idx.size))
            except ValueError:
                continue
            Kb, _, _ = _K_from_s2(wb["s2"], np.ones(idx.size), s_max)
            boots.append(kappa_from_K(Kb, s_max))
        boots = np.asarray(boots, dtype=float)
        finite = boots[np.isfinite(boots)]
        if finite.size >= max(8, n_boot // 4):
            lo, hi = np.percentile(finite, [16.0, 84.0])
            out["kappa_lo"], out["kappa_hi"] = float(lo), float(hi)
            out["kappa_err"] = float(0.5 * (hi - lo))
        elif boots.size:
            # bootstrap dominated by inf (Maxwellian-consistent resamples)
            out["kappa_lo"] = float(np.nanmin(boots))
            out["kappa_hi"] = float("inf")
    return out


def kappa_eff_from_grid(v_par, v_perp, f_weights, *,
                        s_max: float = DEFAULT_S_MAX) -> dict:
    """Same estimator applied to a gyrotropic (v_par, v_perp) quadrature grid.

    `f_weights` must already include the perpendicular Jacobian
    (f * 2 pi v_perp dv_par dv_perp), i.e. each node acts as a weighted
    particle. The perpendicular energy splits evenly across the two
    gyrotropic components, so <v_perp_j^2> = <v_perp^2>/2 defines the
    whitening scale. This is the entry point for the theory closures, so
    theory and simulation share one estimator by construction.
    """
    v_par = np.asarray(v_par, dtype=float).ravel()
    v_perp = np.asarray(v_perp, dtype=float).ravel()
    w = np.asarray(f_weights, dtype=float).ravel()
    keep = w > 0
    v_par, v_perp, w = v_par[keep], v_perp[keep], w[keep]
    if w.size == 0:
        return {"kappa": float("nan"), "K": float("nan"), "s_max": s_max}

    mu_par = np.average(v_par, weights=w)
    var_par = np.average((v_par - mu_par) ** 2, weights=w)
    var_perp_comp = 0.5 * np.average(v_perp ** 2, weights=w)  # gyrotropic
    if var_par <= 0 or var_perp_comp <= 0:
        return {"kappa": float("nan"), "K": float("nan"), "s_max": s_max}

    # v_perp^2 / var_perp_comp plays the role of
    # (v_p1^2 + v_p2^2)/var_perp_comp for the gyrophase-averaged grid.
    s2 = (v_par - mu_par) ** 2 / var_par + v_perp ** 2 / var_perp_comp
    K, _, frac_in = _K_from_s2(s2, w, s_max)
    return {"kappa": kappa_from_K(K, s_max), "K": K, "s_max": s_max,
            "fraction_inside": frac_in,
            "var_par": float(var_par), "var_perp": float(var_perp_comp)}


# ── Loader-consistent synthetic samplers ─────────────────────────────────────

def sample_bikappa(n: int, kappa: float, theta_par: float, theta_perp: float,
                   drift=(0.0, 0.0, 0.0),
                   rng: np.random.Generator | None = None):
    """Multivariate bi-kappa, same construction as the PSC loader.

    One shared radial gamma variate couples all three components
    (g ~ chi^2_(2 kappa - 1); v_j = u_j + sqrt(kappa) theta_j z_j / sqrt(g)),
    reproducing f ~ [1 + (v_par^2/theta_par^2 + v_perp^2/theta_perp^2)/kappa]^-(kappa+1).
    Factorising three independent 1-D kappas would NOT give this distribution.
    """
    if kappa <= 1.5:
        raise ValueError("kappa must exceed 3/2 for finite temperature.")
    rng = rng or np.random.default_rng()
    g = rng.chisquare(2.0 * kappa - 1.0, size=n)
    g = np.maximum(g, _TINY)
    z = rng.standard_normal((3, n))
    scale = np.sqrt(kappa / g)
    v_par = drift[0] + theta_par * z[0] * scale
    v_p1 = drift[1] + theta_perp * z[1] * scale
    v_p2 = drift[2] + theta_perp * z[2] * scale
    return v_par, v_p1, v_p2


def sample_bimaxwellian(n: int, theta_par: float, theta_perp: float,
                        drift=(0.0, 0.0, 0.0),
                        rng: np.random.Generator | None = None):
    """Bi-Maxwellian with the kappa -> inf convention of `sample_bikappa`
    (f ~ exp(-v_par^2/theta_par^2 - v_perp^2/theta_perp^2), so the
    per-component std is theta_j / sqrt(2))."""
    rng = rng or np.random.default_rng()
    z = rng.standard_normal((3, n))
    v_par = drift[0] + theta_par * z[0] / np.sqrt(2.0)
    v_p1 = drift[1] + theta_perp * z[1] / np.sqrt(2.0)
    v_p2 = drift[2] + theta_perp * z[2] / np.sqrt(2.0)
    return v_par, v_p1, v_p2


# ── Self-test ────────────────────────────────────────────────────────────────

def self_test(verbose: bool = True) -> bool:
    """Quick internal consistency checks (full validation in test_kappa_eff.py)."""
    ok = True

    # 1. Truncated curve converges to the closed untruncated formula.
    for kap in (4.0, 6.0, 10.0):
        K_inf = untruncated_K(kap)
        K_50 = truncated_K_kappa(kap, 50.0)
        good = abs(K_50 - K_inf) / K_inf < 0.02
        ok &= good
        if verbose:
            print(f"  K_t(kappa={kap}, s_max=50) = {K_50:.4f}  "
                  f"vs closed form {K_inf:.4f}  {'OK' if good else 'FAIL'}")

    # 2. Monotonicity of K_t in kappa (required by the brentq inversion).
    grid = np.array([truncated_K_kappa(k, DEFAULT_S_MAX)
                     for k in np.linspace(KAPPA_MIN, KAPPA_MAX, 60)])
    mono = bool(np.all(np.diff(grid) < 0))
    ok &= mono
    if verbose:
        print(f"  K_t monotonically decreasing in kappa: "
              f"{'OK' if mono else 'FAIL'}")

    # 3. Round trip on synthetic data, anisotropic on purpose.
    rng = np.random.default_rng(12345)
    for kap in (3.0, 5.0):
        v = sample_bikappa(600_000, kap, 1.0, np.sqrt(2.0), rng=rng)
        res = kappa_eff_from_velocities(*v)
        good = abs(res["kappa"] - kap) < 0.15
        ok &= good
        if verbose:
            print(f"  recovered kappa = {res['kappa']:.3f} "
                  f"(true {kap}, A=2)  {'OK' if good else 'FAIL'}")
    v = sample_bimaxwellian(600_000, 1.0, np.sqrt(2.0), rng=rng)
    res = kappa_eff_from_velocities(*v)
    good = res["kappa"] > 20.0
    ok &= good
    if verbose:
        print(f"  bi-Maxwellian A=2 -> kappa_eff = {res['kappa']:.1f} "
              f"(want > 20)  {'OK' if good else 'FAIL'}")
    return ok


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        print("kappa_eff self-test:")
        passed = self_test()
        print("PASSED" if passed else "FAILED")
        return 0 if passed else 1
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
