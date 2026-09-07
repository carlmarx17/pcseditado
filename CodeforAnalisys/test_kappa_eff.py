#!/usr/bin/env python3
"""Synthetic-distribution validation of the truncated kappa_eff estimator.

Every test draws particles from a distribution whose kappa is known by
construction (using the same multivariate sampling rule as the PSC loader)
and asks the estimator to recover it. The two regression tests that motivate
the design are:

  * anisotropy aliasing — a bi-Maxwellian with A != 1 must NOT report a
    finite kappa; without whitening it does.
  * truncation bias — applying the closed-form (untruncated) inversion to
    truncated data is systematically biased; the truncated inversion is not.
"""

from __future__ import annotations

import numpy as np
import pytest

import kappa_eff as ke

N = 400_000
S_MAX = ke.DEFAULT_S_MAX


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── Theory curve ─────────────────────────────────────────────────────────────

def test_untruncated_formula_is_the_large_smax_limit():
    for kappa in (3.5, 5.0, 8.0, 20.0):
        assert ke.truncated_K_kappa(kappa, 60.0) == pytest.approx(
            ke.untruncated_K(kappa), rel=0.03)


def test_maxwellian_truncated_K_below_any_kappa():
    for s_max in (4.0, 6.0, 10.0):
        K_max = ke.truncated_K_maxwellian(s_max)
        for kappa in (2.0, 3.0, 5.0, 12.0):
            assert ke.truncated_K_kappa(kappa, s_max) > K_max


def test_inversion_round_trip_exact():
    for kappa in (2.0, 3.0, 5.0, 10.0, 50.0):
        K = ke.truncated_K_kappa(kappa, S_MAX)
        assert ke.kappa_from_K(K, S_MAX) == pytest.approx(kappa, rel=1e-6)


def test_maxwellian_K_maps_to_inf():
    assert np.isinf(ke.kappa_from_K(ke.truncated_K_maxwellian(S_MAX), S_MAX))
    assert np.isinf(ke.kappa_from_K(1.0, S_MAX))


# ── Recovery on synthetic particles ──────────────────────────────────────────

@pytest.mark.parametrize("kappa", [3.0, 5.0])
@pytest.mark.parametrize("aniso", [1.0, 2.0, 0.3])
def test_recovers_kappa_regardless_of_anisotropy(kappa, aniso):
    """kappa and A must decouple: same kappa in, same kappa out for any A."""
    theta_par = 1.0
    theta_perp = np.sqrt(aniso) * theta_par        # A = theta_perp^2/theta_par^2
    v = ke.sample_bikappa(N, kappa, theta_par, theta_perp, rng=_rng(1))
    res = ke.kappa_eff_from_velocities(*v)
    assert res["kappa"] == pytest.approx(kappa, abs=0.25)


@pytest.mark.parametrize("aniso", [2.0, 0.3])
def test_bimaxwellian_reports_maxwellian(aniso):
    v = ke.sample_bimaxwellian(N, 1.0, np.sqrt(aniso), rng=_rng(2))
    res = ke.kappa_eff_from_velocities(*v)
    assert res["kappa"] > 25.0        # inf or huge: Maxwellian-consistent


def test_drift_does_not_bias():
    v = ke.sample_bikappa(N, 3.0, 1.0, np.sqrt(2.0),
                          drift=(0.7, -0.4, 0.2), rng=_rng(3))
    res = ke.kappa_eff_from_velocities(*v)
    assert res["kappa"] == pytest.approx(3.0, abs=0.25)


# ── Regression: why whitening is required ────────────────────────────────────

def test_anisotropy_aliases_into_kappa_without_whitening():
    """A naive speed-kurtosis estimator on an A=2 bi-MAXWELLIAN reports a
    spurious finite kappa. This is the failure mode whitening removes."""
    v_par, v_p1, v_p2 = ke.sample_bimaxwellian(N, 1.0, np.sqrt(2.0),
                                               rng=_rng(4))
    # naive: no per-component scaling, just |v|^2 in units of total variance
    v2 = v_par**2 + v_p1**2 + v_p2**2
    s2 = 3.0 * v2 / np.mean(v2)
    cut = s2 <= S_MAX * S_MAX
    K_naive = np.mean(s2[cut] ** 2) / np.mean(s2[cut]) ** 2
    kappa_naive = ke.kappa_from_K(K_naive, S_MAX)
    assert np.isfinite(kappa_naive) and kappa_naive < 50.0, \
        "expected the naive estimator to alias anisotropy into kappa"


# ── Regression: why the truncated inversion is required ──────────────────────

def test_untruncated_formula_biased_on_truncated_data():
    v = ke.sample_bikappa(N, 3.0, 1.0, 1.0, rng=_rng(5))
    res = ke.kappa_eff_from_velocities(*v, s_max=4.0)
    kappa_naive = ke.kappa_from_K_untruncated(res["K"])
    assert res["kappa"] == pytest.approx(3.0, abs=0.3)
    assert kappa_naive - 3.0 > 1.0, \
        "closed-form inversion should overestimate kappa on truncated data"


def test_smax_insensitivity_within_reason():
    """The truncated inversion must give consistent kappa across cut choices."""
    v = ke.sample_bikappa(N, 3.0, 1.0, np.sqrt(2.0), rng=_rng(6))
    got = [ke.kappa_eff_from_velocities(*v, s_max=s)["kappa"]
           for s in (4.0, 6.0, 8.0)]
    assert np.all(np.isfinite(got))
    assert max(got) - min(got) < 0.5


# ── Weights, sample size, bootstrap ──────────────────────────────────────────

def test_weights_equivalent_to_repetition():
    rng = _rng(7)
    v_par, v_p1, v_p2 = ke.sample_bikappa(50_000, 4.0, 1.0, 1.2, rng=rng)
    rep = np.tile(np.arange(v_par.size), 2)     # duplicate every particle
    res_rep = ke.kappa_eff_from_velocities(v_par[rep], v_p1[rep], v_p2[rep])
    res_w = ke.kappa_eff_from_velocities(v_par, v_p1, v_p2,
                                         weights=np.full(v_par.size, 2.0))
    assert res_w["K"] == pytest.approx(res_rep["K"], rel=1e-12)
    assert res_w["kappa"] == pytest.approx(res_rep["kappa"], rel=1e-9)


def test_too_few_particles_returns_nan():
    v = ke.sample_bikappa(50, 3.0, 1.0, 1.0, rng=_rng(8))
    assert np.isnan(ke.kappa_eff_from_velocities(*v)["kappa"])


def test_bootstrap_error_is_calibrated():
    """A 68% interval may miss the truth for any single seed; what must hold
    is that the point estimate sits within ~3 bootstrap sigmas of the truth
    and that the reported error is neither zero nor absurd."""
    v = ke.sample_bikappa(100_000, 3.0, 1.0, np.sqrt(2.0), rng=_rng(9))
    res = ke.kappa_eff_from_velocities(*v, n_boot=60, rng=_rng(10))
    assert 0.0 < res["kappa_err"] < 0.5
    assert abs(res["kappa"] - 3.0) < 3.0 * res["kappa_err"]
    assert res["kappa_lo"] < res["kappa"] < res["kappa_hi"]


# ── Grid (theory) entry point ────────────────────────────────────────────────

def test_grid_estimator_matches_particle_estimator():
    """The quadrature entry point must recover kappa from an analytic
    gyrotropic bi-kappa evaluated on a (v_par, v_perp) grid."""
    kappa, th_par, th_perp = 3.0, 1.0, np.sqrt(2.0)
    v_par = np.linspace(-25.0, 25.0, 900)
    v_perp = np.linspace(0.0, 25.0, 900)[1:]     # avoid the zero-Jacobian node
    VP, VU = np.meshgrid(v_par, v_perp, indexing="ij")
    f = (1.0 + (VP**2 / th_par**2 + VU**2 / th_perp**2) / kappa) ** (-(kappa + 1.0))
    wgt = f * 2.0 * np.pi * VU                   # perpendicular Jacobian
    res = ke.kappa_eff_from_grid(VP, VU, wgt)
    assert res["kappa"] == pytest.approx(kappa, abs=0.05)


def test_grid_estimator_maxwellian_limit():
    th = 1.0
    v_par = np.linspace(-8.0, 8.0, 700)
    v_perp = np.linspace(0.0, 8.0, 700)[1:]
    VP, VU = np.meshgrid(v_par, v_perp, indexing="ij")
    f = np.exp(-(VP**2 + VU**2) / th**2)
    res = ke.kappa_eff_from_grid(VP, VU, f * 2.0 * np.pi * VU)
    assert res["kappa"] > 40.0
