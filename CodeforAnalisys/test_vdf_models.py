"""Model velocity distributions (plasma_physics) and the kappa likelihood estimator.

The VDF figures overlay these models on PIC histograms, so a wrong
normalisation shows up as a model that sits above or below the data by a
constant factor -- the symptom that motivated this file. PSC loads a bi-kappa
as a Gaussian scaled by sqrt((kappa - 3/2)/G), G ~ Gamma(kappa - 1/2): a
Student t with nu = 2 kappa - 1 whose variance is the profile temperature.
"""

import numpy as np
import pytest
from scipy import integrate

from plasma_physics import (
    bi_distribution_3d,
    kappa_marginal_cdf,
    kappa_marginal_pdf,
    kappa_mle,
    kappa_speed_pdf_2d,
    reduced_distribution_2d,
)

KAPPAS = [None, 3.0, 5.0]


def psc_kappa_components(n, kappa, sigmas, rng):
    """Velocity components drawn like the PSC loader (one shared mixing variable)."""
    z = rng.standard_normal((len(sigmas), n))
    scale = (np.sqrt((kappa - 1.5) / rng.gamma(kappa - 0.5, 1.0, n)) if kappa else np.ones(n))
    return [s * zi * scale for s, zi in zip(sigmas, z)]


@pytest.mark.parametrize("kappa", KAPPAS)
def test_marginal_is_normalised_with_the_measured_variance(kappa):
    sigma = 0.7
    norm = integrate.quad(lambda v: kappa_marginal_pdf(v, sigma, kappa), -np.inf, np.inf)[0]
    var = integrate.quad(lambda v: v * v * kappa_marginal_pdf(v, sigma, kappa), -np.inf, np.inf)[0]
    assert norm == pytest.approx(1.0, abs=1e-8)
    assert var == pytest.approx(sigma ** 2, rel=1e-6)
    assert kappa_marginal_cdf(0.4, sigma, kappa) == pytest.approx(
        integrate.quad(lambda v: kappa_marginal_pdf(v, sigma, kappa), -np.inf, 0.4)[0], abs=1e-8)


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")   # kappa = 3 tails
@pytest.mark.parametrize("kappa", KAPPAS)
def test_three_dimensional_model_is_normalised_per_d3v(kappa):
    s_par, s_perp = 0.7, 1.0
    f = lambda vperp, vpar: 2 * np.pi * vperp * bi_distribution_3d(vpar, vperp, s_par, s_perp, kappa)
    moment = lambda g: integrate.dblquad(lambda vperp, vpar: g(vpar, vperp) * f(vperp, vpar),
                                         -np.inf, np.inf, 0, np.inf, epsabs=1e-10)[0]
    assert moment(lambda a, b: 1.0) == pytest.approx(1.0, rel=1e-6)
    assert moment(lambda a, b: a * a) == pytest.approx(s_par ** 2, rel=1e-5)
    assert moment(lambda a, b: b * b) == pytest.approx(2 * s_perp ** 2, rel=1e-5)


@pytest.mark.parametrize("kappa", KAPPAS)
def test_reduced_and_speed_densities_are_marginals_of_the_same_model(kappa):
    s1, s2 = 0.7, 1.0
    norm2 = integrate.dblquad(lambda b, a: reduced_distribution_2d(a, b, s1, s2, kappa),
                              -np.inf, np.inf, -np.inf, np.inf)[0]
    assert norm2 == pytest.approx(1.0, rel=1e-6)
    # Integrating the 2-D density over v2 gives the 1-D marginal.
    v1 = 1.3
    marginal = integrate.quad(lambda b: reduced_distribution_2d(v1, b, s1, s2, kappa), -np.inf, np.inf)[0]
    assert marginal == pytest.approx(float(kappa_marginal_pdf(v1, s1, kappa)), rel=1e-6)
    assert integrate.quad(lambda v: kappa_speed_pdf_2d(v, s2, kappa), 0, np.inf)[0] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("kappa", [3.0, 5.0])
def test_models_match_a_sample_drawn_like_the_psc_loader(kappa):
    rng = np.random.default_rng(7)
    vpar, vx, vy = psc_kappa_components(400_000, kappa, (0.7, 1.0, 1.0), rng)
    edges = np.linspace(-4.0, 4.0, 41)
    counts, _ = np.histogram(vpar, bins=edges)
    expected = len(vpar) * np.diff(kappa_marginal_cdf(edges, 0.7, kappa))
    # Poisson agreement bin by bin, core to the far shoulders (no free factor).
    pulls = (counts - expected) / np.sqrt(expected)
    assert np.max(np.abs(pulls)) < 4.5
    assert np.mean(pulls ** 2) < 2.0
    counts2, _ = np.histogram(np.hypot(vx, vy), bins=np.linspace(0.0, 5.0, 26))
    expected2 = [len(vx) * integrate.quad(lambda v: kappa_speed_pdf_2d(v, 1.0, kappa), a, b)[0]
                 for a, b in zip(np.linspace(0, 5, 26)[:-1], np.linspace(0, 5, 26)[1:])]
    pulls2 = (counts2 - np.array(expected2)) / np.sqrt(expected2)
    assert np.max(np.abs(pulls2)) < 4.5


def test_kappa_likelihood_recovers_the_loaded_index_and_reports_a_maxwellian():
    rng = np.random.default_rng(11)
    (sample,) = psc_kappa_components(200_000, 3.0, (0.7,), rng)
    fit = kappa_mle(sample, rng=np.random.default_rng(1))
    assert fit["sigma"] == pytest.approx(0.7, rel=0.02)
    assert 1 / fit["kappa"] == pytest.approx(1 / 3.0, abs=0.03)
    assert fit["kappa_lo"] <= fit["kappa"] <= fit["kappa_hi"]
    maxwellian = kappa_mle(rng.normal(0.0, 0.7, 200_000), rng=np.random.default_rng(2))
    assert maxwellian["kappa_lo"] > 15.0


def test_kappa_likelihood_interval_is_not_shrunk_by_thinning():
    rng = np.random.default_rng(5)
    (sample,) = psc_kappa_components(60_000, 4.0, (1.0,), rng)
    full = kappa_mle(sample, rng=np.random.default_rng(3))
    thinned = kappa_mle(sample, max_samples=15_000, rng=np.random.default_rng(3))
    width = lambda f: 1 / f["kappa_lo"] - (0.0 if np.isinf(f["kappa_hi"]) else 1 / f["kappa_hi"])
    # A point estimate from 15k points is noisier than one from 60k: its
    # interval must be wider (about twice), not the full-sample one.
    assert width(thinned) > 1.4 * width(full)
