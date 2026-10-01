"""kappa_linear_survey.py: conventions, threshold interpolation and fit."""
import numpy as np
import pytest

import kappa_linear_survey as ks


def test_same_core_convention_is_a_hotter_plasma():
    assert ks.core_beta(5.0, None) == 5.0
    assert ks.core_beta(5.0, 3.0) == pytest.approx(10.0)       # kappa / (kappa - 3/2) = 2


def test_threshold_curve_and_fit_recover_a_prescribed_law():
    # gamma_max rising steeply through the level at A = 1 + 0.5 / beta^0.4
    truth = 1.0 + 0.5 / ks.BETA_GRID ** 0.4
    gmax = np.array([1e-3 * np.exp(12.0 * (ks.A_GRID - a)) for a in truth])
    curve = ks.threshold_curve(gmax, 1e-3)
    np.testing.assert_allclose(curve, truth, atol=0.03)
    a, b = ks.fit_threshold(ks.BETA_GRID, curve)
    assert (a, b) == (pytest.approx(0.5, abs=0.05), pytest.approx(0.4, abs=0.05))


def test_maxwellian_growth_at_the_run_parameters_matches_the_quoted_value():
    gamma, omega = ks.spectrum((5.0, 2.0, None, 1.0, 200))
    assert gamma.max() == pytest.approx(0.127, abs=0.003)
    assert 0 < omega[np.argmax(gamma)] < 1
