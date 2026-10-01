"""mirror_ic_competition.py: threshold distance and the near-threshold growth estimate."""
import pytest

import mirror_ic_competition as mic


def test_cold_electron_drive_is_the_hasegawa_condition():
    assert float(mic.mirror_gamma(5.0, 2.0)) == pytest.approx(9.0)        # beta_perp (A - 1) - 1
    assert float(mic.mirror_gamma(5.0, 1.0)) == pytest.approx(-1.0)


def test_hot_isotropic_electrons_raise_the_threshold():
    # Hellinger (2007), eq. (32): 1 + (A_i - A_e)^2 / [2 (1/beta_i + 1/beta_e)]
    assert float(mic.mirror_gamma(5.0, 2.0, 1.0, 1.0)) == pytest.approx(9.0 - 1.0 / 2.4)
    assert float(mic.mirror_gamma(5.0, 2.0, 8.0, 1.0)) < float(mic.mirror_gamma(5.0, 2.0, 1.0, 1.0))


def test_near_threshold_growth_vanishes_at_threshold_and_scales_as_gamma_squared():
    assert mic.mirror_growth_near_threshold(5.0, 1.1) == 0.0
    near = mic.mirror_growth_near_threshold(5.0, 1.19)                     # Gamma ~ 0.13
    assert 0 < near < 1e-3
