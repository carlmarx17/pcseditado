"""trajectory_linear_theory.py: windowed growth, saturation time and the marginal anisotropy of a mode."""
import numpy as np
import pytest

import trajectory_linear_theory as tlt

STATE = {"beta_i": 5.0, "a_i": 2.0, "beta_e": 1.0, "a_e": 1.0, "mass_ratio": 200.0, "c_over_va": 176.8}
K = 0.3142


def test_windowed_growth_recovers_a_known_rate_that_decays_to_zero():
    t = np.linspace(0.0, 100.0, 2001)
    rate = 0.12 * (1.0 - t / 80.0)                       # crosses zero at t = 80
    amp = 1e-4 * np.exp(0.12 * (t - t ** 2 / 160.0))
    centres = np.arange(10.0, 95.0, 5.0)
    gamma, err = tlt.windowed_growth(t, amp, centres, 10.0)
    assert gamma == pytest.approx(np.interp(centres, t, rate), abs=1e-6)
    assert np.all(err < 1e-3)
    assert tlt.zero_crossing(centres, gamma) == pytest.approx(80.0, abs=1e-3)


def test_zero_crossing_without_a_sign_change_is_nan():
    assert np.isnan(tlt.zero_crossing([0.0, 1.0, 2.0], [0.3, 0.2, 0.1]))


def test_initial_state_gives_the_ion_cyclotron_root():
    root = tlt.linear_root(STATE, None, K)
    assert np.imag(root) == pytest.approx(0.1236, abs=2e-3)
    assert 0.0 < np.real(root) < 1.0
    assert np.imag(tlt.linear_root(STATE, 3.0, K)) < np.imag(root)   # same temperature: the tail lowers it


def test_marginal_anisotropy_is_where_the_mode_stops_growing():
    a_marginal, _ = tlt.marginal_anisotropy(STATE, None, K)
    assert 1.2 < a_marginal < 1.5
    root = tlt.linear_root(STATE, None, K, a_i=a_marginal)
    assert abs(np.imag(root)) < 2e-3
    # Resonance condition of the L mode at marginality: omega_r = Omega_ci (A - 1) / A.
    assert np.real(root) == pytest.approx((a_marginal - 1.0) / a_marginal, abs=0.02)


def test_bi_kappa_marginal_anisotropy_is_extrapolated_from_the_growing_side():
    a_marginal, _ = tlt.marginal_anisotropy(STATE, 3.0, K)
    assert 1.2 < a_marginal < 1.5
    assert np.imag(tlt.linear_root(STATE, 3.0, K, a_i=a_marginal + 0.05)) > 0


def test_c_over_va_uses_the_ion_alfven_speed_not_the_b0_input():
    # The profile key vA_over_c is B0 (historical C++ name): v_A/c = B0 / sqrt(m_i/m_e).
    assert tlt.c_over_va({"mass_ratio": 200.0, "vA_over_c": 0.08}) == pytest.approx(176.78, abs=0.01)
