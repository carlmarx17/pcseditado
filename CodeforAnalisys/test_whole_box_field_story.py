"""Protect the physical definitions and cadence alignment of the summary plots."""
import numpy as np
import pytest
from whole_box_field_story import field_series, pair


def table():
    return [dict(omega_ci_t=t, delta_B_parallel_rms_over_B0=.03,
                 delta_B_perp_rms_over_B0=.04, delta_B_vec_rms_over_B0=.05,
                 B_mean_over_B0=1.03, B_std_over_B0=.04,
                 delta_B_rms_over_B0=.05) for t in [2.,0.,1.]]


def test_field_power_and_magnitude_identities():
    f=field_series(table())
    np.testing.assert_array_equal(f['omega_ci_t'],[0,1,2])
    np.testing.assert_allclose(f['W'],.0025)


def test_inconsistent_power_is_rejected():
    r=table();r[0]['delta_B_vec_rms_over_B0']=.06
    with pytest.raises(ValueError,match='Vector power'):field_series(r)


def test_duplicate_times_are_rejected():
    r=table();r[0]['omega_ci_t']=0
    with pytest.raises(ValueError,match='unique'):field_series(r)


def test_field_interpolation_keeps_particle_times_without_extrapolation():
    f=field_series(table());f['B_mean_over_B0']=np.array([1.,1.1,1.2])
    p=[dict(population='all',omega_ci_t=t,inv_kappa_signed=.2) for t in [-1,.5,1.5,3]]
    p.append(dict(population='hole',omega_ci_t=1,inv_kappa_signed=.5))
    result=pair(f,p)
    np.testing.assert_allclose(result['t'],[.5,1.5])
    np.testing.assert_allclose(result['B_mean_over_B0'],[1.05,1.15])
