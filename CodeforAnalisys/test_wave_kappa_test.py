import numpy as np
import pytest
from wave_kappa_test import integral, interval_mean, window_rates, residual_correlation, fits


def test_integrates_linear_power_with_interpolated_boundaries():
    t=np.array([0.,1.,2.,3.]);w=2*t
    np.testing.assert_allclose(integral(t,w),t*t)
    assert interval_mean(t,w,.25,2.75)==pytest.approx(3.)
    with pytest.raises(ValueError):interval_mean(t,w,-1,2)


def test_window_slopes_are_signed_and_samples_do_not_overlap():
    t=np.arange(0.,41.);q=.3-.002*t;e=np.full_like(t,.001)
    r=window_rates(t,q,e,t,np.ones_like(t),start=0,width=10)
    assert len(r)==4
    assert sum(x['n'] for x in r)==40
    np.testing.assert_allclose([x['rate'] for x in r],.002,atol=1e-12)
    assert all(a['t_end']<b['t_start'] for a,b in zip(r[:-1],r[1:]))


def test_detrending_removes_shared_linear_trend():
    t=np.linspace(0,100,101);a=np.sin(t);b=np.cos(2*t)
    x=np.column_stack([np.ones(len(t)),t])
    a-=x@np.linalg.lstsq(x,a,rcond=None)[0];b-=x@np.linalg.lstsq(x,b,rcond=None)[0]
    b-=a*np.dot(a,b)/np.dot(a,a)
    assert np.corrcoef(t+a,t+b)[0,1]>.99
    assert abs(residual_correlation(t,t+a,t+b,1))<1e-10


def test_future_targets_do_not_affect_first_forecast_fit():
    t=np.arange(100.);z=np.sin(t/20);y=.2-.001*t-.003*z
    original=fits(t,z,y)['time_wave']['forward_fold_rmse'][0]
    changed=y.copy();changed[60:]+=10
    revised=fits(t,z,changed)['time_wave']['forward_fold_rmse'][0]
    assert original==pytest.approx(revised)
