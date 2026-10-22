import numpy as np
import pytest
from kappa_fourier_review import spectrum


def test_recovers_angular_frequency_not_cycles_per_time():
    t=np.arange(256)*.5;w=2*np.pi*12/(256*.5)
    *_,s=spectrum(t,np.sin(w*t)+.03*t,1)
    assert s['omega_peak']==pytest.approx(w)
    assert s['period']==pytest.approx(2*np.pi/w)


def test_refuses_irregular_cadence():
    t=np.arange(40,dtype=float);t[20]+=.1
    with pytest.raises(ValueError):spectrum(t,np.sin(t))


def test_removes_quadratic_trend():
    t=np.arange(100,dtype=float)
    r,*_=spectrum(t,2+.02*t+.001*t*t,2)
    assert np.max(np.abs(r))<1e-12


def test_psd_density_is_per_angular_frequency():
    t=np.arange(512)*.25;w=2*np.pi*32/(512*.25)
    _,_,omega,p,_=spectrum(t,np.sin(w*t),0)
    assert np.sum(p)*(omega[1]-omega[0])==pytest.approx(.5,rel=1e-8)
