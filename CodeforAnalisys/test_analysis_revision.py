"""Regression tests for the evidence-driven v6 changes; independent known answers."""
import json
import numpy as np
import pytest
from analysis_contract import (
    strict_dumps, stable_rng, align_time, cylindrical_density, gap_segments,
    effective_sample_size,
)
from growth_fit import fit_exponential_growth
from physical_diagnostics import _mode_power_map, mode_power, MODE_CANDIDATE_FRACTIONS, _grid_fit_distribution
from heat_flux_analysis import heat_flux_moments, integrated_flux_ratio
from compare_physical_cases import validate_estimators
from structures_analysis import analyse
from quality_report import inspect_run


def test_strict_json_and_endpoint_alignment():
    assert json.loads(strict_dumps({'x': np.nan, 'a': np.array([1., np.inf])})) == {'x': None, 'a': [1., None]}
    y, meta = align_time([0, 1], [0, 4], [0, 1.0001, 2], tolerance=.0002)
    assert y[1] == 4 and np.isnan(y[2]) and meta['endpoint_snaps'] == 1
    y, _ = align_time([0, 1], [0, 4], [1.0001])
    assert np.isnan(y[0])  # Default never silently extrapolates.


def test_cylindrical_probability_and_exact_annular_volume():
    rng = np.random.default_rng(17)
    v = rng.normal(size=(3, 100000))
    par, perp = np.linspace(-5, 5, 41), np.linspace(0, 6, 31)
    radial = np.hypot(v[0], v[1]); w = np.ones(v.shape[1])
    f, meta = cylindrical_density(v[2], radial, w, par, perp)
    volume = np.diff(par)[:, None] * (np.pi * np.diff(perp**2))[None, :]
    assert abs(np.sum(f*volume)-meta['retained_probability']) < 1e-12
    g, gm = cylindrical_density(v[2], radial, w, par, perp, 'probability')
    assert abs(np.sum(g*np.diff(par)[:, None]*np.diff(perp)[None, :])-gm['retained_probability']) < 1e-12
    # Near-axis gyrotropic density should be highest at the core, not an artificial ring.
    assert f[20, :2].mean() > f[20, 10:12].mean()


def test_deterministic_task_rng_and_real_time_gaps():
    first = stable_rng('run', 20, 'ion', 'heat').integers(0, 100000, 20)
    stable_rng('run', 10, 'ion', 'heat').random(200)
    np.testing.assert_array_equal(first, stable_rng('run', 20, 'ion', 'heat').integers(0, 100000, 20))
    assert len(gap_segments([0, 1, 2, 3, 4, 5, 8], [1, np.nan, 2, np.nan, 3, np.nan, 4])) == 2


def test_fourier_translation_amplitude_and_nyquist_invariance():
    rng = np.random.default_rng(4); b = rng.normal(size=(3, 16, 16))
    np.testing.assert_allclose(_mode_power_map(b), _mode_power_map(np.roll(b, (3, 5), (1, 2))), atol=1e-14)
    np.testing.assert_allclose(_mode_power_map(3*b), 9*_mode_power_map(b), atol=1e-14)
    assert np.isclose(_mode_power_map(b).sum(), np.mean(np.sum(b*b, axis=0)))
    ny = np.zeros((3, 16, 16)); ny[0] = (-1.)**np.arange(16)[:, None]
    assert np.isclose(mode_power(ny, [(8, 0)])[0], 1.)
    assert any(f < .01 for f in MODE_CANDIDATE_FRACTIONS)


def test_growth_amplitude_scaling_and_correlated_uncertainty():
    t = np.linspace(0, 20, 300)
    a = np.exp(.2*t + .025*np.sin(t))
    x = fit_exponential_growth(t, a, t_start=0, t_end=20)
    y = fit_exponential_growth(t, 500*a, t_start=0, t_end=20)
    assert abs(x['gamma']-y['gamma']) < 1e-12
    assert x['gamma_hac_stderr'] > x['gamma_stderr']
    assert x['gamma_err'] >= x['gamma_hac_stderr']


def test_weight_and_rotation_invariance_of_flux():
    rng = np.random.default_rng(3); v = rng.normal(size=(3, 1000)); w = rng.uniform(.5, 2, 1000)
    b = np.array([0.,0.,1.]); r = heat_flux_moments(*v, w, 1., b, 6)
    idx = rng.permutation(len(w)); rr = heat_flux_moments(*v[:,idx], w[idx], 1., b, 6)
    assert np.isclose(r['q_par_over_q0'], rr['q_par_over_q0'])
    repeat = heat_flux_moments(*np.repeat(v,2,axis=1), np.repeat(w/2,2), 1., b, 6)
    assert np.isclose(r['q_par_over_q0'], repeat['q_par_over_q0'])
    q,_ = np.linalg.qr(rng.normal(size=(3,3)))
    rotated = heat_flux_moments(*(q@v), w, 1., q@b, 6)
    assert np.isclose(r['q_par_over_q0'], rotated['q_par_over_q0'])
    assert effective_sample_size(w) <= len(w)


def test_matched_flux_floor_on_maxwellian_controls():
    rng = np.random.default_rng(10); measured, predicted = [], []
    for _ in range(120):
        r=heat_flux_moments(*rng.normal(size=(3,1500)),np.ones(1500),1.,np.array([0,0,1.]),6)
        measured.append(abs(r['q_par_over_q0']));predicted.append(r['abs_q_matched_floor'])
    assert .7 < np.mean(measured)/np.mean(predicted) < 1.3
    blocks=[dict(weight=1,q_par_per_particle=1,q0_per_particle=1),dict(weight=3,q_par_per_particle=4,q0_per_particle=2)]
    assert np.isclose(integrated_flux_ratio(blocks),13/7)


def test_uniform_magnitude_increase_does_not_create_structures():
    z=np.zeros((16,16)); b=np.full_like(z,.1)
    row,cat,_=analyse({'x':z,'y':z,'z':b},0,1,.02,1)
    assert not cat and row['peak_area_fraction']==0 and row['hole_area_fraction']==0


def test_missing_evidence_and_mixed_estimators_are_not_passes(tmp_path):
    assert inspect_run(tmp_path)['scientific_status']=='UNVERIFIED'
    a={'gamma_series':'mode','manifest':{'physics':{'analysis_conventions_version':6}}}
    b={'gamma_series':'total','manifest':{'physics':{'analysis_conventions_version':6}}}
    with pytest.raises(ValueError,match='Mixed'):
        validate_estimators([a,b])


def test_fallback_density_objective_recovers_maxwellian_width():
    x=np.linspace(-4,4,100);y=2*np.exp(-.5*(x/1.2)**2)
    m,k=_grid_fit_distribution(x,y,1.2)
    assert abs(m[1]/1.2-1)<.05


def test_structure_tracks_record_split_and_periodic_overlap():
    from structure_tracking import StructureTracker
    tr=StructureTracker();z=np.zeros((8,8),int)
    first=z.copy();first[0,:4]=1;first[-1,:4]=1
    tr.update(0,{'hole_labels':first,'peak_labels':z})
    second=z.copy();second[0,:2]=1;second[-1,:2]=1;second[0,3]=2;second[-1,3]=2
    tr.update(1,{'hole_labels':second,'peak_labels':z})
    assert [r['event'] for r in tr.rows].count('split')==2
    assert all(r['observed_duration']>=0 for r in tr.summary())


def test_early_mode_is_discovered_before_quarter_run(monkeypatch):
    import physical_diagnostics as pd
    n=32;z,y=np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
    def fields(step):
        mode=np.sin(2*np.pi*z/n)*(1. if step<10 else 0.)
        late=np.sin(4*np.pi*y/n)*(1. if step>=10 else 0.)
        return {'Bx':mode[None,...], 'By':late[None,...], 'Bz':np.zeros((1,n,n))}
    monkeypatch.setattr(pd,'load_fields',fields)
    monkeypatch.setattr(pd,'_fluctuation_plane',lambda bx,by,bz:(np.stack([bx[0],by[0],bz[0]]),('z','y'),(1.,1.)))
    modes=pd.mode_candidates({i:i for i in range(100)},1.,per_snapshot=1)
    assert any(np.isclose(abs(m['k_parallel_di']),2*np.pi/n) and m['k_perp_di']==0 for m in modes)


def test_heldout_models_do_not_certify_global_mixture():
    from vdf_validation import predictive_check
    from physical_diagnostics import _fit_density_models
    rng=np.random.default_rng(11)
    u=np.r_[rng.normal(size=15000),rng.normal(scale=3,size=5000)]
    r=predictive_check(u,np.ones(u.size),np.random.default_rng(12),_fit_density_models)
    assert np.isfinite(r['heldout_mse_maxwellian']) and np.isfinite(r['heldout_mse_kappa'])
    assert 'not_physical_identification' in r['model_validation_status']
    assert 'mixture' in r['mixture_ambiguity']


def test_initial_uniform_field_does_not_stretch_log_axes(tmp_path):
    """dB(t=0) is float rounding of the uniform B0; it must not set the log scale."""
    import matplotlib.pyplot as plt
    import compare_physical_cases as cpc
    import plot_style as ps
    t = np.linspace(0.0, 40.0, 81)
    db = np.where(t > 0, 1e-3 * np.exp(0.2 * np.minimum(t, 25.0)), 1e-12)
    assert list(ps.measured_fluctuation([0.0, 0.5, np.nan], settle=0.0)) == [False, True, False]
    assert list(ps.measured_fluctuation([0.0, 0.5, 3.0], settle=2.0)) == [False, False, True]
    rows = [{"omega_ci_t": ti, "delta_B_vec_rms_over_B0": v} for ti, v in zip(t, db)]
    captured = {}
    original = cpc._save
    cpc._save = lambda fig, path: captured.setdefault("ylim", fig.axes[0].get_ylim())
    try:
        cpc.plot_timeseries([{"name": "run", "rows": rows}], ["delta_B_vec_rms_over_B0"],
                            ["dB"], tmp_path / "x.png", "t", yscale="log")
    finally:
        cpc._save = original
        plt.close("all")
    assert captured["ylim"][0] > 1e-4
