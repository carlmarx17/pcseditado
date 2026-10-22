"""Regression coverage for production batch failures and false success states."""
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pytest

import physical_diagnostics as pd
from check_initial_conditions import check_initial_conditions, relative_error
from check_run_completion import completion
from run_pipeline import stage_options

HERE = Path(__file__).resolve().parent


def test_real_adios2_roundtrip_matches_hdf5(tmp_path):
    adios2 = pytest.importorskip('adios2')
    from data_reader import PICDataReader
    field = np.arange(24, dtype=float).reshape(4, 6, 1)
    values = {'psc/fields/hx/p0/3d': field,
              **{f'particles/p0/1d/{name}': np.full(8, value, dtype=float)
                 for name, value in zip(('q', 'm', 'px', 'py', 'pz', 'w'),
                                        (1, 100, .1, .2, .3, .001))}}
    h5 = tmp_path / 'snapshot.h5'
    bp = tmp_path / 'snapshot.bp'
    with h5py.File(h5, 'w') as f, adios2.Stream(str(bp), 'w') as stream:
        particles = np.zeros(8, dtype=[(name, 'f8') for name in ('q', 'm', 'px', 'py', 'pz', 'w')])
        for name, value in values.items():
            if name.startswith('particles/'):
                particles[name.rsplit('/', 1)[-1]] = value
            else:
                f.create_dataset(name, data=value)
            stream.write(name, value, list(value.shape), [0]*value.ndim, list(value.shape))
        f.create_dataset('particles/p0/1d', data=particles)
        for name, bounds in [('lo', [0, 1, 2]), ('hi', [1, 4, 6])]:
            f['particles'].attrs[name] = bounds
            stream.write(f'particles/{name}', np.asarray(bounds, dtype=np.int64))
    for path in (h5, bp):
        np.testing.assert_array_equal(
            PICDataReader.read_multiple_fields_3d(str(path), 'psc', ['fields/hx/p0/3d'])['fields/hx/p0/3d'], field)
    for left, right in zip(PICDataReader.read_particles_snapshot(str(h5), 100),
                           PICDataReader.read_particles_snapshot(str(bp), 100)):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(PICDataReader.read_prt_window(str(h5)),
                           PICDataReader.read_prt_window(str(bp))):
        np.testing.assert_array_equal(left, right)


def test_parallel_snapshot_failure_is_not_a_successful_partial_series(monkeypatch):
    monkeypatch.setattr(pd.concurrent.futures, 'ProcessPoolExecutor', ThreadPoolExecutor)
    monkeypatch.setattr(pd.os, 'cpu_count', lambda: 4)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '4')
    def worker(task):
        if task[0] == 20:
            raise OSError('truncated HDF5 file')
        return task
    with pytest.raises(RuntimeError, match='1/3 snapshots failed.*step 20'):
        pd._run_step_tasks(worker, [(10,), (20,), (30,)], 2, 'Fields')


def test_workers_respect_the_slurm_allocation(monkeypatch):
    monkeypatch.setattr(pd.os, 'cpu_count', lambda: 128)
    if hasattr(pd.os, 'sched_getaffinity'):
        monkeypatch.setattr(pd.os, 'sched_getaffinity', lambda _: set(range(128)))
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '4')
    assert pd._parallel_workers(0, 80) == 4
    assert pd._parallel_workers(32, 80) == 4


@pytest.mark.parametrize('status', ['FAIL', 'RUNNING', 'NOT_RUN'])
def test_failed_physics_does_not_reuse_an_old_linear_phase(tmp_path, status):
    p = tmp_path / '09_physical_diagnostics' / 'linear_phase.json'
    p.parent.mkdir()
    p.write_text(json.dumps({'status': 'PASS', 'start': 2, 'end': 9}))
    options, notes = stage_options('spectral', tmp_path, [], {'stages': {'physics': {'execution_status': status}}})
    assert options == []
    assert notes['window_source'].startswith('script defaults')
    options, _ = stage_options('spectral', tmp_path, [], {'stages': {'physics': {'execution_status': 'PASS'}}})
    assert 'GROWTH_T_START=2' in options and 'DISPERSION_T_END=9' in options


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_initial_values_cannot_pass(value):
    assert relative_error(value, .08) == np.inf


def test_missing_initial_fields_and_moments_are_failures(tmp_path):
    from test_check_initial_conditions import CheckInitialConditionsTests
    factory = CheckInitialConditionsTests()
    particles = factory.particle_file(tmp_path, .01, .02, 1e-4, 2e-4)
    failures = check_initial_conditions(particles, str(tmp_path/'pfd.*.h5'),
                                        str(tmp_path/'pfd_moments.*.h5'),
                                        factory.declared(), 10000, .15, .05)
    assert any('No field snapshots' in s for s in failures)
    assert any('No moment snapshots' in s for s in failures)


def test_evolved_snapshot_is_not_certified_as_initial_state(tmp_path):
    from test_check_initial_conditions import CheckInitialConditionsTests
    factory = CheckInitialConditionsTests()
    particles = factory.particle_file(tmp_path, .01, .02, 1e-4, 2e-4)
    fields = factory.field_file(tmp_path, .08)
    moments = factory.moment_file(tmp_path, 1.)
    p = next(tmp_path.glob('prt*.h5'))
    p.rename(p.with_name('prt_case.000100.h5'))
    failures = check_initial_conditions(particles, fields, moments, factory.declared(), 10000, .15, .05)
    assert any('Initial particle snapshot (step 0) is missing' in s for s in failures)


def test_completion_requires_the_last_scheduled_output():
    # nmax need not be a multiple of the output cadence.
    assert completion({0: '', 1000: ''}, 1249, 500) == (1000, 1000)
    last, expected = completion({0: '', 1000: ''}, 1500, 500)
    assert last < expected


def test_completion_reads_runtime_nmax_and_cadence(tmp_path):
    with h5py.File(tmp_path/'pfd.000001000_p000000.h5', 'w') as f:
        f.create_dataset('jeh-0/hx_fc/p0/3d', data=np.zeros((4, 4, 1)))
    (tmp_path/'analysis_config.json').write_text(json.dumps({'nmax': 1249, 'fields_every': 500}))
    env = {k: v for k, v in os.environ.items() if not k.startswith('PSC_ANALYSIS_')}
    r = subprocess.run([sys.executable, str(HERE/'check_run_completion.py'), '--data-dir',
                        str(tmp_path), '--case', 'mirror_bimaxwellian_moderate'],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == '1000 1000 1249 500'


def test_runner_sets_case_even_with_an_inherited_profile(tmp_path, monkeypatch):
    from test_evidence_tooling import run_runner
    monkeypatch.setenv('PSC_PROFILE', 'whistler_bimaxwellian_strong')
    r, _, calls = run_runner(tmp_path, ['manifest'])
    assert r.returncode == 0, r.stderr
    assert 'PSC_PROFILE=mirror_bimaxwellian_moderate' in calls[0]['args']


def test_runner_rejects_profile_overrides(tmp_path):
    from test_evidence_tooling import run_runner
    r, _, calls = run_runner(tmp_path, ['manifest'], '--make-option', 'PSC_PROFILE=W_S_bM')
    assert r.returncode == 2 and not calls


def test_no_products_is_a_failure_even_when_the_command_exits_zero(tmp_path):
    from test_evidence_tooling import run_runner
    r, state, _ = run_runner(tmp_path, ['manifest', 'fields'], spec={'fields': {'files': []}})
    assert r.returncode == 1
    assert state['stages']['fields']['execution_status'] == 'FAIL'
    assert state['stages']['fields']['returncode'] == 1
    assert 'no output files' in state['stages']['fields']['reason']


@pytest.mark.parametrize('consumer', ['anisotropy', 'spectral', 'field_panels', 'field_limits'])
def test_unreadable_snapshots_fail_instead_of_disappearing(monkeypatch, consumer):
    from data_reader import PICDataReader
    def broken(*args, **kwargs):
        raise OSError('truncated snapshot')
    monkeypatch.setattr(PICDataReader, 'read_multiple_fields_3d', broken)
    with pytest.raises(RuntimeError, match='truncated snapshot'):
        if consumer == 'anisotropy':
            from anisotropy_analysis import process_snapshot
            process_snapshot('moments.h5', 'fields.h5')
        elif consumer == 'spectral':
            from spectral_analysis import SpectralAnalyzer
            SpectralAnalyzer.__new__(SpectralAnalyzer).process_snapshot('fields.h5')
        else:
            from fluctuationofmagneticfiel import FieldImagePlotter
            p = FieldImagePlotter.__new__(FieldImagePlotter)
            p.file_map = {10: 'fields.h5'}
            if consumer == 'field_panels':
                p._build_panels(10)
            else:
                p.compute_global_scales([10], 'yz')


@pytest.mark.parametrize('gif,enabled', [('0', False), ('', False), ('1', True)])
def test_make_uses_profile_b0_and_explicit_gif_boolean(tmp_path, gif, enabled):
    r = subprocess.run(['make', '-n', 'fields', 'CASE=M_S_bM', f'DATA_DIR={tmp_path}',
                        f'PYTHON={sys.executable}', f'FIELDS_GIF={gif}'], cwd=HERE,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert '--B0' not in r.stdout
    assert ('--create_gifs' in r.stdout) == enabled
    assert f'"{sys.executable}" fluctuationofmagneticfiel.py' in r.stdout


@pytest.mark.parametrize('runner_rc,resonant_rc,expected', [(7, 0, 1), (0, 6, 1), (0, 0, 0)])
def test_batch_exit_reflects_runner_and_postprocessing_failures(tmp_path, runner_rc, resonant_rc, expected):
    """Run the actual shell orchestrator with tiny fake compute commands."""
    repo = tmp_path/'repo'; analysis = repo/'CodeforAnalisys'; analysis.mkdir(parents=True)
    (repo/'src').mkdir(); (repo/'src/cosma_adios2_env.sh').write_text('true\n')
    (analysis/'analysis_contract.py').write_text('CONVENTIONS_VERSION = 6\n')
    (analysis/'run_pipeline.py').write_text('# --launcher\n')
    case = 'mirror_bikappa5_moderate'
    run = tmp_path/'data/psc_mirror_bikappa5_moderate_12063822'; run.mkdir(parents=True)
    (run/'pfd.000000000_p000000.h5').touch()
    fake = tmp_path/'fake.py'
    fake.write_text('''import json, os, pathlib, sys
a = sys.argv[1:]
if a[0] == '-': os.execv(sys.executable, [sys.executable, *a])
if a[0] == 'check_run_completion.py': print('1000 1000 1000 100')
if a[0] == 'run_pipeline.py':
    root = pathlib.Path(a[a.index('--results-root')+1]) / a[a.index('--case')+1]
    root.mkdir(parents=True)
    # No successful physics -> cross-run comparisons are correctly skipped.
    (root/'pipeline.json').write_text(json.dumps({'stages': {'physics': {'execution_status': 'FAIL'}}}))
    sys.exit(int(os.environ['TEST_RUNNER_RC']))
if a[0] == 'resonant_anisotropy.py': sys.exit(int(os.environ['TEST_RESONANT_RC']))
''')
    wrapper = tmp_path/'python-wrapper'
    wrapper.write_text('#!/bin/sh\nexec '+shlex.join([sys.executable, str(fake)])+' "$@"\n')
    wrapper.chmod(0o755)
    srun = tmp_path/'srun'
    srun.write_text('#!/bin/sh\nwhile [ "${1#--}" != "$1" ]; do shift; done\nexec "$@"\n')
    srun.chmod(0o755)
    env = dict(os.environ, REPO=str(repo), RUN_ROOT=str(tmp_path/'data'), NEW_ROOT=str(tmp_path/'out'),
               LOG_DIR=str(tmp_path/'logs'), PYTHON=str(wrapper), ONLY=case,
               PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
               TEST_RUNNER_RC=str(runner_rc), TEST_RESONANT_RC=str(resonant_rc))
    for key in ('DRY_RUN', 'RESUME', 'EXTRA_RUNS', 'ALLOW_INCOMPLETE'):
        env.pop(key, None)
    result = subprocess.run(['bash', str(HERE.parent/'cosma_jobs/analisis/reanalysis_v6_all.sh')],
                            env=env, capture_output=True, text=True)
    assert result.returncode == expected, result.stdout+'\n'+result.stderr


def _batch_invocations(tmp_path, cases, nodes, extra_env=None):
    """Run the orchestrator with fake compute commands; return one record per runner."""
    repo = tmp_path/'repo'; analysis = repo/'CodeforAnalisys'; analysis.mkdir(parents=True)
    (repo/'src').mkdir(); (repo/'src/cosma_adios2_env.sh').write_text('true\n')
    (analysis/'analysis_contract.py').write_text('CONVENTIONS_VERSION = 6\n')
    (analysis/'run_pipeline.py').write_text('# --launcher\n')
    extra = []
    for case in cases:
        run = tmp_path/f'data/psc_{case}_1'; run.mkdir(parents=True)
        (run/'pfd.000000000_p000000.h5').touch()
        extra.append(f'{case}:psc_{case}_1')
    fake = tmp_path/'fake.py'
    fake.write_text('''import json, os, pathlib, sys
def _until_flag(argv, start):
    out = []
    for item in argv[start:]:
        if item.startswith('--'): break
        out.append(item)
    return out
a = sys.argv[1:]
if a[0] == '-': os.execv(sys.executable, [sys.executable, *a])
if a[0] == 'check_run_completion.py': print('1000 1000 1000 100')
if a[0] == 'run_pipeline.py':
    case = a[a.index('--case')+1]
    root = pathlib.Path(a[a.index('--results-root')+1]) / case
    root.mkdir(parents=True)
    (root/'pipeline.json').write_text(json.dumps({'stages': {'physics': {'execution_status': 'FAIL'}}}))
    # The runners are concurrent: append one line instead of read-modify-write.
    with open(os.environ['TEST_RECORD'], 'a') as fh:
        fh.write(json.dumps(
            {'case': case, 'jobs': a[a.index('--jobs')+1],
             'stages': _until_flag(a, a.index('--stages')+1)}) + '\\n')
''')
    wrapper = tmp_path/'python-wrapper'
    wrapper.write_text('#!/bin/sh\nexec '+shlex.join([sys.executable, str(fake)])+' "$@"\n')
    wrapper.chmod(0o755)
    srun = tmp_path/'srun'
    srun.write_text('#!/bin/sh\nwhile [ "${1#--}" != "$1" ]; do shift; done\nexec "$@"\n')
    srun.chmod(0o755)
    record = tmp_path/'record.jsonl'; record.write_text('')
    env = dict(os.environ, REPO=str(repo), RUN_ROOT=str(tmp_path/'data'),
               NEW_ROOT=str(tmp_path/'out'), LOG_DIR=str(tmp_path/'logs'),
               PYTHON=str(wrapper), ONLY=' '.join(cases), EXTRA_RUNS=' '.join(extra),
               SLURM_JOB_NUM_NODES=str(nodes), TEST_RECORD=str(record),
               PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
               TEST_RUNNER_RC='0', TEST_RESONANT_RC='0')
    for key in ('DRY_RUN', 'RESUME', 'ALLOW_INCOMPLETE', 'JOBS_PER_RUN', 'CONTROL_STAGES'):
        env.pop(key, None)
    env.update(extra_env or {})
    subprocess.run(['bash', str(HERE.parent/'cosma_jobs/analisis/reanalysis_v6_all.sh')],
                   env=env, capture_output=True, text=True)
    return {json.loads(l)['case']: json.loads(l) for l in record.read_text().splitlines()}


def test_stage_slots_follow_the_work_not_the_number_of_runs(tmp_path):
    """A fixed NODES/#CASES share serialised the long runs; the allocation throttles instead."""
    cases = ['mirror_bimaxwellian_moderate', 'mirror_bikappa3_moderate',
             'firehose_bikappa3_bigbox40', 'mirror_bimaxwellian_isotropic']
    got = _batch_invocations(tmp_path, cases, nodes=12)
    assert set(got) == set(cases)
    # Six slots -- where one run reaches its own critical path -- not 12/4 = 3,
    # and the allocation throttles the rest instead of the run count.
    assert {r['jobs'] for r in got.values()} == {'6'}
    # A small allocation cannot offer more slots than it has nodes.
    small = _batch_invocations(tmp_path/'small', cases, nodes=4)
    assert {r['jobs'] for r in small.values()} == {'4'}
    # And it stays pinnable for a site that refuses queued job steps.
    pinned = _batch_invocations(tmp_path/'pin', cases, nodes=12, extra_env={'JOBS_PER_RUN': '3'})
    assert {r['jobs'] for r in pinned.values()} == {'3'}


def test_isotropic_controls_only_run_the_stages_the_energy_audit_reads(tmp_path):
    """A control measures numerical heating; growth and spectra would describe an absent mode."""
    cases = ['mirror_bikappa3_moderate', 'mirror_bikappa3_isotropic']
    got = _batch_invocations(tmp_path, cases, nodes=8)
    control = got['mirror_bikappa3_isotropic']['stages']
    assert control == ['manifest', 'residuals', 'physics']
    # physical_diagnostics.py (the physics stage) writes every input of
    # energy_audit.audit_run, so the baseline subtraction still has its data.
    for stage in ('spectral', 'brazil', 'structures', 'fields', 'theory'):
        assert stage not in control
    anisotropic = got['mirror_bikappa3_moderate']['stages']
    assert {'spectral', 'brazil', 'structures', 'fields', 'theory'} <= set(anisotropic)
    assert _batch_invocations(tmp_path/'override', cases, nodes=8,
                              extra_env={'CONTROL_STAGES': 'manifest physics'}
                              )['mirror_bikappa3_isotropic']['stages'] == ['manifest', 'physics']


def _audit_stub(name, beta_i_par, A_i, kappa=None):
    """Minimal audit record: the fields match_control and pair_controls read."""
    return {'run': name, 'kappa': kappa,
            'ion_thermal_energy': beta_i_par * (0.5 + A_i),
            'role': 'isotropic_control' if A_i == 1.0 else 'run',
            'setup': {'mass_ratio': 200.0, 'B0': 0.08, 'beta_e_parallel': 1.0, 'A_e': 1.0,
                      'nicell_from_profile': 1000, 'dx_de': 0.49104637582399135,
                      'beta_i_parallel': beta_i_par, 'A_i': A_i, 'domain_di': 20.0,
                      'grid': [576, 576]}}


def test_a_control_of_another_series_does_not_silently_correct_a_run():
    """Every CONTROL_KEYS value is shared across this tree, so only the ion
    thermal energy separates the moderate mirror controls from the firehose runs."""
    import energy_audit
    # The profiles of the runs on COSMA: mirror moderate beta_i_par 5, A_i 2;
    # firehose strong 40 d_i beta_i_par 10, A_i 0.1; controls 25/3 with A_i 1.
    controls = [_audit_stub('mirror_bimaxwellian_isotropic', 25/3, 1.0),
                _audit_stub('mirror_bikappa3_isotropic', 25/3, 1.0, kappa=3.0)]
    mirror = _audit_stub('mirror_bikappa3_moderate', 5.0, 2.0, kappa=3.0)
    firehose = _audit_stub('firehose_bikappa3_bigbox40', 10.0, 0.1, kappa=3.0)
    audits = [mirror, firehose, *controls]

    control, match = energy_audit.match_control(mirror, audits)
    assert control['run'] == 'mirror_bikappa3_isotropic'      # same kappa
    assert match['ion_thermal_energy_ratio'] == pytest.approx(1.0)

    control, match = energy_audit.match_control(firehose, audits)
    assert control is None, 'a mirror control must not correct a firehose run'
    assert match['rejected'] == 'mirror_bikappa3_isotropic'
    assert match['ion_thermal_energy_ratio'] == pytest.approx(2.0833, rel=1e-3)
    assert 'no numerical-heating baseline' in match['reason']

    # An explicit pairing is the user's statement and is not capped.
    control, match = energy_audit.match_control(
        firehose, audits, explicit='mirror_bikappa3_isotropic')
    assert control['run'] == 'mirror_bikappa3_isotropic' and match['explicit']

    # pair_controls records the rejection instead of leaving no trace.
    energy_audit.pair_controls(audits)
    assert 'baseline' in mirror and 'baseline_unavailable' not in mirror
    assert 'baseline' not in firehose
    assert firehose['baseline_unavailable']['status'] == 'UNVERIFIED'


def _uniform_plus_hole_field(path, ngrid=32, b0=0.08, depth=0.5, hole=slice(0, 8)):
    """|B| = b0 everywhere except a compressive hole in the first rows."""
    bz = np.full((ngrid, ngrid), b0, dtype=float)
    bz[hole, :] = b0 * (1.0 - depth)
    zero = np.zeros((ngrid, ngrid), dtype=float)
    crd = (np.arange(ngrid) + 0.5)
    with h5py.File(path, 'w') as f:
        for name, arr in (('ex_ec', zero), ('ey_ec', zero), ('ez_ec', zero),
                          ('jx_ec', zero), ('jy_ec', zero), ('jz_ec', zero),
                          ('hx_fc', zero), ('hy_fc', zero), ('hz_fc', bz)):
            f.create_dataset(f'jeh-0/{name}/p0/3d', data=arr[:, :, None].astype('f4'))
        for axis in (1, 2):
            f.create_dataset(f'crd[{axis}]/p0/1d', data=crd - ngrid / 2)
    return bz


def test_mean_B_over_the_whole_box_is_a_series_of_its_own(tmp_path, monkeypatch):
    """<|B|> over the domain did not exist: only prt_B_mean_over_B0, the same
    quantity inside the particle window (about 4 % of the box)."""
    field = tmp_path/'pfd.000000000_p000000.h5'
    bz = _uniform_plus_hole_field(field, ngrid=32, b0=0.08, depth=0.5, hole=slice(0, 8))
    # _prt_cell_slices compares the field shape against the profile grid.
    monkeypatch.setattr(pd, 'N_GRID_Z', 32); monkeypatch.setattr(pd, 'N_GRID_Y', 32)
    # prt_window is (lo, hi) in grid cells per axis (x, y, z); a window on the
    # second half of z misses the hole in the first rows entirely.
    window = (np.array([0, 16, 16]), np.array([1, 32, 32]))
    metrics = pd.field_metrics(str(field), b0=0.08, prt_window=window)

    expected = float(np.mean(np.abs(bz))) / 0.08          # 1/4 of the box at 0.5 b0
    assert metrics['B_mean_over_B0'] == pytest.approx(expected, rel=1e-3)
    assert metrics['B_mean_over_B0'] == pytest.approx(0.875, rel=1e-3)
    assert metrics['B_mean'] == pytest.approx(0.08 * 0.875, rel=1e-3)
    # The window sees no hole, so the two disagree: that is the whole point.
    assert metrics['prt_B_mean_over_B0'] == pytest.approx(1.0, rel=1e-3)
    assert metrics['B_std_over_B0'] > 0.1
    # And it reaches the CSV: the worker keeps every scalar of field_metrics.
    _, row, _ = pd._field_metrics_worker((0, str(field), 0.08, window, []))
    for key in ('B_mean', 'B_mean_over_B0', 'B_std_over_B0'):
        assert key in row and isinstance(row[key], float)


def test_the_baseline_propagates_to_beta_e_and_A_e_without_replacing_the_raw():
    """The control's isotropic rise comes off both components; both versions stay."""
    import energy_audit
    b0, t_e0 = 0.08, 0.0032            # T_e0 = beta_e0 B0^2 / 2 with beta_e0 = 1
    # Run: T_bar doubles and the electrons end anisotropic at A_e = 1.5.
    # Control: T_bar rises by 60 % of T_e0, isotropically.
    ws = {'omega_ci_t': np.array([0.0, 10.0]),
          'E_internal_e': 1.5 * np.array([t_e0, 2 * t_e0]),
          'A_e': np.array([1.0, 1.5]),
          'T_e_over_T_e0': np.array([1.0, 2.0])}
    ctrl = 1.5 * np.array([t_e0, 1.6 * t_e0])
    ok = np.array([True, True])
    got = energy_audit._corrected_electrons(ws, ctrl, ok, {'setup': {'B0': b0}})

    # Raw: T_bar = 2 T_e0, A_e = 1.5 -> T_par = 3(2 T_e0)/(1 + 3) = 1.5 T_e0.
    assert got['T_e_par'][1] == pytest.approx(1.5 * t_e0)
    assert got['T_e_perp'][1] == pytest.approx(1.5 * 1.5 * t_e0)
    assert got['beta_e_par'][1] == pytest.approx(1.5 * t_e0 * 2 / b0**2)
    # Control rose by 0.6 T_e0, taken off both components.
    assert got['T_e_par_corrected'][1] == pytest.approx((1.5 - 0.6) * t_e0)
    assert got['T_e_perp_corrected'][1] == pytest.approx((2.25 - 0.6) * t_e0)
    # The anisotropy the correction implies is larger than the measured one:
    # removing the same amount from both components raises the ratio.
    assert got['A_e_corrected'][1] == pytest.approx(1.65 / 0.9)
    assert got['A_e_corrected'][1] > got['A_e'][1]
    assert got['beta_e_par_corrected'][1] == pytest.approx(0.9 * t_e0 * 2 / b0**2)
    # At t = 0 the control has not heated yet, so nothing is subtracted.
    assert got['T_e_par_corrected'][0] == pytest.approx(got['T_e_par'][0])
    assert got['A_e_corrected'][0] == pytest.approx(got['A_e'][0])


def test_a_control_that_absorbs_the_whole_temperature_gives_nan_not_a_negative():
    import energy_audit
    t_e0 = 0.0032
    ws = {'omega_ci_t': np.array([0.0, 10.0]),
          'E_internal_e': 1.5 * np.array([t_e0, 1.1 * t_e0]),
          'A_e': np.array([1.0, 1.0]),
          'T_e_over_T_e0': np.array([1.0, 1.1])}
    ctrl = 1.5 * np.array([t_e0, 5.0 * t_e0])      # control hotter than the run
    got = energy_audit._corrected_electrons(ws, ctrl, np.array([True, True]),
                                            {'setup': {'B0': 0.08}})
    assert not np.isfinite(got['T_e_par_corrected'][1])
    assert not np.isfinite(got['A_e_corrected'][1])
    assert np.isfinite(got['T_e_par'][1]), 'the raw column must survive'


def _kappa_field_run(root: Path, kappa0, inv0, t_end=40.0):
    """One run root with the two products field_vs_kappa.py reads."""
    (root/'09_physical_diagnostics').mkdir(parents=True, exist_ok=True)
    (root/'03_particles').mkdir(parents=True, exist_ok=True)
    t_field = np.linspace(0.0, t_end, 161)
    amp = 0.02 * (1.0 - np.exp(-((t_field / 14.0) ** 3)))
    with (root/'09_physical_diagnostics'/'field_fluctuation_table.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['step', 'omega_ci_t', 'delta_B_vec_rms_over_B0',
                    'delta_B_parallel_rms_over_B0', 'B_mean_over_B0', 'B_std_over_B0'])
        for i, t in enumerate(t_field):
            w.writerow([i * 1000, f'{t:.6g}', f'{amp[i]:.6g}', f'{0.6 * amp[i]:.6g}',
                        f'{1.0 - 1.5 * amp[i] ** 2:.6g}', f'{amp[i] * 1.2:.6g}'])
    t_part = np.linspace(0.0, t_end, 21)
    inv = inv0 + 0.05 * (1.0 - np.exp(-t_part / 12.0))
    with (root/'03_particles'/'vdf_kappa_series.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['step', 'omega_ci_t', 'population', 'inv_kappa_signed',
                    'inv_kappa_signed_error', 'A_local_b'])
        for i, t in enumerate(t_part):
            w.writerow([i * 8000, f'{t:.6g}', 'all', f'{inv[i]:.6g}', '0.004', '2.0'])
    with (root/'09_physical_diagnostics'/'growth_rate_summary.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['mode', 'gamma', 'gamma_err', 'linear_phase_end', 'is_reference'])
        w.writerow(['mode_compressive', '0.21', '0.02', '16.0', 'True'])
    return root


def test_field_against_kappa_figures_drop_the_quiet_start_and_pass_layout_qa(tmp_path):
    """Both figures come from products only, and the field series obeys the
    fluctuation rule: no t = 0 and no quiet-start noise build-up."""
    roots = [_kappa_field_run(tmp_path/'mirror_bimaxwellian_moderate', None, 0.0),
             _kappa_field_run(tmp_path/'mirror_bikappa5_moderate', 5.0, 0.20),
             _kappa_field_run(tmp_path/'mirror_bikappa3_moderate', 3.0, 1/3)]
    out = tmp_path/'out'
    env = dict(os.environ, MPLCONFIGDIR='/tmp/psc-mpl', PSC_FIG_THEME='paper')
    r = subprocess.run([sys.executable, str(HERE/'field_vs_kappa.py'), *map(str, roots),
                        '--outdir', str(out)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    names = ('field_vs_kappa_instants.png', 'field_vs_kappa_trajectory.png')
    for name in names:
        assert (out/name).is_file()
        # The vector copy is opt-in (PSC_FIG_PDF), so the default pass must
        # NOT pay for it: a PDF here means the switch leaked back on.
        assert not (out/name).with_suffix('.pdf').exists(), 'PDF is opt-in now'

    # ... and PSC_FIG_PDF=1 still produces it for the submission pass.
    out_pdf = tmp_path/'out_pdf'
    r = subprocess.run([sys.executable, str(HERE/'field_vs_kappa.py'), *map(str, roots),
                        '--outdir', str(out_pdf)],
                       env=dict(env, PSC_FIG_PDF='1'), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    for name in names:
        assert (out_pdf/name).with_suffix('.pdf').is_file(), 'PSC_FIG_PDF=1 writes the PDF'

    # plot_style records a layout check per figure; it has to come back empty.
    qa = [json.loads(l) for f in out.glob('figure_qa_*.jsonl')
          for l in f.read_text().splitlines()]
    assert qa and all(not entry.get('issues') for entry in qa), qa

    rows = list(csv.DictReader((out/'field_vs_kappa.csv').open()))
    assert rows
    times = sorted({float(r['omega_ci_t']) for r in rows})
    assert times[0] > 0.0, 't = 0 is float rounding of the uniform B0'
    # psc_units.noise_settling_time of the box fundamental for these profiles.
    assert times[0] >= 2.5, f'quiet-start build-up still drawn (first t = {times[0]})'
    # Both field measures reach the table, and the index is the measured one.
    assert all(r['B_mean_over_B0'] and r['delta_B_vec_rms_over_B0'] for r in rows)
    assert {r['run'] for r in rows} == {p.name for p in roots}
    k3 = [r for r in rows if r['run'] == 'mirror_bikappa3_moderate']
    assert float(k3[0]['inv_kappa']) > float(
        [r for r in rows if r['run'] == 'mirror_bimaxwellian_moderate'][0]['inv_kappa'])


def test_story_figure_shares_one_scale_per_quantity_across_the_columns(tmp_path):
    """The composite figure is only comparable if the columns share scales.

    Per-column autoscaling is what makes the bi-Maxwellian's 1/kappa_eff --
    zero to within its +-0.004 noise floor -- draw as a large oscillation
    beside the real decline of the kappa runs.
    """
    import paper_story_figure as psf

    roots = [_kappa_field_run(tmp_path/'mirror_bimaxwellian_moderate', None, 0.0),
             _kappa_field_run(tmp_path/'mirror_bikappa5_moderate', 5.0, 0.20),
             _kappa_field_run(tmp_path/'mirror_bikappa3_moderate', 3.0, 1/3)]
    out = tmp_path/'story.png'
    psf.build(roots, {}, (10.0, 30.0), out, with_maps=False)
    assert out.is_file()
    assert not out.with_suffix('.pdf').exists(), 'PDF stays opt-in'

    # Columns are ordered Maxwellian, kappa 3, kappa 5 whatever the input order.
    assert [psf.short_name(r) for r in psf.order_runs(roots)] == \
        ['bimaxwellian', 'bikappa3', 'bikappa5']

    qa = [json.loads(l) for f in tmp_path.glob('figure_qa_*.jsonl')
          for l in f.read_text().splitlines()]
    assert qa and all(not entry.get('issues') for entry in qa), qa


def test_story_figure_parallel_map_is_self_calibrating_and_uses_bz(tmp_path):
    """dB_par comes from Bz (B0 is along z) and B0 defaults to the run's <Bz>.

    Reading hx, or passing a B0 from a different profile, does not fail -- it
    silently rescales every panel -- so both are pinned here.
    """
    import paper_story_figure as psf
    from data_reader import PICDataReader

    run = tmp_path/'synth'
    subprocess.run([sys.executable, str(HERE/'synthetic_run.py'), str(run),
                    '--ngrid', '32', '--t-end', '40', '--snapshots', '4',
                    '--ppc', '20'], check=True, capture_output=True)
    steps = sorted(PICDataReader.find_files(str(run/'pfd.*.h5')))

    first = psf.parallel_fluctuation_map(run, steps[0])
    last = psf.parallel_fluctuation_map(run, steps[-1])
    # At t = 0 the field is the uniform B0, so dB_par is rounding ...
    assert float(np.sqrt((first ** 2).mean())) < 1e-3
    # ... and it has grown by the end of the run.
    assert float(np.sqrt((last ** 2).mean())) > 10 * float(np.sqrt((first ** 2).mean()))
    # A wrong B0 shows up as a constant offset of the whole map.
    skewed = psf.parallel_fluctuation_map(run, steps[0], b0=0.05)
    assert abs(float(skewed.mean())) > 0.1
