"""Regression coverage for production batch failures and false success states."""
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
