#!/usr/bin/env python3
"""Ordered PSC stages with isolated output, atomic completion records and resumability.

Each stage runs ``make <stage>`` into its own staging directory. Its products are
promoted into the results directory only after the command succeeds, and a stage
may never overwrite a product of another stage (that would make every resume
re-run both). Independent stages can run concurrently (``--jobs``), optionally
through a cluster launcher such as ``srun --nodes=1 --ntasks=1 --exclusive``
(``--launcher``); bookkeeping always stays in this one process.
"""
from __future__ import annotations
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from analysis_contract import atomic_json, provenance, inventory_identity
from quality_report import write_report

# Default stage set. `physics` already writes the DiagEnergies budget
# (global_energy_*), so `energy-if-present` is only for a run analysed without it.
STAGES = ('manifest', 'residuals', 'physics', 'brazil', 'spectral', 'structures',
          'energy-exchange', 'estimators', 'heatflux', 'particles', 'validate',
          'diamagnetic', 'fields', 'vdf-spatial')
OPTIONAL = ('energy-if-present', 'theory', 'theory-liouville')
# Ordering constraints, whatever the outcome of the earlier stage: the spectral
# windows come from the accepted linear phase of `physics`, the polarization
# overlay from `theory`. Every stage additionally requires a passed manifest.
AFTER = {'spectral': ('physics', 'theory')}
# Relative cost at 576^2; only the longest-first launch order depends on it.
COST = {'physics': 10, 'fields': 8, 'spectral': 7, 'structures': 6, 'diamagnetic': 6,
        'energy-exchange': 5, 'brazil': 4, 'heatflux': 4, 'vdf-spatial': 3,
        'residuals': 3, 'estimators': 3, 'particles': 3}
RESERVED = ('OUTPUT_DIR', 'DATA_DIR', 'CASE', 'PYTHON', 'RESULTS_ROOT')
MAKE_DIR = Path(__file__).resolve().parent


def case_instability(case):
    """Instability family of the PSC_PROFILE named `case`; also validates it exists."""
    env = dict(os.environ, PSC_PROFILE=case)
    env.pop('PSC_ANALYSIS_DATA_DIR', None)
    r = subprocess.run([sys.executable, '-c', 'import psc_units as u; print(u.INSTABILITY)'],
                       cwd=MAKE_DIR, env=env, capture_output=True, text=True)
    if r.returncode:
        raise ValueError(f'Unknown case/profile {case!r}: {r.stderr.strip().splitlines()[-1:]}')
    return r.stdout.strip()


def execute(command, log, env):
    """Run one stage; per-process resources via wait4, not cumulative RUSAGE_CHILDREN."""
    start = time.monotonic()
    with log.open('w') as f:
        proc = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, env=env)
        _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = (os.WEXITSTATUS(status) if os.WIFEXITED(status)
                       else -os.WTERMSIG(status))
    return proc.returncode, usage.ru_maxrss, time.monotonic() - start


def is_complete(entry, out):
    products = entry.get('products', [])
    return (entry.get('execution_status') == 'PASS'
            and all((out / f).is_file() for f in products)
            and inventory_identity([out / f for f in products]) == entry.get('output_identity'))


def stage_options(stage, out, options, state):
    """Make options of one stage plus a record of where its analysis window came from."""
    options, notes = list(options), {}
    if stage != 'spectral':
        return options, notes
    if any(x.startswith(('GROWTH_T_', 'DISPERSION_T_')) for x in options):
        notes['window_source'] = 'explicit make option'
    else:
        phase_file = out / '09_physical_diagnostics' / 'linear_phase.json'
        phase = json.loads(phase_file.read_text()) if phase_file.exists() else {}
        if phase.get('status') == 'PASS':
            options += [f'GROWTH_T_START={phase["start"]}', f'GROWTH_T_END={phase["end"]}',
                        f'DISPERSION_T_START={phase["start"]}', f'DISPERSION_T_END={phase["end"]}']
            notes['window_source'] = 'accepted linear phase (09_physical_diagnostics/linear_phase.json)'
        else:
            notes['window_source'] = 'script defaults; no accepted linear phase'
    theory = out / '04_spectra' / 'linear_theory.csv'
    if (not any(x.startswith('THEORY_CSV=') for x in options)
            and theory.is_file() and state['stages'].get('theory', {}).get('execution_status') == 'PASS'):
        options.append(f'THEORY_CSV={theory}')
        notes['theory_csv'] = str(theory)
    return options, notes


def promote(stage, scratch, out, state):
    """Move a finished stage's products into place; refuse to overwrite another stage."""
    files = sorted(f for f in scratch.rglob('*') if f.is_file())
    owners = {p: s for s, e in state['stages'].items() if s != stage for p in e.get('products', [])}
    clash = sorted({owners[str(f.relative_to(scratch))] + ': ' + str(f.relative_to(scratch))
                    for f in files if str(f.relative_to(scratch)) in owners})
    if clash:
        raise RuntimeError('would overwrite products of other stages: ' + '; '.join(clash[:5]))
    products = []
    for f in files:
        relative = f.relative_to(scratch); target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if f.name.endswith('_analysis_manifest.json'):
            manifest = json.loads(f.read_text()); manifest['output_directory'] = str(out)
            atomic_json(f, manifest)
        os.replace(f, target)
        products.append(str(relative))
    # A re-run replaces the stage's previous products; drop the ones it no longer makes.
    for old in set(state['stages'].get(stage, {}).get('products', [])) - set(products):
        (out / old).unlink(missing_ok=True)
    shutil.rmtree(scratch, ignore_errors=True)
    return products


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True, type=Path); p.add_argument('--case', required=True)
    p.add_argument('--results-root', type=Path, default=MAKE_DIR.parent / 'analysis_results' / 'v6')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stages', nargs='+', choices=STAGES + OPTIONAL, default=list(STAGES))
    p.add_argument('--make-option', action='append', default=[], help='Explicit KEY=VALUE Make override, recorded in provenance')
    p.add_argument('--jobs', type=int, default=1, help='Stages run concurrently (default 1: strictly sequential)')
    p.add_argument('--launcher', default='', help="Command prefix per stage, e.g. 'srun --nodes=1 --ntasks=1 --exclusive'")
    p.add_argument('--keep-going', action='store_true',
                   help='After a failed stage, still run the stages that do not depend on it')
    a = p.parse_args()
    if any('=' not in x or x.split('=')[0] in RESERVED for x in a.make_option):
        p.error('make options must be KEY=VALUE and cannot override run identity/output')
    if a.jobs < 1:
        p.error('--jobs must be at least 1')
    selected = set(a.stages)
    if {'energy-if-present', 'physics'} <= selected:
        p.error('physics already writes the DiagEnergies budget; energy-if-present is only for runs without physics')
    try:
        instability = case_instability(a.case)
    except ValueError as exc:
        p.error(str(exc))
    if 'theory' in selected and instability == 'mirror':
        p.error('The parallel linear solver is not a mirror prediction; theory is only for parallel-propagating cases')
    data = a.data_dir.resolve(); out = (a.results_root / a.case).resolve()
    if not data.is_dir(): p.error('Data directory does not exist')
    prov = provenance()
    inputs = inventory_identity([x for x in data.iterdir() if x.suffix in ('.bp', '.h5', '.asc', '.out', '.log')])
    signature = {'source': prov['analysis_source_sha256'], 'input': inputs, 'case': a.case,
                 'config': prov['run_config'], 'environment': prov['environment'], 'options': a.make_option}
    record = out / 'pipeline.json'
    previous = json.loads(record.read_text()) if record.exists() else {}
    if out.exists() and any(out.iterdir()) and not a.resume:
        p.error('Output exists; use a new results root or --resume for identical inputs/configuration')
    if a.resume and previous and previous.get('signature') != signature:
        p.error('Source, inputs or configuration changed; use a new results root')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.pipeline.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('Another pipeline is using this output directory')
    except OSError as exc:  # e.g. Lustre mounted without flock support
        print(f'warning: no exclusive lock on this filesystem ({exc}); '
              f'never run two pipelines on {out}', file=sys.stderr)
    state = previous or {'signature': signature, 'provenance': prov, 'stages': {}}
    state['instability'] = instability
    atomic_json(record, state)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PSC_ANALYSIS_RUN_ID=os.environ.get('PSC_ANALYSIS_RUN_ID', data.name))
    launcher = shlex.split(a.launcher)

    # Manifest is always a real dependency; science is never launched before preflight.
    stages = ['manifest'] + [s for s in STAGES[1:] + OPTIONAL if s in selected]
    pending = [s for s in stages if not (a.resume and is_complete(state['stages'].get(s, {}), out))]
    changed = True
    while changed:  # a re-run earlier stage invalidates the stages ordered after it
        changed = False
        for s in stages:
            if s not in pending and any(d in pending for d in AFTER.get(s, ())):
                pending.append(s); changed = True
    for s in stages:
        if s not in pending:
            print(f'{s}: up to date', flush=True)
    finished, failed, running = {s for s in stages if s not in pending}, set(), {}
    stop = False

    def ready():
        if 'manifest' in pending:
            return ['manifest'] if not running else []
        if 'manifest' in failed:
            return []
        busy = set(pending) | {s for s, *_ in running.values()}
        return sorted((s for s in pending if not any(d in busy for d in AFTER.get(s, ()))),
                      key=lambda s: -COST.get(s, 1))

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        while pending or running:
            for stage in ([] if stop else ready()):
                if len(running) >= a.jobs:
                    break
                pending.remove(stage)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
                scratch = out / '.staging' / (stage + '-' + stamp)
                scratch.mkdir(parents=True)
                options, notes = stage_options(stage, out, a.make_option, state)
                command = [*launcher, 'make', '-C', str(MAKE_DIR), stage, f'PYTHON={sys.executable}',
                           f'DATA_DIR={data}', f'CASE={a.case}', f'OUTPUT_DIR={scratch}', *options]
                log = out / 'logs' / (stage + '-' + stamp + '.log'); log.parent.mkdir(exist_ok=True)
                state['stages'][stage] = {'execution_status': 'RUNNING', 'command': command, 'log': str(log),
                                          'started_utc': stamp, 'scientific_status': 'UNVERIFIED', **notes,
                                          'products': state['stages'].get(stage, {}).get('products', [])}
                atomic_json(record, state)
                print(f'{stage}: started', flush=True)
                running[pool.submit(execute, command, log, env)] = (stage, scratch)
            if not running:
                break  # nothing left that can start
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                stage, scratch = running.pop(future)
                entry = state['stages'][stage]
                try:
                    rc, maxrss, elapsed = future.result()
                except OSError as exc:
                    rc, maxrss, elapsed, entry['reason'] = 127, None, 0.0, f'could not start: {exc}'
                entry.update(elapsed_seconds=elapsed, returncode=rc, maxrss_native_units=maxrss,
                             maxrss_scope='launcher process' if launcher else 'make process tree peak')
                if rc == 0:
                    try:
                        products = promote(stage, scratch, out, state)
                    except RuntimeError as exc:
                        rc, entry['reason'] = 1, str(exc)
                if rc:
                    entry['execution_status'] = 'FAIL'
                    failed.add(stage)
                    print(f'FAILED {stage}: {entry.get("reason") or entry["log"]}', file=sys.stderr, flush=True)
                    stop = stop or not a.keep_going
                else:
                    entry.update(execution_status='PASS', products=products,
                                 output_identity=inventory_identity([out / f for f in products]))
                    print(f'{stage}: completed ({elapsed:.1f}s)', flush=True)
                finished.add(stage)
                atomic_json(record, state)
    for stage in pending:
        reason = 'manifest preflight failed' if 'manifest' in failed else 'stopped after an earlier failure'
        state['stages'][stage] = {**state['stages'].get(stage, {}), 'execution_status': 'NOT_RUN', 'reason': reason}
        print(f'{stage}: not run ({reason})', file=sys.stderr)
    atomic_json(record, state)
    write_report([out], out / 'report')
    return 1 if failed or pending else 0


if __name__ == '__main__':
    raise SystemExit(main())
