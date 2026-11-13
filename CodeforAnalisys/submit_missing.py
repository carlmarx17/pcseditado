"""Plan one resumable COSMA analysis per completed periodic PSC history."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from analysis_contract import atomic_json, inventory_identity, provenance
from psc_units import list_profiles
from run_naming import FAMILIES, family_of
from run_pipeline import STAGES, is_complete

HERE = Path(__file__).resolve().parent
WORKER = HERE.parent/'cosma_jobs/analisis/tesis_pendientes.sh'


def inventory(raw, archives, out):
    by_input = {}
    for manifest in sorted(archives.rglob('*_analysis_manifest.json')):
        if '.staging' in manifest.parts or out in manifest.parents:
            continue
        if (manifest.parent/'pipeline.json').is_file():
            meta = json.loads(manifest.read_text())
            by_input.setdefault(meta.get('input_directory'), []).append(str(manifest.parent))
    profiles = sorted((p for p in list_profiles() if family_of(p)), key=len, reverse=True)
    source = provenance()['analysis_source_sha256']
    runs = []
    for data in sorted(raw.glob('psc_*')):
        if not data.is_dir() or not family_of(data.name[4:]):
            continue
        case = next((p for p in profiles if data.name == 'psc_'+p or data.name.startswith('psc_'+p+'_')), None)
        if not case:
            raise ValueError(f'Unknown profile: {data}')
        bp = next(data.glob('pfd.*.bp'), None)
        h5 = next(data.glob('pfd.*.h5'), None)
        if not (bp or h5):
            print(f'{data.name}: sin campos')
            continue
        if bp and h5:
            raise ValueError(f'Mixed field formats in {data}')
        check = subprocess.check_output([sys.executable, str(HERE/'check_run_completion.py'),
                                         '--case', case, '--data-dir', str(data)], text=True)
        last, expected, *_ = map(int, check.split())
        control = case.endswith('_isotropic')
        if last == 0 or (not control and last < expected):
            print(f'{data.name}: incompleta ({last}/{expected})')
            continue
        family = family_of(case)
        stages = ['manifest', 'physics', 'residuals', 'energy-exchange'] if control else [
            *STAGES, 'theory', 'anisotropy-dispersion', 'resonant-anisotropy']
        options = ['FIELDS_GIF=0', 'SNAPSHOT_EVERY=100000', 'RUN_TAG='+data.name.removeprefix('psc_'+case+'_')]
        if bp:
            options += [f'FIELD_PATTERN={data}/pfd.*.bp', f'MOMENT_PATTERN={data}/pfd_moments.*.bp',
                        f'PARTICLE_PATTERN={data}/prt_{case}.*.bp']
        if family == 'mirror' and not control:
            stages.append('theory-liouville')
            options.append('STRUCTURES_FLAGS=--attribution')
        elif not control:
            options.append('STRUCTURES_FLAGS=--budget')
        if family == 'whistler':
            options.append('RESONANT_FLAGS=--species electron')
        stages.append('heating-budget')
        dest = out/'runs'/data.name/case
        record = dest/'pipeline.json'
        previous = json.loads(record.read_text()) if record.exists() else {}
        sig = previous.get('signature', {})
        done = False
        if previous:
            config = data/'analysis_config.json'
            inputs = inventory_identity(p for p in data.iterdir() if p.suffix in ('.bp','.h5','.asc','.out','.log'))
            if (sig.get('source') != source or sig.get('input') != inputs or sig.get('options') != options
                    or sig.get('case') != case or sig.get('config') != (json.loads(config.read_text()) if config.exists() else None)):
                raise ValueError(f'{dest}: changed inputs or code; choose another OUT inside ARCHIVES to reuse compatible stages')
            done = all(is_complete(previous['stages'].get(s, {}), dest) for s in stages) and (dest/'report/index.html').is_file()
        item = dict(data=str(data), case=case, dest=str(dest), stages=stages, options=options,
                    archives=by_input.get(str(data), []), done=done)
        runs.append(item)
        print(f'{data.name}: {"terminada" if done else "pendiente"}; {len(item["archives"])} archivos candidatos')
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submit', action='store_true')
    p.add_argument('--run-root', type=Path, default=os.getenv('RUN_ROOT','/cosma7/data/dp433/dc-mart18/anisotropy_adios2'))
    p.add_argument('--archives', type=Path, default=os.getenv('ARCHIVES', HERE.parent/'analysis_results'))
    p.add_argument('--out', type=Path, default=os.getenv('OUT', HERE.parent/'analysis_results/tesis_pendientes'))
    p.add_argument('--worker', type=Path)
    args = p.parse_args()
    if args.worker:
        plan = json.loads(args.worker.read_text())
        if os.environ.get('PSC_REVIEW') == '1':
            from energy_audit import audit_run, pair_controls, group_runs, write_outputs
            audits = [audit_run(r['dest'], Path(r['data']).name) for r in plan['runs']]
            pair_controls(audits)
            write_outputs(audits, group_runs(audits), Path(plan['out'])/'revision')
            return
        row = [r for r in plan['runs'] if not r['done']][int(os.environ['SLURM_ARRAY_TASK_ID'])]
        cmd = [sys.executable, str(HERE/'run_pipeline.py'), '--case', row['case'], '--data-dir', row['data'],
               '--results-root', str(Path(row['dest']).parent), '--resume', '--stages', *row['stages']]
        for option in row['options']:
            cmd += ['--make-option', option]
        for arc in row['archives']:
            cmd += ['--reuse-from', arc]
        subprocess.run(cmd, check=True)
        return
    raw, archives, out = (v.resolve() for v in (args.run_root, args.archives, args.out))
    if not raw.is_dir() or not archives.is_dir():
        p.error('RUN_ROOT and ARCHIVES must exist')
    rows = inventory(raw, archives, out)
    if not rows:
        p.exit(1, 'Sin historias analizables.\n')
    count = sum(not r['done'] for r in rows)
    for family in FAMILIES:
        if not any(family_of(r['case']) == family for r in rows):
            print(f'{family}: sin historias analizables')
    print(f'{count} historias con etapas pendientes de {len(rows)}.')
    if not args.submit:
        return
    # Persist each submission separately so a new inventory cannot change queued tasks.
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out/'.submission.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    # Prevent a second invocation from reserving another array for this output.
    active = out/'submitted.json'
    if active.exists():
        jobids = json.loads(active.read_text())
        queued = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i'], text=True)
        if {job.split("_")[0] for job in queued.split()} & set(jobids):
            print('Ya hay un envío activo para este directorio', ', '.join(jobids))
            return
    folder = Path(tempfile.mkdtemp(prefix='envio-', dir=out))
    plan = folder/'plan.json'
    atomic_json(plan, {'runs': rows, 'out': str(out)})
    base = ['sbatch', '--parsable', '--chdir='+str(HERE.parent), '--output='+str(folder/'%A_%a.log')]
    dep = []
    if count:
        jid = subprocess.check_output([*base, f'--array=0-{count-1}%1', str(WORKER), str(plan)], text=True).strip().split(';')[0]
        dep = ['--dependency=afterok:'+jid, '--kill-on-invalid-dep=yes']
        atomic_json(active, [jid])
        print('Array Slurm', jid)
    elif (out/'revision/energy_audit.json').exists():
        # An earlier report is current only for the very same set of histories.
        prior = json.loads((out/'revision/energy_audit.json').read_text())
        if {a['root'] for a in prior['runs']} == {r['dest'] for r in rows}:
            return
    review = subprocess.check_output([*base, *dep, '--time=00:30:00', '--export=ALL,PSC_REVIEW=1',
                                      str(WORKER), str(plan)], text=True).strip().split(';')[0]
    atomic_json(active, ([jid] if count else []) + [review])
    print('Revisión Slurm', review)


if __name__ == '__main__':
    main()
