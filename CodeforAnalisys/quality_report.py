#!/usr/bin/env python3
"""Read-only scientific audit of result products; writes a separate traceable report."""
from __future__ import annotations
import argparse
import csv
import html
import json
import os
from pathlib import Path
import numpy as np
from analysis_contract import atomic_json, ALGORITHMS, CONVENTIONS_VERSION
from growth_fit import reference_growth_row
import energy_audit

#: Worst first; a run's overall status is the worst of its checks.
SEVERITY = ('FAIL', 'UNVERIFIED', 'WARN', 'PASS')


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def figure_issues(root):
    """Latest content record per figure (a re-run appends), from plot_style.save."""
    latest = {}
    for registry in sorted(Path(root).rglob('figure_qa_*.jsonl')):
        for line in registry.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                latest[str(registry.parent / entry['file'])] = entry
    issues = [f'{Path(f).relative_to(root)}: {i}' for f, e in sorted(latest.items())
              for i in e.get('issues', []) + ([e['qa_error']] if 'qa_error' in e else [])]
    return len(latest), issues


def inspect_run(root, energy_tolerance=None):
    root = Path(root)
    phys = root / '09_physical_diagnostics'
    checks = []
    def check(name, status, reason, path, metrics=None):
        checks.append(dict(check=name, status=status, reason=reason, source=str(path), metrics=metrics or {}))
    manifests = sorted(root.glob('*_analysis_manifest.json'))
    manifest = read_json(manifests[0]) if len(manifests) == 1 else {}
    current = (manifest.get('physics', {}).get('analysis_conventions_version') == CONVENTIONS_VERSION
               and manifest.get('provenance', {}).get('algorithms') == ALGORITHMS)
    check('provenance', 'PASS' if current else 'UNVERIFIED',
          'Current algorithm manifest' if current else 'Missing or legacy algorithm provenance', root)
    check('runtime', 'PASS' if manifest.get('runtime_verified') else 'UNVERIFIED',
          'Explicit runtime configuration' if manifest.get('runtime_verified') else 'dt/nicell require runtime verification', root)
    pipeline = read_json(root / 'pipeline.json')
    if pipeline:
        bad = {s: e.get('execution_status') for s, e in pipeline.get('stages', {}).items()
               if e.get('execution_status') != 'PASS'}
        check('execution', 'FAIL' if bad else 'PASS',
              'Stages not completed: ' + ', '.join(f'{s} ({v})' for s, v in sorted(bad.items()))
              if bad else 'All selected stages completed', root / 'pipeline.json', bad)
    # Global energy: the audit reads DiagEnergies when present and otherwise the
    # prt-window moments, so a missing diag.asc no longer hides a heating that
    # the moments show just as clearly.
    audit = energy_audit.audit_run(root)
    path = phys / 'global_energy_summary.json'
    energy = read_json(path)
    drift = energy.get('max_abs_relative_change')
    status, reason = audit['status'], audit['reason']
    if status != 'FAIL' and energy_tolerance is not None and drift is not None and np.isfinite(drift):
        status = 'PASS' if drift <= energy_tolerance else 'FAIL'
        reason = f'Maximum drift {drift:.6g}; configured tolerance {energy_tolerance:g}'
    check('global_energy', status, reason, phys,
          {k: v for k, v in audit.items() if k in ('global', 'window', 'early_time', 'window_vs_global')})
    path = phys / 'growth_rate_summary.csv'
    if path.exists():
        with path.open(newline='') as handle:
            growth = reference_growth_row(list(csv.DictReader(handle)))
    else:
        growth = None
    accepted = bool(growth) and str(growth.get('fit_ok')) in ('1', 'True', 'true')
    series = (growth or {}).get('series') or 'total'
    if not growth:
        status, reason = 'UNVERIFIED', 'Missing growth measurement'
    elif not accepted:
        status, reason = 'UNVERIFIED', growth.get('fit_reject_reason') or 'Reference fit rejected'
    elif series != 'mode':
        # The domain rms of dB mixes every Fourier mode and is biased low; an
        # accepted fit of it is still not the modal growth rate of the run.
        status, reason = 'UNVERIFIED', f"Legacy '{series}' estimator (domain rms, biased low); regenerate with the modal reference"
    else:
        status, reason = 'PASS', 'Accepted modal fit; mode identity assessed separately'
    check('growth_fit', status, reason, path, growth)
    path = phys / 'field_residuals_summary.json'
    residuals = read_json(path)
    for key in ('gauss', 'continuity'):
        value, limit = residuals.get(key + '_max_err'), residuals.get('thresholds', {}).get(key)
        known = value is not None and limit is not None
        status = 'FAIL' if known and value > limit else ('PASS' if known and residuals.get('log_reaches_nmax') else 'UNVERIFIED')
        reason = {'FAIL': f'Maximum error {value:g} above PSC threshold {limit:g}' if known else '',
                  'PASS': f'Maximum error {value:g} within PSC threshold {limit:g} over the whole run' if known else '',
                  'UNVERIFIED': 'Log coverage and solver threshold required'}[status]
        check(key, status, reason, path, {'maximum': value, 'threshold': limit})
    mode_paths = sorted((root / '04_spectra').glob('dispersion_modes*.json'))
    modes = [read_json(p) for p in mode_paths]
    confirmed = any(m.get('dominant') and m.get('dominant', {}).get('accepted') for m in modes)
    check('mode_identity', 'PASS' if confirmed else 'UNVERIFIED',
          'Coherent mode characterized; branch assignment still requires physical interpretation' if confirmed else 'No accepted dominant-mode evidence', root / '04_spectra')
    path = phys / 'energy_exchange_summary.json'
    exchange = read_json(path)
    check('energy_closure', exchange.get('scientific_status', 'UNVERIFIED'),
          exchange.get('reason', 'Closure not independently validated'), path, exchange)
    init = list((root / '08_validation').glob('*validation_summary.json'))
    initial = [read_json(p) for p in init]
    if not initial:
        status, reason = 'UNVERIFIED', 'Measured initial-state validation required'
    elif all(p.get('status') == 'PASS' for p in initial):
        status, reason = 'PASS', 'Initial particle moments match the profile within tolerance and shot noise'
    else:
        failed = [f"{c['species']} {c['quantity']}" for p in initial for c in p.get('checks', [])
                  if c.get('status') not in ('PASS', 'NOT_INITIAL')]
        status = 'FAIL' if failed else 'UNVERIFIED'
        reason = ('Initial moments off profile: ' + ', '.join(failed)) if failed else 'No step-0 particle snapshot'
    check('initial_state', status, reason, root / '08_validation')
    count, issues = figure_issues(root)
    check('figures', 'UNVERIFIED' if not count else ('WARN' if issues else 'PASS'),
          'No figure content records (delivery predates them)' if not count else
          (f'{len(issues)} problem(s) in {count} figures: ' + '; '.join(issues[:5]) if issues
           else f'{count} figures, every panel draws data'), root, {'figures': count, 'issues': issues})
    # Explicit blockers; a successful executable is never a scientific certification.
    check('convergence', 'UNVERIFIED', 'Independent controlled runs required; no convergence inferred from one trajectory', root)
    overall = next(s for s in SEVERITY if any(c['status'] == s for c in checks))
    return {'run': root.name, 'root': str(root.resolve()), 'scientific_status': overall, 'checks': checks,
            '_audit': audit}


def write_report(roots, outdir, energy_tolerance=None):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    runs = [inspect_run(r, energy_tolerance) for r in roots]
    audits = [r.pop('_audit') for r in runs]
    # An isotropic control among the runs supplies the numerical-heating
    # baseline of the anisotropic run it matches (energy_audit.py).
    energy_audit.pair_controls(audits)
    for run, audit in zip(runs, audits):
        baseline = audit.get('baseline')
        if baseline:
            run['checks'].insert(next(i for i, c in enumerate(run['checks']) if c['check'] == 'global_energy') + 1,
                                 dict(check='energy_baseline', status=baseline['status'], reason=baseline['reason'],
                                      source=str(outdir / 'energy_audit'), metrics=energy_audit.strip_private(baseline)))
            run['scientific_status'] = next(s for s in SEVERITY if any(c['status'] == s for c in run['checks']))
    groups = energy_audit.group_runs(audits)
    if any(a.get('window') for a in audits):
        energy_audit.write_outputs(audits, groups, outdir / 'energy_audit')
    atomic_json(outdir / 'validation_matrix.json', {'runs': runs, 'energy_tolerance': energy_tolerance,
                                                    'energy_groups': groups})
    rows = [{**c, 'run': r['run']} for r in runs for c in r['checks']]
    with (outdir / 'validation_matrix.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'check', 'status', 'reason', 'source'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    content = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>PSC evidence report</title>',
               '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:0 1em}td,th{padding:.5em;text-align:left;border-bottom:1px solid #ddd;vertical-align:top}img{max-width:100%}.FAIL{color:#b3261e;font-weight:600}.WARN{color:#8a5a00}.PASS{color:#1b6e2e}</style>',
               '<h1>PSC scientific evidence</h1><p>Execution success is separate from scientific validity. Missing evidence remains UNVERIFIED.</p>']
    if (outdir / 'energy_audit' / 'energy_audit.png').exists():
        content.append('<h2>Energy audit across runs</h2>')
        content += [f'<p>{html.escape(g["interpretation"])}.</p>' for g in groups]
        content.append('<figure><a href="energy_audit/energy_audit.png"><img src="energy_audit/energy_audit.png" alt="energy audit"></a>'
                       '<figcaption>Electron heating, Debye-length resolution and global budget; '
                       'table in energy_audit/energy_audit.csv</figcaption></figure>')
        if (outdir / 'energy_audit' / 'energy_audit_baseline.png').exists():
            content.append('<figure><a href="energy_audit/energy_audit_baseline.png"><img src="energy_audit/energy_audit_baseline.png" '
                           'alt="baseline-corrected energy"></a><figcaption>Electron heating and total energy change '
                           'after subtracting the isotropic control of each run</figcaption></figure>')
    for run in runs:
        content += [f'<h2>{html.escape(run["run"])} — <span class="{run["scientific_status"]}">{run["scientific_status"]}</span></h2><table><tr><th>Check</th><th>Status</th><th>Evidence</th></tr>']
        for c in run['checks']:
            content.append(f'<tr><td>{c["check"]}</td><td class="{c["status"]}">{c["status"]}</td><td>{html.escape(c["reason"])}<br><small>{html.escape(c["source"])}</small></td></tr>')
        content.append('</table>')
        # Curated preview, leave the complete archive accessible by directory.
        for name in ('global_energy_conservation.png', 'growth_rate_fit_mode.png', 'estimator_consistency.png', 'structures_vs_time.png', 'heat_flux_vs_time.png'):
            for p in Path(run['root']).rglob(name):
                rel = html.escape(os.path.relpath(p, outdir), quote=True)
                content.append(f'<figure><a href="{rel}"><img src="{rel}" alt="{name}" style="max-width:500px"></a><figcaption>{name}</figcaption></figure>')
    content.append('</html>')
    (outdir / 'index.html').write_text('\n'.join(content))
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path); p.add_argument('--outdir', required=True, type=Path)
    p.add_argument('--energy-tolerance', type=float, default=None, help='Independently justified maximum relative energy drift')
    a = p.parse_args()
    if a.energy_tolerance is not None and (not np.isfinite(a.energy_tolerance) or a.energy_tolerance < 0):
        p.error('Energy tolerance must be finite and nonnegative')
    write_report(a.runs, a.outdir, a.energy_tolerance)

if __name__ == '__main__':
    main()
