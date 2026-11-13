"""Reuse intact archived stages when the recipe, input and local Python dependencies match."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

from analysis_contract import atomic_json, inventory_identity

# Entry points invoked by each Make target, including the recursive spectral recipe.
ENTRY = {
    'residuals': 'field_residuals', 'physics': 'physical_diagnostics energy_conservation',
    'brazil': 'anisotropy_analysis', 'fields': 'fluctuationofmagneticfiel',
    'spectral': 'spectral_analysis dispersion_analysis growth_rate_map polarization_dispersion',
    'structures': 'structures_analysis', 'energy-exchange': 'energy_exchange',
    'estimators': 'estimator_consistency', 'heatflux': 'heat_flux_analysis', 'particles': 'plot_prt',
    'validate': 'validate_moments', 'diamagnetic': 'diamagnetic_current',
    'vdf-spatial': 'vdf_spatial', 'theory': 'linear_theory', 'theory-liouville': 'liouville_kappa',
    'resonant-anisotropy': 'resonant_anisotropy', 'anisotropy-dispersion': 'temperature_anisotropy_dispersion',
}


def changed_modules(root, record):
    """Verify the recorded source digest even for commits marked dirty at submission."""
    import re
    rev = record.get('provenance', {}).get('git_commit') or ''
    if not re.fullmatch('[0-9a-f]{40}', rev):
        return None
    result = subprocess.run(['git', '-C', str(root.parent), 'archive', rev + ':CodeforAnalisys'],
                            capture_output=True)
    if result.returncode:
        return None
    digest = hashlib.sha256()
    changed = set()
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        files = sorted(m.name for m in archive if m.isfile() and '/' not in m.name
                       and (m.name.endswith('.py') or m.name in ('Makefile', 'requirements.txt')))
        for name in files:
            raw = archive.extractfile(name).read()
            digest.update(name.encode() + b'\0' + raw)
            if not (root/name).exists() or raw != (root/name).read_bytes():
                # A new standalone target appended below the historical Makefile
                # leaves its existing recipes and defaults intact.
                if name != 'Makefile' or not (root/name).read_bytes().startswith(raw):
                    changed.add(name)
    changed.update(p.name for p in root.glob('*.py') if p.name not in files)
    return changed if digest.hexdigest() == record['signature']['source'] else None


def dependencies(root, stage):
    pending = ENTRY[stage].split() + ['check_resolution', 'psc_units']
    seen = set()
    while pending:
        name = pending.pop()
        path = root / (name + '.py')
        if path.name in seen or not path.is_file():
            continue
        seen.add(path.name)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                pending.append(node.module.split('.')[0])
            elif isinstance(node, ast.Import):
                pending.extend(a.name.split('.')[0] for a in node.names)
    return seen | {'Makefile', 'requirements.txt'}


def comparable_options(options, stage):
    # Data identity and snapshot format are checked separately. A field movie
    # already archived satisfies a request that needs only its static products.
    ignored = {'RUN_TAG', 'FIELD_PATTERN', 'MOMENT_PATTERN', 'PARTICLE_PATTERN'}
    for owner, keys in [('fields', ('GIF_EVERY', 'FIELDS_GIF', 'SNAPSHOT_EVERY')),
                        ('structures', ('STRUCTURES_FLAGS',)), ('resonant-anisotropy', ('RESONANT_FLAGS',))]:
        if stage != owner:
            ignored.update(keys)
    return {k: v for k, v in (x.split('=', 1) for x in options) if k not in ignored}


def import_stages(archives, data, out, state, selected, root, complete):
    sig = state['signature']
    deps = {}
    for src in archives:
        record = json.loads((src/'pipeline.json').read_text())
        old = record.get('signature', {})
        if any(old.get(k) != sig[k] for k in ('input', 'config', 'case', 'environment')):
            continue
        manifests = list(src.glob('*_analysis_manifest.json'))
        if len(manifests) != 1 or Path(json.loads(manifests[0].read_text())['input_directory']).resolve() != data:
            continue
        changed = set() if old['source'] == sig['source'] else changed_modules(root, record)
        if changed is None:
            continue
        for stage in selected:
            if stage not in ENTRY or complete(state['stages'].get(stage, {}), out):
                continue
            entry = record.get('stages', {}).get(stage, {})
            if not entry.get('products') or not complete(entry, src):
                continue
            a = comparable_options(old['options'], stage)
            b = comparable_options(sig['options'], stage)
            if stage == 'fields' and b.get('FIELDS_GIF') == '0':
                for opts in (a, b):
                    for key in ('GIF_EVERY', 'FIELDS_GIF'):
                        opts.pop(key, None)
            if a != b:
                continue
            if changed and stage not in deps:
                deps[stage] = dependencies(root, stage)
            if changed & deps.get(stage, set()):
                continue
            # Spectral windows/overlay depend on upstream products. Only import
            # them when the upstream products came from this same archive.
            if stage == 'spectral' and any(state['stages'].get(k, {}).get('archive') != str(src)
                                           for k in ('physics', 'theory')):
                continue
            owners = {p for s, e in state['stages'].items() if s != stage for p in e.get('products', [])}
            for rel in entry['products']:
                if Path(rel).is_absolute() or '..' in Path(rel).parts or rel in owners:
                    raise ValueError(f'Unsafe or shared archive product {rel}')
                dest = out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                link = dest.with_name(dest.name + '.reusing')
                link.unlink(missing_ok=True)
                link.symlink_to(src / rel)
                os.replace(link, dest)
            state['stages'][stage] = {**entry, 'archive': str(src),
                                      'output_identity': inventory_identity([out/p for p in entry['products']])}
            atomic_json(out/'pipeline.json', state)
            print(f'{stage}: reused {src}', flush=True)
