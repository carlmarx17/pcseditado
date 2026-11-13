"""Resumable double-Harris analysis with an independent periodic-box energy audit."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from analysis_contract import atomic_json, inventory_identity, provenance
from data_reader import PICDataReader
from energy_conservation import write_energy_analysis
from reconnection_analysis import RECONNECTION_PROFILES, resolve_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('profile', choices=RECONNECTION_PROFILES)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    data, out = args.data.resolve(), args.out.resolve()
    files = PICDataReader.discover_outputs(str(data))
    fields = files['fields']
    if not fields:
        parser.error('No field snapshots')
    geometry = resolve_geometry(fields[min(fields)], args.profile)
    here = Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    source = provenance()['analysis_source_sha256']
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('10_reconnection', '11_heating'):
            inputs = ([*fields.values(), *(p for series in files['particles'].values() for p in series.values())]
                      if name == '10_reconnection' else sorted(data.glob('diag*.asc')))
            marker = out/(name+'.json')
            identity = {'source': source, 'data': str(data), 'profile': args.profile, 'omega_ci': geometry.omega_ci,
                        'inputs': inventory_identity(inputs), 'steps': [min(fields), max(fields)]}
            if marker.exists():
                prev = json.loads(marker.read_text())
                if prev['identity'] != identity:
                    parser.error(f'Inputs changed for {name}. Use a new OUT for this history.')
                products = [out/name/p for p in prev['files']]
                if all(p.is_file() for p in products) and inventory_identity(products) == prev['products']:
                    print(f'{name}: up to date')
                    continue
            target = out/name
            if target.exists() and not (target/'.completed.json').exists():
                parser.error(f'{target} already exists without an intact completion record; inspect before recomputing')
            if target.exists():
                saved = json.loads((target/'.completed.json').read_text())
                files_saved = [target/p for p in saved['files']]
                if (saved['identity'] == identity and all(p.is_file() for p in files_saved)
                        and inventory_identity(files_saved) == saved['products']):
                    atomic_json(marker, saved)
                    continue
                parser.error(f'{target} differs from its completion record')
            with tempfile.TemporaryDirectory(dir=out) as scratch:
                tmp = Path(scratch)/name
                tmp.mkdir()
                if name == '10_reconnection':
                    subprocess.run([sys.executable, str(here/'reconnection_analysis.py'), '--data-dir', str(data),
                                    '--profile', args.profile, '--outdir', str(tmp)], check=True)
                elif inputs:
                    write_energy_analysis(inputs, tmp, omega_ci=geometry.omega_ci)
                else:
                    atomic_json(tmp/'global_energy_summary.json', {'scientific_status':'UNVERIFIED',
                                'reason':'No DiagEnergies segments for this history'})
                products = sorted(p for p in tmp.iterdir() if p.is_file())
                stamp = {'identity':identity, 'files':[p.name for p in products],
                         'products':inventory_identity(products)}
                atomic_json(tmp/'.completed.json', stamp)
                tmp.rename(target)
                atomic_json(marker, stamp)


if __name__ == '__main__':
    main()
