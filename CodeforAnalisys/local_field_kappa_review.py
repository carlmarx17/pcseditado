#!/usr/bin/env python3
"""Review delivered CSV/PNG products without raw snapshots; no pooled-time inference.

Usage: python local_field_kappa_review.py RESULTS_ROOT [--gifs]
Outputs PNG diagnostics, a paired CSV, and optionally chronological GIFs.
The signed inverse kappa is a truncated-moment shape diagnostic, not an MLE.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def rows(path):
    with path.open(newline='') as f:
        return [{k: (v if k == 'population' else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]


def physical_b(row):
    """Saved b_mean is |B|/B_ref; B_ref varies with snapshot."""
    return row['b_mean'] * row['b_ref_over_B0']


def animate(files, target, note):
    files = sorted(files, key=lambda p: int(re.search(r'step(\d+)', p.name)[1]))
    if len(files) < 2:
        return None
    steps = [int(re.search(r'step(\d+)', p.name)[1]) for p in files]
    durations = [max(20, int(round(d / np.median(np.diff(steps)) * 450 / 10)) * 10)
                 for d in np.diff(steps)]
    durations.append(durations[-1])
    frames = []
    for path in files:
        with Image.open(path) as original:
            im = original.convert('RGB')
            im.thumbnail((1400, 1000), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (1400, 1040), 'white')
        canvas.paste(im, ((1400-im.width)//2, 40))
        ImageDraw.Draw(canvas).text((15, 12), note, fill='black')
        frames.append(canvas.quantize(colors=256))
    frames[0].save(target, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=2, optimize=False)
    with Image.open(target) as check:
        assert check.n_frames == len(files)
    return {'file': str(target), 'frames': [str(p) for p in files],
            'steps': steps, 'duration_ms': durations, 'note': note}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--gifs', action='store_true')
    args = parser.parse_args()
    out = args.root / 'local_field_kappa_review'
    out.mkdir(exist_ok=True)
    runs = sorted(p for p in args.root.glob('mirror_*') if p.is_dir())
    if not runs:
        raise ValueError('No mirror runs found')
    fig, axes = plt.subplots(1, len(runs), figsize=(15, 4.8), squeeze=False)
    evo, ea = plt.subplots(2, len(runs), figsize=(15, 7), squeeze=False, sharex='col')
    contrast, ca = plt.subplots(1, len(runs), figsize=(15, 4.8), squeeze=False)
    paired, summary, animations = [], [], []
    for i, run in enumerate(runs):
        label = run.name.replace('mirror_', '').replace('_moderate', '')
        directory = run / '03_particles'
        binned = rows(directory / 'vdf_kappa_b_series.csv')
        populations = rows(directory / 'vdf_kappa_series.csv')
        allrows = sorted((r for r in populations if r['population'] == 'all'), key=lambda r:r['step'])
        steps = sorted({r['step'] for r in binned if r['step'] > 0})
        selected = [steps[j] for j in sorted(set(np.linspace(0, len(steps)-1, 4).round().astype(int)))]
        ax = axes[0, i]
        for step in selected:
            rr = sorted((r for r in binned if r['step'] == step and r['count'] >= 2000
                         and np.isfinite(r['inv_kappa_signed']) and np.isfinite(physical_b(r))),
                        key=physical_b)
            if not rr:
                continue
            ax.errorbar([physical_b(r) for r in rr], [r['inv_kappa_signed'] for r in rr],
                        yerr=[r['inv_kappa_signed_err'] for r in rr], fmt='o-', ms=4,
                        capsize=2, label=f"t={rr[0]['omega_ci_t']:.1f}")
            paired.extend({'run': run.name, 'B_over_B0': physical_b(r), **r} for r in rr)
        ax.set(title=label, xlabel=r'Local $|B|/B_0$', ylabel=r'Signed $1/\kappa_{eff}$')
        ax.legend(fontsize=8)
        t = [r['omega_ci_t'] for r in allrows]
        ea[0,i].errorbar(t, [r['inv_kappa_signed'] for r in allrows],
                        yerr=[r['inv_kappa_signed_error'] for r in allrows], lw=1, label='all prt particles')
        ea[0,i].set(title=label, ylabel=r'Signed $1/\kappa_{eff}$')
        ea[1,i].plot(t, [r['b_mean_over_B0'] for r in allrows])
        ea[1,i].set(xlabel=r'$t\Omega_{ci}$', ylabel=r'Particle-sampled $\langle|B|\rangle/B_0$')
        grouped = {(r['step'], r['population']): r for r in populations}
        delta_b, delta_q, times = [], [], []
        for r in allrows:
            hole, peak = grouped.get((r['step'],'hole')), grouped.get((r['step'],'peak'))
            if not hole or not peak or r['step'] == 0:
                continue
            delta_b.append(peak['b_mean_over_B0']-hole['b_mean_over_B0'])
            delta_q.append(peak['inv_kappa_signed']-hole['inv_kappa_signed'])
            times.append(r['omega_ci_t'])
        sc = ca[0,i].scatter(delta_b, delta_q, c=times, s=18, cmap='viridis')
        ca[0,i].axhline(0, color='gray', lw=.7)
        ca[0,i].set(title=label, xlabel=r'$\langle|B|\rangle_{peak}/B_0-\langle|B|\rangle_{hole}/B_0$',
                    ylabel=r'$(1/\kappa)_{peak}-(1/\kappa)_{hole}$')
        contrast.colorbar(sc, ax=ca[0,i], label=r'$t\Omega_{ci}$')
        summary.append({'run':run.name, 'initial_kappa':allrows[0]['kappa_eff'] if np.isfinite(allrows[0]['kappa_eff']) else None,
                        'final_kappa':allrows[-1]['kappa_eff'] if np.isfinite(allrows[-1]['kappa_eff']) else None,
                        'final_time':t[-1], 'final_peak_minus_hole_inverse_kappa':delta_q[-1]})
        if args.gifs:
            for pattern, directory, suffix, note in [
                ('Bmag_fluct_step*_yz.png', run/'02_fields', 'field',
                 '|B - mean(B)| / B0 (vector fluctuation); read the original color bars'),
                ('vdf_spatial_step*.png', run/'03_particles', 'prt_window',
                 'Full domain + prt window: delta |B| / B0; color scales vary between frames')]:
                result = animate(list(directory.glob(pattern)), out/f'{run.name}_{suffix}.gif', note)
                if result:
                    animations.append(result)
    for f, name, title in [(fig, 'local_kappa_vs_B', 'Same-time field bins inside the prt window; bars: estimator errors'),
                            (evo, 'prt_time_evolution', 'Same prt population; magnetic mean is particle sampled, not volume averaged'),
                            (contrast, 'peak_hole_contrast', 'Spatial contrast at each time; descriptive points, no independent-sample significance')]:
        f.suptitle(title, fontsize=11)
        f.tight_layout(rect=(0,0,1,.94))
        f.savefig(out/f'{name}.png', dpi=170)
        plt.close(f)
    with (out/'local_kappa_vs_B.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    if args.gifs:
        (out/'animation_manifest.json').write_text(json.dumps(animations, indent=2))
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f'Created diagnostics and {len(animations)} animations in {out}')


if __name__ == '__main__':
    main()
