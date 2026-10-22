#!/usr/bin/env python3
"""Whole-domain magnetic diagnostics and their co-evolution with prt kappa.

Reads delivered tables only. Keeps the global/particle-window distinction and
uses the exact definitions of physical_diagnostics.field_metrics. No causal fit.
Usage: python whole_box_field_story.py RESULTS_ROOT [--gifs]
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
from run_naming import find_runs, families_in, distribution_of
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def col(rows, name):
    return np.array([float(r[name]) for r in rows])


def field_series(rows):
    order = np.argsort(col(rows, 'omega_ci_t'))
    data = {k: col(rows, k)[order] for k in rows[0]}
    t = data['omega_ci_t']
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('Field times must be finite and unique')
    par = data['delta_B_parallel_rms_over_B0']**2
    perp = data['delta_B_perp_rms_over_B0']**2
    total = data['delta_B_vec_rms_over_B0']**2
    if not np.allclose(total, par+perp, rtol=1e-9, atol=1e-14):
        raise ValueError('Vector power does not equal parallel plus perpendicular')
    # RMS of |B|-B0 is not std(|B|): it also contains the mean offset.
    if not np.allclose(data['delta_B_rms_over_B0']**2,
                       data['B_std_over_B0']**2+(data['B_mean_over_B0']-1)**2,
                       rtol=1e-7, atol=1e-12):
        raise ValueError('Magnitude RMS / standard-deviation identity failed')
    data.update(W=total, Wpar=par, Wperp=perp)
    return data


def pair(field, particles):
    """Interpolate fields to real particle times, never extrapolate kappa."""
    rows = sorted((r for r in particles if r['population']=='all'), key=lambda r:float(r['omega_ci_t']))
    t=col(rows,'omega_ci_t')
    keep=(t>=field['omega_ci_t'][0]) & (t<=field['omega_ci_t'][-1])
    q=col(rows,'inv_kappa_signed')
    keep &= np.isfinite(q)
    return {'t':t[keep], 'q':q[keep],
            **{key:np.interp(t[keep],field['omega_ci_t'],field[key])
               for key in ['B_mean_over_B0','delta_B_vec_rms_over_B0','W','Wpar','Wperp']}}


def save(fig, path, title):
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(path,dpi=165)
    plt.close(fig)


def map_animation(run, out):
    """Full-domain |B| and magnitude fluctuations from already-rendered PNGs."""
    folder=run/'09_physical_diagnostics'
    paths=sorted(folder.glob('mirror_holes_map_step_*.png'),
                 key=lambda p:int(re.search(r'step_(\d+)',p.name)[1]))
    frames=[]; sources=[]
    for path in paths:
        step=int(re.search(r'step_(\d+)',path.name)[1])
        other=folder/f'deltaB_map_step_{step}.png'
        if not other.exists():
            continue
        canvas=Image.new('RGB',(1600,870),'white')
        draw=ImageDraw.Draw(canvas)
        draw.text((15,10),f'{run.name} | whole domain | step {step}',fill='black')
        draw.text((15,30),'Left: |B| (code units). Right: (|B|-B0)/B0. Read each color bar; scales may change.',fill='black')
        draw.text((15,50),'Saved maps are smoothed for display (sigma=2 cells); scalar time series use raw fields.',fill='black')
        for j,p in enumerate([path,other]):
            with Image.open(p) as im:
                im=im.convert('RGB');im.thumbnail((800,790),Image.Resampling.LANCZOS)
                canvas.paste(im,(j*800+(800-im.width)//2,80))
        frames.append(canvas.quantize(colors=256))
        sources.append({'step':step,'sources':[str(path),str(other)]})
    if len(frames)<2:
        return None
    steps=np.array([r['step'] for r in sources]);dt=np.diff(steps)
    durations=[max(20,round(float(d)/np.median(dt)*700/10)*10) for d in dt]
    durations.append(durations[-1])
    target=out/f'{run.name}_whole_box_B_and_deltaB.gif'
    frames[0].save(target,save_all=True,append_images=frames[1:],duration=durations,loop=0,disposal=2,optimize=False)
    with Image.open(target) as im:
        assert im.n_frames==len(frames)
    return {'gif':str(target),'frames':sources,'duration_ms':durations}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path);ap.add_argument('--gifs',action='store_true')
    ap.add_argument('--family',default=None,help='mirror | firehose | whistler (default: the one present)')
    args=ap.parse_args();out=args.root/'whole_box_field_story';out.mkdir(exist_ok=True)
    # Every instability family: the old 'mirror_*' glob made this raise on a
    # firehose or whistler results root.
    runs=find_runs(args.root,family=args.family)
    n=len(runs)
    if not n:raise ValueError(f'No case runs under {args.root}')
    families=families_in(runs)
    if len(families)>1:raise SystemExit(f'Mixed families {sorted(families)}; pass --family')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(3,n,figsize=(15,10),sharex=True,squeeze=False,sharey='row')
    story,sx=plt.subplots(4,n,figsize=(15,11),sharex=True,squeeze=False,sharey='row')
    phase,px=plt.subplots(2,n,figsize=(15,7),squeeze=False)
    geometry,gx=plt.subplots(1,n,figsize=(15,4),squeeze=False,sharey=True)
    summaries=[];exports=[];animations=[]
    for i,run in enumerate(runs):
        label=distribution_of(run.name)
        folder=run/'09_physical_diagnostics'
        f=field_series(read(folder/'field_fluctuation_table.csv'));t=f['omega_ci_t']
        parts=read(run/'03_particles/vdf_kappa_series.csv')
        pr=sorted((r for r in parts if r['population']=='all'),key=lambda r:float(r['omega_ci_t']))
        p=pair(f,parts)
        ax[0,i].plot(t,f['B_mean_over_B0'],label='whole-box mean',color='#0072B2')
        ax[0,i].plot(t,f['prt_B_mean_over_B0'],label='prt-volume mean',ls='--',color='#D55E00')
        ax[0,i].axhline(1,color='gray',lw=.6);ax[0,i].set_title(label)
        for key,label2,color in [('delta_B_vec_rms_over_B0','vector total','#222222'),
                                 ('delta_B_perp_rms_over_B0','perpendicular','#0072B2'),
                                 ('delta_B_parallel_rms_over_B0','parallel to B0','#D55E00'),
                                 ('delta_B_rms_over_B0','magnitude minus B0','#009E73')]:
            ax[1,i].plot(t,f[key],label=label2,color=color,lw=1.5)
        for key,label2,color in [('W','total','#222222'),('Wperp','perpendicular','#0072B2'),('Wpar','parallel','#D55E00')]:
            positive=(t>0)&(f[key]>0)
            ax[2,i].semilogy(t[positive],f[key][positive],label=label2,color=color)
        for row in range(3):ax[row,i].legend(fontsize=8,loc='best');ax[row,i].grid(alpha=.15)
        ax[2,i].set_xlabel(r'$t\Omega_{ci}$')
        pt=col(pr,'omega_ci_t')
        sx[0,i].plot(pt,col(pr,'A_local_b'),color='#D55E00');sx[0,i].set_title(label)
        sx[1,i].plot(pt,col(pr,'inv_kappa_signed'),color='#009E73')
        sx[2,i].plot(t,f['B_mean_over_B0'],color='#0072B2')
        sx[3,i].plot(t,f['Wperp'],label='whole-box perpendicular',color='#0072B2')
        sx[3,i].plot(t,f['Wpar'],label='whole-box parallel',color='#D55E00')
        sx[3,i].legend(fontsize=8);sx[3,i].set_xlabel(r'$t\Omega_{ci}$')
        modes=read(folder/'mode_growth_table.csv')
        dominant=next(r for r in modes if r['dominant']=='1')
        for row in range(4):
            sx[row,i].axvspan(float(dominant['linear_phase_start']),float(dominant['linear_phase_end']),color='gray',alpha=.12)
            sx[row,i].grid(alpha=.15)
        for row,key in enumerate(['B_mean_over_B0','delta_B_vec_rms_over_B0']):
            a=px[row,i];a.plot(p[key],p['q'],color='lightgray',lw=.7)
            sc=a.scatter(p[key],p['q'],c=p['t'],cmap='viridis',s=13,vmin=0,vmax=t[-1])
            phase.colorbar(sc,ax=a,label=r'$t\Omega_{ci}$')
            a.set_xlabel(r'Whole-box $\langle|B|\rangle/B_0$' if row==0 else r'Whole-box $\delta B_{vec,rms}/B_0$')
            a.set_ylabel(r'prt signed $1/\kappa_{eff}$')
        px[0,i].set_title(label)
        gx[0,i].plot(t,f['B_mean_over_B0']-1,label=r'measured $\langle|B|\rangle/B_0-1$')
        gx[0,i].plot(t,.5*f['Wperp'],ls='--',label=r'$W_\perp/2$ (small-amplitude estimate)')
        gx[0,i].set(title=label,xlabel=r'$t\Omega_{ci}$');gx[0,i].legend(fontsize=8)
        summaries.append({'run':run.name,'final_time':float(t[-1]),'B_mean_over_B0':float(f['B_mean_over_B0'][-1]),
                          'vector_rms_over_B0':float(f['delta_B_vec_rms_over_B0'][-1]),
                          'W':float(f['W'][-1]),'transverse_power_fraction':float(f['Wperp'][-1]/f['W'][-1]),
                          'magnitude_offset_over_half_Wperp':float((f['B_mean_over_B0'][-1]-1)/(.5*f['Wperp'][-1])),
                          'dominant_mode_classification':dominant['classification'],'saved_branch_label':dominant['physical_branch']})
        for j in range(len(p['t'])):
            exports.append({'run':run.name,**{k:float(v[j]) for k,v in p.items()}})
        if args.gifs:
            animation=map_animation(run,out)
            if animation:animations.append(animation)
    for row,label in enumerate([r'$\langle|B|\rangle/B_0$',r'RMS amplitude / $B_0$',r'Normalized power $W=\langle\delta B^2\rangle/B_0^2$']):ax[row,0].set_ylabel(label)
    for row,label in enumerate([r'prt $T_\perp/T_\parallel$ (local B)',r'prt signed $1/\kappa_{eff}$',r'Whole-box $\langle|B|\rangle/B_0$',r'Whole-box power / $B_0^2$']):sx[row,0].set_ylabel(label)
    gx[0,0].set_ylabel('Normalized magnitude increase')
    save(fig,out/'whole_box_field_and_fluctuations.png','Full simulation box: magnetic magnitude, fluctuation amplitude and power (raw-field statistics)')
    save(story,out/'anisotropy_field_kappa_story.png','Coupled evolution: particle-window diagnostics + whole-box fields; shading = fitted dominant-mode linear phase')
    save(phase,out/'whole_box_field_vs_prt_kappa.png','Whole-box fields against prt kappa: time trajectories, not spatial correlations')
    save(geometry,out/'mean_magnitude_geometry.png','Does transverse power explain the increase in mean magnetic magnitude?')
    (out/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
    if args.gifs:(out/'animation_manifest.json').write_text(json.dumps(animations,indent=2))
    with (out/'paired_global_field_prt_kappa.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(exports[0]));writer.writeheader();writer.writerows(exports)
    print(json.dumps(summaries,indent=2));print('Output:',out)

if __name__=='__main__':main()
