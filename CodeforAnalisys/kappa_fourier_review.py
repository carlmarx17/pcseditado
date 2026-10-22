#!/usr/bin/env python3
"""Exploratory Fourier analysis of the signed inverse-kappa diagnostic.

Uniform cadence required; Hann-windowed PSD without zero padding. Polynomial
trend/window sensitivity is exported. A peak is not a periodicity detection.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.signal import periodogram
from run_naming import find_runs, families_in, distribution_of
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))


def col(r,k):return np.array([float(x[k]) for x in r])


def spectrum(t,y,degree=2):
    dt=np.diff(t)
    if len(t)<12 or not np.allclose(dt,dt[0],rtol=1e-6,atol=1e-9):
        raise ValueError('At least 12 uniformly spaced samples required')
    u=(t-t.mean())/np.ptp(t)
    trend=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyfit(u,y,degree))
    residual=y-trend
    f,p=periodogram(residual,fs=1/dt[0],window='hann',detrend=False,scaling='density')
    j=1+np.argmax(p[1:])
    return residual,trend,2*np.pi*f,p/(2*np.pi),dict(omega_peak=float(2*np.pi*f[j]),period=float(1/f[j]),
        cycles_in_span=float(np.ptp(t)*f[j]),omega_resolution=float(2*np.pi/(len(t)*dt[0])),
        omega_nyquist=float(np.pi/dt[0]),n=len(t),duration=float(np.ptp(t)))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path)
    ap.add_argument('--family',default=None,help='mirror | firehose | whistler (default: the one present)')
    args=ap.parse_args()
    out=args.root/'kappa_fourier_review';out.mkdir(exist_ok=True)
    # Every instability family: a 'mirror_*' glob silently produced an empty
    # figure on a firehose or whistler results root.
    runs=find_runs(args.root,family=args.family)
    if not runs:raise SystemExit(f'No case runs under {args.root}')
    families=families_in(runs)
    if len(families)>1:raise SystemExit(f'Mixed families {sorted(families)}; pass --family')
    family=families.pop()
    fig,axes=plt.subplots(len(runs),2,figsize=(13,10),squeeze=False)
    sensitivity=[];summary=[];psdrows=[]
    for i,run in enumerate(runs):
        pr=sorted([r for r in read(run/'03_particles/vdf_kappa_series.csv') if r['population']=='all'],key=lambda r:float(r['omega_ci_t']))
        t=col(pr,'omega_ci_t');q=col(pr,'inv_kappa_signed');e=col(pr,'inv_kappa_signed_error')
        axes[i,0].errorbar(t,q,yerr=e,fmt='.',ms=3,elinewidth=.5,color='black',label='signed inverse kappa')
        for degree,color in [(1,'#888888'),(2,'#0072B2'),(3,'#D55E00')]:
            residual,trend,omega,p,info=spectrum(t,q,degree)
            axes[i,0].plot(t,trend,color=color,lw=1,label=f'trend degree {degree}')
            axes[i,1].semilogy(omega[1:],p[1:],color=color,label=f'trend degree {degree}')
            for j in range(1,len(omega)):
                psdrows.append(dict(run=run.name,degree=degree,omega=float(omega[j]),psd=float(p[j])))
            if degree==2:summary.append({'run':run.name,**info})
            for start in [0,20,60]:
                keep=t>=start
                *_,stats=spectrum(t[keep],q[keep],degree)
                sensitivity.append(dict(run=run.name,start=start,degree=degree,**stats))
        theory=[r for r in read(args.root/f'trajectory_linear_theory_{family}_moderate_kappa'/'trajectory_linear_theory.csv') if r['run']==run.name]
        theory_w=col(theory,'omega_r');lo,hi=np.nanmin(theory_w),np.nanmax(theory_w)
        axes[i,1].axvspan(lo,hi,alpha=.15,color='green',label='saved IC theory frequency')
        axes[i,1].set_xlim(0,1.0)
        axes[i,0].set(title=distribution_of(run.name),xlabel=r'$t\Omega_{ci}$',ylabel=r'Signed $q=1/\kappa_{eff}$')
        axes[i,1].set(xlabel=r'Angular frequency $\omega/\Omega_{ci}$',ylabel='PSD of detrended q')
        for ax in axes[i]:ax.legend(fontsize=8);ax.grid(alpha=.15)
        if 'bimaxwellian' in run.name:
            comparison,aa=plt.subplots(2,1,figsize=(11,8))
            fields=read(run/'09_physical_diagnostics/field_fluctuation_table.csv');ft=col(fields,'omega_ci_t')
            signals={'q':q,'local anisotropy':col(pr,'A_local_b'),'tail fraction':col(pr,'tail_fraction_par_3sigma'),
                     'prt mean |B|':np.interp(t,ft,col(fields,'prt_B_mean_over_B0')),
                     'global mean |B|':np.interp(t,ft,col(fields,'B_mean_over_B0'))}
            comparisons=[]
            qr=spectrum(t,q,2)[0]
            for label,y in signals.items():
                r,_,w,p,inf=spectrum(t,y,2)
                aa[0].plot(t,r/np.std(r),label=label,lw=1.2)
                aa[1].plot(w[1:],p[1:]/max(p[1:]),label=label)
                comparisons.append({'signal':label,'residual_correlation_with_q':float(np.corrcoef(qr,r)[0,1]),**inf})
            aa[0].set(xlabel=r'$t\Omega_{ci}$',ylabel='Residual / residual std (quadratic trend removed)')
            aa[1].set(xlabel=r'$\omega/\Omega_{ci}$',ylabel='PSD / maximum PSD',xlim=(0,.6))
            aa[1].axvspan(lo,hi,alpha=.15,color='green',label='IC theory frequency')
            for a in aa:a.legend(fontsize=8);a.grid(alpha=.15)
            comparison.suptitle('Bi-Maxwellian reference: shape, anisotropy and magnetic modulation')
            comparison.tight_layout();comparison.savefig(out/'maxwellian_signal_comparison.png',dpi=170);plt.close(comparison)
            (out/'maxwellian_comparison.json').write_text(json.dumps(comparisons,indent=2))
    fig.suptitle('Fourier spectra of q(t): trend sensitivity, Hann window, no zero padding',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'kappa_fourier.png',dpi=170);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    for name,rows in [('sensitivity.csv',sensitivity),('spectra.csv',psdrows)]:
        with (out/name).open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print(json.dumps(summary,indent=2))
    print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
