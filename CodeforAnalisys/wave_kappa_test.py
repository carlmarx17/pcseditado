#!/usr/bin/env python3
"""Exploratory dominant-mode / prt shape coupling, with chronological prediction.

No causal inference or iid p-values. Non-overlapping local slope windows;
comparison against linear and quadratic time trends. Runs never pooled.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from run_naming import find_runs, families_in, distribution_of
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def column(rows,key):return np.array([float(r[key]) for r in rows])


def integral(t,w):
    if np.any(np.diff(t)<=0):raise ValueError('Times must strictly increase')
    return np.r_[0,np.cumsum(np.diff(t)*(w[1:]+w[:-1])/2)]


def interval_mean(t,w,left,right):
    if left<t[0] or right>t[-1] or right<=left:raise ValueError('Invalid integration bounds')
    inside=(t>left)&(t<right)
    tt=np.r_[left,t[inside],right];ww=np.interp(tt,t,w)
    return float(np.sum(np.diff(tt)*(ww[1:]+ww[:-1])/2)/(right-left))


def window_rates(t,q,e,mt,w,start=10.,width=10.):
    result=[]
    for left in np.arange(max(start,t[0]),min(t[-1],mt[-1])-width+1e-8,width):
        keep=(t>=left)&(t<left+width)&np.isfinite(q)&np.isfinite(e)&(e>0)
        if keep.sum()<4:continue
        tt=t[keep];x=np.column_stack([np.ones(len(tt)),tt-tt.mean()])
        a=x/e[keep,None];b=q[keep]/e[keep]
        beta=np.linalg.lstsq(a,b,rcond=None)[0]
        covariance=np.linalg.inv(a.T@a)
        result.append(dict(t=float(tt.mean()),t_start=float(tt[0]),t_end=float(tt[-1]),
                           rate=float(-beta[1]),formal_error=float(np.sqrt(covariance[1,1])),
                           W=interval_mean(mt,w,tt[0],tt[-1]),n=int(keep.sum())))
    return result


def design(t,z,kind):
    # Fixed rescaling improves conditioning, with no use of target values.
    t=np.asarray(t)/100.;z=np.asarray(z)
    columns={'constant':[], 'time':[t], 'time2':[t,t*t],
             'wave':[z], 'time_wave':[t,z]}
    return np.column_stack([np.ones(len(t)),*columns[kind]])


def fits(t,z,y):
    out={}
    for kind in ['constant','time','time2','wave','time_wave']:
        x=design(t,z,kind);beta=np.linalg.lstsq(x,y,rcond=None)[0]
        pred=x@beta;ss=float(np.sum((y-y.mean())**2))
        # Expanding training prefix, disjoint future validation blocks.
        edges=np.unique((np.array([.5,.6,.7,.8,.9,1])*len(y)).astype(int))
        errors=[];folds=[]
        for a,b in zip(edges[:-1],edges[1:]):
            if a<=x.shape[1]+1 or b<=a:continue
            bf=np.linalg.lstsq(x[:a],y[:a],rcond=None)[0]
            residual=y[a:b]-x[a:b]@bf
            errors.extend(residual.tolist());folds.append(float(np.sqrt(np.mean(residual**2))))
        out[kind]={'beta':beta.tolist(),'r2':1-float(np.sum((y-pred)**2))/ss if ss else None,
                   'rmse':float(np.sqrt(np.mean((y-pred)**2))),
                   'forward_rmse':float(np.sqrt(np.mean(np.array(errors)**2))) if errors else None,
                   'forward_fold_rmse':folds}
    return out


def residual_correlation(t,w,y,degree):
    x=np.column_stack([np.asarray(t)**j for j in range(degree+1)])
    a=w-x@np.linalg.lstsq(x,w,rcond=None)[0]
    b=y-x@np.linalg.lstsq(x,y,rcond=None)[0]
    return float(np.corrcoef(a,b)[0,1])


def dump_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path)
    ap.add_argument('--family',default=None,help='mirror | firehose | whistler (default: the one present)')
    args=ap.parse_args();out=args.root/'wave_kappa_test';out.mkdir(exist_ok=True)
    # Every instability family, not just mirror: globbing 'mirror_*' made
    # this script produce an empty figure on a firehose or whistler root.
    runs=find_runs(args.root,family=args.family);n=len(runs)
    if not n:raise SystemExit(f'No case runs under {args.root}')
    families=families_in(runs)
    if len(families)>1:raise SystemExit(f'Mixed families {sorted(families)}; pass --family')
    family=families.pop()
    fig,ax=plt.subplots(3,n,figsize=(15,11),squeeze=False)
    cum,cx=plt.subplots(2,n,figsize=(15,8),squeeze=False)
    shape,sx=plt.subplots(1,n,figsize=(15,4.8),squeeze=False,layout='constrained')
    theory=read(args.root/f'trajectory_linear_theory_{family}_moderate_kappa'/'trajectory_linear_theory.csv')
    results=[];allrates=[];sensitivity=[];pairs=[]
    for i,run in enumerate(runs):
        label=distribution_of(run.name)
        modes=read(run/'09_physical_diagnostics/mode_growth_table.csv')
        dominant=next(r for r in modes if r['dominant']=='1')
        kpar=float(dominant['k_parallel_di']);kperp=float(dominant['k_perp_di'])
        key=f'amp_over_B0_kpar{kpar:.3f}_kperp{kperp:+.3f}'
        mode=read(run/'09_physical_diagnostics/mode_amplitude_timeseries.csv')
        mt=column(mode,'omega_ci_t');order=np.argsort(mt);mt=mt[order];w=column(mode,key)[order]**2
        F=integral(mt,w)
        pr=sorted((r for r in read(run/'03_particles/vdf_kappa_series.csv') if r['population']=='all'),key=lambda r:float(r['omega_ci_t']))
        t=column(pr,'omega_ci_t');q=column(pr,'inv_kappa_signed');e=column(pr,'inv_kappa_signed_error')
        keep=(t>=mt[0])&(t<=mt[-1])&np.isfinite(q)&np.isfinite(e)&(e>0)
        t,q,e=t[keep],q[keep],e[keep]
        rates=window_rates(t,q,e,mt,w);rt=column(rates,'t');rw=column(rates,'W');ry=column(rates,'rate')
        rf=fits(rt,rw,ry)
        raw=float(np.corrcoef(rw,ry)[0,1]);partial=residual_correlation(rt,rw,ry,1)
        partial2=residual_correlation(rt,rw,ry,2)
        for width in [6.,10.,16.]:
            for start in [0.,10.,20.]:
                rr=window_rates(t,q,e,mt,w,start,width)
                tx,wx,yx=column(rr,'t'),column(rr,'W'),column(rr,'rate')
                ff=fits(tx,wx,yx)
                sensitivity.append(dict(run=run.name,start=start,width=width,n=len(rr),
                    raw_r=float(np.corrcoef(wx,yx)[0,1]),partial_r_time=residual_correlation(tx,wx,yx,1),
                    wave_coefficient=ff['time_wave']['beta'][-1],
                    time_forward_rmse=ff['time']['forward_rmse'],time2_forward_rmse=ff['time2']['forward_rmse'],
                    time_wave_forward_rmse=ff['time_wave']['forward_rmse']))
        for r in rates:allrates.append({'run':run.name,**r})
        mask=t>=10;ct=t[mask];cq=q[mask];cf=np.interp(ct,mt,F);cf-=cf[0]
        cf_fit=fits(ct-ct[0],cf,cq)
        results.append({'run':run.name,'mode_column':key,'kpar_di':kpar,'kperp_di':kperp,
                        'rate_windows':len(rates),'rate_raw_r':raw,'rate_partial_r_linear_time':partial,
                        'rate_partial_r_quadratic_time':partial2,'rate_models':rf,'cumulative_models':cf_fit})
        for j in range(len(ct)):pairs.append(dict(run=run.name,t=float(ct[j]),q=float(cq[j]),F=float(cf[j]),W=float(np.interp(ct[j],mt,w))))
        a=ax[0,i];a.errorbar(rw,ry,yerr=column(rates,'formal_error'),fmt='none',ecolor='gray',alpha=.5)
        sc=a.scatter(rw,ry,c=rt,cmap='viridis',s=35,vmin=0,vmax=mt[-1])
        fig.colorbar(sc,ax=a,label=r'$t\Omega_{ci}$')
        a.set(title=f'{label}: raw r={raw:.2f}',xlabel=r'Mean dominant-mode power $W_k$',ylabel=r'$-d(1/\kappa_{eff})/d(t\Omega_{ci})$')
        a.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax[1,i].errorbar(rt,ry,yerr=column(rates,'formal_error'),fmt='o',color='black',ms=3,label='window slopes')
        for kind,color in [('time','#888888'),('time2','#D55E00'),('time_wave','#0072B2')]:
            ax[1,i].plot(rt,design(rt,rw,kind)@rf[kind]['beta'],color=color,label=kind)
        ax[1,i].set(xlabel=r'$t\Omega_{ci}$',ylabel='Rate and full-sample fits')
        ax[1,i].ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax[1,i].legend(fontsize=8)
        trend=design(rt,rw,'time')
        wr=rw-trend@np.linalg.lstsq(trend,rw,rcond=None)[0]
        yr=ry-trend@np.linalg.lstsq(trend,ry,rcond=None)[0]
        ax[2,i].scatter(wr,yr,c=rt,cmap='viridis',s=30,vmin=0,vmax=mt[-1])
        ax[2,i].axhline(0,color='gray',lw=.6);ax[2,i].axvline(0,color='gray',lw=.6)
        ax[2,i].set(title=f'After linear time trend: r={partial:.2f}',
                    xlabel='Mode power residual',ylabel='Rate residual')
        ax[2,i].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ca=cx[0,i];sc=ca.scatter(cf,cq[0]-cq,c=ct,cmap='viridis',s=14)
        cum.colorbar(sc,ax=ca,label=r'$t\Omega_{ci}$')
        ca.set(title=label,xlabel=r'$F_k=\int_{t_0}^t W_k\,d(t\Omega_{ci})$',ylabel=r'$q(t_0)-q(t)$, $q=1/\kappa_{eff}$')
        ca=cx[1,i];ca.plot(ct,cq,'k.',ms=3,label='observed q')
        for kind,color in [('time','#888888'),('time2','#D55E00'),('time_wave','#0072B2')]:
            ca.plot(ct,design(ct-ct[0],cf,kind)@cf_fit[kind]['beta'],color=color,label=kind)
        ca.set(xlabel=r'$t\Omega_{ci}$',ylabel='q and full-sample fits');ca.legend(fontsize=8)
        with np.load(run/'03_particles/vdf_shape_series.npz') as sh:
            st=sh['t'];u=(sh['edges'][1:]+sh['edges'][:-1])/2
            density=sh['par_density'];counts=sh['par_counts'];sigma=sh['par_sigma_vA']
            valid=(counts>=30)&(counts[0]>=30)[None,:]&(density>0)&(density[0]>0)[None,:]
            ratio=np.divide(density,density[0],out=np.full_like(density,np.nan),where=density[0]>0)
            change=np.log10(np.where(valid,ratio,np.nan))
            a=sx[0,i];image=a.pcolormesh(st,u,change.T,shading='nearest',cmap='RdBu_r',vmin=-.35,vmax=.35)
            tr=sorted((r for r in theory if r['run']==run.name),key=lambda r:float(r['t']))
            tt=column(tr,'t');omega=column(tr,'omega_r');valid_t=(tt>=st[0])&(tt<=st[-1])
            tt=tt[valid_t];omega=omega[valid_t]
            # Theory omega is normalized to Omega_ci, k to 1/di; vA=Omega_ci*di.
            ur=np.abs((omega-1)/kpar)/np.interp(tt,st,sigma)
            a.plot(tt,ur,'k--',lw=1,label='theory |resonance| / sigma')
            a.plot(tt,-ur,'k--',lw=1)
            a.set(title=label,xlabel=r'$t\Omega_{ci}$',ylabel=r'$u=(v_\parallel-\langle v_\parallel\rangle)/\sigma_\parallel(t)$',ylim=(-4,4))
            a.legend(fontsize=7,loc='upper right')
    for f,name,title in [(fig,'mode_power_vs_kappa_rate','Dominant global mode vs prt kappa rate | disjoint 10-unit windows | bars: formal particle errors only'),
                         (cum,'mode_fluence_vs_kappa_change','Accumulated mode power and shape change | t0 = first particle snapshot at or after 10')]:
        f.suptitle(title,fontsize=12);f.tight_layout(rect=(0,0,1,.95));f.savefig(out/f'{name}.png',dpi=170);plt.close(f)
    shape.suptitle('VDF shape change from t=0 (variance normalized); dashed curves: theoretical resonance guides',fontsize=12)
    shape.colorbar(image,ax=sx.ravel().tolist(),label=r'$\log_{10}[g(u,t)/g(u,0)]$',shrink=.8)
    shape.savefig(out/'vdf_shape_resonance.png',dpi=170);plt.close(shape)
    (out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    dump_csv(out/'rate_windows.csv',allrates);dump_csv(out/'sensitivity.csv',sensitivity);dump_csv(out/'paired_series.csv',pairs)
    for r in results:
        print(r['run'],'raw / linear-time partial / quadratic-time partial r:',*[round(r[k],3) for k in ['rate_raw_r','rate_partial_r_linear_time','rate_partial_r_quadratic_time']])
        for category in ['rate_models','cumulative_models']:
            print(category,{k:round(v['forward_rmse'],7) for k,v in r[category].items()})
    print(out)

if __name__=='__main__':main()
