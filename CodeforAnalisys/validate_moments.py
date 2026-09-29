#!/usr/bin/env python3
"""Validate initial particle moments using the actual output window.

PSC stores u=gamma*v. Temperatures below are the initialized m*Var(u), not
relativistic pressure. Density uses weights, runtime cori and file lo/hi.
HDF5 is reduced in chunks; no particle subsampling is used for density.
"""
import argparse
import csv
import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_style as ps
from data_reader import PICDataReader
from psc_units import (SIM_PROFILE, DOMAIN_DI_Y, DOMAIN_DI_Z, N_GRID_Y, N_GRID_Z,
                       NICELL, CORI, N0, TI_PAR, TI_PERP, TE_PAR, TE_PERP,
                       M_ION, M_ELEC, RUN_PARAMETER_SOURCES)


def particle_chunks(path, chunk_size=250_000):
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if PICDataReader.is_adios2_path(path):
        arrays = PICDataReader.read_particles_snapshot(str(path), max_particles=2**63-1)
        n = len(arrays[0])
        for start in range(0, n, chunk_size):
            yield dict(zip(('q','m','px','py','pz','w'), (a[start:start+chunk_size] for a in arrays)))
    else:
        with h5py.File(path, 'r') as f:
            ds = f['particles/p0/1d']
            required = ['q','m','px','py','pz']
            if not set(required) <= set(ds.dtype.names or ()):
                raise ValueError('Particle dataset lacks charge/mass/momentum fields')
            columns = required + (['w'] if 'w' in ds.dtype.names else [])
            for start in range(0, len(ds), chunk_size):
                data = ds.fields(columns)[start:start+chunk_size]
                row = {k: np.asarray(data[k], dtype=float) for k in columns}
                row.setdefault('w', np.ones(len(data)))
                yield row


def measure_particles(path, cori=CORI, chunk_size=250_000):
    lo, hi = PICDataReader.read_prt_window(str(path))
    if (lo.shape != (3,) or hi.shape != (3,) or np.any(lo < 0)
            or np.any(hi <= lo) or np.any(hi > [1,N_GRID_Y,N_GRID_Z])):
        raise ValueError(f'Invalid particle window lo={lo}, hi={hi} for current grid')
    cells = int(np.prod(hi-lo))
    if not np.isfinite(cori) or cori <= 0:
        raise ValueError('cori must be positive')
    stats = {name: dict(count=0, weight=0., weight2=0., sum=np.zeros(3), sum2=np.zeros(3),
                        sum3=np.zeros(3), sum4=np.zeros(3))
             for name in ('ion','electron')}
    for row in particle_chunks(path, chunk_size):
        if any(not np.all(np.isfinite(v)) for v in row.values()) or np.any(row['w'] < 0):
            raise ValueError('Nonfinite particle values or negative weights')
        if np.any(row['q'] == 0):
            raise ValueError('Neutral particles are unsupported by this two-species validator')
        for name, mask, mass in [('ion',row['q']>0,M_ION),('electron',row['q']<0,M_ELEC)]:
            if not np.allclose(np.abs(row['m'][mask]), mass):
                raise ValueError(f'{name}: particle mass disagrees with profile')
            w = row['w'][mask]; u = np.stack([row[k][mask] for k in ('px','py','pz')],axis=1)
            st = stats[name]; st['count'] += len(w); st['weight'] += float(w.sum())
            st['weight2'] += float(np.dot(w, w))
            for power in (1, 2, 3, 4):
                st['sum' if power == 1 else f'sum{power}'] += np.sum(w[:,None]*u**power,axis=0)
    result = {}
    for name, mass in [('ion',M_ION),('electron',M_ELEC)]:
        st = stats[name]
        if st['weight'] <= 0:
            raise ValueError(f'No positive-weight {name} particles')
        W=st['weight']; mean=st['sum']/W; m2=np.maximum(st['sum2']/W-mean**2,0)
        # Central fourth moment: the sampling error of a variance is
        # sqrt((m4 - m2^2)/N_eff), which kappa tails make much larger than
        # the Gaussian sqrt(2/N_eff).
        m4=st['sum4']/W-4*mean*st['sum3']/W+6*mean**2*st['sum2']/W-3*mean**4
        n_eff=W**2/st['weight2']; temp=mass*m2
        se=mass*np.sqrt(np.maximum(m4-m2**2,0)/n_eff)
        t_par, t_perp = float(temp[2]), float((temp[0]+temp[1])/2)
        se_par, se_perp = float(se[2]), float(.5*np.hypot(se[0], se[1]))
        result[name] = {'count':st['count'], 'total_weight':st['weight'], 'n_effective':float(n_eff),
                        'n':st['weight']*cori/cells, 'mean_u':mean.tolist(),
                        'T_parallel':t_par, 'T_perp':t_perp,
                        'A':t_perp/t_par if t_par>0 else None,
                        'sampling_se_rel':{'T_parallel':se_par/t_par if t_par>0 else None,
                                           'T_perp':se_perp/t_perp if t_perp>0 else None,
                                           'A':float(np.hypot(se_par/t_par, se_perp/t_perp)) if t_par>0 and t_perp>0 else None,
                                           'bulk_u_over_vth':float(np.sqrt(np.sum(m2)/(n_eff*m2[2]))) if m2[2]>0 else None}}
    return result, {'lo':lo.tolist(),'hi':hi.tolist(),'sampled_cells':cells,
                    'domain_cells':N_GRID_Y*N_GRID_Z,'cori':cori}


#: Allowed deviation in units of the sampling standard error. The fixed
#: tolerances below are what a production window (~10^7 particles) must meet;
#: a small window is instead limited by its own shot noise, and a correct
#: initialization must not FAIL for having few particles.
SAMPLING_SIGMAS = 4.


def validation_rows(measured, step):
    rows=[]
    for species,tp,tt,mass in [('ion',TI_PAR,TI_PERP,M_ION),('electron',TE_PAR,TE_PERP,M_ELEC)]:
        se=measured[species].get('sampling_se_rel',{})
        for key,expected,tol in [('n',N0,1.),('T_parallel',tp,2.),('T_perp',tt,2.),('A',tt/tp,2.)]:
            val=measured[species][key]
            err=abs(val/expected-1)*100 if val is not None and step==0 else None
            stat=100*se[key] if se.get(key) is not None else None
            allowed=max(tol,SAMPLING_SIGMAS*stat) if stat is not None else tol
            rows.append(dict(species=species,quantity=key,measured=val,initial_expected=expected,
                             relative_error_pct=err,tolerance_pct=tol,sampling_se_pct=stat,
                             effective_tolerance_pct=allowed,
                             status='NOT_INITIAL' if step!=0 else 'PASS' if err is not None and err<allowed else 'FAIL'))
        drift=float(np.linalg.norm(measured[species]['mean_u'])/np.sqrt(tp/mass))
        stat=se.get('bulk_u_over_vth')
        allowed=max(.01,SAMPLING_SIGMAS*stat) if stat is not None else .01
        rows.append(dict(species=species,quantity='bulk_u_over_initial_vth',measured=drift,
                         initial_expected=0.,relative_error_pct=None,tolerance_pct=1.,
                         sampling_se_pct=100*stat if stat is not None else None,
                         effective_tolerance_pct=100*allowed,
                         status='NOT_INITIAL' if step!=0 else 'PASS' if drift<allowed else 'FAIL'))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('filepath',type=Path)
    ap.add_argument('--outdir',type=Path,default=Path('validation_plots'))
    ap.add_argument('--run-name',default='')
    ap.add_argument('--chunk-size',type=int,default=250_000)
    args=ap.parse_args()
    step=PICDataReader.get_step_from_filename(str(args.filepath))
    measured,window=measure_particles(args.filepath,chunk_size=args.chunk_size)
    rows=validation_rows(measured,step)
    status='NOT_INITIAL' if step!=0 else 'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL'
    args.outdir.mkdir(parents=True,exist_ok=True)
    prefix=(args.run_name+'_') if args.run_name else ''
    report=dict(profile=SIM_PROFILE,source_file=str(args.filepath.resolve()),step=step,status=status,
                initial_state_available=step==0,window=window,grid=[N_GRID_Y,N_GRID_Z],
                domain_di=[DOMAIN_DI_Y,DOMAIN_DI_Z],nicell=NICELL,parameter_sources=RUN_PARAMETER_SOURCES,
                temperature_definition='m Var(u); u=gamma*v; initialization diagnostic',species=measured,checks=rows)
    (args.outdir/f'{prefix}validation_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (args.outdir/f'{prefix}validation_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    ps.apply()
    fig,ax=plt.subplots(figsize=(12,6));ax.axis('off')
    data=[[r['species'],r['quantity'],f"{r['measured']:.6g}" if r['measured'] is not None else 'undefined',
           f"{r['initial_expected']:.6g}",r['status']] for r in rows]
    table=ax.table(cellText=data,colLabels=['Species','Quantity','Measured (code)','Initial reference','Status'],loc='center',cellLoc='center',colWidths=[.12,.36,.18,.19,.15])
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.8)
    for i,r in enumerate(rows,1):
        table[i,4].set_facecolor({'PASS':'#d7efdf','FAIL':'#f7d4d4','NOT_INITIAL':'#f6e7be'}[r['status']])
    ax.set_title(f'{SIM_PROFILE}: {status} · step={step}\n{DOMAIN_DI_Y:g} × {DOMAIN_DI_Z:g} dᵢ; grid {N_GRID_Y} × {N_GRID_Z}; sampled cells={window["sampled_cells"]}',fontsize=12)
    ps.save(fig,args.outdir/f'{prefix}validation_summary.png')
    print(f'Validation status: {status}; window cells={window["sampled_cells"]}')
    if step!=0:
        print('Initial snapshot missing: evolved moments were measured but not tested against initialization.')


if __name__=='__main__':
    main()
