"""Health-bounded frozen/live startup figures, without extrapolated science claims."""
import csv,json,sys
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from health import require_window,read_rows
root=Path(sys.argv[1]);PREF=461.99570043659014
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(12,8.5),layout='constrained')
for arm,color,style in [('frozen','#17649a','--'),('live','#be6d28','-')]:
    label='preflight_'+arm;run=root/'evidence/amd_runs'/label
    window=require_window(run,allow_stopped=False);out=root/'reduced'/label
    meta=json.loads((out/'reduction.json').read_text());limit=min(meta['tmax'],window['last_healthy_time'])
    def load(name):return [{k:float(v) for k,v in row.items()} for row in read_rows(out/name) if float(row['time'])<=limit+1.e-9]
    totals=load('integrated.csv');t=np.array([r['time'] for r in totals])
    for key,marker in [('Sr','o'),('Stheta','s'),('Sphi','^')]:
        value=np.array([r[key] for r in totals]);axes[0,0].plot(t,100*(value/value[0]-1),style,marker=marker,ms=3,color=color,label=arm+' '+key)
    radii=load('radii.csv')
    for q,marker in [(.1,'o'),(.5,'s'),(.9,'^')]:
        values=[r for r in radii if abs(r['quantile']-q)<1.e-10]
        rt=np.array([r['time'] for r in values]);initial=values[0]['r_areal']
        axes[0,1].plot(rt,[100*(r['r_areal']/initial-1) for r in values],style,marker=marker,ms=3,color=color,label=f'{arm} q{int(q*100)}')
        lo=[100*(r['r_lower']/values[0]['r_upper']-1) for r in values]
        hi=[100*(r['r_upper']/values[0]['r_lower']-1) for r in values]
        axes[0,1].fill_between(rt,lo,hi,color=color,alpha=.06)
    h=load('health.csv');ht=np.array([r['time'] for r in h])
    axes[0,2].plot(ht,[r['alpha_min'] for r in h],style,color=color,label=arm)
    axes[1,0].semilogy(ht,[r['mass_error'] for r in h],style,color=color,label=arm)
    if arm=='live':
        axes[1,1].plot(ht,[r['H_core_L2'] for r in h],color=color,label='H core L2')
        axes[1,1].plot(ht,[r['M_core_L2'] for r in h],':',color=color,label='M core L2')
    if (out/'invariants.csv').exists():
        inv=load('invariants.csv');it=[r['time'] for r in inv]
        for key,label in [('energy_relative_rms','Relative energy RMS'),('angular_vector_rms_normalized','Angular-vector RMS')]:
            axes[1,2].semilogy(it,[max(r[key],1.e-18) for r in inv],label=label)
axes[1,0].axhline(1.e-10,color='.4',lw=.8,label='Health limit');axes[1,0].set_ylim(1.e-15,2.e-10)
titles=['Physical-stress drift / %','Mass-radius drift / % (histogram bounds shaded)','Minimum lapse',
        'Relative rest-mass error','Live constraints (r ≤ 100M)','Frozen orbit error']
for ax,title in zip(axes.flat,titles):
    ax.set_title(title,pad=50,fontsize=10);ax.set_xlabel('Time / M');ax.legend(fontsize=7)
    # Every figure displays both the physical time and common reference clock.
    top=ax.secondary_xaxis('top',functions=(lambda t:t/PREF,lambda p:p*PREF));top.set_xlabel('Time / P_ref',fontsize=8)
fig.suptitle('Matched AMD 100-step preflights · full startup · insufficient for equilibrium classification')
for ext in ('png','pdf'):fig.savefig(root/'figures'/('preflight_health.'+ext),dpi=160)
plt.close(fig)
