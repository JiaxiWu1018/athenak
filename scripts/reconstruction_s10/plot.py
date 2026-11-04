"""Standalone, source-bound scientific figures; run only in an allocation."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from campaign import ROOT
from analyze import read_csv, PRIMARY_SCHEME, BASE_H, PRIMARY_H_LABEL
from state_io import require_job, write_json, sha

NEW_LABEL='RK3 predictor, fixed nodes, no fallback' if PRIMARY_SCHEME!='predictor' else 'RK3 predictor'
COLORS={1:'#0072B2',2:'#D55E00',4:'#009E73',8:'#7B3294'}
CAVEAT=f'Archived RK4: h=0.2M, lagged sources. New: h={BASE_H:g}M, synchronized endpoints; fixed nodes for the revised scheme.'
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':10,
    'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180,'pdf.fonttype':42})


def arrays(path,key,until=500):
    rows=read_csv(path)
    rows=[r for r in rows if str(r.get('constraint_usable','True')).lower() in ('true','1') and float(r['time'])<=until+1e-9]
    t=np.array([float(r['time']) for r in rows]);v=np.array([float(r[key]) for r in rows])
    assert np.all(np.diff(t)>0) and np.all(np.isfinite(v)) and np.all(v>0),(path,key)
    return t,v


def save(fig,out,name,sources):
    for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),bbox_inches='tight',facecolor='white')
    plt.close(fig)
    write_json(out/(name+'.sources.json'),dict(files=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(sources))],script_sha256=sha(__file__)))


def constraint_figures(data,out):
    sources=[]
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
    ratio,ra=plt.subplots(2,3,figsize=(15,8),sharex=True)
    for col,(region,label,name) in enumerate([('r10','r < 10M','health.csv'),('r16','r < 16M','health.csv'),('global','Full domain','global_constraints.csv')]):
        for row,metric in enumerate(('H','M')):
            ax=axes[row,col];rx=ra[row,col]
            key=f'{metric}_rms'+('' if region=='global' else '_'+region)
            for n in (1,2,4,8):
                old=ROOT/f'reference/session7/reduced/mc_{n}m_s1'/name
                new=data/f'{PRIMARY_SCHEME}_{n}m_h{PRIMARY_H_LABEL}'/name
                t0,v0=arrays(old,key);t1,v1=arrays(new,key)
                assert t1[-1]>=500-1e-8
                ax.semilogy(t0,v0,color=COLORS[n],ls='--',lw=1.15,alpha=.8)
                ax.semilogy(t1,v1,color=COLORS[n],lw=1.4)
                valid=(t1>=t0[0])&(t1<=t0[-1]);tt=t1[valid]
                rx.plot(tt,v1[valid]/np.interp(tt,t0,v0),color=COLORS[n],lw=1.4)
                sources.extend((old,new))
            ax.set_title(('Hamiltonian' if metric=='H' else 'Momentum')+' · '+label)
            ax.set_ylabel(r'Constraint RMS [$M^{-2}$]');ax.grid(alpha=.18);ax.set_xlim(0,500)
            rx.set_title(('Hamiltonian' if metric=='H' else 'Momentum')+' · '+label)
            rx.set_ylabel('New / archived RMS');rx.axhline(1,color='.4',lw=1,ls=':');rx.grid(alpha=.18);rx.set_xlim(0,500)
            if row==1:ax.set_xlabel('Time [M]');rx.set_xlabel('Time [M]')
    handles=[Line2D([],[],color=COLORS[n],label=f'{n}M particles') for n in COLORS]
    handles.extend([Line2D([],[],color='.25',ls='-',label=NEW_LABEL),Line2D([],[],color='.25',ls='--',label='Archived RK4')])
    fig.legend(handles=handles,loc='upper center',ncol=3,bbox_to_anchor=(.5,.98),frameon=False)
    fig.suptitle('ST yc = 0.819 · constraint evolution at fixed grid spacing 0.5M',y=1.02)
    fig.text(.5,.01,CAVEAT,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.88));save(fig,out,'constraint_evolution',sources)
    ratio.legend(handles=handles[:4],loc='upper center',ncol=4,bbox_to_anchor=(.5,.98),frameon=False)
    ratio.suptitle('ST yc = 0.819 · constraint ratios relative to the archived scheme',y=1.02)
    ratio.text(.5,.01,CAVEAT,ha='center',fontsize=10)
    ratio.tight_layout(rect=(0,.035,1,.93));save(ratio,out,'constraint_ratios',sources)


def particle_scaling(data,out):
    source=data/'constraint_windows.csv';rows=read_csv(source)
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for col,(lo,hi) in enumerate(((20,100),(100,300),(300,500))):
        for row,metric in enumerate(('H','M')):
            ax=axes[row,col]
            group=sorted([r for r in rows if r['region']=='r10' and r['constraint']==metric and int(r['start'])==lo and int(r['end'])==hi],key=lambda r:int(r['N']))
            assert len(group)==4
            n=np.array([int(r['N'])/1e6 for r in group])
            for scheme,key,color,style in [('Predictor','predictor_mean','#0072B2','o-'),('Archived','archived_mean','#D55E00','s--')]:
                v=np.array([float(r[key]) for r in group]);alpha=-np.polyfit(np.log(n),np.log(v),1)[0]
                ax.loglog(n,v,style,color=color,label=f'{scheme}: α = {alpha:.2f}')
                if scheme=='Predictor':ax.loglog(n,v[0]*n**(-.5),':',color='.4',label=r'$N^{-1/2}$ reference')
            ax.set_title(f'{metric} RMS · {lo}–{hi}M');ax.set_ylabel(r'Time-mean RMS [$M^{-2}$]');ax.grid(alpha=.18)
            ax.set_xticks([1,2,4,8]);ax.set_xticklabels(['1','2','4','8']);ax.legend(frameon=False,fontsize=9)
            if row==1:ax.set_xlabel('Particle count [million]')
    fig.suptitle('Particle-number scaling inside r < 10M · RMS ∝ N⁻ᵅ',y=1.01)
    fig.text(.5,.01,'One nested seed at fixed mesh and time step. Particle-number scaling is not temporal order.\n'+CAVEAT,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.98));save(fig,out,'particle_number_scaling',[source])


def temporal_figures(data,out):
    source=data/'temporal_order.csv';rows=read_csv(source)
    choices=[('grid','chi','r10','Conformal factor χ'),('grid','alpha','r10','Lapse α'),
             ('particles','position','all','Particle positions'),('particles','covariant_momentum','all','Particle momenta')]
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for ax,(kind,field,region,title) in zip(axes.flat,choices):
        for scheme,color,marker in [('legacy','#D55E00','s'),(PRIMARY_SCHEME,'#0072B2','o')]:
            subset=[r for r in rows if (r['scheme'],r['kind'],r['field'],r['region'])==(scheme,kind,field,region) and r['observed_p']]
            assert len(subset)==3,(scheme,field)
            ax.plot([float(r['time']) for r in subset],[float(r['observed_p']) for r in subset],marker+'-',color=color,label='Legacy RK4 (fallback allowed)' if scheme=='legacy' else NEW_LABEL)
        ax.axhline(1,color='.45',ls=':',lw=1);ax.axhline(2,color='.45',ls='--',lw=1)
        ax.set_title(title);ax.set_ylabel('Observed temporal order p');ax.grid(alpha=.15);ax.set_xticks([20,50,100]);ax.legend(frameon=False)
    for ax in axes[-1]:ax.set_xlabel('Matched endpoint [M]')
    fig.suptitle(f'Temporal self-convergence · 1M particles · legacy h=.2/.1/.05; new h={BASE_H:g}/{BASE_H/2:g}/{BASE_H/4:g}M',y=1.01)
    fig.text(.5,.01,'p = log₂(‖Qₕ − Qₕ/₂‖ / ‖Qₕ/₂ − Qₕ/₄‖). Double-precision fields and matched particle IDs.\nNominal maximum timesteps include restart transients; legacy runs allow Euler fallback.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.98));save(fig,out,'temporal_order',[source])
    # Differences as a function of h make a reported p inspectable, with reference slopes.
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(kind,field,region,title) in zip(axes.flat,choices):
        for scheme,color,marker in [('legacy','#D55E00','s'),(PRIMARY_SCHEME,'#0072B2','o')]:
            r=next(r for r in rows if (r['scheme'],r['kind'],r['field'],r['region'],float(r['time']))==(scheme,kind,field,region,100.))
            h=float(r.get('nominal_h',.2));x=np.array([h/2,h]);y=np.array([float(r['diff_h2_h4']),float(r['diff_h_h2'])])
            ax.loglog(x,y,marker+'-',color=color,label='Legacy RK4 (fallback allowed)' if scheme=='legacy' else NEW_LABEL)
            if scheme==PRIMARY_SCHEME:
                for p,style in [(1,':'),(2,'--')]:ax.loglog(x,y[0]*(x/x[0])**p,style,color='.5',label=f'Order {p} reference')
        ax.set_title(title);ax.set_xlabel('Coarser time step h [M]');ax.set_ylabel('RMS difference ‖Qₕ − Qₕ/₂‖');ax.grid(alpha=.15)
        ticks=sorted(set([.1,.2,BASE_H/2,BASE_H]));ax.set_xticks(ticks);ax.set_xticklabels([f'{v:g}' for v in ticks]);ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Matched solution differences at 100M',y=1.01);fig.tight_layout();save(fig,out,'temporal_differences',[source])


def controls(data,out):
    sources=[];fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharex=True)
    for case,label,color,style in [(f'legacy_1m_h{PRIMARY_H_LABEL}','Legacy RK4 (fallback allowed)','#D55E00','--'),(f'{PRIMARY_SCHEME}_1m_h{PRIMARY_H_LABEL}',NEW_LABEL,'#0072B2','-')]:
        path=data/case/'health.csv';sources.append(path)
        for ax,metric in zip(axes,('H','M')):
            t,v=arrays(path,f'{metric}_rms_r10',100);ax.semilogy(t,v,style,color=color,label=label)
            ax.set_title(('Hamiltonian' if metric=='H' else 'Momentum')+' · r < 10M');ax.set_xlabel('Time [M]');ax.set_ylabel(r'Constraint RMS [$M^{-2}$]');ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.suptitle(f'Integrator/coupling controls · 1M particles · matched nominal h={BASE_H:g}M',y=1.02)
    fig.text(.5,.01,'Synchronized endpoint constraints. Frozen-source RK3 at h=.2M failed at 18.4M; no valid 20M endpoint.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.98));save(fig,out,'short_controls',sources)


def main():
    require_job();ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    constraint_figures(a.data,a.out);particle_scaling(a.data,a.out);temporal_figures(a.data,a.out);controls(a.data,a.out)
    write_json(a.out/'FIGURES.json',dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(a.out.iterdir()) if p.suffix in ('.png','.pdf')]))

if __name__=='__main__':main()
