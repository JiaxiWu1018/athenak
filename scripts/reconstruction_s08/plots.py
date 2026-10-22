"""Publication files from fully audited Session 8 reductions; no synthetic rows."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from common import *
from reduce import csv_rows,write_csv

def style():
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'dejavuserif','font.size':10,
      'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':9,'axes.linewidth':.7,
      'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
      'savefig.dpi':220,'pdf.fonttype':42,'ps.fonttype':42})

def save(fig,name):
    for ext in ['pdf','png']:fig.savefig(ROOT/'plots'/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

def array(rows,key):return np.array([r[key] for r in rows])

def shell_limits(*datasets):
    values=np.array([row[f'r{q}'] for rows in datasets for row in rows
                     for q in range(10,100,10)])
    assert values.size and np.isfinite(values).all() and (values>0).all()
    # Keep the historical reference range, expanding it to include every curve.
    return (min(.05,float(values.min())/1.08),max(30.,float(values.max())*1.08))


def main():
    job();style();(ROOT/'plots').mkdir(exist_ok=True);(ROOT/'reports').mkdir(exist_ok=True)
    cases=CASES[::-1];data={}
    for c in cases:
        d=ROOT/'reduced'/c['id'];s=load(d/'summary.json');assert s['complete']
        assert (d/'REDUCTION_OK.json').exists()
        data[c['id']]=dict(summary=s,shell=csv_rows(d/'shells.csv'),mass=csv_rows(d/'mass.csv'),history=csv_rows(d/'history.csv'))
        if c['unstable']:data[c['id']]['ah']=csv_rows(d/'ah.csv')
    radius_limits=shell_limits(*(d['shell'] for d in data.values()))
    fig,axs=plt.subplots(2,5,figsize=(15,5.8),sharex=True,sharey='row',layout='constrained')
    cm=plt.get_cmap('cividis');norm=Normalize(.1,.9)
    for j,c in enumerate(cases):
        d=data[c['id']];s=d['summary'];rows=d['shell'];h=d['history']
        for q in range(10,100,10):axs[0,j].plot(array(rows,'time'),array(rows,f'r{q}'),color=cm(norm(q/100)),lw=1)
        axs[0,j].set(yscale='log',ylim=radius_limits,title=rf'$y_c={c["yc"]:.3f}$')
        axs[1,j].plot(array(h,'time'),array(h,'core_lapse'),color='#275d7f',lw=1.2)
        axs[1,j].set(ylim=(0,1),xlabel=r'$t/M$')
        for ax in axs[:,j]:
            ax.set(xlim=(0,300),xticks=[0,100,200,300]);ax.grid(alpha=.15,lw=.5)
            if s['first_removal'] is not None:ax.axvline(s['first_removal'],color='.45',ls=':',lw=.8)
            if s['ah_first'] is not None:ax.axvline(s['ah_first'],color='.15',ls='--',lw=.8)
            if s['endpoint']<299.99:ax.axvspan(s['endpoint'],300,color='.96',zorder=-5)
        if s['endpoint']<299.99:axs[1,j].text(.97,.95,f"Run ends at {s['endpoint']:.1f}M",ha='right',va='top',transform=axs[1,j].transAxes,fontsize=8)
    axs[0,0].set_ylabel(r'COM rest-mass shell radius $r_q/M$')
    axs[1,0].set_ylabel(r'Core lapse $\langle\alpha\rangle_{r<0.75M}$')
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cm),ax=axs[0,:],fraction=.018,pad=.01,ticks=[.1,.5,.9]);cb.set_label('Enclosed rest-mass fraction')
    save(fig,'s08_shells_lapse')
    fig,axs=plt.subplots(2,5,figsize=(15,5.4),sharex=True,sharey='row',layout='constrained')
    maximum=max(abs(r['closure']) for d in data.values() for r in d['mass']);extent=max(1e-15,maximum*1.2)
    for j,c in enumerate(cases):
        d=data[c['id']];m=d['mass'];t=array(m,'time');m0=m[0]['m_initial'];w=m0/2000000
        axs[0,j].plot(t,array(m,'m_alive')/m0,color='#275d7f',label='Alive',lw=1.2)
        removed=sum(array(m,'n_'+k) for k in ['sphere','lapse','horizon'])/2000000
        axs[0,j].plot(t,removed,color='#b35c26',label='Removed',lw=1.1)
        axs[0,j].plot(t,array(m,'n_exit')/2000000,color='#686868',ls=':',label='Escaped',lw=1.1)
        axs[0,j].plot(t,1+array(m,'closure'),color='.15',ls='--',label='Accounted total',lw=.8)
        axs[0,j].set(title=rf'$y_c={c["yc"]:.3f}$',ylim=(-.03,1.04))
        if np.all(removed==0) and np.all(array(m,'n_exit')==0):
            axs[0,j].text(.5,.16,'Removed = escaped = 0',ha='center',transform=axs[0,j].transAxes,fontsize=7.5)
        if np.allclose(array(m,'m_alive')/m0,1,rtol=0,atol=1e-12):
            axs[0,j].text(.5,.84,'Alive = accounted total',ha='center',transform=axs[0,j].transAxes,fontsize=7.5)
        axs[1,j].plot(t,array(m,'closure'),color='#275d7f',lw=.9)
        axs[1,j].axhline(0,color='.6',lw=.6);axs[1,j].set(ylim=(-extent,extent),xlabel=r'$t/M$')
        axs[1,j].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        for ax in axs[:,j]:ax.set(xlim=(0,300),xticks=[0,100,200,300]);ax.grid(alpha=.15,lw=.5)
    axs[0,0].set_ylabel(r'Rest mass / initial $M_0$');axs[1,0].set_ylabel(r'$M_{0,\mathrm{accounted}}/M_0(0)-1$')
    axs[0,2].legend(loc='center',frameon=False)
    save(fig,'s08_mass_conservation')
    fig,axs=plt.subplots(1,3,figsize=(11,3.4),sharey=True,layout='constrained')
    for ax,c in zip(axs,[c for c in cases if c['unstable']]):
        d=data[c['id']];ah=d['ah'];s=d['summary'];t=array(ah,'time');ok=array(ah,'accepted')==1
        for key,color,ls,label in [('mirr','#275d7f','-',r'$M_{\rm irr}$'),('mass','#b35c26','--',r'$M_{\rm CH}$')]:
            v=array(ah,key);v[~ok]=np.nan;ax.plot(t,v,color=color,ls=ls,lw=1.2,label=label)
        ax.set(title=rf'$y_c={c["yc"]:.3f}$',xlabel=r'$t/M$',ylim=(0,1.15),xlim=(0,s['endpoint']))
        ax.grid(alpha=.15,lw=.5)
        if not ok.any():ax.text(.5,.5,'No accepted AH',ha='center',transform=ax.transAxes)
    axs[0].set_ylabel(r'AH mass / initial gravitational $M$');axs[-1].legend(frameon=False)
    save(fig,'s08_ah_masses')
    table=[]
    for c in cases:
        s=data[c['id']]['summary'];tail=s['ah_endpoint_window']
        table.append(dict(yc=c['yc'],t_end=s['endpoint'],t_first_removal=s['first_removal'],
            t_first_accepted_AH=s['ah_first'],M0_initial=s['rest_mass_initial'],max_mass_closure=s['max_closure'],
            M_irr_last5M_median=None if tail is None else tail['mirr']['median'],
            M_CH_last5M_median=None if tail is None else tail['mass']['median']))
    write_csv(ROOT/'reports/production_summary.csv',table)
    captions='''# Session 8 figure captions

All times and radii use the initial gravitational mass M=1. Each run has 2,000,000 equal-rest-mass particles, proper-profile stratified sampling, and seed 1. Shell radii are coordinate radii about the instantaneous rest-mass COM, not areal radii or tracked particle surfaces.

**s08_shells_lapse.** Top: radii enclosing 10–90% of the rest mass, truncated strictly before the first removal. Bottom: coordinate-volume average of lapse within origin-centered r<0.75M. Dotted lines mark first removal; dashed lines mark first accepted AH. Grey areas lie beyond the actual run endpoint. All columns share the same axes. The common radius range includes every audited shell radius with an 8% margin beyond data outside the reference range.

**s08_mass_conservation.** Alive particle rest mass plus cumulative removed and escaped rest mass, normalized by the initial rest mass. Removed particles include lapse, sphere, and AH removal categories (production uses lapse removal only). Lower panels show the double-precision ledger residual. This is rest-mass bookkeeping, not ADM-energy conservation; AH gravitational mass is not added to this ledger.

**s08_ah_masses.** Irreducible mass sqrt(A/16pi) and Christodoulou mass from accepted finder surfaces. Missing or rejected searches are gaps. The three panels use their actual run endpoints, so horizontal ranges differ. The accompanying table gives accepted-sample medians in the final 5M, not a claim of settled mass. Acceptance requires a fully on-grid, resolved surface, scaled expansion residual <=0.05, and three successive passing searches.

Session 7 supplies particle-number sensitivity evidence. This session contains no new convergence study and does not establish convergence of the adaptive production mesh.
'''
    (ROOT/'reports/FIGURES.md').write_text(captions)
    atomic(ROOT/'evidence/PLOTS_RENDERED.json',dict(utc=now(),job=job(),files=[receipt(p) for p in sorted((ROOT/'plots').glob('*'))],shell_radius_limits_M=radius_limits,visual_review='pending'))
if __name__=='__main__':main()
